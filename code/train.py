# 内置
import os
import warnings
# dgl
import dgl
import hydra
# 机器学习
import numpy as np
# torch
import torch
import torch.nn.functional as F
# 工程化、自建和其他
import wandb
from dgl import function as fn
from dgl.data.utils import load_graphs
from dgl.utils import expand_as_pair, dgl_warning
from omegaconf import DictConfig, OmegaConf
# pl
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.metrics import (roc_auc_score, average_precision_score)
from torch import nn
from tqdm import tqdm

from myutils import describe, mask_to_index, set_all_seed, cal_metrics, bin_encoding2

warnings.filterwarnings("ignore")
print(os.getcwd())


class IntraConv_single(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 add_self=True,
                 bias=True,
                 norm=None,
                 activation=None):
        super(IntraConv_single, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype=None, edge_weight=None):
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.srcdata['degree'] = torch.ones((graph.num_src_nodes(), 1)).to(feat.device)
                graph.edata['_edge_weight'] = edge_weight
                msg_fn1 = fn.u_mul_e('h', '_edge_weight', 'm')
                msg_fn2 = fn.u_mul_e('degree', '_edge_weight', 'degree')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feaddts).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            if edge_weight is not None:
                graph.update_all(msg_fn1, fn.sum('m', 'neigh'))
                graph.update_all(msg_fn2, fn.sum('degree', 'degree'))
                h_neigh = graph.dstdata['neigh'] / (graph.dstdata['degree'] + torch.FloatTensor([1e-8]).to(feat.device))
            else:
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']

            # h_neigh = torch.concat([h_neigh_mean,h_neigh_sum],axis=-1)
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
            h_self = self.fc_self(h_self)
            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh
            # rst = torch.concat([rst,graph.dstdata['degree']],-1)
            # bias term
            if self.bias is not None:
                rst = rst + self.bias
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst  # , h_self


class IntraConv_multi(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 add_self=True,
                 bias=True,
                 norm=None,
                 activation=None):
        super(IntraConv_multi, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.add_self = add_self
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, etype, edge_weight=None):
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges(etype=etype)
                graph.srcdata['degree'] = torch.ones((graph.num_src_nodes(), 1)).to(feat.device)
                graph.edata['_edge_weight'] = {etype: edge_weight}
                msg_fn1 = fn.u_mul_e('h', '_edge_weight', 'm')
                msg_fn2 = fn.u_mul_e('degree', '_edge_weight', 'degree')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats
            msg_fn = fn.copy_u('h', 'm')
            # Message Passing
            graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
            if edge_weight is not None:
                graph.multi_update_all({
                    etype: (msg_fn1, fn.sum('m', 'neigh'))
                },
                    'sum'
                )
                graph.multi_update_all({
                    etype: (msg_fn2, fn.sum('degree', 'degree'))
                },
                    'sum'
                )
                h_neigh = graph.dstdata['neigh'] / (graph.dstdata['degree'] + torch.FloatTensor([1e-8]).to(feat.device))
            else:
                graph.multi_update_all({
                    etype: (msg_fn, fn.mean('m', 'neigh'))
                },
                    'sum'
                )

                h_neigh = graph.dstdata['neigh']

            # h_neigh = torch.concat([h_neigh_mean,h_neigh_sum],axis=-1)
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
            h_self = self.fc_self(h_self)
            if self.add_self:
                rst = h_self + h_neigh
            else:
                rst = h_neigh
            # rst = torch.concat([rst,graph.dstdata['degree']],-1)
            # bias term
            if self.bias is not None:
                rst = rst + self.bias
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst  # , h_self


# 构建dataloader
class DataModule(LightningDataModule):
    def __init__(self, graph, batch_size, n_classes):
        super().__init__()

        trn_sampler = dgl.dataloading.NeighborSampler(
            [-1]  # , prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )
        val_sampler = dgl.dataloading.NeighborSampler(
            [-1]  # , prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )
        self.g = graph
        self.trn_idx, self.val_idx, self.tst_idx = mask_to_index(graph.ndata['trn_msk']), mask_to_index(
            graph.ndata['val_msk']), mask_to_index(graph.ndata['tst_msk'])
        self.trn_sampler = trn_sampler
        self.val_sampler = val_sampler
        self.batch_size = batch_size
        self.n_classes = n_classes

    def train_dataloader(self):
        loader = dgl.dataloading.DataLoader(
            self.g,
            self.trn_idx.to('cuda'),
            self.trn_sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )
        return loader

    def val_dataloader(self):
        loader = dgl.dataloading.DataLoader(
            self.g,
            torch.arange(self.g.num_nodes()).to('cuda'),
            self.val_sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )
        return loader


class DGA(nn.Module):
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes, p=0.3, n_head=1,
                  unclear_up=0.1, unclear_down=0.1):
        """Initialize the SAGE model with the given parameters."""
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_etypes = n_etypes
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('super_mask', torch.ones((num_nodes, self.n_classes)))
        self.n_head = n_head

        # 特征编码
        hidden_units = [n_hidden, n_hidden]
        input_size = in_feats
        hidden_unit = input_size
        all_layers = []
        for hidden_unit in hidden_units:
            layer = nn.Linear(input_size, hidden_unit)
            all_layers.append(nn.Dropout(p))
            all_layers.append(layer)
            all_layers.append(nn.BatchNorm1d(hidden_unit))  # 加入bn
            all_layers.append(nn.ReLU())
            input_size = hidden_unit
            self.last_dim = hidden_unit
        self.emb_layer = nn.Sequential(*all_layers)

        # 分组器损失
        all_layers = []
        all_layers.append(nn.Linear(n_hidden, self.n_classes))
        self.emb_layer_fc = nn.Sequential(*all_layers)

        # 输出层
        all_layers = []
        all_layers.append(nn.Linear(self.n_head * n_hidden, n_hidden // 2))
        all_layers.append(nn.ReLU())
        all_layers.append(nn.Linear(n_hidden // 2, self.n_classes))
        self.final_fc_layer = nn.Sequential(*all_layers)

        self.attn_fn = nn.Tanh()
        self.W_f = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)
        self.W_x = nn.Sequential(nn.Linear(n_hidden, n_hidden * self.n_head), self.attn_fn)

        self.reset_parameters()
        #
        if n_etypes == 1:
            intra_conv = IntraConv_single
        else:
            intra_conv = IntraConv_multi

        dgas = []
        for r in range(self.n_etypes):
            m = nn.ModuleDict({
                'all': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(),
                                  bias=False),
                'gp0': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(),
                                  bias=False, add_self=False),
                'gp1': intra_conv(self.last_dim, n_hidden, "mean", norm=nn.BatchNorm1d(n_hidden), activation=nn.ReLU(),
                                  bias=False, add_self=False)
            })
            dgas.append(m)
        self.dgas = nn.ModuleList(dgas)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)

    def dynamic_grouping(self, mask, block, unclear_down, unclear_up):
        mask0 = (mask[:, 1] <= unclear_down)[block.srcdata[dgl.NID][block.edges()[0]]].float()
        mask1 = (mask[:, 1] > unclear_up)[block.srcdata[dgl.NID][block.edges()[0]]].float()
        return mask0, mask1

    def forward(self, blocks, x):
        """Forward pass of the SAGE model."""
        batch_size = blocks[-1].dstdata['feat'].shape[0]
        x = self.emb_layer(x)
        emb_out = self.emb_layer_fc(x)

        mask0_dict = {}
        mask1_dict = {}
        block = blocks[0]

        for etype in block.etypes:
            mask0_dict[etype], mask1_dict[etype] = \
                self.dynamic_grouping(self.super_mask, block.edge_type_subgraph(etypes=[etype]),
                                      self.unclear_up, self.unclear_down)

        h_list = []
        for idx, etype in enumerate(block.etypes):
            h_list.append(self.dgas[idx]['all'](block, x, etype))
            h_list.append(self.dgas[idx]['gp0'](block, x, etype, mask0_dict[etype]))
            h_list.append(self.dgas[idx]['gp1'](block, x, etype, mask1_dict[etype]))

        s_len = len(h_list)
        h_list = torch.stack(h_list, dim=1)

        h_list_proj = self.W_f(h_list).view(batch_size, s_len, self.n_head, self.n_hidden)
        h_list_proj = h_list_proj.permute(0, 2, 1, 3).contiguous().view(-1, s_len, self.n_hidden)

        x_proj = self.W_x(x[:batch_size]).view(batch_size, self.n_head, self.n_hidden, 1)
        x_proj = x_proj.view(-1, self.n_hidden, 1)

        attention_logit = torch.bmm(h_list_proj, x_proj)
        soft_attention = F.softmax(attention_logit, dim=1).transpose(1, 2)
        h_list_rep = h_list.repeat([self.n_head, 1, 1])
        weighted_features = torch.bmm(soft_attention, h_list_rep).squeeze(-2)
        h = weighted_features.view(batch_size, -1)
        o = self.final_fc_layer(h)

        return o, emb_out[:batch_size]


# 构建网络结构
class pl_DGA(LightningModule):
    def __init__(self, in_feats, n_hidden, num_nodes, n_classes, n_etypes,
                 lr=1e-3, weight_decay=5e-4, p=0.3, n_head=1, w=None,
                 unclear_up=0.1, unclear_down=0.1,
                 trn_idx=None, val_idx=None, tst_idx=None):
        super().__init__()
        self.save_hyperparameters()
        self.n_etypes = n_etypes
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dga = DGA(in_feats[0], n_hidden, num_nodes, n_classes, n_etypes,  p, n_head, unclear_up, unclear_down)
        self.trn_idx = trn_idx
        self.val_idx = val_idx
        self.tst_idx = tst_idx
        self.unclear_up = unclear_up
        self.unclear_down = unclear_down
        self.register_buffer('w', torch.FloatTensor(w))
        self.ps = []

    def forward(self, blocks, x):
        o, emb_out = self.dga(blocks, x)
        return o, emb_out

    def training_step(self, batch, batch_idx):  # , optimizer_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["aux_label"]
        logits, emb_logits = self(blocks, x)
        loss = F.cross_entropy(logits, y, self.w)
        # emb_loss = F.cross_entropy(emb_logits, y, self.w)
        loss = loss  # + 0.5*emb_loss
        self.log("trn_loss0", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(output_nodes))
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
        logits, emb_logits = self(blocks, x)
        loss = F.cross_entropy(logits, y, self.w)
        # emb_loss = F.cross_entropy(emb_logits, y, self.w)
        loss = loss  # + 0.5*emb_loss

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(output_nodes))
        return logits, y

    def validation_epoch_end(self, outs):
        """Calculate validation AUC at the end of the epoch."""
        y = torch.cat([x[1] for x in outs]).cpu().numpy()
        if self.trainer.sanity_checking:
            prob = torch.cat([x[0] for x in outs]).softmax(-1).cpu().numpy()[:, 1]
        else:
            p = torch.cat([x[0] for x in outs]).softmax(-1).cpu().numpy()
            self.ps.append(p)
            prob = p[:, 1]
            prob_ = self.ps[-2][:, 1] if len(self.ps) > 1 else 1 - prob
            self.dga.super_mask.copy_(torch.FloatTensor(np.mean(self.ps[-10:], axis=0)))

            # if self.trainer.current_epoch > 50:
            # self.super_mask[trn_idx, y[trn_idx]] = 1.0  # 训练集的部分硬标
            # print((self.super_mask.argmax(-1)[trn_idx].cpu().numpy() == y[trn_idx]).mean())
            trn_auc = roc_auc_score(y[self.trn_idx], prob[self.trn_idx])
            val_auc = roc_auc_score(y[self.val_idx], prob[self.val_idx])
            tst_auc = roc_auc_score(y[self.tst_idx], prob[self.tst_idx])
            trn_aps = average_precision_score(y[self.trn_idx], prob[self.trn_idx])
            val_aps = average_precision_score(y[self.val_idx], prob[self.val_idx])
            tst_aps = average_precision_score(y[self.tst_idx], prob[self.tst_idx])
            self.log('g0', int((prob <= self.unclear_down).sum()) / 100000, prog_bar=True, on_step=False, on_epoch=True)
            self.log('g1', int((prob > self.unclear_up).sum()) / 100000, prog_bar=True, on_step=False, on_epoch=True)
            self.log('flip_r', ((prob_ <= self.unclear_down) ^ (prob <= self.unclear_down)).sum() / len(prob),
                     prog_bar=True, on_step=False, on_epoch=True)
            self.log("pmean", prob.mean(), prog_bar=True, on_step=False, on_epoch=True)
            self.log("trn_auc", trn_auc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_auc", val_auc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("tst_auc", tst_auc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("trn_aps", trn_aps, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_aps", val_aps, prog_bar=True, on_step=False, on_epoch=True)
            self.log("tst_aps", tst_aps, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        optimizer = torch.optim.Adam(self.dga.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                               verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            }
        }

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        epoch = self.trainer.current_epoch
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in self.trainer.logged_metrics.items()}
        formatted_metrics = {k: f"{v:.5f}" for k, v in metrics.items()}
        print(f"Epoch {epoch}: {formatted_metrics}")

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        """Inference function for the model."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size * 5,
            shuffle=False,
            drop_last=False,
            use_uva=True,
            num_workers=0,
        )

        if buffer_device is None:
            buffer_device = device

        y = torch.zeros(
            g.num_nodes(),
            self.n_classes,
            device=buffer_device,
        )
        for input_nodes, output_nodes, blocks in tqdm(dataloader):
            x = blocks[0].srcdata["feat"]
            logits, emb_logits = self(blocks, x)
            y[output_nodes] = logits.to(buffer_device)
        return y


@hydra.main(config_path="configs", config_name="amazon", version_base=None)
def run(args: DictConfig):
    # 设置GPU, 设置随机种子，修改模型名称
    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() and args.usegpu else 'cpu')
    accelerator = 'gpu' if torch.cuda.is_available() and args.usegpu else 'cpu'
    set_all_seed(args.seed)
    args.model = f'{args.model}'
    suffix = '_bin' if args.bin_encoding else ''
    args.model = args.model + suffix

    # logger设置在读取数据之前，可以保存describe的信息
    mode = 'offline' if args.nowandb else 'online'
    wandb_run = wandb.init(project=args.project, config=OmegaConf.to_container(args), mode=mode)

    # 读取数据，输出图的统计信息
    DATA_PATH = '../data/processed/'
    graph, split_dict = load_graphs(DATA_PATH + args.dname + '.dgldata')
    graph = graph[0]
    y = graph.ndata['label'].cpu().numpy()
    graph.ndata['aux_label'] = graph.ndata['label']
    w = [1, 1]
    n_classes = 2

    n_etypes = len(graph.etypes)

    trn_idx, val_idx, tst_idx = mask_to_index(graph.ndata['trn_msk']), mask_to_index(
        graph.ndata['val_msk']), mask_to_index(graph.ndata['tst_msk'])
    # 信息打印
    print("==" * 20)
    print("数据名称", args.dname)
    describe(graph)
    print("==" * 20)
    print("超参数设定：")
    print(OmegaConf.to_yaml(args))
    print("n_etypes", n_etypes)
    print("n_classes", n_classes)
    print("==" * 20)

    if args.bin_encoding:
        feature = bin_encoding2(graph, trn_idx, n_bins=args.k)
        graph.ndata['feat'] = torch.FloatTensor(feature.values).contiguous()
        print("after bin_encoding：", feature.shape)
        print("==" * 20)
    else:
        feature = graph.ndata['feat']
        graph.ndata['feat'] = feature.contiguous()
        print("after bin_encoding：", feature.shape)

    in_feats = [graph.ndata['feat'].shape[1]]
    unclear_up = unclear_down = args.z
    wandb.log({
        'unclear_up': unclear_up,
        'unclear_down': unclear_down
    })
    print({'unclear_up': unclear_up, 'unclear_down': unclear_down})
    # 构建 datamodule 和 modelaccelerator
    datamodule = DataModule(graph, args.bs, n_classes)
    model = pl_DGA(in_feats, args.n_hidden, graph.num_nodes(), n_classes=n_classes,
                   n_etypes=n_etypes, lr=args.lr, weight_decay=args.weight_decay, p=args.p,
                   n_head=args.n_head, w=w,
                   unclear_up=unclear_up, unclear_down=unclear_down,
                   trn_idx=trn_idx, val_idx=val_idx, tst_idx=tst_idx)

    # 进行训练前的设置，timer，checkpoint，early_stopping，logger
    timer = Timer()
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min', save_top_k=1, verbose=False)
    early_stopping = EarlyStopping('val_loss', verbose=False, mode='min', patience=args.patience)
    logger = WandbLogger(wandb=wandb_run)

    trainer = Trainer(
        accelerator=accelerator, devices=[args.gpuid],
        max_epochs=args.max_epochs,
        logger=logger,
        enable_progress_bar=False,
        callbacks=[checkpoint_callback, early_stopping, timer],
    )
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    print(f'training time elapsed {timer.time_elapsed("train"):.2f}s')

    # 读取最优checkpoint 并且进行推理
    print("Evaluating model in", trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        y_hat = model.inference(graph, device, 5120, 0, "cpu")
    prob = y_hat.softmax(-1).cpu().numpy()[:, 1]
    dic = cal_metrics(prob, y, trn_idx, val_idx, tst_idx, verbose=True)
    wandb.log(dic)
    wandb.finish()
    print("===" * 10)
    print("===" * 10)


if __name__ == "__main__":
    run()
