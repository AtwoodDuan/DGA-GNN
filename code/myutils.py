import multiprocessing
import random

import numpy as np
import pandas as pd
import toad
import torch
from sklearn.metrics import (average_precision_score,
                             f1_score,
                             precision_score,
                             recall_score, roc_auc_score, )
from sklearn.metrics import confusion_matrix
from sklearn.metrics._ranking import _binary_clf_curve


def index_to_mask(index_list, length):
    """
    将给定的索引列表转换为一个长度为length的掩码张量。

    参数:
        index_list (list): 包含要转换为掩码的索引的列表。
        length (int): 掩码张量的长度。

    返回:
        mask (torch.Tensor): 一个长度为length的布尔掩码张量，其中给定索引的位置为True，其他位置为False。
    """
    mask = torch.zeros(length, dtype=torch.bool)
    mask[index_list] = True
    return mask


def mask_to_index(mask):
    """
    将给定的掩码张量转换为一个索引列表。

    参数:
        mask (torch.Tensor): 一个布尔掩码张量。

    返回:
        index_list (list): 包含掩码中True值对应的索引的列表。
    """
    index_list = torch.nonzero(mask).squeeze()
    return index_list


# 设置随机种子
def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def describe(graph):
    # 统计输出
    num_nodes = graph.ndata['feat'].shape[0]
    num_features = graph.ndata['feat'].shape[1]
    # 计算正类（欺诈样本）和负类（非欺诈样本）的数量
    num_positive = torch.sum(graph.ndata['label'] == 1)
    num_negative = torch.sum(graph.ndata['label'] == 0)
    # 计算欺诈样本占比
    fraud_ratio = num_positive / (num_positive + num_negative)
    print(f"总样本数: {num_nodes}")
    print(f"欺诈样本数: {num_positive}")
    print(f"非欺诈样本数: {num_negative}")
    print(f"欺诈样本占比: {fraud_ratio:.2%}")
    print(f"特征个数: {num_features}")
    print()
    for etype in graph.etypes:
        subgraph = graph.edge_type_subgraph([etype])
        num_edges = subgraph.number_of_edges()
        avg_out_degree = np.mean(subgraph.out_degrees().numpy())

        # 输出统计信息
        print(f"{etype}边关系下的统计信息:")
        print(f"边的个数: {num_edges}")
        print(f"平均出度: {avg_out_degree:.2f}")

    print("\n数据划分情况:")
    print(
        f"训练集: {graph.ndata['trn_msk'].sum()} 验证集: {graph.ndata['val_msk'].sum()} 测试集: {graph.ndata['tst_msk'].sum()}")
    print(
        f"训练集: {graph.ndata['trn_msk'].sum() / num_nodes:.2%} 验证集: {graph.ndata['val_msk'].sum() / num_nodes:.2%} 测试集: {graph.ndata['tst_msk'].sum() / num_nodes:.2%}")


# 计算获得最优的macrof1,gmean和对应的阈值
def get_max_macrof1_gmean(true, prob):
    fps, tps, thresholds = _binary_clf_curve(true, prob)
    n_pos = np.sum(true)
    n_neg = len(true) - n_pos
    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2 * tps / (2 * tps + fns + fps)
    f10 = 2 * tns / (2 * tns + fns + fps)
    marco_f1 = (f11 + f10) / 2

    idx = np.argmax(marco_f1)
    best_marco_f1 = marco_f1[idx]
    best_marco_f1_thr = thresholds[idx]

    gmean = np.sqrt(tps / n_pos * tns / n_neg)
    idx = np.argmax(gmean)
    best_gmean = gmean[idx]
    best_gmean_thr = thresholds[idx]
    return best_marco_f1, best_marco_f1_thr, best_gmean, best_gmean_thr


# 计算所有metrics指标
def cal_metrics(prob, y, trn_idx, val_idx, tst_idx, verbose=False):
    out_dic = {}
    val_th1 = 0
    val_th2 = 0
    for prefix, idx in zip(['final_trn/', 'final_val/', 'final_tst/'], [trn_idx, val_idx, tst_idx]):
        prob_ = prob[idx]
        y_ = y[idx]

        if prefix in ['final_trn/', 'final_val/']:
            mf1, th1, gme, th2 = get_max_macrof1_gmean(y_, prob_)
            val_th1 = th1
            val_th2 = th2
            pred = np.where(prob_ > th1, 1, 0)
        elif 'tst' in prefix:
            th1 = val_th1
            th2 = val_th2
            pred = np.where(prob_ > th1, 1, 0)
            mf1 = f1_score(y_true=y_, y_pred=pred, average='macro')
            tn, fp, fn, tp = confusion_matrix(y_, pred).ravel()
            gme = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

        rec = recall_score(y_, pred)
        pre = precision_score(y_, pred)
        auc = roc_auc_score(y_, prob_)
        aps = average_precision_score(y_, prob_)

        dic = {
            f'{prefix}auc': np.round(auc, 5),
            f'{prefix}aps': np.round(aps, 5),  # AP score
            f'{prefix}mf1': np.round(mf1, 5),
            f'{prefix}th1': np.round(th1, 5),
            f'{prefix}gme': np.round(gme, 5),
            f'{prefix}th2': np.round(th2, 5),
            f'{prefix}rec': np.round(rec, 5),
            f'{prefix}pre': np.round(pre, 5),
        }
        formatted_dic = {k: f"{v:.5f}" for k, v in dic.items()}
        if verbose == True:
            print(formatted_dic)
        out_dic.update(dic)
    return out_dic


# 决策树分箱编码
def bin_encoding2(graph, trn_idx, n_bins, BCD=False, col_index=None):
    X = graph.ndata['feat'].numpy()
    y = graph.ndata['label'].numpy()
    X = pd.DataFrame(X)
    trn_X = X.iloc[trn_idx]
    trn_y = pd.DataFrame(y[trn_idx])
    combiner = toad.transform.Combiner()
    combiner.fit(trn_X, trn_y, method='dt', min_samples=0.01, n_bins=n_bins, )
    bins = combiner.export()
    if col_index is None or col_index == 'None':
        col_index = X.columns
    bin_encoded_X = combiner.transform(X[col_index])

    bin_encoded_X_dummies = pd.get_dummies(bin_encoded_X, columns=col_index)
    feature = pd.concat([X, bin_encoded_X_dummies], axis=1)

    feature = feature.astype(float)
    return feature
