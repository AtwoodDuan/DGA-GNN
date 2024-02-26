import torch
import numpy as np
import pandas as pd
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score,)
import random
import toad
from joblib import Parallel, delayed
import multiprocessing
from sklearn.metrics import confusion_matrix


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
            mf1 = f1_score(y_true=y_,y_pred=pred,average='macro')
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

# 计算获得最优的macrof1,gmean和对应的阈值
def get_max_macrof1_gmean(true,prob):
    fps, tps, thresholds = _binary_clf_curve(true,prob)
    n_pos = np.sum(true)
    n_neg = len(true) - n_pos
    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2*tps/(2*tps+fns+fps)
    f10 = 2*tns/(2*tns+fns+fps)
    marco_f1 = (f11+f10)/2

    idx = np.argmax(marco_f1)
    best_marco_f1 = marco_f1[idx]
    best_marco_f1_thr = thresholds[idx]

    gmean = np.sqrt(tps/n_pos * tns/n_neg)
    idx = np.argmax(gmean)
    best_gmean = gmean[idx]
    best_gmean_thr = thresholds[idx]

    return best_marco_f1,best_marco_f1_thr,best_gmean,best_gmean_thr


# 计算获得最优的macrof1和对应的阈值
def get_max_macrof1(true,prob):
    fps, tps, thresholds = _binary_clf_curve(true,prob)
    n_pos = np.sum(true)
    n_neg = len(true) - n_pos
    fns = n_pos - tps
    tns = n_neg - fps

    f11 = 2*tps/(2*tps+fns+fps)
    f10 = 2*tns/(2*tns+fns+fps)
    marco_f1 = (f11+f10)/2

    idx = np.argmax(marco_f1)

    best_marco_f1 = marco_f1[idx]
    best_thresholds = thresholds[idx]
    return best_marco_f1,best_thresholds

# 设置随机种子
def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


'''获取辅助label'''
def get_aux_label(graph,stat='black_ratio', stat_edge=3):
    y = graph.ndata['label'].numpy()
    # 处理一下多个关系的情况，先把关系合并成homo
    if len(graph.etypes) > 1:
        if stat_edge==3:
            edge_l = []
            for etype in graph.etypes:
                g = graph[etype]
                edge = np.stack([g.edges()[0].numpy(),g.edges()[1].numpy()]).T
                edge_l.append(edge)
            edge = np.concatenate(edge_l, axis=0)
            edge = np.unique(edge, axis=0)
        else:
            g = graph[graph.etypes[stat_edge]]
            edge = np.stack([g.edges()[0].numpy(), g.edges()[1].numpy()]).T
    else:
        edge = np.stack([graph.edges()[0].numpy(),graph.edges()[1].numpy()]).T
    trn_idx = np.where(graph.ndata['trn_msk'].numpy() == True)[0]

    y_map = dict(zip(range(len(y)), y))
    trn_map = dict(zip(trn_idx, [True]*len(trn_idx)))
    edge_df = pd.DataFrame(edge, columns=['n0', 'n1'])
    edge_df['n0_in_trn'] = edge_df['n0'].map(trn_map)
    edge_df['n1_in_trn'] = edge_df['n1'].map(trn_map)

    # 保证两个节点都在训练集中
    edge_df = edge_df.query("n0_in_trn == True and n1_in_trn == True")
    edge_df['l0'] = edge_df['n0'].map(y_map)
    edge_df['l1'] = edge_df['n1'].map(y_map)
    edge_df_0 = edge_df.query("l0 == 0")

    if stat == 'black_ratio':
        value = edge_df_0.groupby('n0')['l1'].mean()
    elif stat == 'black_cnt':
        value = edge_df_0.groupby('n0')['l1'].sum()
    elif stat == 'cnt':
        value = edge_df_0.groupby('n0')['l1'].size()
    tmp = pd.Series(y[trn_idx]).value_counts()
    tmp = tmp[1]/tmp[0]
    aux_thr = value.quantile(0.5)
    aux_y_map = y_map.copy()
    aux_y_map.update(((value>aux_thr)*2).astype(int).to_dict())
    aux_y = np.array([aux_y_map[i] for i in range(len(y))])

    print(pd.Series(aux_y[trn_idx]).value_counts().sort_index())
    print(pd.Series(y[trn_idx]).value_counts().sort_index())
    return aux_y

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
    print(f"训练集: {graph.ndata['trn_msk'].sum()} 验证集: {graph.ndata['val_msk'].sum()} 测试集: {graph.ndata['tst_msk'].sum()}")
    print(f"训练集: {graph.ndata['trn_msk'].sum() / num_nodes:.2%} 验证集: {graph.ndata['val_msk'].sum() / num_nodes:.2%} 测试集: {graph.ndata['tst_msk'].sum() / num_nodes:.2%}")


import math
def bin_encoding(graph, trn_idx, n_bins, col_index=None):
    X = graph.ndata['feat'].numpy()
    # if col_index is not None:
    #     X = graph.ndata['feat'].numpy()[:,col_index]
    y = graph.ndata['label'].numpy()
    X = pd.DataFrame(X)
    trn_X = X.iloc[trn_idx]
    trn_y = pd.DataFrame(y[trn_idx])
    combiner = toad.transform.Combiner()
    combiner.fit(trn_X, trn_y, method='dt', min_samples=0.01, n_bins=n_bins)
    bins = combiner.export()
    if col_index is None:
        col_index = X.columns
    bin_encoded_X = combiner.transform(X[col_index])

    df = bin_encoded_X.copy()

    df['y'] = y
    for col in df.columns.difference(['y']):
        ind = df.iloc[trn_idx].groupby(col)['y'].mean()
        map_dict = dict(zip(ind.index,ind.values))
        df[col] = df[col].map(map_dict)
    df_pm = df.drop(columns=['y'])

    # df['y'] = y
    # for col in df.columns.difference(['y']):
    #     ind = df.iloc[trn_idx].groupby(col)['y'].mean().sort_values().index
    #     map_dict = dict(zip(ind,range(len(ind))))
    #     df[col] = df[col].map(map_dict)
    # df = df.drop(columns=['y'])
    #
    # def process_column(col):
    #     col = df[col].astype(int).copy()
    #     # 计算唯一值的个数，并根据个数计算二进制编码的位数
    #     n_unique = col.nunique()
    #     n_bits = np.ceil(np.log2(n_unique)).astype(int)
    #     # 将列的整数转化为二进制字符串，并填充至 n_bits 位
    #     bin_func = np.vectorize(lambda x: format(x, f'0{n_bits}b'))
    #     binary_data = bin_func(col.values)
    #     # 将二进制字符串分解为独立的位
    #     binary_cols = pd.DataFrame(map(list, binary_data.tolist()), columns=[f'{col.name}_b{i}' for i in range(n_bits)]).astype(int)
    #     return binary_cols
    # df = pd.concat(Parallel(n_jobs=1)(delayed(process_column)(col) for col in df.columns), axis=1)

    # for col in df.columns:
    #     df[col] = df[col].astype(int)
    #     # 计算唯一值的个数，并根据个数计算二进制编码的位数
    #     n_unique = df[col].nunique()
    #     n_bits = math.ceil(np.log2(n_unique))
    #     # 将列的整数转化为二进制字符串，并填充至 n_bits 位
    #     binary_data = np.array([list(format(x, f'0{n_bits}b')) for x in df[col].values])
    #     # 将二进制字符串分解为独立的位
    #     for i in range(n_bits):
    #         df[f'{col}_b{i}'] = binary_data[:, i].astype(int)
    #     # 可以选择删除原列
    #     df = df.drop(columns=[col])
    # bin_encoded_X = df
    df = df.drop(columns=['y']).astype(str)
    df = pd.get_dummies(df)
    feature = pd.concat([X, df], axis=1)
    return feature

# 根据 y.mean 对序号进行替换
def process_column(col):
    # 计算唯一值的个数，并根据个数计算二进制编码的位数
    n_unique = col.nunique()
    n_bits = np.ceil(np.log2(n_unique)).astype(int)
    # 将列的整数转化为二进制字符串，并填充至 n_bits 位
    bin_func = np.vectorize(lambda x: format(x, f'0{n_bits}b'))
    binary_data = bin_func(col.values)
    # 将二进制字符串分解为独立的位
    binary_cols = pd.DataFrame(map(list, binary_data.tolist()), columns=[f'{col.name}_b{i}' for i in range(n_bits)]).astype(int)
    return binary_cols




def bin_encoding2(graph, trn_idx, n_bins, BCD=False, col_index=None):
    X = graph.ndata['feat'].numpy()
    y = graph.ndata['label'].numpy()
    X = pd.DataFrame(X)
    trn_X = X.iloc[trn_idx]
    trn_y = pd.DataFrame(y[trn_idx])
    combiner = toad.transform.Combiner()
    combiner.fit(trn_X, trn_y, method='dt', min_samples=0.01, n_bins=n_bins,)
    bins = combiner.export()
    if col_index is None or col_index=='None':
        col_index = X.columns
    bin_encoded_X = combiner.transform(X[col_index])
    if BCD == False:
        bin_encoded_X_dummies = pd.get_dummies(bin_encoded_X, columns=col_index)
        feature = pd.concat([X, bin_encoded_X_dummies], axis=1)
    else:
        df = bin_encoded_X.copy()
        df['y'] = y
        for col in df.columns.difference(['y']):
            ind = df.iloc[trn_idx].groupby(col)['y'].mean().sort_values().index
            map_dict = dict(zip(ind, range(len(ind))))
            df[col] = df[col].map(map_dict)

        # 创建一个进程池
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # 使用 multiprocessing 的 map 函数来并行执行函数
        results = pool.map(process_column, [df[col] for col in df.columns.difference(['y'])])
        df_b = pd.concat(results, axis=1)
        feature = pd.concat([X, df_b], axis=1)
    feature = feature.astype(float)
    return feature


def bin_encoding_equal(graph, trn_idx, n_bins, BCD=False, col_index=None):
    X = graph.ndata['feat'].numpy()
    y = graph.ndata['label'].numpy()

    df = pd.DataFrame(X)
    for col in df.columns:
        n_bins = pd.cut(df[col], bins=10, labels=False).nunique()
        n_bins = pd.cut(df[col], bins=n_bins, labels=False).nunique()
        df[col] = pd.cut(df[col], bins=n_bins, labels=False)
    if col_index is None or col_index=='None':
        col_index = df.columns
    bin_encoded_X_dummies = pd.get_dummies(df, columns=col_index)
    feature = pd.concat([pd.DataFrame(X), bin_encoded_X_dummies], axis=1)
    feature = feature.astype(float)
    return feature