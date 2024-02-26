import pickle
import random
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix
from collections import defaultdict
# from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import networkx as nx
import dgl
from dgl.data.utils import load_graphs, save_graphs
from myutils import describe, mask_to_index,index_to_mask, set_all_seed, cal_metrics, bin_encoding2
import os

# graph,_ = dgl.load_graphs(DATA_PATH+'yelpchi_homo.dgldata')
# graph = graph[0]

RAW_PATH = '../data/raw/'
DATA_PATH = '../data/processed/'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
# 数据的基本形式 ['feat', 'label', 'train_msk', 'val_msk', 'tst_msk']
# X为torch.float32，使用标准化统一进行归一化
# y和index为 torch.long
# mask 为 torch.bool
# 进行一版简单的设定，分割方式，yelp 和 amazon  按照 PCGNN 是 4 2 4
# amnet 的分割yelp的 方式是 7 2 1  分割 elliptic的方式应该是按照时间分的
# dgraph 的官方方式是 7 1.5 1.5
# elliptic 应该是根据原始数据的时间序列切的

# ====================================================================
# ====================================================================
# # books
# Books = pd.read_csv(RAW_PATH+'Books/AmazonFailNumeric.true', sep=';', header=None)
# AmazonFailNumericIDS = pd.read_csv(RAW_PATH+'Books/AmazonFailNumeric.graphml',  header=None)
# graph = nx.read_graphml(RAW_PATH+'Books/AmazonFailNumeric.graphml')
# # 遍历节点和特征
# dfs = []
# for node in graph.nodes():
#     dfs.append(pd.DataFrame(graph.nodes()[node], index=[int(node)]))
# X = pd.concat(dfs).sort_index()
# # 特征标准化
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# y = pd.read_csv(RAW_PATH+'Books/AmazonFailNumeric.true',sep=';',header=None)
# y = y.sort_values(0)[1].values
# num_nodes = len(y)
# # 数据划分
# trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y, train_size=0.4, random_state=3,shuffle=True)
# val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst, test_size=0.67, random_state=3,shuffle=True)
#
# trn_idx = torch.LongTensor(np.sort(trn_idx))
# val_idx = torch.LongTensor(np.sort(val_idx))
# tst_idx = torch.LongTensor(np.sort(tst_idx))
#
# # 创建掩码
# trn_msk = index_to_mask(trn_idx,num_nodes)
# val_msk = index_to_mask(val_idx,num_nodes)
# tst_msk = index_to_mask(tst_idx,num_nodes)
#
# # 构建图
# src_nodes = []
# dst_nodes = []
# for edge in graph.edges():
#     src_nodes.append(int(edge[0]))  # 出发点
#     dst_nodes.append(int(edge[1]))  # 结束点
# graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
# graph = dgl.remove_self_loop(graph)
# graph = dgl.to_bidirected(graph)
# graph.create_formats_()
# # 进一步丰富图数据要素
# graph.ndata['feat'] = torch.FloatTensor(X_std)
# graph.ndata['label'] = torch.LongTensor(y)
# graph.ndata['trn_msk'] = trn_msk
# graph.ndata['val_msk'] = val_msk
# graph.ndata['tst_msk'] = tst_msk
#
# # 保存图数据
# save_graphs(DATA_PATH + 'books.dgldata', graph)
# describe(graph)

# ====================================================================
# ====================================================================
# tfinance
print('====================================================================')
print('tfinance...')
# 加载图数据
g, label_dict = load_graphs(RAW_PATH+'tfinance')
g = g[0]
num_nodes = g.num_nodes()

# 特征标准化
X = g.ndata['feature'].numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 提取标签
y = g.ndata['label'][:,1].numpy()

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y, train_size=0.4, random_state=20230415,shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst, test_size=0.67, random_state=20230415,shuffle=True)

trn_idx = torch.LongTensor(np.sort(trn_idx))
val_idx = torch.LongTensor(np.sort(val_idx))
tst_idx = torch.LongTensor(np.sort(tst_idx))

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

# 构建图
src_nodes = g.edges()[0]
dst_nodes = g.edges()[1]
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
# 进一步丰富图数据要素
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk
<<<<<<< HEAD
describe(graph)
split_dict = dict()
split_dict['trn_msk'] = trn_msk
split_dict['val_msk'] = val_msk
split_dict['tst_msk'] = tst_msk
split_dict['trn_idx'] = torch.LongTensor(trn_idx)
split_dict['val_idx'] = torch.LongTensor(val_idx)
split_dict['tst_idx'] = torch.LongTensor(tst_idx)

# 保存图数据
save_graphs(DATA_PATH + 'tfinance.dgldata', graph,split_dict)
=======

# 保存图数据
save_graphs(DATA_PATH + 'tfinance.dgldata', graph, label_dict)

>>>>>>> origin/main
describe(graph)

# ====================================================================
# ====================================================================
# tsocial ,这两个数据集应该是没有争议的
print('====================================================================')
print('tsocial...')
g, label_dict = load_graphs(RAW_PATH+'tsocial')
g = g[0]
num_nodes = g.num_nodes()

# 特征标准化
X = g.ndata['feature'].numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 提取标签
y = g.ndata['label'].numpy()

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y, train_size=0.4, random_state=20230415,shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst, test_size=0.67, random_state=20230415,shuffle=True)

trn_idx = torch.LongTensor(np.sort(trn_idx))
val_idx = torch.LongTensor(np.sort(val_idx))
tst_idx = torch.LongTensor(np.sort(tst_idx))

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

# 构建图
src_nodes = g.edges()[0]
dst_nodes = g.edges()[1]
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
# 进一步丰富图数据要素
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk

<<<<<<< HEAD
describe(graph)
split_dict = dict()
split_dict['trn_msk'] = trn_msk
split_dict['val_msk'] = val_msk
split_dict['tst_msk'] = tst_msk
split_dict['trn_idx'] = torch.LongTensor(trn_idx)
split_dict['val_idx'] = torch.LongTensor(val_idx)
split_dict['tst_idx'] = torch.LongTensor(tst_idx)

# 保存图数据
save_graphs(DATA_PATH + 'tsocial.dgldata', graph,split_dict)
describe(graph)



=======
# 保存图数据
save_graphs(DATA_PATH + 'tsocial.dgldata', graph)

describe(graph)

>>>>>>> origin/main
# ====================================================================
# ====================================================================
# elliptic_of_amnet
print('====================================================================')
print('elliptic...')
data = pickle.load(open(RAW_PATH+'elliptic.dat', 'rb'))

# 特征标准化
X = data.x.numpy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


#构建图
num_nodes = data.num_nodes
src_nodes = data.edge_index[0]
dst_nodes = data.edge_index[1]
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = data.y
graph.ndata['trn_msk'] = data.train_mask
graph.ndata['val_msk'] = data.val_mask
graph.ndata['tst_msk'] = data.test_mask
<<<<<<< HEAD
describe(graph)
split_dict = dict()
split_dict['trn_msk'] = trn_msk
split_dict['val_msk'] = val_msk
split_dict['tst_msk'] = tst_msk
split_dict['trn_idx'] = torch.LongTensor(trn_idx)
split_dict['val_idx'] = torch.LongTensor(val_idx)
split_dict['tst_idx'] = torch.LongTensor(tst_idx)

# 保存图数据
save_graphs(DATA_PATH + 'elliptic_of_amnet.dgldata', graph,split_dict)
describe(graph)

=======

# 保存图数据
save_graphs(DATA_PATH + 'elliptic_of_amnet.dgldata', graph)

describe(graph)


>>>>>>> origin/main
# ====================================================================
# ====================================================================
# yelpchi
print('====================================================================')
print('yelpchi...')
# yelpchi_net_rur
# yelpchi_net_rtr
# yelpchi_net_rsr
# yelpchi_homo
# 仅仅保留 yelpchi_net_rur 和 yelpchi_homo  目前先以
# 读取数据
yelpchi = loadmat(RAW_PATH+'YelpChi.mat')
net_rur = yelpchi['net_rur']
net_rtr = yelpchi['net_rtr']
net_rsr = yelpchi['net_rsr']
net_hom = yelpchi['homo']
y = yelpchi['label'].reshape(-1)
num_nodes = yelpchi['features'].shape[0]

# 特征标准化
X = np.asarray(yelpchi['features'].todense())
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y,train_size=0.4, random_state=2, shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst,test_size=0.67, random_state=2, shuffle=True)

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

#构建图
src_nodes = net_rur.tocoo().col
dst_nodes = net_rur.tocoo().row
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk
split_dict = dict()
split_dict['trn_msk'] = trn_msk
split_dict['val_msk'] = val_msk
split_dict['tst_msk'] = tst_msk
split_dict['trn_idx'] = torch.LongTensor(trn_idx)
split_dict['val_idx'] = torch.LongTensor(val_idx)
split_dict['tst_idx'] = torch.LongTensor(tst_idx)

# 保存图数据
save_graphs(DATA_PATH + 'yelpchi_rur.dgldata', graph, split_dict)
describe(graph)

# ====================================================================
# ====================================================================
# yelpchi
# yelpchi_net_rur
# yelpchi_net_rtr
# yelpchi_net_rsr
# yelpchi_homo
# 仅仅保留 yelpchi_net_rur 和 yelpchi_homo  目前先以424做划分
# 读取数据
yelpchi = loadmat(RAW_PATH+'YelpChi.mat')
net_rur = yelpchi['net_rur']
net_rtr = yelpchi['net_rtr']
net_rsr = yelpchi['net_rsr']
net_hom = yelpchi['homo']
y = yelpchi['label'].reshape(-1)
num_nodes = yelpchi['features'].shape[0]

# 特征标准化
X = np.asarray(yelpchi['features'].todense())
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y,train_size=0.4, random_state=2, shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst,test_size=0.67, random_state=2, shuffle=True)

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

#构建图
src_nodes = net_hom.tocoo().col
dst_nodes = net_hom.tocoo().row
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk
<<<<<<< HEAD
split_dict = dict()
split_dict['trn_msk'] = trn_msk
split_dict['val_msk'] = val_msk
split_dict['tst_msk'] = tst_msk
split_dict['trn_idx'] = torch.LongTensor(trn_idx)
split_dict['val_idx'] = torch.LongTensor(val_idx)
split_dict['tst_idx'] = torch.LongTensor(tst_idx)
# 保存图数据
save_graphs(DATA_PATH + 'yelpchi_homo.dgldata', graph, split_dict)
=======

# 保存图数据
save_graphs(DATA_PATH + 'yelpchi_homo.dgldata', graph)
>>>>>>> origin/main
describe(graph)


# ====================================================================
# ====================================================================
# yelpchi 构建异构图
# 目前先以424做划分
# 读取数据
yelpchi = loadmat(RAW_PATH+'YelpChi.mat')
net_rur = yelpchi['net_rur']
net_rtr = yelpchi['net_rtr']
net_rsr = yelpchi['net_rsr']
net_hom = yelpchi['homo']
y = yelpchi['label'].reshape(-1)
num_nodes = yelpchi['features'].shape[0]

# 特征标准化
X = np.asarray(yelpchi['features'].todense())
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y,train_size=0.4, random_state=2, shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst,test_size=0.67, random_state=2, shuffle=True)

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

#构建图

graph_data = {
    ("review", "net_rur", "review") : (net_rur.tocoo().col,net_rur.tocoo().row),
    ("review", "net_rtr", "review") : (net_rtr.tocoo().col,net_rtr.tocoo().row),
    ("review", "net_rsr", "review") : (net_rsr.tocoo().col,net_rsr.tocoo().row),
}
graph = dgl.heterograph(graph_data)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk

# 保存图数据
save_graphs(DATA_PATH + 'yelpchi.dgldata', graph)
describe(graph)

# ====================================================================
# ====================================================================
print('====================================================================')
print('amazon...')
# amazon
# amazon_net_upu
# amazon_net_usu
# amazon_net_uvu
# amazon_homo
# 读入原始数据
amazon = loadmat(RAW_PATH+'Amazon.mat')
net_upu = amazon['net_upu']
net_usu = amazon['net_usu']
net_uvu = amazon['net_uvu']
net_hom = amazon['homo']
num_nodes = amazon['features'].shape[0]
y = amazon['label'].reshape(-1)

# 特征标准化
X = np.asarray(amazon['features'].todense())
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y,train_size=0.4, random_state=2, shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst,test_size=0.67, random_state=2, shuffle=True)


trn_idx = pd.Series(trn_idx)
trn_idx = trn_idx[~trn_idx.isin(np.arange(3305))].tolist()
val_idx = pd.Series(val_idx)
val_idx = val_idx[~val_idx.isin(np.arange(3305))].tolist()
tst_idx = pd.Series(tst_idx)
tst_idx = tst_idx[~tst_idx.isin(np.arange(3305))].tolist()

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

# 构建图
src_nodes = net_hom.tocoo().col
dst_nodes = net_hom.tocoo().row
graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
# 进一步丰富图数据要素
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk

<<<<<<< HEAD
split_dict = dict()
split_dict['trn_msk'] = trn_msk
split_dict['val_msk'] = val_msk
split_dict['tst_msk'] = tst_msk
split_dict['trn_idx'] = torch.LongTensor(trn_idx)
split_dict['val_idx'] = torch.LongTensor(val_idx)
split_dict['tst_idx'] = torch.LongTensor(tst_idx)

# 保存图数据
save_graphs(DATA_PATH + 'amazon_homo.dgldata', graph,split_dict)
=======
# 保存图数据
save_graphs(DATA_PATH + 'amazon_homo.dgldata', graph)
>>>>>>> origin/main
describe(graph)


# ====================================================================
# ====================================================================
# amazon 异构图

# 读入原始数据
amazon = loadmat(RAW_PATH+'Amazon.mat')
net_upu = amazon['net_upu']
net_usu = amazon['net_usu']
net_uvu = amazon['net_uvu']
net_hom = amazon['homo']
num_nodes = amazon['features'].shape[0]
y = amazon['label'].reshape(-1)

# 特征标准化
X = np.asarray(amazon['features'].todense())
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 数据划分
trn_idx, rst_idx, y_trn, y_rst = train_test_split(range(len(y)), y, stratify=y,train_size=0.4, random_state=2, shuffle=True)
val_idx, tst_idx, y_val, y_tst = train_test_split(rst_idx, y_rst, stratify=y_rst,test_size=0.67, random_state=2, shuffle=True)


trn_idx = pd.Series(trn_idx)
trn_idx = trn_idx[~trn_idx.isin(np.arange(3305))].tolist()
val_idx = pd.Series(val_idx)
val_idx = val_idx[~val_idx.isin(np.arange(3305))].tolist()
tst_idx = pd.Series(tst_idx)
tst_idx = tst_idx[~tst_idx.isin(np.arange(3305))].tolist()

# 创建掩码
trn_msk = index_to_mask(trn_idx,num_nodes)
val_msk = index_to_mask(val_idx,num_nodes)
tst_msk = index_to_mask(tst_idx,num_nodes)

# 构建图
graph_data = {
    ("user", "net_upu", "user") : (net_upu.tocoo().col,net_upu.tocoo().row),
    ("user", "net_usu", "user") : (net_usu.tocoo().col,net_usu.tocoo().row),
    ("user", "net_uvu", "user") : (net_uvu.tocoo().col,net_uvu.tocoo().row),
}
len (net_hom.tocoo().col)

graph = dgl.heterograph(graph_data)
graph = dgl.to_bidirected(graph)
graph.create_formats_()
graph.ndata['feat'] = torch.FloatTensor(X_std)
graph.ndata['label'] = torch.LongTensor(y)
graph.ndata['trn_msk'] = trn_msk
graph.ndata['val_msk'] = val_msk
graph.ndata['tst_msk'] = tst_msk

# 保存图数据
save_graphs(DATA_PATH + 'amazon.dgldata', graph)
describe(graph)


