import numpy as np
import scipy.sparse as sp
import torch
import os
import yaml
import dill

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return [1. / (r[0] + 1) if r.size else 0. for r in rs]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def randint():
    return np.random.randint(2**32 - 1)



def feature_OAG(layer_data, graph):
    """
    用于合并图的特征；
    :param layer_data: [type][id]:[len(layer_data),time]，采样节点类型：ID的字典
    :param graph:Graph()类型的图
    :return:图的特征
    """
    feature = {}
    times   = {}
    indxs   = {}
    texts   = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue

        #
        idxs  = np.array(list(layer_data[_type].keys()))
        # 对应节点的时间
        tims  = np.array(list(layer_data[_type].values()))[:,1]
        # graph.node_feature[type]:pd.DataFrame
        # 如果graph.node_feature[type]包括node_emb列，则赋给feature[type]
        # 如果不包括，则默认长度为400的向量为特征向量
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
            # 合并，node_emb,emb,ciation数为整体特征向量
            # graph.node_feature: a DataFrame containing all features
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),\
            np.log10(np.array(list(graph.node_feature[_type].loc[idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)
        
        times[_type]   = tims
        indxs[_type]   = idxs
        # 如果节点类型为paper，则
        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts


def save_data_as_yaml(data,dir,name):
    name = ''.join(name, '.yml')
    with open(os.path.join(dir,name),'w',encoding='utf-8') as f:
        f.write(yaml.dump(data, default_flow_style=False))


def save_data_as_pk(data, dir, name):
    name = ''.join(name, '.pk')
    dill.dump(data, open(os.path.join(dir, name), 'wb'))

