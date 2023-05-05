# -*-  coding: utf-8 -*-
# @Time      :2021/2/3 18:13
# @Author    :huangzg28153
# @File      :DESampleGenerate.py
# @Software  :PyCharm

import torch
import numpy as np
import networkx as nx
import torch_geometric.utils as tgu
from itertools import combinations
from torch_geometric.data import DataLoader, Data


def get_features_sp_sample(G, node_set, max_sp):
    """
    生成最短路径距离特征,最终返回的特征矩阵shape为G.num_of_nodes*(max_sp+2);
    :param G: 图
    :param node_set: 给定的节点集，其作用是什么？？？
    :param max_sp: shortest_path允许的最大最短路径
    :return:shortest_path距离编码的特征矩阵
    """

    dim = max_sp + 2  # 维度为最大路径最短路径+2,防止共线性，在索引的基础上+1？
    set_size = len(node_set)  # 节点集容量
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1  # 生成图中节点vs给定节点集的空节点矩阵
    for i, node in enumerate(node_set):
        # 针对给定节点，nx.shortest_path_length生成的是邻居节点：最短路径长度的字典
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length  # 求每个节点与其他节点间的最短距离
    sp_length = np.minimum(sp_length, max_sp)  # 对于矩阵的每个元素，如值大于max_sp，则该位置上赋值为max_sp；
    # 其原作用为取两个矩阵对应位置上的最小值
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    temp = onehot_encoding[sp_length]  # 将onehot_encoding的每一行都视为一个元素，根据sp_length的值取对应元素，组成一个len(G)*set_size*dim的新数组
    features_sp = onehot_encoding[sp_length].sum(axis=1)  # 然后对第二维求和，构造为一个len(G)*dim的数组；
    return features_sp


def get_features_rw_sample(adj, node_set, rw_depth):
    """
    生成随机游走距离编码特征矩阵的函数，最终返回的特征矩阵与node_set的size无关，为G.num_of_nodes()*(rw_depth+1)
    :param adj: G.adjacency().todense(),致密的邻接矩阵；
    :param node_set:给定的节点集,
    :param rw_depth:随机游走的深度
    :return:随机游走距离编码的特征矩阵；
    """
    epsilon = 1e-6
    # keepdims，保持维度，即维度仍为2，shape为len(G)*1，得到每一行都初一行和的二维数组,得到的度分布；

    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    # 生成一个与adj节点数一致的单位矩阵，然后取出节点集对应的行，构成矩阵G.num_of_nodes*len(node_set)，作为随机游走的初始状态；
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)  # set_size*G.num_of_nodes 与 G.num_of_nodes*G.num_of_nodes相乘，即每次往前一步;
        rw_list.append(rw)  # 记录所有游走的状态矩阵，作为一个list，最终为len(node_set)*G.num_of_nodes*(rw_depth+1)
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, G.nun_of_nodes, rw_depth+1],合并为一个数组；
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)  # shape [G.num_of_nodes,set_size],求和得到访问每个节点的频数？
    return features_rw


def get_data_sample(G, set_index, hop_num, feature_flags, max_sprw, label, debug=False):
    """
    生成包含距离编码、节点属性、节点标签的torch_geometric，只针对给定的节点的索引set_index生成，
    DE-p，set_index的数量决定了DE-p中的p
    :param G: 图,
    :param set_index:节点在节点集中的索引，只针对一个节点，其为1*1的向量；
    :param hop_num:生成子图的跳数；
    :param feature_flags: 距离编码类型，shortest_path,random_walk的元组或列表；
    :param max_sprw: 距离编码的参数，针对shortest_path_length，rw_depth的元组或列表；
    :param label:节点标签集合；
    :param debug:
    :return: 重新组装的图数据，以n-跳子图为基准，包括属性、距离编码属性、标签；
    """
    # first, extract subgraph
    set_index = list(set_index) # 数字向量索引
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    # 如果不是DE-1，则需要删除节点set_index构造的边；
    if len(set_index) > 1:
        G = G.copy()
        G.remove_edges_from(combinations(set_index, 2))  # 排列组合，删除节点集set_index内的所有边；
    # torch.Tensor()是Python类，更明确的说，是默认张量类型torch.FloatTensor()的别名，
    # torch.Tensor([1,2]) 会调用Tensor类的构造函数__init__，生成单精度浮点类型的张量。

    # torch.tensor()
    # 仅仅是Python的函数，函数原型是：
    # torch.tensor(data, dtype=None, device=None, requires_grad=False)
    # 其中data可以是：list, tuple, array, scalar等类型。
    # torch.tensor()
    # 可以从data中的数据部分做拷贝（而不是直接引用），根据原始数据类型生成相应的torch.LongTensor，torch.FloatTensor，torch.DoubleTensor。
    # long长类型tensor,t转置，contiguous深拷贝；
    edge_index = torch.tensor(list(G.edges)).long().t().contiguous() # G.edges()为边元组构成的列表,转换为nx的边列表形式，一个2*N的致密矩阵
    # # 只保留的是节点集内与节点集外的连接；
    # 然后构造为一个无向图；
    edge_index = torch.cat([edge_index, edge_index[[1, 0], ]], dim=-1)  # concatenate，拼接,按列拼接，将构造为一个无向图边列表致密矩阵；
    #  生成k-hop子图,Computes the `k`-hop subgraph of `edge_index` around node `node_idx`.

    #   # 子图包含的点；
    #   # 过滤后的连接性；
    #   # 从（原节点索引）到新位置的映射，原节点集在新子图内的索引，如果图与节点集使用同一套索引则，则仍为原节点索引；否则其指向新图位置的索引
    #   # 表示保留边的边掩码；
    # #
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(torch.tensor(set_index).long(), hop_num, edge_index, num_nodes=G.number_of_nodes(), relabel_nodes=True)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    # 从边列表创建图，create_use，创建的图的类型；
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G))
    # 如果生成的子图中存在孤立点，则需加入子图集
    new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))  # to add disconnected nodes
    assert(new_G.number_of_nodes() == num_nodes)

    # Construct x from x_list，x_list属性列表；
    x_list = []
    attributes = G.graph['attributes']
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)[subgraph_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)  # unsqueeze对数据进行扩展，0为行方向；1为列方向；
        x_list.append(new_attributes)
    # if deg_flag:
    #     x_list.append(torch.log(tgu.degree(new_edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)+1))
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index.numpy(), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index.numpy(), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)
    # 将属性和距离编码特征直接拼接起来整体作为属性，G.num_of_nodes*(new_attributes.shape[-1]+max_sp+2(optional)+rw_depth+1(optional))
    x = torch.cat(x_list, dim=-1)
    # 将标签组装起来
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    # 将新节点集的索引扩展为二维，按行方向扩展；
    new_set_index = new_set_index.long().unsqueeze(0)
    if not debug:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index)
    else:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index,
                    old_set_indices=torch.tensor(set_index).long().unsqueeze(0), old_subgraph_indices=subgraph_node_old_index)


###########################################################################################################################################################################################


############################################################################################################################################
row=np.array(["a0","a0","a0","a1","a2","a3","a6"])
col=np.array(["b1","b2","b3","b4","b5","b6","b7"])

print('生成一个空的有向图')
G=nx.Graph()

print('在网络中添加带权中的边...')
for i in range(np.size(row)):
    G.add_edges_from([(row[i],col[i])])
list(np.expand_dims(np.arange(G.number_of_nodes()),1)[3])
G.graph["attributes"] = None
set_index=[2]
adj = np.asarray(nx.adjacency_matrix(G, nodelist=np.arange(G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))
get_features_rw_sample(adj,set_index,rw_depth=4)

adj = np.asarray(nx.adjacency_matrix(G, nodelist=np.arange(G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))
node_set = [2]
max_sp = 3

sp_result = get_features_sp_sample(G,node_set,max_sp)

rw_result = get_features_rw_sample(adj,node_set,rw_depth=max_sp)
sprw_para = [2,2]

data = get_data_sample(G,set_index=node_set,hop_num=3,feature_flags=("sp", "rw"),max_sprw=sprw_para,label=None,debug=False)

print(sp_result.shape)
print(rw_result.shape)

