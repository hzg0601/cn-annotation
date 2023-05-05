# -*-  coding: utf-8 -*-
# @Time      :2021/3/2 15:47
# @Author    :huangzg28153
# @File      :distance_feature.py
# @Software  :PyCharm
from collections import defaultdict
from itertools import combinations
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def get_features_sp_sample(G, node_set, max_sp):
    """
    生成最短路径距离特征
    :param G: 图,传入的是new_G,以数字为ID；
    :param node_set: 给定的节点集，
    :param max_sp: shortest_path允许的最大最短路径
    :return:shortest_path距离编码的特征矩阵
    """

    dim = max_sp + 2  # 维度为最大路径最短路径+2,防止共线性，在索引的基础上+1？
    set_size = len(node_set)  # 节点集容量，DE-p中的p；
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1  # 生成图中节点vs给定节点集的空节点矩阵
    for i, node in enumerate(node_set):
        # 针对给定节点，nx.shortest_path_length生成的是邻居节点：最短路径长度的字典
        # node_neighbour,node_ngh即距离为length的邻接节点在图中的索引
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
    生成随机游走距离编码特征矩阵的函数
    :param adj: G.adjacency().todense(),致密的邻接矩阵；
    :param node_set:给定的节点集
    :param rw_depth:随机游走的深度
    :return:随机游走距离编码的特征矩阵；
    """
    epsilon = 1e-6
    # keepdims，保持维度，即维度仍为2，shape为len(G)*1，得到每一行都初一行和的二维数组,得到的度分布；

    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    # 生成一个与adj节点数一致的单位矩阵，然后取出节点集对应的行，构成矩阵G.num_of_nodes*node_set_size，作为随机游走的初始状态；
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)  # node_set_size*G.num_of_nodes 与 G.num_of_nodes*G.num_of_nodes相乘，即每次往前一步;
        rw_list.append(rw)  # 记录所有游走的状态矩阵，作为一个list
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, G.nun_of_nodes, rw_depth+1],合并为一个数组；
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)  # shape [G.num_of_nodes,node_set_size],求和得到访问每个节点的频数？
    return features_rw


# def get_k_hop_subgraph(G,set_index,):
#     pass

# def get_adjacent_matrix_dict(graph):
#     """
#
#     :param graph: 完整的原始图
#     :return: 元关系邻接矩阵的字典
#     """
#     edge_list = graph.edge_list()
#     adjacent_matrix_dict = defaultdict(  # target_type
#         lambda: defaultdict(  # source_type
#             lambda: defaultdict(  # relation_type
#                 lambda: np.array([])  # 默认类型np.array()
#             )))
#     for target_type in edge_list:
#         target_type_dict = edge_list[target_type]
#         for source_type in target_type_dict:
#             source_type_dict = target_type_dict[source_type]
#             for relation_type in source_type_dict:
#                 relation_type_matrix = []
#                 relation_type_dict = source_type_dict[relation_type]
#                 for target_id in relation_type_dict:
#                     target_id_dict = relation_type_dict[target_id]
#                     for source_id in target_id_dict:
#                         relation_type_matrix = [[target_id, source_id]]
#
#                 adjacent_matrix_dict[target_type][source_type][relation_type] = relation_type_matrix
#
#     return adjacent_matrix_dict
#
#
# def get_graph_and_set_indices(node_pair_list):
#     """
#     1,重命名节点；
#     2，合并节点list,作为一个set;
#     3,构造node_mapping,
#     4,加入边；
#     5，
#     :param node_pair_list: 元关系邻接矩阵，如[（0，0），（0，4）]
#     :return: 同质邻接矩阵
#     """
#     target_nodes = [item[0] for item in node_pair_list]  # 0，0，2，2，5；
#     source_nodes = [item[1] for item in node_pair_list]  # 1，1，4，5，5；
#
#     original_target_index = np.unique(target_nodes)  # 0,2,5
#     original_source_index = np.unique(source_nodes)  # 1,4,5
#
#     target_node_new_id = ["t" + str(item[0]) for item in node_pair_list]
#
#     source_node_new_id = ["s" + str(item[1]) for item in node_pair_list]
#
#     node_mapping = {}
#     for idx, target in enumerate(original_target_index):
#         node_mapping["t"+str(target)] = idx
#
#     for idx, source in enumerate(original_source_index):
#         node_mapping["s"+str(source)] = idx + len(original_target_index)
#
#
#     edges = [[node_mapping[target_node_new_id[i]],
#               node_mapping[source_node_new_id[i]]]
#              for i in range(len(target_node_new_id))]
#     G = nx.Graph(edges)
#     # 如0，1，4，。。。
#     str_target_index = ["t" + str(index) for index in original_target_index]
#     set_indices = list(range(len(original_target_index)))
#
#     set_indices = np.expand_dims(np.array(set_indices), 1)
#
#     return G, set_indices, original_target_index
def get_data_sample(G, set_index, hop_num, feature_flags, max_sprw, label, debug=False):
    """
    生成一个样本的子图和特征
    根据set_index抽取子图，set_index为一个样本(通常对于边级任务为一条边，对于节点级任务为一个节点，对于子图级任务，
    则为一个子图，因此set_index可能为一维，也可能为二维，甚至高维)
    如果为二维或高维，则是边，则需要先将边或子图从图中删除边和子图；
    无论输入的图是否为有向图，我们都将其作为无向图处理，首先拷贝边列表，然后交换边列表节点的顺序，作为无向图的边列表；
    然后从图抽取set_index中上的k-hop子图，其返回的结果为subgraph_node_old_index, new_edge_index, new_set_index, edge_mask
    然后基于生成的new_edge_index生成new_G,new_set_index为样本在生成子图中的新索引
    基于new_G,set_index生成distance_encoding,故生成的distance_encoding应为
    最后将特征、抽取的new_edge_index、标签组装为torch_geometric.data.Data
    补充：1，抽取子图后，会重新安排索引，生成新索引下的new_edge_list,new_set_index，同时保留旧索引下抽取节点的索引集合；
         2，每个节点的distance_encoding特征仍为一个向量，只是从新子图出发，一次生成的是新子图所有节点的特征向量构成的矩阵
         3，
    :param G:删除val_test边的图
    :param set_index:原图中的set_index;
    :param hop_num:抽取子图的hop
    :param feature_flags:特征类型列表
    :param max_sprw:特征超参数
    :param label:标签集合
    :param debug:
    :return:Data
    """
    # first, extract subgraph
    set_index = list(set_index)
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if len(set_index) > 1:
        G = G.copy()
        G.remove_edges_from(combinations(set_index, 2))
    edge_index = torch.tensor(list(G.edges)).long().t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], ]], dim=-1)
    # sub_graph_node_old_index，为原图中抽取的节点索引
    # new_edge_index为抽取的边，其也为新索引集合下的边列表
    # new_set_index为样本的新索引，即抽取后节点会重新安排索引
    # edge_mask，为抽取边的掩码,长度与原图G的边列表长度
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = tgu.k_hop_subgraph(torch.tensor(set_index).long(), hop_num, edge_index, num_nodes=G.number_of_nodes(), relabel_nodes=True)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G))
    new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))  # to add disconnected nodes
    assert(new_G.number_of_nodes() == num_nodes)

    # Construct x from x_list
    x_list = []
    attributes = G.graph['attributes']
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)[subgraph_node_old_index]
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes)
    # if deg_flag:
    #     x_list.append(torch.log(tgu.degree(new_edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)+1))
    # num_of_nodes * distance_encoding超参数，如feature_sp_sample为286*5，feature_rw_smaple为286*4
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index.numpy(), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index.numpy(), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)
    # 每个节点的distance_encoding特征仍为一个相连，只是从新子图出发，一次生成的是新子图所有节点的特征向量构成的矩阵
    #
    x = torch.cat(x_list, dim=-1)
    y = torch.tensor([label], dtype=torch.long) if label is not None else torch.tensor([0], dtype=torch.long)
    # 将new_set_index恢复为set_index相同的tensor向量，从而可以由Data组装，Data类似一个dict，见原代码；
    new_set_index = new_set_index.long().unsqueeze(0)
    if not debug:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index)
    else:
        return Data(x=x, edge_index=new_edge_index, y=y, set_indices=new_set_index,
                    old_set_indices=torch.tensor(set_index).long().unsqueeze(0),
                    old_subgraph_indices=subgraph_node_old_index)


def get_distance_feature(G, set_index, hop_num, feature_flags, max_sprw):
    """
    构造nx.Graph(),生成包含距离编码特征的属性,只针对一个节点；
    :param node_pair_list: 保存节点对的list
    :param set_index:节点的索引集合
    :param hop_num:生成子图的跳数；
    :param feature_flags: 距离编码类型，shortest_path,random_walk的元组或列表；
    :param max_sprw: 距离编码的参数，针对shortest_path_length，rw_depth的元组或列表；
    :return:

    """

    """
    是否add_edges_from是有向图？
    原文应该也是有向图
    """
    # first, extract subgraph
    set_index = list(set_index)
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
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
    edge_index = torch.tensor(list(G.edges)).long().t().contiguous()  # G.edges()为边元组构成的列表,转换为nx的边列表形式，一个2*N的致密矩阵
    # # 只保留的是节点集内与节点集外的连接；
    # 然后构造为一个无向图；
    edge_index = torch.cat([edge_index, edge_index[[1, 0],]], dim=-1)  # concatenate，拼接,按列拼接，将构造为一个无向图边列表致密矩阵；
    #  生成k-hop子图,Computes the `k`-hop subgraph of `edge_index` around node `node_idx`.

    #   # 子图包含的点；
    #   # 过滤后的连接性；
    #   # 从（原节点索引）到新位置的映射，原节点集在新子图内的索引，如果图与节点集使用同一套索引则，则仍为原节点索引；否则其指向新图位置的索引
    #   # 表示保留边的边掩码；
    # #
    subgraph_node_old_index, new_edge_index, new_set_index, edge_mask = get_k_hop_subgraph(
        torch.tensor(set_index).long(), hop_num, edge_index,
        num_nodes=G.number_of_nodes(), relabel_nodes=True)

    # reconstruct networkx graph object for the extracted subgraph
    num_nodes = subgraph_node_old_index.size(0)
    # 从边列表创建图，create_use，创建的图的类型；
    new_G = nx.from_edgelist(new_edge_index.t().numpy().astype(dtype=np.int32), create_using=type(G))
    # 如果生成的子图中存在孤立点，则需加入子图集
    new_G.add_nodes_from(np.arange(num_nodes, dtype=np.int32))  # to add disconnected nodes
    assert (new_G.number_of_nodes() == num_nodes)

    # Construct x from x_list，x_list属性列表；
    x_list = []

    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index.numpy(), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(), dtype=np.int32)
                                             ).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index.numpy(), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)

    x = torch.cat(x_list, dim=-1)
    return x

#
# def get_all_distance_feature(graph, hop_num, feature_flags, max_sprw):
#     """
#     返回一个距离编码特征字典，针对每个target_id抽取距离编码特征；
#     :param adjacent_matrix_dict: 邻接矩阵字典，
#     :param hop_num: 抽取距离特征子图的超参数，
#     :param feature_flags: 使用的距离编码特征的类型，list格式，"sp","rw"
#     :param max_sprw: 距离编码特征的超参数，与feature_flags对应，也为一个list,建议为3，4等
#     :return: 距离编码特征字典；
#     """
#
#     distance_feature_dict = defaultdict( # target_type
#         lambda: defaultdict( #source_type
#             lambda : defaultdict( # relation_type
#                 lambda : torch.Tensor()
#             )
#         )
#     )
#
#     for target_type in graph.edge_list:
#         target_type_dict =graph.edge_list[target_type]
#         for source_type in target_type_dict:
#             source_type_dict = target_type_dict[source_type]
#
#             for relation_type in source_type_dict:
#                 adjacent_matrix = source_type_dict[relation_type]
#                 G, set_indices, original_target_index = get_graph_and_set_indices(adjacent_matrix)
#                 for i in range(set_indices.shape[0]):
#                     set_index = set_indices[i]
#                     distance_feature = get_distance_feature(G,
#                                                             set_index,
#                                                             hop_num=hop_num,
#                                                             feature_flags=feature_flags,
#                                                             max_sprw=max_sprw)
#                     # graph.node_feature[target_type].loc[original_target_index[i],"distance_emb"] += [distance_feature]
#                     distance_feature_dict[target_type][source_type][relation_type] += [distance_feature]
#
#


