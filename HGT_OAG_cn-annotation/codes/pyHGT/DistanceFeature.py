# -*-  coding: utf-8 -*-
# @Time      :2021/4/15 21:42
# @Author    :huangzg28153
# @File      :DistanceFeature.py
# @Software  :PyCharm
import numpy as np
import networkx as nx
import oaglog
import torch


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
    # temp = onehot_encoding[sp_length]  # 将onehot_encoding的每一行都视为一个元素，根据sp_length的值取对应元素，组成一个len(G)*set_size*dim的新数组
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


def get_distance_feature(new_set_index,
                         edge_index,
                         node_feature,
                         node_num,
                         feature_flags,
                         max_sprw):
    """
    生成distance encoding特征；
    :param new_set_index: 节点在edge_index中的索引，在SubgraphToTorch单节点抽样中，固定为np.array([0]);批抽样中为
                          np.arrange(args.batch_size);
    :param edge_index:边列表
    :param node_feature:节点的固有属性
    :param node_num:子图包含的节点数
    :param feature_flags:距离特征的类型
    :param max_sprw:距离特征的参数
    :return: 包含距离特征的全部节点特征；
    """
    oaglog.logger.debug("计算distance encoding特征开始。。。")
    # new_set_index = np.array([0])
    # 生成distance_encoding特征,根据每个子图的新的edge_index,
    edge_index = edge_index.numpy().transpose() if isinstance(edge_index, torch.LongTensor) else edge_index
    edge_index = np.array([row for row in edge_index if row[0] != row[1]])
    node_feature = node_feature.numpy() if isinstance(node_feature,torch.FloatTensor) else node_feature
    # edge_index = []
        # 传入的edge_index的维度为2*x,但构造new_G需要的维度为x*2,且nx不接受torch.Tensor类型的输出
    new_G = nx.from_edgelist(edge_index, create_using=type(nx.Graph()))

    new_G.add_nodes_from(np.arange(node_num), dtype=np.int32)

    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index, max_sp=max_sp)

        node_feature = np.concatenate((np.array(node_feature), features_sp_sample), axis=1)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G,
                                             nodelist=np.arange(
                                                 new_G.number_of_nodes(),
                                                 dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index, rw_depth=rw_depth)

        node_feature = np.concatenate((np.array(node_feature), features_rw_sample), axis=1)

    oaglog.logger.debug("计算distance encoding特征完毕。")
    return node_feature