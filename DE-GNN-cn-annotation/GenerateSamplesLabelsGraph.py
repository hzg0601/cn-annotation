# -*-  coding: utf-8 -*-
# @Time      :2021/2/5 14:22
# @Author    :huangzg28153
# @File      :GenerateSamplesLabelsGraph.py
# @Software  :PyCharm
import numpy as np
import random
from itertools import combinations
from sklearn.model_selection import train_test_split


def sample_neg_sets(G, n_samples, set_size):
    """
    返回一个n_samples个元组的随机样本list,每个元组包含set_size个节点，这些节点必须包括至少一对不相连的节点；
    :param G:给定的图
    :param n_samples: 样本集的大小，对应为batch的大小？
    :param set_size: 每个样本对应set_size个随机节点，从set_size个节点中选择不相连的两个节点；
    :return:
    """
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    # 抽样至多n_samples个节点
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        # 每个样本集含有 set_size个负节点,节点的索引是随机选择的；
        candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        for node1, node2 in combinations(candid_set, 2):
            # 如果两个节点在边集合里，则不保留，只要有不相连的边，就保留，并返回；
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break

    return neg_sets


def collect_tri_sets(G):
    """
    针对每个节点，找出其邻居节点中有边相连的所有节点对，构造三元组；
    :param G:给定的图
    :return:
    """
    # 从节点1的邻居节点中选择两个节点，且这两个节点间有边相连，遍历满足图中每个节点1，每个节点1对应多个(节点2,节点3)
    tri_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def retain_partial(indices, ratio):
    """
    随机丢弃一些样本；
    :param indices:原节点索引集合
    :param ratio:保留比例
    :return:保留的样本索引和样本索引的索引
    """
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)  # 选择的数字
    return indices[sample_i], sample_i


def sample_pos_neg_sets(G, task, data_usage=1.0):
    """
    构造正负样本集，正样本与负样本的个数相同；
    针对链路预测，则负样本是不相连的样本对；
    针对三元组预测问题，则负样本是一对相连、一对不相连的节点；
    :param G:给定的图
    :param task:任务类型link_prediction或triplet_prediction
    :param data_usage:是否进行随机丢弃样本，如为1则不丢弃
    :return:
    """
    # 如果链路预测问题，则每个元组只需要两个节点，且这两个样本没有边相连；
    # 所有的边都是正样本
    if task == 'link_prediction':
        pos_edges = np.array(list(G.edges), dtype=np.int32)
        set_size = 2
    # 如果是三元组预测问题，则每个元组需要三个节点，且三个样本中有一对是相连的，一对是不相连的；
    # 针对图中每个节点，找出其邻居节点中有边相连的所有节点对，构造三元组，则该三元组必然具有两条边；
    elif task == 'triplet_prediction':
        pos_edges = np.array(collect_tri_sets(G))
        set_size = 3
    else:
        raise NotImplementedError
    # 如果data_usage<1,则随机丢弃一些样本；
    if data_usage < 1-1e-6:
        pos_edges, sample_i = retain_partial(pos_edges, ratio=data_usage)
        # 负样本与正样本的数量相等
    neg_edges = np.array(sample_neg_sets(G, pos_edges.shape[0], set_size=set_size), dtype=np.int32)
    return pos_edges, neg_edges


def get_mask(idx, length):
    """
    构造掩码向量，针对给定的idx，其值为1，否则为0；
    :param idx: 节点索引
    :param length: 整个掩码向量的长度
    :return:掩码向量，针对给定的idx，其值为1，否则为0；
    """
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.int8)


def split_dataset(n_samples, test_ratio, stratify=None):
    """
    使用train_test_split生成训练和测试标签索引，并生成对应的掩码，应用于有标签的节点级任务；
    :param n_samples: 样本容量
    :param test_ratio: 测试比例
    :param stratify:是否分层，即按标签比例抽取训练测试集
    :return:训练和测试样本的掩码向量
    """
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio, stratify=stratify)
    train_mask = get_mask(train_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, test_mask


def generate_set_indices_labels(G, task, test_ratio, data_usage=1.0):
    """
    生成样本和标签集
    :param G: 给定的图
    :param task: 任务类型，link_prediction或triplet_prediction;
    :param test_ratio: 测试样本的比例；
    :param data_usage:数据利用比例
    :return:无向图G、正负样本标签向量、正负样本集、训练和测试样本掩码
    """
    # 首先将图修改为无向图
    G = G.to_undirected()  # the prediction task completely ignores directions
    # 构造正负样本集；
    pos_edges, neg_edges = sample_pos_neg_sets(G, task, data_usage=data_usage)  # each shape [n_pos_samples, set_size], note hereafter each "edge" may contain more than 2 nodes
    n_pos_edges = pos_edges.shape[0]
    assert(n_pos_edges == neg_edges.shape[0])
    # 测试样本的容量
    pos_test_size = int(test_ratio * n_pos_edges)
    # 拼接正样本、负样本；
    set_indices = np.concatenate([pos_edges, neg_edges], axis=0)
    # 测试集正样本的索引
    test_pos_indices = random.sample(range(n_pos_edges), pos_test_size)  # randomly pick pos edges for test
    # 测试集负样本的索引
    test_neg_indices = list(range(n_pos_edges, n_pos_edges + pos_test_size))  # pick first pos_test_size neg edges for test
    # 构造针对测试集的掩码向量
    test_mask = get_mask(test_pos_indices + test_neg_indices, length=2*n_pos_edges)
    # 构造测试样本值为0，非测试样本值为1的掩码向量
    train_mask = np.ones_like(test_mask) - test_mask
    # 构造正样本索引为1、负样本索引为0的标签向量
    labels = np.concatenate([np.ones((n_pos_edges, )), np.zeros((n_pos_edges, ))]).astype(np.int32)
    # 从G中删除正样本测试集中所有涉及节点构成的边
    G.remove_edges_from([node_pair for set_index in list(set_indices[test_pos_indices]) for node_pair in combinations(set_index, 2)])

    # permute everything for stable training
    permutation = np.random.permutation(2*n_pos_edges)
    set_indices = set_indices[permutation]
    labels = labels[permutation]
    train_mask, test_mask = train_mask[permutation], test_mask[permutation]

    return G, labels, set_indices, (train_mask, test_mask)


def generate_samples_labels_graph(G, labels, task, args, logger):
    """
    如果未指定节点的标签，则是边级任务，则使用对比学习的思路，构造正负样本；
    如果指定了节点的标签，则是节点级任务，则按标签生成标签集、样本集、掩码向量
    :param G:给定的图
    :param labels:标签集合，边级任务则为None,且并不是所有节点都有标签；
    :param task:任务类型，link_prediction，或triplet_prediction;
    :param args:传入的参数；
    :param logger:日志
    :return:G, labels, set_indices, (train_mask, val_test_mask)
    """
    if labels is None:
        logger.info('Labels unavailable. Generating training/test instances from dataset ...')
        G, labels, set_indices, (train_mask, val_test_mask) = generate_set_indices_labels(G, task, test_ratio=2*args.test_ratio, data_usage=args.data_usage)
    else:
        # training on nodes or running on synthetic data
        logger.info('Labels provided (node-level task).')
        assert(G.number_of_nodes() == labels.shape[0])
        # 根据标签数量与使用率定义使用节点的数量
        n_samples = int(round(labels.shape[0] * args.data_usage))
        # 在图中抽样节点，构造节点索引集合；
        set_indices = np.random.choice(G.number_of_nodes(), n_samples, replace=False)

        labels = labels[set_indices]
        set_indices = np.expand_dims(set_indices, 1)
        train_mask, val_test_mask = split_dataset(set_indices.shape[0], test_ratio=2*args.test_ratio, stratify=labels)
    logger.info('Generate {} train+val+test instances in total. data_usage: {}.'.format(set_indices.shape[0], args.data_usage))
    return G, labels, set_indices, (train_mask, val_test_mask)