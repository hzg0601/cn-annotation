# -*-  coding: utf-8 -*-
# @Time      :2021/2/5 13:54
# @Author    :huangzg28153
# @File      :DataLoader.py
# @Software  :PyCharm
"""

1，根据任务模式决定返回数据的类型，如果是simulation,则无需构造正负样本集；
否则：
根据是否指定标签构造正负样本集合：
如果未指定节点的标签，则是边级任务，则使用对比学习的思路，构造正负样本；
如果指定了节点的标签，则是节点级任务，则按标签生成标签集、样本集、掩码向量
返回无向图、标签向量集、正负样本集、训练-测试掩码向量，
主函数为GenerateSamplesLabelsGraph中的generate_samples_labels_graph

2，首先生成distance encoding特征，
然后将属性和距离编码特征直接拼接起来整体作为属性，
针对每个节点抽取一个k-hop子图，
然后再将子图封装为一个torch_geometric.data.Data类数据，
主函数为get_data_sample,在DESampleGenerate中；

3，针对每个节点构造如上的torch_geometric.data.Data类数据，
构成一个子图list，主函数为ExtractSubgraph中的Extract_Subgraph；

4，将子图列表划分为train,val,test数据集，再用DataLoader封装为mini-batch；
最终返回train,val,test数据集，与mini-batch size*标签类型数的输出形式

"""
from copy import deepcopy
from ExtractSubgraph import extract_subgraphs
from GenerateSamplesLabelsGraph import generate_samples_labels_graph
from torch_geometric.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


def split_datalist(data_list, masks):
    """

    :param data_list: ExtractSubgraph生成的数据list;
    :param masks: 训练、验证、测试样本的掩码
    :return:
    """
    # generate train_set
    train_mask, val_test_mask = masks
    num_graphs = len(data_list)
    assert((train_mask.sum()+val_test_mask.sum()).astype(np.int32) == num_graphs)
    assert(train_mask.shape[0] == num_graphs)
    train_indices = np.arange(num_graphs)[train_mask.astype(bool)]
    train_set = [data_list[i] for i in train_indices]
    # generate val_set and test_set
    val_test_indices = np.arange(num_graphs)[val_test_mask.astype(bool)]
    val_test_labels = np.array([data.y for data in data_list], dtype=np.int32)[val_test_indices]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=int(0.5*len(val_test_indices)), stratify=val_test_labels)
    val_set = [data_list[i] for i in val_indices]
    test_set = [data_list[i] for i in test_indices]
    return train_set, val_set, test_set


def load_datasets(train_set, val_set, test_set, bs):
    """
    封装为batch
    :param train_set:
    :param val_set:
    :param test_set:
    :param bs:
    :return:
    """
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def print_dataset(dataset, logger):
    for i in range(len(dataset)):
        data = dataset[i]
        keys = ['old_set_indices', 'old_subgraph_indices', 'set_indices', 'edge_index', 'x', 'y']
        for key in keys:
            logger.info(key)
            logger.info(data.__dict__[key])

def get_data(G, task, args, labels, logger):
    """
    生成模型所需的数据
    :param G: 图，节点映射后的节点ID，以数字为顺序
    :param task: 任务类型，simulation ，link_prediction,triplet_prediction;
    :param args: 传入参数
    :param labels: 节点标签，针对边级任务为None
    :param logger:日志
    :return:
    """
    G = deepcopy(G)  # to make sure original G is unchanged
    if args.debug:
        logger.info(list(G.edges))
    # di_flag = isinstance(G, nx.classes.digraph.DiGraph)
    # deg_flag = args.use_degree
    sp_flag = 'sp' in args.feature
    rw_flag = 'rw' in args.feature
    # norm_flag = args.adj_norm
    feature_flags = (sp_flag, rw_flag)
    # TODO: adapt the whole branch for simulation
    if task == 'simulation':
        # 将其扩展为G.number_of_nodes() * 1维的矩阵，
        # 即针对图的每个节点抽取子图，
        #
        set_indices = np.expand_dims(np.arange(G.number_of_nodes()), 1)
        # 每个节点都对应一个k-hop子图，针对每个节点构造的一个List
        data_list = extract_subgraphs(G, labels, set_indices, prop_depth=args.prop_depth, layers=args.layers,
                                 feature_flags=feature_flags, task=task,
                                 max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, logger=logger, debug=args.debug)
        # DataLoader,将data.dataset类型数据封装为一个mini-batch;
        # DataListLoader,将data.dataset类型数据封装为python list；
        # ClusterLoader,见Cluster-GCN;
        # DenseDataLoader,将data.dataset封装为一个mini-batch。
        # ① 数据的shuffle和batch处理
        #
        # RandomSampler(dataset)
        # SequentialSampler(dataset)
        # BatchSampler(sampler, batch_size, drop_last)
        # 可以看出RandomSampler等方法返回的就是DataSet中的索引位置(indices)
        # BatchSampler是wrap一个sampler，并生成mini-batch的索引(indices)的方式

        # DataLoaderIter
        # 这个_DataLoaderIter其实就是DataLoader类的__iter__()
        # 方法的返回值：
        #
        # 注意，这个_DataLoaderIter中 * init(self, loader) * 中的loader就是对应的DataLoader类的实例。
        loader = DataLoader(data_list, batch_size=args.bs, shuffle=False, num_workers=0)
        return loader
    #     如果未指定节点的标签，则是边级任务，则使用对比学习的思路，构造正负样本；
    #     如果指定了节点的标签，则是节点级任务，则按标签生成标签集、样本集、掩码向量
    # 生成节点集，节点集对应的，掩码索引
    G, labels, set_indices, (train_mask, val_test_mask) = generate_samples_labels_graph(G, labels, task, args, logger)

    if args.debug:
        logger.info(list(G.edges))
    # 仍为原图，但set_indices为以数字向量为索引的节点索引集合；
    data_list = extract_subgraphs(G, labels, set_indices, prop_depth=args.prop_depth, layers=args.layers,
                                 feature_flags=feature_flags, task=task,
                                 max_sprw=(args.max_sp, args.rw_depth), parallel=args.parallel, logger=logger, debug=args.debug)
    # 首先分割为train和val_test两个数据集，再使用train_test_split分割val_test数据集
    train_set, val_set, test_set = split_datalist(data_list, (train_mask, val_test_mask))

    if args.debug:
        print_dataset(train_set, logger)
        print_dataset(val_set, logger)
        print_dataset(test_set, logger)
    # 封装为三个mini-batch，外加一个batch size* 标签类别数的输出维度形式;
    train_loader, val_loader, test_loader = load_datasets(train_set, val_set, test_set, bs=args.bs)
    logger.info('Train size :{}, val size: {}, test size: {}, val ratio: {}, test ratio: {}'.format(len(train_set), len(val_set), len(test_set), args.test_ratio, args.test_ratio))
    return (train_loader, val_loader, test_loader), len(np.unique(labels))


#  ###########################################################
