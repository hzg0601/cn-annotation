# -*-  coding: utf-8 -*-
# @Time      :2021/2/5 14:19
# @Author    :huangzg28153
# @File      :ExtractSubgraph.py
# @Software  :PyCharm
import tqdm
import time
import multiprocessing as mp
from DESampleGenerate import get_data_sample


def parallel_worker(x):
    return get_data_sample(*x)


def get_hop_num(prop_depth, layers, max_sprw, feature_flags):
    # TODO: may later use more rw_depth to control as well?
    return int(prop_depth * layers) + 1   # in order to get the correct degree normalization for the subgraph


def extract_subgraphs(G, labels, set_indices, prop_depth, layers, feature_flags, task, max_sprw, parallel, logger, debug=False):
    """
    针对set_inddices中每个节点生成带有标签、特征、距离编码特征的完整Data数据，
    每个Data数据是该节点在G上的一个k_hop_subgraph
    :param G: 图，原图，有自己的节点ID
    :param labels: 标签
    :param set_indices: 待生成数据的节点索引集合，由于GetData中，其维度维G.number_of_nodes() * 1,因此是针对每一个节点生成的
    :param prop_depth: ？？
    :param layers:
    :param feature_flags: 距离编码类型，shortest_path,random_walk的元组或列表；
    :param task: 任务，simulation or not；
    :param max_sprw: 距离编码的参数，针对shortest_path_length，rw_depth的元组或列表；
    :param parallel: 是否多线程
    :param logger:日志
    :param debug:
    :return: datalist
    """
    # deal with adj and features
    logger.info('Encode positions ... (Parallel: {})'.format(parallel))
    data_list = []
    # propgation_depth,信息传递的深度*层数，即得到子图的hop数，
    hop_num = get_hop_num(prop_depth, layers, max_sprw, feature_flags)
    n_samples = set_indices.shape[0]  # set_indices中的节点数
    if not parallel:
        for sample_i in tqdm(range(n_samples)):
            # 针对每个节点生成带有标签、特征、距离编码特征的完整Data数据，每个Data数据是该节点在G上的一个k_hop_subgraph；
            # set_indices[sample_i], 对应为np.array([sample_i])
            # data.keys = ['x','edge_index','y','set_indices']
            data = get_data_sample(G, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   label=labels[sample_i] if labels is not None else None, debug=debug)
            data_list.append(data)
    else:
        pool = mp.Pool(4)
        results = pool.map_async(parallel_worker,
                                 [(G, set_indices[sample_i], hop_num, feature_flags, max_sprw,
                                   labels[sample_i] if labels is not None else None, debug) for sample_i in range(n_samples)])
        remaining = results._number_left
        pbar = tqdm(total=remaining)
        while True:
            pbar.update(remaining - results._number_left)
            if results.ready():
                break
            remaining = results._number_left
            time.sleep(0.2)
        data_list = results.get()
        pool.close()
        pbar.close()
    return data_list