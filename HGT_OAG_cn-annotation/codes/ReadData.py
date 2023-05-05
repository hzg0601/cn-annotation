# -*-  coding: utf-8 -*-
# @Time      :2021/4/1 21:02
# @Author    :huangzg28153
# @File      :train_data_process.py
# @Software  :PyCharm
import oaglog
from pyHGT.data import renamed_load
import os
import numpy as np

ABSULUTE_DIR = '/data1/huangzg/research/pyHGT_OAG'


def read_data(args, chosen_relation=("venue","paper","PV_Journal")):
    """

    :param args: 输入参数
    :param chosen_relation: 选定的目标节点类型、源节点类型、关系类型构成的列表
    :param random_sampling_flag: 是否以随机方式进行target节点选择
    :param batch_size: 每个target_info_choice中target节点数
    :return:
    """
    oaglog.logger.info("读取数据开始。。。")
    graph = renamed_load(open(os.path.join(ABSULUTE_DIR + args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
    oaglog.logger.info("读取数据完毕。")

    train_range = {t: True for t in graph.times if t != None and t < 2015}
    valid_range = {t: True for t in graph.times if t != None and t >= 2015 and t <= 2016}
    test_range = {t: True for t in graph.times if t != None and t > 2016}
    target_type,source_type,relation_type = chosen_relation

    cand_list = list(graph.edge_list[target_type][source_type][relation_type].keys())

    target_relation = [[source_type, target_type, "rev_"+relation_type],
                       [target_type, source_type, relation_type]]
    types = list(graph.get_types())
    meta_graph = graph.get_meta_graph()
    # 元关系字典，最后一个为"self"类型
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    train_pairs = {}
    valid_pairs = {}
    test_pairs = {}

    for target_id in graph.edge_list[source_type][target_type]["rev_"+relation_type]:
        for source_id in graph.edge_list[source_type][target_type]["rev_"+relation_type][target_id]:
            _time = graph.edge_list[source_type][target_type]["rev_"+relation_type][target_id][source_id]
            if _time in train_range:
                if target_id not in train_pairs:
                    train_pairs[target_id] = [source_id, _time]
            elif _time in valid_range:
                if target_id not in valid_pairs:
                    valid_pairs[target_id] = [source_id, _time]
            else:
                if target_id not in test_pairs:
                    test_pairs[target_id] = [source_id, _time]

    np.random.seed(43)

    sel_train_pairs = {p: train_pairs[p] for p in
                       np.random.choice(list(train_pairs.keys()), int(len(train_pairs) * args.data_percentage),
                                        replace=False)}
    sel_valid_pairs = {p: valid_pairs[p] for p in
                       np.random.choice(list(valid_pairs.keys()), int(len(valid_pairs) * args.data_percentage),
                                        replace=False)}

    # 首先生成所有train、val、test样本
    ## 全部样本的list
    pairs_list = [sel_train_pairs, sel_valid_pairs, test_pairs]
    time_range_list = [train_range, valid_range, test_range]
    data_class_list = ["train",'val','test']
    target_info_list = []

    # target_info_choice组装，形式为target
    for pairs,data_class in zip(pairs_list, data_class_list):
        target_ids = list(pairs.keys())
        # target_info = []
        # if random_sampling_flag:
        #     num_batches = args.n_epoch * args.batch_size
        #
        #     target_ids_func = lambda i : np.random.choice(list(pairs.keys()), batch_size, replace=False)
        # else:
        #     num_batches = int(np.floor(len(target_ids)/batch_size))
        #     target_ids_func = lambda i : pairs[i*batch_size:(i+1)*batch_size]
        #
        # for i in range(num_batches):
        #     target_ids = target_ids_func(i)
        #     for target_id in target_ids:
        #         _, _time = pairs[target_id]
        #         target_info += [[target_id, _time]]
        #     target_info_list.append(target_info)

        oaglog.logger.info("数据类型 %s 共 %d 个节点" % (data_class, len(target_ids)))

    return time_range_list, cand_list, target_relation, graph, types, meta_graph, edge_dict, pairs_list

