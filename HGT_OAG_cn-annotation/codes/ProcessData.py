# -*-  coding: utf-8 -*-
# @Time      :2021/3/26 13:15
# @Author    :huangzg28153
# @File      :ProcessData.py
# @Software  :PyCharm
from pyHGT.SubgraphToTorch import SubgraphToTorch
import ray
import numpy as np
from collections import defaultdict
import torch

# ray_graph = ray.put(graph)

#
# def sample_n_batch_ray(target_info,
#                        time_range,
#                        target_relation=None,
#                        edge_dict=None,
#                        types=None,
#                        args=None,
#                        ray_graph=None):
#     """
#     将输入的batch的结果组装起来，返回
#     :return:
#     """
#     # 采样n_batch*batch_size个节点
#     target_info_choice = np.random.choice(target_info, args.batch_size*args.n_batch,replace=False)
#     # 共享graph内存地址
#
#     # 所有节点返回值的列表
#     result_n_batch = []
#     # 启动多进行程采样的次数
#     N = int(np.ceil(args.batch_size*args.n_batch/args.sample_n_pool))
#     for i in np.arange(N):
#         # 在前N-1次，建立args.sample_n_pool个actor,最后一次为batch_size*n_batch-(N-1)*args.sample_n_pool
#         # 记每次启动的进程数为X
#         target_info_n_pool = target_info_choice[i:(i+1)*args.sample_n_pool]
#         # 建立X个actor
#         sample_actors = [SubgraphToTorch.remote(
#                                   types=types,
#                                   edge_dict=edge_dict,
#                                   time_range=time_range,
#                                   sampled_depth=args.sample_depth,
#                                   sampled_number=args.sample_width,
#                                   sample_n_pool=args.sample_n_pool,
#                                   target_relation=target_relation,
#                                   feature_flag=args.feature_flags,
#                                   max_sprw=args.max_sprw) for _ in target_info_n_pool]
#         # 建立分布式采样X个节点的ray对象；
#         ray_n_batch = [actor.full_result_one_sample.remote(inp, ray_graph)
#                        for inp, actor in zip(target_info_n_pool, sample_actors)]
#         # 启动多进程，
#         result_one_batch = ray.get(ray_n_batch)
#         # 将结果加入 列表
#         result_n_batch += result_one_batch
#
#     return result_n_batch, target_info_choice

#
# def sample_n_batch_single(target_info,
#                        time_range,
#                        target_relation=None,
#                        edge_dict=None,
#                        types=None,
#                        args=None,
#                        original_graph=None):
#     target_info_choice = np.random.choice(target_info, args.batch_size * args.n_batch, replace=False)
#
#     result_list = []
#     for inp in target_info_choice:
#         result = SubgraphToTorch(
#             types=types,
#             edge_dict=edge_dict,
#             time_range=time_range,
#             sampled_depth=args.sample_depth,
#             sampled_number=args.sample_width,
#             # sample_n_pool=args.sample_n_pool,
#             target_relation=target_relation,
#             feature_flag=args.feature_flags,
#             max_sprw=args.max_sprw).full_result_one_sample(inp,original_graph)
#         result_list.append(result)
#
#     return result_list, target_info_choice
#
#
# def assemble_n_batch(result_n_batch,
#                      cand_list,
#                      args,
#                      types,
#                      targe_info_choice,
#                      pairs=None):
#
#     """
#
#     :param result_n_batch:
#     :param cand_list:
#     :param args:
#     :param types:
#     :param targe_info_choice: 选定的
#     :param pairs:完整的pairs组合，如sel_train_pairs等
#     :return:
#     """
#     torch_n_batch = []
#     for i in np.arange(args.n_batch):
#         result = result_n_batch[i*args.batch_size:(i+1)*args.batch_size]
#         node_feature_list = []
#         node_num_list = [0]
#         node_type_list = []
#
#         node_dict_list = []
#         node_num_dict = defaultdict(lambda: [])
#
#         edge_time_list = []
#         edge_index_list = []
#         edge_type_list = []
#
#         for num, result_sample in enumerate(result):
#
#             # 每个节点的完整索引，为一个dataframe,本次未用到
#             # indxs["sample"] = num
#             # indxs_list.append(indxs)
#             # node_feature 为一个array;
#             node_feature_list += list(result_sample[0])
#             # node_num为一个数字
#             node_num_list.append(result_sample[1])
#             # node_type为一个一层列表
#             node_type_list += result_sample[2]
#             # node_dict为类型：节点个数字典
#             node_dict_list.append(result_sample[3])
#             # edge_time为一个一层列表
#             edge_time_list += result_sample[4]
#             # edge_index为一个二层列表
#             edge_index = result_sample[5]
#             # 重新安排索引，每个样本以前一个样本的节点数为新索引的起始
#             edge_index = [[pair[0] + np.sum(node_num_list),
#                            pair[1] + np.sum(node_num_list)]
#                           for pair in edge_index]
#
#             edge_index_list += edge_index
#             # edge_type为一个一层列表
#             edge_type_list += result_sample[6]
#             # oaglog.logger.info("%s 个节点特征生成完毕" % str(num))
#
#         # # 用于节点重新排序，
#         # indxs_list = pd.concat(indxs_list, axis=0)
#         # tyep_dict = {_type:idx for idx,_type in enumerate(types)}
#         # indxs_list["type"] = indxs_list.map(tyep_dict)
#         # indxs_list = indxs_list.sort_values(by=['layer',"type","sample","ser"], axis=0).reset_index()
#
#         # 计算每个类型下的节点数
#         num_len = 0
#         for t in types:
#             # 首先确定每个类型的节点数目
#             # 然后再按类型累加
#             temp = [node_dict[t][2] for node_dict in node_dict_list]
#             temp = np.sum(temp)
#             node_num_dict[t] += [num_len, len(node_num_dict)]
#             num_len += temp
#
#         node_dict = node_num_dict
#         # node_num_list的最后一个并非传入target_id的新索引，须删除
#         node_num_list.pop(-1)
#
#         node_feature = torch.FloatTensor(node_feature_list)
#         node_type = torch.LongTensor(node_type_list)
#         edge_time = torch.LongTensor(edge_time_list)
#         edge_index = torch.LongTensor(edge_index_list).t()
#         edge_type = torch.LongTensor(edge_type_list)
#
#         if args.task_name == "PV":
#             ylabel = torch.zeros(args.batch_size, dtype = torch.long)
#             for x_id, target_id_dict in enumerate(targe_info_choice):
#                 target_id = target_id_dict['paper'][0][0]
#
#                 # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1
#                 ylabel[x_id] = cand_list.index(pairs[target_id][0])
#
#             # x_ids = np.arange(args.batch_size) + node_dict['paper'][0]
#
#         elif args.task_name == "PF":
#             ylabel = np.zeros([args.batch_size, len(cand_list)])
#             for x_id, target_id_dict in enumerate(targe_info_choice):
#                 target_id = target_id_dict['paper'][0][0]
#                 for source_id in pairs[target_id][0]:
#                     # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1,
#                     ylabel[x_id][cand_list.index(source_id)] = 1
#                     # 求ylabel的列归一化值,作为连接概率值的标签
#             ylabel /= ylabel.sum(axis=1).reshape(-1, 1)
#
#         else:
#             pass
#         x_ids = np.array(node_num_list)
#         torch_one_batch = (node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel)
#
#         torch_n_batch.append(torch_one_batch)
#     return torch_n_batch


def assemble_one_batch(result,
                     cand_list,
                     args,
                     types,
                     targe_id_choice,
                     pairs=None):
    """
    组装一个batch数据的函数，可被包装为multiprocessing 共享内存的多进程程序
    :param result_one_batch: 采样的batch_size个节点；
    :param cand_list: 待遇测的关系下的边列表；
    :param args: 参数
    :param types: 关系下节点的类型
    :param targe_info_choice: 选定的batch_size个{"paper":[target_id,time]}
    :param pairs: 原始训练节点对
    :param i: batch的轮次，0.。。。，n_batch-1;
    :return:
    """

    node_feature_list = []
    node_num_list = [0]
    node_type_list = []

    node_dict_list = []
    node_num_dict = defaultdict(lambda: [])

    edge_time_list = []
    edge_index_list = []
    edge_type_list = []

    for num, result_sample in enumerate(result):
        # 每个节点的完整索引，为一个dataframe,本次未用到
        # indxs["sample"] = num
        # indxs_list.append(indxs)
        # node_feature 为一个array;
        node_feature_list += list(result_sample[0])
        # node_num为一个数字
        node_num_list.append(result_sample[1])
        # node_type为一个一层列表
        node_type_list += result_sample[2]
        # node_dict为类型：节点个数字典
        node_dict_list.append(result_sample[3])
        # edge_time为一个一层列表
        edge_time_list += result_sample[4]
        # edge_index为一个二层列表
        edge_index = result_sample[5]
        # 重新安排索引，每个样本以前一个样本的节点数为新索引的起始
        edge_index = [[pair[0] + np.sum(node_num_list),
                       pair[1] + np.sum(node_num_list)]
                      for pair in edge_index]

        edge_index_list += edge_index
        # edge_type为一个一层列表
        edge_type_list += result_sample[6]
        # oaglog.logger.info("%s 个节点特征生成完毕" % str(num))

    # # 用于节点重新排序，
    # indxs_list = pd.concat(indxs_list, axis=0)
    # tyep_dict = {_type:idx for idx,_type in enumerate(types)}
    # indxs_list["type"] = indxs_list.map(tyep_dict)
    # indxs_list = indxs_list.sort_values(by=['layer',"type","sample","ser"], axis=0).reset_index()

    # 计算每个类型下的节点数
    num_len = 0
    for t in types:
        # 首先确定每个类型的节点数目
        # 然后再按类型累加
        temp = [node_dict[t][2] for node_dict in node_dict_list]
        temp = np.sum(temp)
        node_num_dict[t] += [num_len, len(node_num_dict)]
        num_len += temp

    node_dict = node_num_dict
    # node_num_list的最后一个并非传入target_id的新索引，须删除
    node_num_list.pop(-1)

    node_feature = torch.FloatTensor(node_feature_list)
    node_type = torch.LongTensor(node_type_list)
    edge_time = torch.LongTensor(edge_time_list)
    edge_index = torch.LongTensor(edge_index_list).t()
    edge_type = torch.LongTensor(edge_type_list)

    if args.task_name == "PV":
        ylabel = torch.zeros(args.batch_size, dtype=torch.long)
        for x_id, target_id in enumerate(targe_id_choice):
            # target_id = target_id_dict['paper'][0][0]

            # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1
            ylabel[x_id] = cand_list.index(pairs[target_id][0])

        # x_ids = np.arange(args.batch_size) + node_dict['paper'][0]

    elif args.task_name == "PF":
        ylabel = np.zeros([args.batch_size, len(cand_list)])
        for x_id, target_id in enumerate(targe_id_choice):
            # target_id = target_id_dict['paper'][0][0]
            for source_id in pairs[target_id][0]:
                # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1,
                ylabel[x_id][cand_list.index(source_id)] = 1
                # 求ylabel的列归一化值,作为连接概率值的标签
        ylabel /= ylabel.sum(axis=1).reshape(-1, 1)

    else:
        pass
    x_ids = np.array(node_num_list)
    torch_one_batch = (node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel)
    return torch_one_batch


