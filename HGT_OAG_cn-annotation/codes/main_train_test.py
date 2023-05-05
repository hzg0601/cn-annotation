# -*-  coding: utf-8 -*-
# @Time      :2021/4/6 15:40
# @Author    :huangzg28153
# @File      :MainFunction.py
# @Software  :PyCharm

from Args import args, device
from ReadData import read_data
from pyHGT.SubgraphToTorch import SubgraphToTorch, get_id_label
import numpy as np
from multiprocessing import Pool
import torch
import os
import oaglog
import torch.nn as nn
from CreateModel import create_model
from TrainModel import train_eval_model
from TestModel import test_model

# 读取和返回的数据
if args.task_name == "PV":
    chosen_relation = ("venue", "paper", "PV_Journal")
elif args.task_name == "PF":
    chosen_relation = ('field', 'paper', 'PF_in_L2')
else:
    chosen_relation = []
time_range_list, cand_list, target_relation, \
graph, types, meta_graph, edge_dict, pair_list = read_data(args, chosen_relation=chosen_relation)
# 1 代表引用次数
inp_dimension = len(graph.node_feature['paper']['emb'].values[0]) + args.emb_len + 1
if args.feature_flags[0]:
    inp_dimension += (args.max_sprw[0] + 2)
if args.feature_flags[1]:
    inp_dimension += (args.max_sprw[1] + 1)

use_distance_feature = True

if (not args.feature_flags[0]) and (not args.feature_flags[1]):
    use_distance_feature = False
cand_list_len = len(cand_list)
# model的相关组件，scheduler, model, optimizer, gnn, classifier
models = create_model(in_dim=inp_dimension, types=types, meta_graph=meta_graph, cand_list_len=cand_list_len, args=args)
# 训练的相关参数
stats = []
best_val = 0
train_step = 1500
criterion = nn.NLLLoss()
train_params = stats, best_val, train_step, criterion


# 定义子进程的函数
# 定义声明global变量的子进程函数
# 定义多进程函数


def sub_process_sample(target_ids, data_type_flag):
    """
    定义子进程函数，包装sample_one_batch
    :return:
    """
    global graph, target_relation, edge_dict, types, args, cand_list, pair_list, time_range_list
    oaglog.logger.debug("子进程采样一个batch节点开始...")
    inp = {"paper": [[target_id, pair_list[data_type_flag][target_id][1]] for target_id in target_ids]}
    result_one_batch = SubgraphToTorch(
        types=types,
        edge_dict=edge_dict,
        time_range=time_range_list[data_type_flag],
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        target_relation=target_relation,
        feature_flag=args.feature_flags,
        max_sprw=args.max_sprw,
        emb_len=args.emb_len,
        use_distance_feature=use_distance_feature
    ).full_result_sampling(inp, graph)

    x_ids, ylabel = get_id_label(task_name=args.task_name,
                                 target_id_choice=target_ids,
                                 cand_list=cand_list,
                                 pairs=pair_list[data_type_flag])
    result_one_batch.append(x_ids)
    result_one_batch.append(ylabel)
    oaglog.logger.debug("子进程采样一个batch节点开始...")

    # dill.dump(result_one_sample,open(os.path.join(args.subgraphs_dir,"以SubgraphToTorch采样一个节点的大小.pk"),'wb'))
    # print("采样节点为",inp)
    return result_one_batch


def mp_process_sample(
        data_type_flag,
        batch_size=args.batch_size,
        n_batch=args.n_batch,
        epoch_order="random"
                    ):
    """
    以多进程进行采样，采样n_batch个batch_size大小的batch

    :param data_type_flag:数据类型的索引，0为训练，1为验证，2为测试
    :param batch_size:采样的大小,对于测试集可以令batch_size为len(test_data),n_batch=1;
    :param n_batch: 一个epoch采样的batch数目
    :param epoch_order:采样顺序，如果从target_info中随机采样，则为“random";否则为数值
    :return:
    """
    oaglog.logger.info("多进程采样开始。。。")
    # 采样时至少开32个进程，可固定为args.sample_n_pool
    pool = Pool(args.sample_n_pool)
    jobs = []
    keys = list(pair_list[data_type_flag].keys())
    # 如果没有order，则随机采样；否则按顺序采样；
    # if epoch_order == 'random':
    #     target_ids_func = lambda i: np.random.choice(keys, batch_size, replace=False)
    # else:
    #     target_ids_func = lambda i: keys[i * batch_size:(i + 1) * batch_size]

    if epoch_order != "random" and (epoch_order + 1)*n_batch*batch_size <= len(keys):
        target_ids_choices = [keys[i * batch_size:(i + 1) * batch_size] for i in
                              range(n_batch * epoch_order, n_batch * (epoch_order + 1))]
    # 如果epoch所代表的节点总数超过pairs.keys()的总数，则随机抽样
    else:
        target_ids_choices = [np.random.choice(keys, batch_size, replace=False) for _ in range(n_batch)]
    for target_ids in target_ids_choices:
        oaglog.logger.debug("多进程采样中，epoch数要求的节点超过总节点数")

        p = pool.apply_async(sub_process_sample, args=(target_ids, data_type_flag))
        jobs.append(p)
    result_n_batch = [job.get(timeout=9600) for job in jobs]
    # 关闭pool，使其不在接受新的（主进程）任务
    pool.close()
    # 主进程阻塞后，让子进程继续运行完成，子进程运行完后，再把主进程全部关掉
    pool.join()
    oaglog.logger.info("多进程采样结束。")
    return result_n_batch


def train_eval(epoch=args.n_epoch, models=None, train_params=None):
    """
    :param epoch: 训练的epoch数
    :param models: model的组件，scheduler, model, optimizer, gnn, classifier
    :param train_params: 训练的相关参数，stats, res, best_val, train_step, criterion
    """

    for epoch_order in range(epoch):
        oaglog.logger.info("epoch %d 训练数据生成开始。。。" % epoch_order)
        torch_train = mp_process_sample(
            data_type_flag=0,
            batch_size=args.batch_size,
            n_batch=args.n_batch,
            epoch_order="random"
        )
        # torch_train = mp_process_assemble(
        #     result_train,
        #     target_ids_choices_train,
        #     data_type_index=0
        # )
        oaglog.logger.info("epoch %d 训练数据生成完毕" % epoch_order)
        oaglog.logger.info("epoch %d 验证数据生成开始。。。" % epoch_order)
        torch_valid = mp_process_sample(
            data_type_flag=1,
            batch_size=args.batch_size,
            n_batch=1,
            epoch_order="random"
        )
        # torch_valid = mp_process_assemble(
        #     result_valid,
        #     target_ids_choices_valid,
        #     data_type_index=1
        # )
        oaglog.logger.info("epoch %d 验证数据生成完毕" % epoch_order)
        # 每次训练完成后，要更新model的组件和训练相关参数
        models, train_params = train_eval_model(models,
                                                train_params,
                                                torch_train,
                                                torch_valid,
                                                args,
                                                device,
                                                epoch_order)
    scheduler, model, optimizer, gnn, classifier = models
    return gnn, classifier


gnn, classifier = train_eval(epoch=args.n_epoch, train_params=train_params, models=models)

# 根据最后一次训练的结果进行测试
torch_test = mp_process_sample(
    data_type_flag=2,
    batch_size=args.batch_size,
    n_batch=10,
    epoch_order="random"
)
# torch_test = mp_process_assemble(
#     result_test,
#     target_ids_choices_test,
#     data_type_index=2
# )
test_model(torch_test, model=(gnn, classifier), model_flag='last')

# 根据最好的model进行测试
best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))

torch_test = mp_process_sample(
    data_type_flag=2,
    batch_size=args.batch_size,
    n_batch=10,
    epoch_order="random"
)
# torch_test = mp_process_assemble(
#     result_test,
#     target_ids_choices_test,
#     data_type_index=2
# )

test_model(torch_test, model=best_model, model_flag='best')

# def generate_utility_sample(target_info, time_range, args, data_flag="train"):
#     """
#     整体抽样，由于占用空间过大，不可行；
#     :param target_info:
#     :param time_range:
#     :param args:
#     :param data_flag:
#     :return:
#     """
#     for i in range(int(np.floor(len(target_info)/(args.batch_size * args.n_batch)))):
#         i += 1
#         oaglog.logger.info("采样第 %d epoch节点开始" % i)
#         result_n_batch, target_info_choice = mp_process_sample(
#                                               target_info=target_info ,
#                                               time_range=time_range,
#                                               sample_order=i)
#
#         oaglog.logger.info("第 %d 个epoch节点采样结束" % i)
#         dill.dump(result_n_batch, open(args.subgraphs_dir + '/graph%s.pk' % (
#                 args.domain + str(i) + "epoch" + data_flag), 'wb'))
#         oaglog.logger.info("第 %d 个epoch节点写入结束" % i)
#         return result_n_batch
#
# generate_utility_sample(target_info_list[0],time_range_list[0],args,data_flag='train')

# result_n_batch = dill.load(open(args.subgraphs_dir + '/graph%s.pk' % (
#                 args.domain + str(1) + "epoch" + "train"), 'rb'))
# def sub_process_assemble(result_one_batch,
#                          target_info_choice_one_batch,
#                          data_type_index=0):
#     """
#     组装一个batch数据的子进程函数,
#     所有作为条件输入多进程的方法的都会复制n份
#     :param result_one_batch:
#     :param target_info_choice_one_batch:
#     :param data_type_index: 数据类型的索引，“train"为0，"valid"为1，"test"为2
#     :return:
#     """
#     global cand_list, pair_list
#     oaglog.logger.info("子进程组装一个batch节点开始...")
#     torch_one_batch = assemble_one_batch(result_one_batch,
#                                          cand_list,
#                                          args,
#                                          types,
#                                          target_info_choice_one_batch,
#                                          pair_list[data_type_index])
#     oaglog.logger.info("子进程组装一个batch节点结束.")
#     return torch_one_batch
#
#
# def mp_process_assemble(
#                       result_n_batch,
#                       target_ids_choices,
#                       data_type_index=0
#                       ):
#     """
#     多线程组装数据为torch
#     :param result_n_batch:n_batch*batch_size个节点数据,每个batch一个list
#     :param target_ids_choices:n_batch{"paper":[target_id,time]*batch_size}
#     :param data_type_index: 数据类型的索引，“train"为0，"valid"为1，"test"为2
#     :return:
#     """
#     oaglog.logger.info("多进程组装数据开始。。。")
#     pool = Pool(args.n_pool)
#     jobs = []
#     for i in range(args.n_batch):
#         # result_one_batch = result_n_batch[i * args.batch_size:(i + 1) * args.batch_size]
#         # target_info_choice_one_batch = target_info_choice[i * args.batch_size:(i + 1) * args.batch_size]
#         result_one_batch = result_n_batch[i]
#         target_ids_choice = target_ids_choices[i]
#         p = pool.apply_async(sub_process_assemble, args=(result_one_batch, target_ids_choice, data_type_index))
#         jobs.append(p)
#
#     torch_n_batch = [job.get(timeout=9600) for job in jobs]
#     pool.close()
#     pool.join()
#     oaglog.logger.info("多进程组装数据结束。")
#     return torch_n_batch
