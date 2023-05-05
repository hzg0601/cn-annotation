# -*-  coding: utf-8 -*-
# @Time      :2021/4/6 15:40
# @Author    :huangzg28153
# @File      :MainFunction.py
# @Software  :PyCharm

from Args import args
from ReadData import read_data, graph_to_dict
from ProcessData import assemble_n_batch
from pyHGT.SubgraphToTorch import SubgraphToTorch
import numpy as np
from multiprocessing import Pool
import os
# import ray
# import dill
# ray.init(ignore_reinit_error=True)
target_info_list, time_range_list, cand_list, target_relation, graph, types, edge_dict, pair_list = read_data(args)

# 定义子进程的函数
# 定义声明global变量的子进程函数
# 定义多进程函数


def sub_process(inp,time_range):
    """
    定义子进程函数，包装sample_n_batch
    :return:
    """
    global graph, time_range_list, target_relation, edge_dict, types, args
    result_one_sample = SubgraphToTorch(
        types=types,
        edge_dict=edge_dict,
        time_range=time_range,
        sampled_depth=args.sample_depth,
        sampled_number=args.sample_width,
        sample_n_pool=args.sample_n_pool,
        target_relation=target_relation,
        feature_flag=args.feature_flags,
        max_sprw=args.max_sprw).full_result_one_sample(inp, graph)
    # print(os.getpid(), id(graph))
    return result_one_sample


def mp_process(pool, target_info, time_range,sample_size=args.batch_size * args.n_batch):
    jobs = []
    target_info_choice = np.random.choice(target_info, sample_size, replace=False)
    for inp in target_info_choice:
        p = pool.apply_async(sub_process, args=(inp,time_range))
        jobs.append(p)
    return jobs, target_info_choice


pool = Pool(args.sample_n_pool)
jobs, target_info_choice = mp_process(pool, target_info=target_info_list[0], time_range=time_range_list[0])
result_n_batch = [job.get(timeout=7200) for job in jobs]
pool.close()
pool.join()
torch_n_batch = assemble_n_batch(result_n_batch,cand_list,args,types,target_info_choice,pair_list[0])









