# # # -*-  coding: utf-8 -*-
# # # @Time      :2021/3/22 20:56
# # # @Author    :huangzg28153
# # # @File      :test.py
# # # @Software  :PyCharm
# # import numpy as np
# # import pandas as pd
# # # type = [0,1,1,1,2,0,1,0,1,2,2,0]
# # # ser = [0,1,2,3,4,5,6,0,1,2,3,4]
# # # layer = [0,0,0,0,0,1,1,0,0,0,0,1]
# # # sample = [0,0,0,0,0,0,0,1,1,1,1,1]
# # #
# # # df = pd.DataFrame({"type":type,"ser":ser,"layer":layer,"sample":sample})
# # #
# # #
# # # df.sort_values(by=["ser",'type',"sample","layer"],axis=0)
# # # df.sort_values(by=["layer","sample","type","ser"],axis=0)
# # # df.sort_values(by=["type","layer","sample","ser"],axis=0)
# # # df['order'] = [0,2,4,5,6,9,11,1,3,7,8,10]
# # # df = df.sort_values(by=['order'],axis=0)
# # # df.sort_values(by=['layer','ser','type','sample'],axis=0)
# # # df.sort_values(by=["sample","type",'ser',"layer"],axis=0)
# # #
# # # ########################################################
# # # df.sort_values(by=['layer',"type","sample","ser"],axis=0).reset_index().index
# # # #######################################################
# # # from multiprocess import Process,Manager
# # # from pyHGT.data import Graph, renamed_load
# # # from pyHGT.data import renamed_load
# # # import os
# # # import ray
# # # Manager().register("Graph", Graph)
# # # dir(Manager())
# # # ABSULUTE_DIR = '/data1/huangzg/research/pyHGT_OAG'
# # # graph = renamed_load(open(os.path.join(ABSULUTE_DIR + '/data/oag_output', 'graph_CS.pk'), 'rb'))
# # # func = lambda graph,inp: print(graph.__dir__())
# # #
# # # # graph = Manager().Graph(graph)
# # # ray_graph = ray.put(graph)
# # ###########################
# #
# # import oaglog
# # from pyHGT.data import renamed_load
# # from pyHGT.model import *
# # from pyHGT.SubgraphToTorch import SubgraphToTorch
# # from warnings import filterwarnings
# # filterwarnings("ignore")
# # import ray
# # import os
# # import numpy as np
# # import dill
# # from collections import defaultdict
# # import sys
# # import argparse
# # oaglog.logger.info("流程开始。。。")
# # parser = argparse.ArgumentParser(description='Training GNN on Paper-Venue (Journal) classification task')
# #
# # '''
# #     Dataset arguments
# # '''
# # parser.add_argument('--data_dir', type=str, default='/data/oag_output/',
# #                     help='The address of preprocessed graph.')
# # parser.add_argument('--subgraphs_dir',type=str,default='/data/sampled_subgraphs/',
# #                     help='The adress of sampled subgraph.')
# # parser.add_argument('--model_dir', type=str, default='/model_save/',
# #                     help='The address for storing the models and optimization results.')
# # parser.add_argument('--task_name', type=str, default='PV',
# #                     help='The name of the stored models and optimization results.')
# # parser.add_argument('--cuda', type=int, default=0,
# #                     help='Avaiable GPU ID')
# # parser.add_argument('--domain', type=str, default='_CS',
# #                     help='CS, Medicion or All: _CS or _Med or (empty)')
# # '''
# #    Model arguments
# # '''
# # parser.add_argument('--conv_name', type=str, default='hgt',
# #                     choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
# #                     help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
# # parser.add_argument('--n_hid', type=int, default=400,
# #                     help='Number of hidden dimension')
# # parser.add_argument('--n_heads', type=int, default=8,
# #                     help='Number of attention head')
# # parser.add_argument('--n_layers', type=int, default=4,
# #                     help='Number of GNN layers')
# # parser.add_argument('--dropout', type=float, default=0.2,
# #                     help='Dropout ratio')
# # parser.add_argument('--sample_depth', type=int, default=6,
# #                     help='How many numbers to sample the graph')
# # parser.add_argument('--sample_width', type=int, default=128,
# #                     help='How many nodes to be sampled per layer per type')
# # parser.add_argument('--feature_flags', type=tuple, default=('rw','sp'),
# #                     help='which kind of distance feature to use,"random walk","shortest path" or both')
# # parser.add_argument('--max_sprw', type=tuple, default=(4, 4),
# #                     help='parameters of distance feature')
# # parser.add_argument('--if_sample_mp',type=bool, default=True,
# #                     help="whether sample subgraph with multiprocessing or not")
# # parser.add_argument('--sample_n_pool',type=int,default=16,
# #                     help="how many pools to sample subgraph")
# # '''
# #     Optimization arguments
# # '''
# # parser.add_argument('--optimizer', type=str, default='adamw',
# #                     choices=['adamw', 'adam', 'sgd', 'adagrad'],
# #                     help='optimizer to use.')
# # parser.add_argument('--data_percentage', type=float, default=1.0,
# #                     help='Percentage of training and validation data to use')
# # parser.add_argument('--n_epoch', type=int, default=200,
# #                     help='Number of epoch to run')
# # parser.add_argument('--n_pool', type=int, default=4,
# #                     help='Number of process to sample subgraph')
# # parser.add_argument('--n_batch', type=int, default=32,
# #                     help='Number of batch (sampled graphs) for each epoch')
# # parser.add_argument('--repeat', type=int, default=2,
# #                     help='How many time to train over a singe batch (reuse data)')
# # parser.add_argument('--batch_size', type=int, default=256,
# #                     help='Number of output nodes for training')
# # parser.add_argument('--clip', type=float, default=0.25,
# #                     help='Gradient Norm Clipping')
# #
# # args = parser.parse_args()
# #
# # if args.cuda != -1:
# #     device = torch.device("cuda:" + str(args.cuda))
# # else:
# #     device = torch.device("cpu")
# #
# # ABSULUTE_DIR = '/data1/huangzg/research/pyHGT_OAG'
# #
# # ###############################################data_preparing#########################################################
# # # oaglog.logger.info("读取数据开始。。。")
# # # graph = renamed_load(open(os.path.join(ABSULUTE_DIR + args.data_dir, 'graph%s.pk' % args.domain), 'rb'))
# # # oaglog.logger.info("读取数据完毕。")
# # #
# # # from ReadData import read_data, graph_to_dict
# # #
# # # dict_graph = graph_to_dict(graph)
# #
# # from multiprocess import Manager, Pool, SharedMemoryManager
# # # manager = Manager()
# # # graph_temp = manager.dict(dict_graph)
# #
# # graph = [np.ones(10**8) for i in range(20)]
# #
# # def mp_test(graph):
# #     print(id(graph))
# #     return 1
# #
# # p = Pool(6)
# #
# # result = p.apply_async(mp_test,graph_temp)
# # # @ray.remote
# # # class Counter(object):
# # #     def __init__(self,a):
# # #         self.n = 0
# # #         self.a = a
# # #     def increment(self):
# # #         self.n += 1
# # #
# # #     def read(self,b,m_graph):
# # #         print("a")
# # #         self.increment()
# # #         print(id(m_graph))
# # #         del m_graph
# # #         return self.n * b
# # #
# # # counters = [Counter.remote(a=0) for i in range(8)]
# # # futures = [c.read.remote(2, ray_graph) for c in counters]
# # #
# # # print('******************************')
# # # print(ray.get(futures))
# #
# # ray.init()
# # @ray.remote
# # def func(array, param):
# #     # print(array.job_id)
# #     # print(array.task_id)
# #     # print(array.size)
# #     # print(type(array))
# #     print(id(array))
# #     return 1
# #
# # # array = np.ones(10**6)
# # # Store the array in the shared memory object store once
# # # so it is not copied multiple times.
# # # graph =  {i: np.ones(10**8) for i in range(20)}
# # graph = [np.ones(10**8) for i in range(20)]
# # array_id = ray.put(graph)
# #
# # result_ids = [func.remote(array_id, i) for i in range(40)]
# # output = ray.get(result_ids)
# # #################################################################
# # #
# # # ray.get(ray_graph)
# # # import ray
# # # import asyncio
# # # ray.init()
# # #
# # # import asyncio
# # #
# # # @ray.remote
# # # class AsyncActor:
# # #     async def run_task(self):
# # #         print("started")
# # #         await asyncio.sleep(1) # Network, I/O task here
# # #         print("ended")
# # #
# # # actor = AsyncActor.remote()
# # # # All 50 tasks should start at once. After 1 second they should all finish.
# # # # they should finish at the same time
# # # ray.get([actor.run_task.remote() for _ in range(50)])
# # ###################################################################
# # # import ray
# # # import asyncio
# # # ray.init()
# # #
# # # @ray.remote(num_cpus=40)
# # # class AsyncActor:
# # #     # multiple invocation of this method can be running in
# # #     # the event loop at the same time
# # #     async def run_concurrent(self):
# # #         print("started")
# # #         await asyncio.sleep(2) # concurrent workload here
# # #         print("finished")
# # #
# # # actor = AsyncActor.remote()
# # #
# # # # regular ray.get
# # # ray.get([actor.run_concurrent.remote() for _ in range(80)])
# #
# # # # async ray.get
# # # await actor.run_concurrent.remote()
# #
# # ########################################################################
# from multiprocessing import Pool,Manager,shared_memory
# from multiprocessing.managers import SharedMemoryManager
# import numpy as np
#
#
# a = np.array([np.ones(10**8) for i in range(20)])
#
# shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
# b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
# b[:] = a[:]
#
# def mp_test(graph):
#     print(id(graph))
#     return 1
#
# p = Pool(6)
# results = []
# for i in range(3):
#     result = p.apply_async(mp_test, args=(b,))
#     results.append(result)
#
# re = [job.get() for job in results]
# ############################################################################
from multiprocessing import Pool
import multiprocessing as mp
from collections import defaultdict
import pandas as pd
import os

class NewClass(object):
    def __init__(self,
                 a):
        self.a = a
        self.b = {"a":a}
        self.c = pd.DataFrame(self.b)
        self.d = {"c":self.c, "b":self.b, "a":a}

    def my_method(self,e):
        print(id(self.a))
        print(id(self.b))
        print(id(self.c))
        print(id(self.d))
        print(id(e))
        defaultdict(lambda :[])
        return 1

graph = NewClass([1,3,6])

global graph

def my_fun(param,graph):

    print(os.getpid(), id(graph))

    return 1


def my_mp(param):
    my_fun(param, graph)


if __name__ == '__main__':

    p = Pool(5)
    jobs = []
    for i in range(mp.cpu_count()-1):
        job = p.apply_async(my_mp, args=(['a','b'],))
        jobs.append(job)
    result = [job.get() for job in jobs]

    print(result)







