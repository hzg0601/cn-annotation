# -*-  coding: utf-8 -*-
# @Time      :2021/3/23 10:42
# @Author    :huangzg28153
# @File      :SubGraphToTorch.py
# @Software  :PyCharm
from collections import defaultdict
import networkx as nx
import oaglog
import numpy as np
import pandas as pd
import torch
from pathos.multiprocessing import ProcessPool as Pool
from pyHGT.data import Graph
import multiprocessing as mp
mp.Manager().register("Graph", Graph)

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


class SubgraphToTorch(object):
    """
    抽取子图，并生成特征，最后转化为torch.Tensor
    """
    def __init__(self,
                 # graph,
                 types=None,
                 edge_dict=None,
                 time_range=None,
                 # inp_list=None,
                 sampled_depth=2,
                 sampled_number=8,
                 if_sample_mp=False,
                 sample_n_pool=4,
                 target_relation=None,
                 feature_flag=('rw', 'sp'),
                 max_sprw=(4, 4)):
        # self.graph = graph
        self.types = types
        self.edge_dict = edge_dict
        self.time_range = time_range
        # self.inp_list = inp_list
        self.sampled_depth = sampled_depth
        self.sampled_number = sampled_number
        self.if_sample_mp = if_sample_mp
        self.sample_n_pool = sample_n_pool
        self.target_relation = target_relation
        self.feature_flag = feature_flag
        self.max_sprw = max_sprw

        # self.types = list(self.graph.get_types())
        # # 元关系字典，最后一个为"self"类型
        # self.edge_dict = {e[2]: i for i, e in enumerate(self.graph.get_meta_graph())}
        # self.edge_dict['self'] = len(self.edge_dict)

        oaglog.logger.info("开始子图抽取与组装模块处理。。。")

    def add_budget(self,
                   te,
                   target_id,
                   target_time,
                   layer_data,
                   budget,
                   sampled_number,
                   time_range):
        """
        从inp亦即layer_data中的type为target_type,每个source_type下所有relation_type的至多
        sampled_number个节点
        :param te: target_edge,即Graph.edge_list[target_type]
        :param target_id: target_id, 节点的ID，并非节点的索引ser;ser为特定类型下进入该类型的顺序，以顺序为索引；
        :param target_time:目标节点的时间
        :param layer_data:抽样的节点类型对应layer_data[node_type][node_id]=[node_ser,time]
        :param budget: 抽样的预算集合，budget[source_type][source_id] = [sampled_score,time]
        :return:目标类型下、目标节点target_id在每个关系下的至多sampled_number个节点组成的字典，
                第一轮，budget[source_type,所有与paper相连的类型数len][source_id,至多len*sampled_number个]
        """
        # 针对给定的target_type，target_id，抽取每个source_type下target_id的至多sampled_number个节点，
        # 记录每类source_type的sampled_score，和时间
        # te,graph.edge_list['paper']
        # source_type,graph.edge_list['paper']下的所有类型
        for source_type in te:
            # tes，graph.edge_list['paper']下的所有类型
            tes = te[source_type]
            for relation_type in tes:
                # 只取异质关系，且抽样节点所在的关系下的边
                if relation_type == 'self' or target_id not in tes[relation_type]:
                    continue
                # 针对目标节点所在关系下，目标节点的所有邻居节点
                # 如果邻居节点的数量小于预定抽样节点数，则全部抽取，sampled_ids即节点的全局ID;
                # 否则随机选择sampled_number个邻居节点
                adl = tes[relation_type][target_id]
                if len(adl) < sampled_number:
                    sampled_ids = list(adl.keys())
                else:
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace=False)
                # 对于每个被抽样节点，取出其source_time,如果没有，则将预定的target_time赋给source_time
                for source_id in sampled_ids:
                    source_time = adl[source_id]
                    if source_time == None:
                        source_time = target_time
                    # 如果抽样的source_id,已经在layer_data[source_type]中，则不再将其加入budget中；
                    # 否则将其加入budget中，其sampled_score为抽样节点数的倒数，time为source_time
                    if source_id in layer_data[source_type] or source_time > np.max(list(time_range.keys())):
                        continue
                    budget[source_type][source_id][0] += 1. / len(sampled_ids)
                    budget[source_type][source_id][1] = source_time

        return budget

    def extract_subgraph_one_sample(self, inp, graph):
        """
        为一个输入样本抽取子图
        :param inp: 输入样本，dict格式{type:[original_id,time]}
        :return:抽样的一个子图的字典，其格式为{type:{id:[ser,time]}}
        """
        oaglog.logger.debug("采样一个节点的子图开始。。。")
        layer_data = defaultdict(  # target_type
            lambda: {} ) # {target_id: [ser, time]} # ser即其索引，加入dict的顺序
        budget = defaultdict(  # source_type
            lambda: defaultdict(  # source_id
                lambda: [0., 0]  # [sampled_score, time]
            ))

        for _type in inp:
            for _id, _time in inp[_type]:
                layer_data[_type][_id] = [len(layer_data[_type]), _time, 0]

        for _type in inp:
            te = graph.edge_list[_type]
            for _id, _time in inp[_type]:
                budget = self.add_budget(te, _id, _time, layer_data, budget, self.sampled_number, self.time_range)

        for layer in range(self.sampled_depth):
            sts = list(budget.keys())
            for source_type in sts:
                te = graph.edge_list[source_type]
                keys = np.array(list(budget[source_type].keys()))
                if self.sampled_number > len(keys):

                    sampled_ids = np.arange(len(keys))
                else:
                    score = np.array(list(budget[source_type].values()))[:, 0] ** 2
                    score = score / np.sum(score)
                    sampled_ids = np.random.choice(len(score),
                                                   self.sampled_number,
                                                   p=score,
                                                   replace=False)
                sampled_keys = keys[sampled_ids]

                for k in sampled_keys:
                    layer_data[source_type][k] = [len(layer_data[source_type]),
                                                  budget[source_type][k][1],
                                                  layer + 1]
                for k in sampled_keys:
                    self.add_budget(   te,
                                       k,
                                       budget[source_type][k][1],
                                       layer_data,
                                       budget,
                                       self.sampled_number,
                                       self.time_range)
                    budget[source_type].pop(k)
        oaglog.logger.debug("采样一个节点的子图完毕")
        return layer_data

    def get_masked_edge_list(self, layer_data, graph, target_relation):
        """
        掩码edge_list,即起始节点在特定关系下的边全部删除；
        :param layer_data:
        :param graph:
        :return:
        """
        oaglog.logger.debug("获取掩码后的边列表开始。。。")
        edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: []  # [target_ser, source_ser]
                )))
        for _type in layer_data:
            for _key in layer_data[_type]:
                _ser = layer_data[_type][_key][0]
                edge_list[_type][_type]['self'] += [[_ser, _ser]]

        for target_type in graph.edge_list:

            te = graph.edge_list[target_type]

            tld = layer_data[target_type]
            for source_type in te:

                tes = te[source_type]

                sld = layer_data[source_type]
                for relation_type in tes:

                    tesr = tes[relation_type]

                    for target_key in tld:
                        if target_key not in tesr:
                            continue
                        target_ser = tld[target_key][0]
                        for source_key in tesr[target_key]:
                            if source_key in sld:

                                source_ser = sld[source_key][0]
                                if not (([target_type,source_type,relation_type] ==
                                         target_relation[0] and target_ser == 0) or
                                        ([target_type, source_type, relation_type] ==
                                         target_relation[1] and source_ser == 0)):

                                    edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
        oaglog.logger.debug("获取掩码后的边列表结束。")
        return edge_list

    def get_intrinsic_feature(self, layer_data, graph, types):
        """
        用于合并图的特征；返回的是有类型的特征、时间、索引、文本
        :param layer_data: [type][id]:[len(layer_data),time]，采样节点类型：ID的字典
        :param graph:Graph()类型的图
        :return:图的特征
        """
        oaglog.logger.debug("组装子图的固有属性开始。。。")
        feature = defaultdict(lambda :np.array([]))
        times = defaultdict(lambda :[])
        indxs = []
        texts = []
        for _type in types:
            if len(layer_data[_type]) == 0:
                continue
            idxs, ser, layers, tims = [], [], [], []
            for key, value in layer_data[_type].items():
                idxs.append(key)
                ser.append(value[0])
                tims.append(value[1])
                layers.append(value[2])
            # graph.node_feature[type]:pd.DataFrame
            # 如果graph.node_feature[type]包括node_emb列，则赋给feature[type]
            # 如果不包括，则默认长度为400的向量为特征向量
            if 'node_emb' in graph.node_feature[_type]:
                feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
            else:
                feature[_type] = np.zeros([len(idxs), 400])
                # 合并，node_emb,emb,ciation数为整体特征向量
                # graph.node_feature: a DataFrame containing all features
            feature[_type] = np.concatenate((feature[_type],
                                             list(graph.node_feature[_type].loc[idxs, 'emb']),
                                             np.log10(np.array(list(graph.node_feature[_type].loc[
                                                       idxs, 'citation'])).reshape(-1, 1) + 0.01)), axis=1)

            # 如果节点类型为paper，则
            if _type == 'paper':
                texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
            times[_type] = tims
            indx_pd = pd.DataFrame({"original_index": idxs, "layer": layers,"ser": ser})
            indx_pd["type"] = _type
            indxs.append(indx_pd)
        #
        indxs = pd.concat(indxs,axis=0,ignore_index=True)
        oaglog.logger.debug("组装子图的固有属性完毕")
        return feature, times, indxs, texts

    def to_numpy_one_sample(self, types, feature, time, edge_list, edge_dict):
        """
        返回无类型的edge_index、列表形式的feture,time,并统计各类型下的节点数目
        此处的edge_list也为掩码后的edge_list；
        :param types:
        :param feature:
        :param time:
        :param edge_list:
        :param edge_dict:
        :return:
        """
        oaglog.logger.debug("组装为无类型的特征、边列表开始。。。")
        node_dict = {}
        node_feature = []
        edge_index = []

        edge_type = []
        edge_time = []
        node_time = []

        node_num = 0
        node_type = []
        # 须定义好类型的顺序；

        # 构造节点数字典，node_num为每个类型下子图的节点数，
        # len(node_dict)作为类型的索引
        for t in types:
            type_num = len(feature[t])
            node_dict[t] = [node_num, len(node_dict), type_num]
            node_num += type_num

        for t in types:
            node_feature += list(feature[t])
            node_time += list(time[t])
            # 节点类型掩码
            node_type += [node_dict[t][1] for _ in range(len(feature[t]))]

        for target_type in edge_list:
            for source_type in edge_list[target_type]:
                for relation_type in edge_list[target_type][source_type]:
                    # ti,target_id;si,source_id，新体系的索引，每个都是从0开始；
                    for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                        # 加入上一个类型节点数，作为新的节点索引，构成无类型的边列表
                        tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                        edge_index += [[sid, tid]]
                        # 边类型掩码
                        edge_type += [edge_dict[relation_type]]
                        # 以起始时间为参考点，记录时间差
                        edge_time += [node_time[tid] - node_time[sid] + 120]

        result = [node_feature, node_num, node_type, node_dict,
                  edge_time, edge_index, edge_type]
        oaglog.logger.debug("组装为无类型的特征、边列表完毕。。")
        return result

    # @timeit_method
    def get_distance_feature(self,
                             edge_index,
                             node_feature,
                             node_num,
                             feature_flags,
                             max_sprw):
        """
        生成distance encoding特征；
        :param edge_index:
        :param node_feature:
        :param node_num:
        :param feature_flags:
        :param max_sprw:
        :return: 包含距离特征的全部特征；
        """
        oaglog.logger.debug("计算distance encoding特征开始。。。")
        new_set_index = np.array([0])
        # 生成distance_encoding特征,根据每个子图的新的edge_index,
        try:
            new_G = nx.from_edgelist(edge_index, create_using=type(nx.Graph()))
        except Exception as e:
            print(e)
            print(len(edge_index))
        new_G.add_nodes_from(np.arange(node_num), dtype=np.int32)

        sp_flag, rw_flag = feature_flags
        max_sp, rw_depth = max_sprw
        if sp_flag:
            features_sp_sample = get_features_sp_sample(new_G, new_set_index, max_sp=max_sp)

            node_feature = np.concatenate((np.array(node_feature), features_sp_sample), axis=1)
        if rw_flag:
            adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(),
                                                                           dtype=np.int32)).todense().astype(
                np.float32))  # [n_nodes, n_nodes]
            features_rw_sample = get_features_rw_sample(adj, new_set_index, rw_depth=rw_depth)

            node_feature = np.concatenate((np.array(node_feature), features_rw_sample), axis=1)

        oaglog.logger.info("计算distance encoding特征完毕。")
        return node_feature

    def full_result_one_sample(self, inp, graph):
        """
        返回一个节点的全部输出；
        :param inp:
        :param types:
        :param edge_dict:
        :return:
        """

        layer_data = self.extract_subgraph_one_sample(inp, graph)
        feature, times, indxs, texts = self.get_intrinsic_feature(layer_data, graph, self.types)
        edge_list = self.get_masked_edge_list(layer_data, graph, self.target_relation)
        result_sample = self.to_numpy_one_sample(self.types, feature, times, edge_list, self.edge_dict)
        node_feature = self.get_distance_feature(result_sample[5],
                                                 result_sample[0],
                                                 result_sample[1],
                                                 self.feature_flag,
                                                 self.max_sprw)
        result_sample[0] = node_feature
        oaglog.logger.debug("一个节点的全部数据生成完毕.")
        return result_sample

    def mp_sample_subgraphs(self, inp_list, graph):
        """
        多线程采样所有输入节点的子图
        :return:
        """
        pool = Pool(self.sample_n_pool)
        jobs = []


        for num, inp in enumerate(inp_list):
            oaglog.logger.info("第 %d 个节点处理开始。。。" % num)
            p = pool.amap(self.full_result_one_sample, inp, graph)
            jobs.append(p)
        result = [job.get(36000) for job in jobs]
        pool.close()
        pool.join()
        return result

    def sample_all_subgraphs(self, inp_list, graph):
        """
        单线程采样所有输入节点的子图
        :return:
        """

        result_list = []
        for num, inp in enumerate(inp_list):
            oaglog.logger.info("第 %d 个节点处理开始。。。" % num)
            result = self.full_result_one_sample(inp, graph)
            result_list.append(result)
            oaglog.logger.info("第 %d 个节点处理完成。" %num)
        return result_list

    # @timeit_method
    def assemble_result(self, inp_list, graph):
        """
        将输入的batch的结果组装起来，返回
        :return:
        """
        # # 类型列表
        # types = list(self.graph.get_types())
        # # 元关系字典，最后一个为"self"类型
        # edge_dict = {e[2]: i for i, e in enumerate(self.graph.get_meta_graph())}
        # edge_dict['self'] = len(edge_dict)

        # indxs_list = []
        node_feature_list = []
        node_num_list = [0]
        node_type_list = []

        # {type:[num_type,idx_type]}类型与类型下节点数的字典
        node_dict_list = []
        node_num_dict = defaultdict(lambda :[])

        edge_time_list = []
        edge_index_list = []
        edge_type_list = []
        # 多进程抽样子图；
        # if self.if_sample_mp:
        #     result_sample_all = self.mp_sample_subgraph(self.sample_n_pool, self.inp_list, types, edge_dict)
        # else:
        #     result_sample_all = [self.full_result_one_sample(inp, types, edge_dict) for inp in self.inp_list]
        #
        # for num, result_sample in enumerate(result_sample_all):
        for num, inp in enumerate(inp_list):
            result_sample = self.full_result_one_sample(inp, graph)

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
            edge_index = [[pair[0]+np.sum(node_num_list),
                           pair[1]+np.sum(node_num_list)]
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
        for t in self.types:
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

        return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, self.edge_dict,node_num_list











