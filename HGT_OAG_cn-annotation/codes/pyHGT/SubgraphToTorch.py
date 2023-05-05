# -*-  coding: utf-8 -*-
# @Time      :2021/3/23 10:42
# @Author    :huangzg28153
# @File      :SubGraphToTorch.py
# @Software  :PyCharm
from collections import defaultdict
from pyHGT.DistanceFeature import get_distance_feature
from pyHGT.IdentityFeature import compute_identity
import oaglog
import numpy as np
import os
import torch
# from multiprocess import Pool, Manager
# from pyHGT.data import Graph
# import ray
# Manager().register("Graph", Graph)


class SubgraphToTorch(object):
    """
    抽取子图，并生成特征，最后转化为torch.Tensor
    """
    def __init__(self,
                 types=None,
                 edge_dict=None,
                 time_range=None,
                 sampled_depth=2,
                 sampled_number=8,
                 target_relation=None,
                 feature_flag=('rw', 'sp'),
                 max_sprw=(4, 4),
                 emb_len=400,
                 use_distance_feature=True,
                 use_identity_feature=False
                 ):
        self.types = types
        self.edge_dict = edge_dict
        self.time_range = time_range
        self.sampled_depth = sampled_depth
        self.sampled_number = sampled_number
        self.target_relation = target_relation
        self.feature_flag = feature_flag
        self.max_sprw = max_sprw
        self.emb_len = emb_len
        self.use_distance_feature = use_distance_feature
        self.use_identity_feature = use_identity_feature

        oaglog.logger.debug("【进程 %d】开始子图抽取与组装模块处理。。。" % os.getpid())

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
            # tes = te[source_type]
            for relation_type in te[source_type]:
                # 只取异质关系，且抽样节点所在的关系下的边
                if relation_type == 'self' or target_id not in te[source_type][relation_type]:
                    continue
                # 针对目标节点所在关系下，目标节点的所有邻居节点
                # 如果邻居节点的数量小于预定抽样节点数，则全部抽取，sampled_ids即节点的全局ID;
                # 否则随机选择sampled_number个邻居节点
                adl = te[source_type][relation_type][target_id]
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
        # layer_data存储每个类型下，每个采样节点被采样时进入layer_data的顺序，以及
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
            # te = graph.edge_list[_type]
            for _id, _time in inp[_type]:
                budget = self.add_budget(graph.edge_list[_type],
                                         _id,
                                         _time,
                                         layer_data,
                                         budget,
                                         self.sampled_number,
                                         self.time_range)

        for layer in range(self.sampled_depth):
            sts = list(budget.keys())
            for source_type in sts:
                # te = graph.edge_list[source_type]
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
                    self.add_budget(   graph.edge_list[source_type],
                                       k,
                                       budget[source_type][k][1],
                                       layer_data,
                                       budget,
                                       self.sampled_number,
                                       self.time_range)
                    budget[source_type].pop(k)
        oaglog.logger.debug("采样一个节点的子图完毕")
        return layer_data

    def get_masked_edge_list(self, layer_data, graph, target_relation=None, mask_idx=None):
        """
        掩码edge_list,即起始节点在特定关系下的边全部删除；
        :param layer_data:
        :param graph:
        :return:
        """
        oaglog.logger.debug("获取未掩码后的边列表开始。。。")
        # 以ser为index的edge_list
        edge_list = defaultdict(  # target_type
            lambda: defaultdict(  # source_type
                lambda: defaultdict(  # relation_type
                    lambda: []  # [target_ser, source_ser]
                )))
        # 以原索引为Index的edge_list;
        # edge_list_original = defaultdict(lambda :defaultdict(lambda :defaultdict(lambda :[])))

        for _type in layer_data:
            for _key in layer_data[_type]:
                _ser = layer_data[_type][_key][0]
                edge_list[_type][_type]['self'] += [[_ser, _ser]]

        for target_type in graph.edge_list:

            # te = graph.edge_list[target_type]

            # tld = layer_data[target_type]
            for source_type in graph.edge_list[target_type]:

                # tes = graph.edge_list[target_type][source_type]

                # sld = layer_data[source_type]
                for relation_type in graph.edge_list[target_type][source_type]:

                    # tesr = graph.edge_list[target_type][source_type][relation_type]

                    for target_key in layer_data[target_type]:
                        if target_key not in graph.edge_list[target_type][source_type][relation_type]:
                            continue
                        target_ser = layer_data[target_type][target_key][0]
                        for source_key in graph.edge_list[target_type][source_type][relation_type][target_key]:
                            if source_key in layer_data[source_type]:

                                source_ser = layer_data[source_type][source_key][0]
                                # mask setting
                                # if not (([target_type,source_type,relation_type] ==
                                #          target_relation[0] and target_ser < mask_idx) or
                                #         ([target_type, source_type, relation_type] ==
                                #          target_relation[1] and source_ser < mask_idx)):

                                edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
                                    # edge_list_original[target_type][source_type][relation_type] += [[target_key, source_key]]
        oaglog.logger.debug("获取未掩码后的边列表结束。")
        return edge_list

    def get_intrinsic_feature(self, layer_data, graph, types, emb_len=400):
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
            #     ser.append(value[0])
                tims.append(value[1])
            #     layers.append(value[2])
            # graph.node_feature[type]:pd.DataFrame
            # 如果graph.node_feature[type]包括node_emb列，则赋给feature[type]
            # 如果不包括，则默认长度为400的向量为特征向量
            if 'node_emb' in graph.node_feature[_type]:
                feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
            else:
                feature[_type] = np.zeros([len(idxs), emb_len])
                # 合并，node_emb,emb,ciation数为整体特征向量
                # graph.node_feature: a DataFrame containing all features
            feature[_type] = np.concatenate((feature[_type],
                                             list(graph.node_feature[_type].loc[idxs, 'emb']),
                                             np.log10(np.array(list(graph.node_feature[_type].loc[
                                                       idxs, 'citation'])).reshape(-1, 1) + 0.01)
                                             ), axis=1)

            # 如果节点类型为paper，则
            # if _type == 'paper':
            #     texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
            times[_type] = tims
            # indx_pd = pd.DataFrame({"original_index": idxs, "layer": layers,"ser": ser})
            # indx_pd["type"] = _type
            # indxs.append(indx_pd)
        #
        # indxs = pd.concat(indxs,axis=0,ignore_index=True)
        oaglog.logger.debug("组装子图的固有属性完毕")
        return feature, times, indxs, texts

    def mask_edge_list(self,edge_list,targe_relation,mask_idx=256):
        """

        :param edge_list:
        :param targe_relation:
        :param mask_idx:
        :return:
        """
        masked_edge_list = []
        for i in edge_list[targe_relation[0][0]][targe_relation[0][1]][targe_relation[0][2]]:
            if i[0] >= mask_idx:
                masked_edge_list += [i]
        edge_list[targe_relation[0][0]][targe_relation[0][1]][targe_relation[0][2]] = masked_edge_list

        masked_edge_list = []
        for i in edge_list[targe_relation[1][0]][targe_relation[1][1]][targe_relation[1][2]]:
            if i[0] >= mask_idx:
                masked_edge_list += [i]
        edge_list[targe_relation[1][0]][targe_relation[1][1]][targe_relation[1][2]] = masked_edge_list

        return edge_list

    def to_torch(self, types, feature, time, edge_list, edge_dict):
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

        # 构造节点数字典，node_num为每个类型下的节点数的累加和，
        # len(node_dict)作为类型的索引
        # # 自定义，type_num为该类型下的节点数，
        for t in types:
            type_num = len(feature[t])
            node_dict[t] = [node_num, len(node_dict), type_num]
            node_num += type_num

        for t in types:
            # 节点和时间特征
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
                        try:
                            edge_time += [node_time[tid] - node_time[sid] + 120]
                        except Exception as e:
                            print(e)
                            print(node_time)
                            print(node_time[tid],node_time[sid])

        node_feature = torch.FloatTensor(node_feature)
        node_type = torch.LongTensor(node_type)
        edge_time = torch.LongTensor(edge_time)
        # 构造networkx.Graph()的edge_index必须为x*2,但
        edge_index = torch.LongTensor(edge_index).t()
        edge_type = torch.LongTensor(edge_type)
        # node_dict用于确定节点的起始点索引，但在本代码中“paper"节点的起始点必然为0
        result = [node_feature, node_type,
                  edge_time, edge_index, edge_type]
        oaglog.logger.debug("组装为无类型的特征、边列表完毕。。")
        return result, node_num

    # @ray.method(num_returns=1)
    def full_result_sampling(self, inp, graph):
        """
        返回一次采样的全部输出；
        :param inp:输入的{"paper":[target_id,time]}构成的列表
        :param graph: 输入的完整图
        :return:
        """
        batch_size = len(list(inp.values())[0])
        layer_data = self.extract_subgraph_one_sample(inp, graph)
        feature, times, indxs, texts = self.get_intrinsic_feature(layer_data, graph, self.types, emb_len=self.emb_len)
        edge_list = self.get_masked_edge_list(layer_data, graph, self.target_relation, mask_idx=batch_size)
        edge_list = self.mask_edge_list(edge_list, self.target_relation, mask_idx=batch_size)
        result_sample, node_num = self.to_torch(self.types, feature, times, edge_list, self.edge_dict)
        if self.use_distance_feature:
            node_feature = get_distance_feature(
                                                np.arange(batch_size),
                                                result_sample[3],
                                                result_sample[0].numpy(),
                                                node_num,
                                                self.feature_flag,
                                                self.max_sprw
                                                )
            result_sample[0] = torch.FloatTensor(node_feature)
        if self.use_identity_feature:
            identity_feature = compute_identity(torch.Tensor(edge_list), node_num, self.sampled_depth)

            result_sample[0] = torch.cat([result_sample[0], identity_feature],dim=1)

        oaglog.logger.debug("【进程 %d】 上一个节点的全部数据生成完毕." % os.getpid())
        return result_sample


def get_id_label(task_name, target_id_choice, cand_list, pairs):

    batch_size = len(target_id_choice)
    if task_name == "PV":
        ylabel = torch.zeros(batch_size, dtype=torch.long)
        for x_id, target_id in enumerate(target_id_choice):
            # target_id = target_id_dict['paper'][0][0]

            # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1
            ylabel[x_id] = cand_list.index(pairs[target_id][0])

        # x_ids = np.arange(args.batch_size) + node_dict['paper'][0]

    elif task_name == "PF":
        ylabel = np.zeros([batch_size, len(cand_list)])
        for x_id, target_id in enumerate(target_id_choice):
            # target_id = target_id_dict['paper'][0][0]
            for source_id in pairs[target_id][0]:
                # 从cand_list中搜寻source_id的index,然后令ylabel[x_id]中对应的index值为1,
                ylabel[x_id][cand_list.index(source_id)] = 1
                # 求ylabel的列归一化值,作为连接概率值的标签
        ylabel /= ylabel.sum(axis=1).reshape(-1, 1)

    else:
        pass
    x_ids = np.arange(batch_size)
    return x_ids, ylabel

    # def mp_sample_subgraphs(self, inp_list, graph):
    #     """
    #     多线程采样所有输入节点的子图
    #     :return:
    #     """
    #     pool = Pool(self.sample_n_pool)
    #     jobs = []
    #
    #     for num, inp in enumerate(inp_list):
    #         oaglog.logger.info("第 %d 个节点处理开始。。。" % num)
    #         p = pool.apply_async(self.full_result_one_sample.remote, (inp, graph))
    #         jobs.append(p)
    #     result = [job.get(36000) for job in jobs]
    #     result = ray.get(result)
    #     pool.close()
    #     pool.join()
    #
    #     return result

    # def ray_sample_subgraphs(self, inp_list, graph):
    #     """
    #     使用ray进行采样
    #     :param inp_list:
    #     :param graph:
    #     :return:
    #     """
    #
    #     ray_graph = ray.put(graph)
    #     oaglog.logger.info("使用ray进行多进程采样")
    #     jobs = [self.full_result_one_sample.remote(inp, ray_graph) for inp in inp_list]
    #     result = ray.get(jobs)
    #     return result

    # def single_sample_subgraphs(self, inp_list, graph):
    #     """
    #     单线程采样所有输入节点的子图
    #     :return:
    #     """
    #
    #     result_list = []
    #     for num, inp in enumerate(inp_list):
    #         oaglog.logger.info("第 %d 个节点处理开始。。。" % num)
    #         result = self.full_result_one_sample(inp, graph)
    #         result_list.append(result)
    #         oaglog.logger.info("第 %d 个节点处理完成。" %num)
    #     return result_list












