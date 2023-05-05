# -*-  coding: utf-8 -*-
# @Time      :2021/3/18 13:47
# @Author    :huangzg28153
# @File      :data_my.py
# @Software  :PyCharm

from collections import defaultdict
# import sys
# import os
# sys.path.append(os.getcwd()+'/utils.py')
from pyHGT.utils import *

import networkx as nx
import dill
from pyHGT.data import Graph
import oaglog

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
    temp = onehot_encoding[sp_length]  # 将onehot_encoding的每一行都视为一个元素，根据sp_length的值取对应元素，组成一个len(G)*set_size*dim的新数组
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
    features_rw = features_rw_tmp.sum(axis=0)  # shape [G.num_of_nodes,rw_depth+1],求和得到访问每个节点的频数？
    return features_rw


def add_budget(te, target_id, target_time, layer_data, budget,sampled_number):
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
                if source_id in layer_data[source_type]:
                    continue
                budget[source_type][source_id][0] += 1. / len(sampled_ids)
                budget[source_type][source_id][1] = source_time

    return budget


def feature_OAG(layer_data, graph):
    """
    用于合并图的特征；返回的是有类型的特征、时间、索引、文本
    :param layer_data: [type][id]:[len(layer_data),time]，采样节点类型：ID的字典
    :param graph:Graph()类型的图
    :return:图的特征
    """
    feature = {}
    times = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue

        idxs = np.array(list(layer_data[_type].keys()))
        new_idxs = np.array(list(layer_data[_type].values())[0])
        layer = np.array(list(layer_data[_type].values())[-1])
        idxs = np.concatenate((idxs, new_idxs, layer), axis=1)
        # 对应节点的时间
        tims = np.array(list(layer_data[_type].values()))[:, 1]
        # graph.node_feature[type]:pd.DataFrame
        # 如果graph.node_feature[type]包括node_emb列，则赋给feature[type]
        # 如果不包括，则默认长度为400的向量为特征向量
        if 'node_emb' in graph.node_feature[_type]:
            feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'node_emb']), dtype=np.float)
        else:
            feature[_type] = np.zeros([len(idxs), 400])
            # 合并，node_emb,emb,ciation数为整体特征向量
            # graph.node_feature: a DataFrame containing all features
        feature[_type] = np.concatenate((feature[_type], list(graph.node_feature[_type].loc[idxs, 'emb']),
                                         np.log10(np.array(list(graph.node_feature[_type].loc[idxs,
                                                    'citation'])).reshape(-1,1) + 0.01)), axis=1)

        times[_type] = tims
        indxs[_type] = idxs
        # 如果节点类型为paper，则
        if _type == 'paper':
            texts = np.array(list(graph.node_feature[_type].loc[idxs, 'title']), dtype=np.str)
    return feature, times, indxs, texts


def sample_subgraph_for_one_sample(graph, sampled_depth=2, sampled_number=8, inp=None):
    """
    :param graph:
    :param sampled_depth:
    :param sampled_number:
    :param inp: 需要把inp内的列表拆开变为[{type:[target_id,target_time]},for i in 0:len(pair)]
    :param feature_extractor:
    :return:
    """

    layer_data  = defaultdict( #target_type
                        lambda: {}  # {target_id: [ser, time]} # ser即其索引，加入dict的顺序
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))

    for _type in inp:
        for _id, _time in inp[_type]:
            # target_id,在_type(paper)下的数字顺序(node_forward下)，
            # len(layer_data[_type])作为新的id，0-batch_size,
            #
            # 初始为layer_data["paper"]["paper_target_id_range"] = [len,_time]
            layer_data[_type][_id] = [len(layer_data[_type]), _time, 0]

    for _type in inp:
        # te, target edge_list;
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            # 针对inp["paper"]下的每个节点，以paper为target_type,更新budget
            # 包括所有以paper为target_type的类型，paper,field,venue,author
            budget = add_budget(te, _id, _time, layer_data, budget,sampled_number)

    for layer in range(sampled_depth):
        # sts,source_types的list
        # papaer,field,venue,author
        sts = list(budget.keys())
        # 针对抽样budget中的每种source_type,
        # 取出其作为target_type的边列表，即te;
        # 再取出source_type下所有source_id;
        # 如果每种source_type下抽样得到的节点数大于sampled_number,
        # 则按累积归一化累积sampled_score随机抽取sampled_number个
        # 否则全部抽取
        for source_type in sts:
            te = graph.edge_list[source_type]
            # keys为节点ID的集合，而sampled_ids为ser,即节点在该节点类型下的顺序index；
            # 还是paper,field,venue,author
            # 此处的keys还是node_forward里的节点ID
            keys  = np.array(list(budget[source_type].keys()))
            if sampled_number > len(keys):
                '''
                    Directly sample all the nodes
                '''
                sampled_ids = np.arange(len(keys))
            else:
                '''
                    Sample based on accumulated degree
                '''
                score = np.array(list(budget[source_type].values()))[:,0] ** 2
                score = score / np.sum(score)
                sampled_ids = np.random.choice(len(score), sampled_number, p = score, replace = False)
                # 根据sampled_ids抽取节点ID，forward里的ID
            sampled_keys = keys[sampled_ids]
            '''
                First adding the sampled nodes then updating budget.
            '''
            # 根据抽取的节点ID更新layer_data,budget
            # 首先将采样到的所有节点放入layer_data中，如venue下sampled_number个节点，也可能会采样到
            # paper,但该paper下的节点必然不在inp[paper]下，因此仍可以插入节点

            # 然后以采样到的节点为target节点，采样对应所有source_type下len(realation_type)*sampled_data个节点的采样
            # 最后在budget中删除该节点ID，防止在第二层抽样中反复被抽样？？
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1],layer+1]
            #
            for k in sampled_keys:
                # budget[source_type][k][1]为target_time;
                # 如source_type,为venue, 采样到venue下的某个ID，就会以venue为target_type,取出边；
                # 增加venue为出发target_type, source_type下的采样节点
                add_budget(te, k, budget[source_type][k][1], layer_data, budget,sampled_number)
                budget[source_type].pop(k)

    edge_list = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_ser, source_ser]
            )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
        根据layer_data，构造layer_data子图的边列表，对于layer_data中的任意两个节点，
        如果它们存在于graph.edge_list中，则记录二者的ser,作为节点的新index.
        因此，节点在节点类型下是连续的，在关系下是不连续的；
    '''
    # for target_type in graph.edge_list:
    #     tld = layer_data[target_type]
    #     for source_type in graph.edge_list[target_type]:
    #         sld = layer_data[source_type]
    #         for relation_type in graph.edge_list[target_type][source_type]:
    #             for target_id in tld:
    #                 if target_id not in graph.edge_list[target_type][source_type][relation_type]:
    #                     continue
    #                 for source_id in graph.edge_list[target_type][source_type][relation_type][target_id]:
    #                     if source_id in sld:
    #                         target_ser = tld[target_id][0]
    #                         source_ser = sld[source_id][0]
    #                         edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]

    for target_type in graph.edge_list:
        # 目标类型下的target_edge字典，[source_type][relation_type][target_id][soruce_id]=time
        te = graph.edge_list[target_type]
        # layer_data, [target_type][target_id] = [[ser,time]],
        # 目标类型下的layer_data, target_id:[[ser,time]],
        tld = layer_data[target_type]
        for source_type in te:
            # tes,[relation_type][target_id][source_id]=time
            tes = te[source_type]
            # sld ,source_layer_data, source_id:[[ser,time]],
            sld = layer_data[source_type]
            for relation_type in tes:
                # tesr, [target_id][soruce_id]=time，target_edge_list_source_relation
                tesr = tes[relation_type]
                # 如果元关系下，target_layer_data中的target_id不在元关系边列表下，则不考虑该情况
                # 否则抽取taget_id在target_layer_data中对应的ser,即索引；
                for target_key in tld:
                    if target_key not in tesr:
                        continue
                    target_ser = tld[target_key][0]
                    for source_key in tesr[target_key]:
                        '''
                            Check whether each link (target_id, source_id) exist in original adjacency matrix
                        '''
                        # 如果元关系指定目标节点ID下的源节点ID在source_layer_data中，
                        # 则取出节点ID对应的target_layer_data中对应的ser,即索引；
                        # 索引的二层列表；
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]

    return layer_data, edge_list

def distance_feature_to_numpy(edge_list, feature, time, types, edge_dict,
                              feature_flags, max_sprw):
    '''
    1，生成distance_feature;2,把不同类型的数据转换为无类型numpy
    edge_list:edge_list
    feature:
    graph:
    inp:
    time:
    feature_flags:
    max_sprw:

    '''
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
        node_dict[t] = [node_num, len(node_dict)]
        node_num += len(feature[t])
    # # 将所有节点取出，组成为一个无类型的node_feature、node_time列表
    # # node_type为每个节点类型的掩码
    # for t in types:
    #     node_feature += list(feature[t])
    for t in types:
        node_feature += list(feature[t])
        node_time += list(time[t])
        node_type += [node_dict[t][1] for _ in range(len(feature[t]))]

    # 记录输入节点的新索引，由于抽取子图时，待抽样样本必然第一个被加入layer_data,其ser必然为0
    # 因此其新索引为 node_num[type]
    # node_num[inp.keys()[0]]
    new_set_index = np.array([0])



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

    # 生成distance_encoding特征
    # 根据每个子图的新的edge_index,
    new_G = nx.from_edgelist(edge_index,create_using=type(nx.Graph()))
    new_G.add_nodes_from(np.arange(node_num), dtype=np.int32)

    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw
    if sp_flag:
        features_sp_sample = get_features_sp_sample(new_G, new_set_index, max_sp=max_sp)

        node_feature = np.concatenate(np.concatenate(node_feature, axis=0), features_sp_sample,axis=1)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(new_G,nodelist=np.arange(new_G.number_of_nodes(),
                      dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj, new_set_index, rw_depth=rw_depth)

        node_feature = np.concatenate(np.concatenate(node_feature, axis=0), features_rw_sample,axis=1)
    # # feature类型字典
    # for i, t in enumerate(types):
    #     if i <= len(types) - 2:
    #         feature[t] = node_feature[node_dict[types[i]]:node_dict[types[i+1]], :]
    # feature[types[-1]] = node_feature[node_dict[types[-1]]:, :]
    #

    return node_feature, node_type, node_dict, edge_time, edge_index, edge_type


def generate_full_data_one_node(graph, sampled_depth, sampled_number,
                                inp, types, edge_dicts,
                                feature_flags, max_sprw):
    """

    :param graph:
    :param sampled_depth:
    :param sampled_number:
    :param inp:
    :param feature_flags:
    :param max_sprw:
    :return:
    """
    layer_data, edge_list = sample_subgraph_for_one_sample(graph,
                                                          sampled_depth=sampled_depth,
                                                          sampled_number=sampled_number,
                                                          inp=inp)
    feature, times, indxs, texts = feature_OAG(layer_data, graph)
    node_feature, node_type, node_dict, edge_time, edge_index, edge_type = distance_feature_to_numpy(
        edge_list=edge_list, feature=feature, types=types,
        time=times, edge_dict=edge_dicts,
        feature_flags=feature_flags, max_sprw=max_sprw)
    return node_feature, node_type, node_dict, edge_time, edge_index, edge_type, edge_list

# 将单个节点重新排列为to_torch的形式

def assemble_to_torch(inp_list,
                      graph,
                      sampled_depth,
                      sampled_number,
                      feature_flags,
                      max_sprw):
    """
    :param inp_list:
    :param graph:
    :param sampled_depth:
    :param sampled_number:
    :param feature_flags:
    :param max_sprw:
    :return:
    """
    # meta_graph,[(target_type, source_type, r_type)],元关系列表
    # 元关系：索引字典 ; 最后增加一个自关系类型
    edge_dicts = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dicts['self'] = len(edge_dicts)

    types = graph.get_types()
    # 节点特征的np.array
    node_features = defaultdict(lambda: np.array([]))
    # 节点类型的掩码np.array
    node_types = defaultdict(lambda: np.array([]))
    # 节点类型：数目字典

    node_dicts = defaultdict(lambda: [])
    # 节点的相对时间差np.array
    edge_times = defaultdict(lambda: np.array([]))
    # 边元关系类型类型np.array
    edge_types = defaultdict(lambda: np.array([]))

    ser_index = np.array([0, 0])
    edge_indexs = []

    edge_lists = defaultdict(  # target_type
        lambda: defaultdict(  # source_type
            lambda: defaultdict(  # relation_type
                lambda: []  # [target_ser, source_ser]
            )))
    # # edge_index构造为大edge_index的参数
    # for target_type in graph.edge_list:
    #     for source_type in graph.edge_list[target_type]:
    #         for relation in graph.edge_list[target_type][source_type]
    #             if edge_list[target_type][source_type][relation]:
    #                 edge_list_all[target_type][source_type][relation] = edge_list[
    #                             target_type][source_type][relation] +
    edge_list_list = []

    for inp in inp_list:
        # 把每个节点的
        node_feature, node_type, node_dict, edge_time, edge_index, edge_type, edge_list = generate_full_data_one_node(
            graph, sampled_depth, sampled_number, inp, types, edge_dicts, feature_flags, max_sprw)
        # node_features,node_types,edge_times,edge_types的组装，其为np.array
        for full, sub in zip([node_features, node_types, edge_times, edge_types],
                             [node_feature, node_type, edge_time, edge_type]):
            full = np.concatenate(full, sub, axis=0)
        # 拼接node_dict,其为字典，key为type,value为[
        for t, value in node_dict.items():
            node_dicts[t][0] = node_dicts[t][0] + value[0]
            node_dict[t][1] = value[1]
        # edge_index的组装
        edge_indexs += [[pair[0] + ser_index[0], pair[1]+ser_index[1]] for pair in edge_index]
        ser_index = np.array(edge_index).max(axis=0)

        edge_list_list.append(edge_list)

    for i,edge_list in enumerate(edge_list_list):
        pass


    # to_torch
    node_features = torch.FloatTensor(node_features)
    node_types = torch.LongTensor(node_types)
    edge_times = torch.LongTensor(edge_times)
    edge_indexs = torch.LongTensor(edge_indexs).t()
    edge_types = torch.LongTensor(edge_types)

    return node_features, node_types, edge_times, edge_indexs, edge_types, node_dicts, edge_dicts

# def to_torch_new(feature, time, edge_index,graph):
    # edge_lists = defaultdict(  # target_type
    #     lambda: defaultdict(  # source_type
    #         lambda: defaultdict(  # relation_type
    #             lambda: []  # [target_ser, source_ser]
    #         )))
    # # edge_index构造为大edge_index的参数
# for target_type in graph.edge_list:
#     for source_type in graph.edge_list[target_type]:
#         for relation in graph.edge_list[target_type][source_type]
#             if edge_list[target_type][source_type][relation]:
#                 edge_list_all[target_type][source_type][relation] = edge_list[
#                             target_type][source_type][relation] +

# 编码所有edge_index,构造为一个大的edge_index_all

#     node_feature = torch.FloatTensor(node_feature)
#     node_type = torch.LongTensor(node_type)
#     edge_time = torch.LongTensor(edge_time)
#     edge_index = torch.LongTensor(edge_index).t()
#     edge_type = torch.LongTensor(edge_type)
#     return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict


# def to_torch(feature, time, edge_list, graph):
#     '''
#         Transform a sampled sub-graph into pytorch Tensor
#         node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
#         edge_dict: {edge_type: edge_type_ID}
#     '''
#     node_dict = {}
#     node_feature = []
#     node_type = []
#     node_time = []
#     edge_index = []
#     edge_type = []
#     edge_time = []
#
#     node_num = 0
#     types = graph.get_types()
#     # 构造节点数字典，node_num为每个类型下子图的节点数，
#     # len(node_dict)作为类型的索引
#     for t in types:
#         node_dict[t] = [node_num, len(node_dict)]
#         node_num += len(feature[t])
#     # 将所有节点取出，组成为一个无类型的node_feature、node_time列表
#     # node_type为每个节点类型的掩码
#     for t in types:
#         node_feature += list(feature[t])
#         node_time += list(time[t])
#         node_type += [node_dict[t][1] for _ in range(len(feature[t]))]
#     # meta_graph,[(target_type, source_type, r_type)],元关系列表
#     # 元关系：索引字点
#     # 最后增加一个自关系类型
#     edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
#     edge_dict['self'] = len(edge_dict)
#
#     for target_type in edge_list:
#         for source_type in edge_list[target_type]:
#             for relation_type in edge_list[target_type][source_type]:
#                 # ti,target_id;si,source_id，新体系的索引，每个都是从0开始；
#                 for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
#                     # 加入上一个类型节点数，作为新的节点索引，构成无类型的边列表
#                     tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
#                     edge_index += [[sid, tid]]
#                     # 边类型掩码列表；
#                     edge_type += [edge_dict[relation_type]]
#                     '''
#                         Our time ranges from 1900 - 2020, largest span is 120.
#                     '''
#                     edge_time += [node_time[tid] - node_time[sid] + 120]
#     node_feature = torch.FloatTensor(node_feature)
#     node_type = torch.LongTensor(node_type)
#     edge_time = torch.LongTensor(edge_time)
#     edge_index = torch.LongTensor(edge_index).t()
#     edge_type = torch.LongTensor(edge_type)
#     return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict

class SubgraphExtractToTorch(object):
    """
    抽取子图，并生成特征，最后转化为torch.Tensor
    """
    def __init__(self,
                 graph,
                 inp_list=None,
                 sampled_depth=2,
                 sampled_number=8,
                 feature_flag=['rw','sp'],
                 max_sprw=[4,4]):
        self.graph = graph
        self.inp_list = inp_list
        self.sampled_depth = sampled_depth
        self.sampled_number = sampled_number
        self.feature_flag = feature_flag
        self.max_sprw = max_sprw


        oaglog.logger.info("开始子图抽取与组装模块处理。。。")

    def extract_subgraph_one_sample(self,inp):
        """
        为一个输入样本抽取子图
        :param inp: 输入样本，dict格式{type:[original_id,time]}
        :return:抽样的一个子图的字典，其格式为{type:{id:[ser,time]}}
        """
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
            te = self.graph.edge_list[_type]
            for _id, _time in inp[_type]:
                budget = add_budget(te, _id, _time, layer_data, budget, self.sampled_number)

        for layer in range(self.sampled_depth):
            sts = list(budget.keys())
            for source_type in sts:
                te = self.graph.edge_list[source_type]
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
                    add_budget(te,
                               k,
                               budget[source_type][k][1],
                               layer_data,
                               budget,
                               self.sampled_number)
                    budget[source_type].pop(k)
        return layer_data

    def edge_list_generator(self, layer_data, graph):

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
                    # 如果元关系下，target_layer_data中的target_id不在元关系边列表下，则不考虑该情况
                    # 否则抽取taget_id在target_layer_data中对应的ser,即索引；
                    for target_key in tld:
                        if target_key not in tesr:
                            continue
                        target_ser = tld[target_key][0]
                        for source_key in tesr[target_key]:
                            # 如果元关系指定目标节点ID下的源节点ID在source_layer_data中，
                            # 则取出节点ID对应的target_layer_data中对应的ser,即索引；
                            # 索引的二层列表；
                            if source_key in sld:
                                source_ser = sld[source_key][0]
                                edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]
        return edge_list

    def intrinsic_feature_generator(self, layer_data, graph, types):
        pass

    def distance_feature_generator(self,
                                   edge_list,
                                   types,
                                   node_feature,
                                   feature_flags,
                                   max_sprw):

        edge_index = []
        node_dict = {}
        node_num = 0
        new_set_index = np.array([0])

        for target_type in edge_list:
            for source_type in edge_list[target_type]:
                for relation_type in edge_list[target_type][source_type]:
                    # ti,target_id;si,source_id，新体系的索引，每个都是从0开始；
                    for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                        # 加入上一个类型节点数，作为新的节点索引，构成无类型的边列表
                        tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                        edge_index += [[sid, tid]]

        # 生成distance_encoding特征,根据每个子图的新的edge_index,
        new_G = nx.from_edgelist(edge_index, create_using=type(nx.Graph()))
        new_G.add_nodes_from(np.arange(node_num), dtype=np.int32)

        sp_flag, rw_flag = feature_flags
        max_sp, rw_depth = max_sprw
        if sp_flag:
            features_sp_sample = get_features_sp_sample(new_G, new_set_index, max_sp=max_sp)

            node_feature = np.concatenate(np.concatenate(node_feature, axis=0), features_sp_sample, axis=1)
        if rw_flag:
            adj = np.asarray(nx.adjacency_matrix(new_G, nodelist=np.arange(new_G.number_of_nodes(),
                                                                           dtype=np.int32)).todense().astype(
                np.float32))  # [n_nodes, n_nodes]
            features_rw_sample = get_features_rw_sample(adj, new_set_index, rw_depth=rw_depth)

            node_feature = np.concatenate(np.concatenate(node_feature, axis=0), features_rw_sample, axis=1)
        # # feature类型字典
        # for i, t in enumerate(types):
        #     if i <= len(types) - 2:
        #         feature[t] = node_feature[node_dict[types[i]]:node_dict[types[i+1]], :]
        # feature[types[-1]] = node_feature[node_dict[types[-1]]:, :]
        #









