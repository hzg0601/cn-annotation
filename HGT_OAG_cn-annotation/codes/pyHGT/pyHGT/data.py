from collections import defaultdict
# import sys
# import os
# sys.path.append(os.getcwd()+'/utils.py')
from pyHGT.utils import *
import networkx as nx
import dill


class Graph():
    def __init__(self):
        super(Graph, self).__init__()
        '''
            node_forward and bacward are only used when building the data. 
            Afterwards will be transformed into node_feature by DataFrame
            
            node_forward: name -> node_id
            node_bacward: node_id -> feature_dict
            node_feature: a DataFrame containing all features
        '''
        self.node_forward = defaultdict(lambda: {})
        self.node_bacward = defaultdict(lambda: [])
        self.node_feature = defaultdict(lambda: [])

        '''
            edge_list: index the adjacancy matrix (time) by 
            <target_type, source_type, relation_type, target_id, source_id>
        '''
        self.edge_list = defaultdict( #target_type
                            lambda: defaultdict(  #source_type
                                lambda: defaultdict(  #relation_type
                                    lambda: defaultdict(  #target_id
                                        lambda: defaultdict( #source_id(
                                            lambda: int # time
                                        )))))
        self.times = {}

    def add_node(self, node):
        """
        node['type']= type
        node['id']= id;

       node_forward[type] = [node], 即记录每个节点进入该dict时，dict的长度，作为node_bacward的索引。
       node_bacward[type] = [node]

        :param node: 节点，封装为一个dict,包含type,id关键字；
        :return: 该节点进入node_forward前dict的长度；
        """
        # node_forward，dict格式，以具体节点类型为key,具体节点类型对应该类型下的所有节点id：节点属性的dict;
        nfl = self.node_forward[node['type']]
        # 首先取出该类型下的所有节点id，如果传入节点的id在传入节点type所对应的所有节点id dict内，则返回该节点进入该dict前，dict的长度；

        if node['id'] not in nfl:
            # 否则在传入节点类型下，将该节点加入node_bacward dict；
            self.node_bacward[node['type']] += [node]
            # 记录该节点进入该dict前，dict的长度，作为该节点在bacward的索引，并返回
            ser = len(nfl)
            nfl[node['id']] = ser
            return ser
        return nfl[node['id']]

    def add_edge(self, source_node, target_node, time = None, relation_type = None, directed = True):
        edge = [self.add_node(source_node), self.add_node(target_node)]
        '''
            Add bi-directional edges with different relation type
            加入边列表
        '''
        self.edge_list[target_node['type']][source_node['type']][relation_type][edge[1]][edge[0]] = time
        if directed:
            self.edge_list[source_node['type']][target_node['type']]['rev_' + relation_type][edge[0]][edge[1]] = time
        else:
            self.edge_list[source_node['type']][target_node['type']][relation_type][edge[0]][edge[1]] = time
        self.times[time] = True
        
    def update_node(self, node):
        """
        将加入的节点更新入node_bacward;
        :param node: 待更新的节点；
        :return: None
        """
        # 取出待更新节点类型下的所有节点构成的list
        nbl = self.node_bacward[node['type']]
        # 取出其节点id
        ser = self.add_node(node)
        # dict的key可以进行循环
        for k in node:
            # 如果bnode_bacward没有节点的key，则加入之；
            if k not in nbl[ser]:
                nbl[ser][k] = node[k]

    def get_meta_graph(self):
        """
        取出边列表中所有的元关系，构造为一个列表
        :return: 元关系三元组构成的列表；
        """
        types = self.get_types()
        metas = []
        for target_type in self.edge_list:
            for source_type in self.edge_list[target_type]:
                for r_type in self.edge_list[target_type][source_type]:
                    metas += [(target_type, source_type, r_type)]
        return metas
    
    def get_types(self):
        return list(self.node_feature.keys())


def sample_subgraph(graph, sampled_depth = 2, sampled_number = 8, inp = None, feature_extractor = feature_OAG):
    '''
        Sample Sub-Graph based on the connection of other nodes with currently sampled nodes
        We maintain budgets for each node type, indexed by <node_id, time>.
        Currently sampled nodes are stored in layer_data.

        首先从inp开始，将inp[type]下的每个节点的ID和时间添加至layer_data[type]下，更新内容为[ser,time]
        然后针对inp每种type,和每个id，以其为target_type,target_id,在graph.edge_list中找到source_type下与其相邻的边，
        抽样每种关系下至多sampled_number个节点，以source_type为key，将采样的邻接节点加入budget;

        然后针对每层sampled_depth,
          对于budget抽样的每种source_type(针对inp中的target_type而言),为其抽样至多sampled_number个节点，
            对于抽样的每个节点，将其加入layer_data中，然后更新budget，
            budget的更新是以已存在的budget中的每种type为target_type,取出对应的边列表，以采样的节点为target_id，
            进行更新，采样与采样节点在每种source_type中相邻的边，然后在budget的source_type下删除该节点，
            即同类已采样节点不再二次采样；

        After nodes are sampled, we construct the sampled adjacancy matrix.
        graph: 图，以图属性，边列表的形式存在；
        sampled_depth: 抽样的层数，二度邻居？
        sampled_number: 对于给定的抽样节点，每种relation_type下最多抽取的邻居节点数
        inp: input_node,inp[type] = [id,time],节点ID与时间。
        feature_extractor: 特征抽取函数

        :return feature, times, edge_list, indxs, texts
    '''
    # 初始轮：layer_data["paper"]= [ser,time]* batch_size个
    # 第一轮：layer_data['venue','paper','field','author']下每个至多sampled_number个,
    # 至多k*sampled_number个,假设为k*sampled_number
    # 第二轮：sum_k{k_i*sampled_number*sampled_number}个
    layer_data  = defaultdict( #target_type
                        lambda: {}  # {target_id: [ser, time]} # ser即其索引，加入dict的顺序
                    )
    budget     = defaultdict( #source_type
                                    lambda: defaultdict(  #source_id
                                        lambda: [0., 0] #[sampled_score, time]
                            ))
    new_layer_adj  = defaultdict( #target_type
                                    lambda: defaultdict(  #source_type
                                        lambda: defaultdict(  #relation_type
                                            lambda: [] #[target_id, source_id]
                                )))
    '''
        For each node being sampled, we find out all its neighborhood, 
        adding the degree count of these nodes in the budget.
        Note that there exist some nodes that have many neighborhoods
        (such as fields, venues), for those case, we only consider 
    '''
    def add_budget(te, target_id, target_time, layer_data, budget):
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
                    sampled_ids = np.random.choice(list(adl.keys()), sampled_number, replace = False)
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

    '''
        First adding the sampled nodes then updating budget.
        首先将针对inp中每个_type类型下，每个节点按照加入顺序记录在layer_data[_type]下
        然后针对inp中每个_type类型下，每个节点进行采样，更新budget，
        budget[source_type][source_id] = [sampled_score,source_time],即给定的target_type下，
        每种source_type所有关系抽样至多sampled_number个节点，故每种target_type下，
        节点最多有len(source_type)*len(relation_type)*sampled_number个节点
        
        inp为初始target_type类型：target_id节点ID字典，本脚本只涉及”paper“类型
        
        由于inp为一个{节点类型：ID}的字典，可能包括了所有类型，因此budget就变为一个所有类型每个节点的抽样集合，
        且每个对应len(relation_type)*sampled_number个邻居节点；
        
    '''
    for _type in inp:
        for _id, _time in inp[_type]:
            # target_id,在_type(paper)下的数字顺序(node_forward下)，
            # len(layer_data[_type])作为新的id，
            # 初始为layer_data["paper"]["paper_target_id_range"] = [len,_time]
            layer_data[_type][_id] = [len(layer_data[_type]), _time]

    for _type in inp:
        # te, target edge_list;
        te = graph.edge_list[_type]
        for _id, _time in inp[_type]:
            # 针对inp["paper"]下的每个节点，以paper为target_type,更新budget
            # 包括所有以paper为target_type的类型，paper,field,venue,author
            add_budget(te, _id, _time, layer_data, budget)
    '''
        We recursively expand the sampled graph by sampled_depth.
        Each time we sample a fixed number of nodes for each budget,
        based on the accumulated degree.
        针对初始的target_type:target_id字典，更新初始的layer_data,budget字典；
        buget抽取的是针对target_type下，每个target_id对应的每种source_type在所有关系下的至多sampled_number个节点，
        
        然后针对每种source_type，抽取至多sampled_data个节点
        layer_data为target_type:target_id字典，经过抽样我们获得了source_type下的至多sampled_number个source_id;
        由于不考虑自关系，因此budget_type必然不包括layer_data中的target_type
    '''
    # layer_data,从inp里的type开始，找到 以inp的type为target_type各source_type下每类关系至多sampled_number个节点
    # 然后针对budget中source_type抽取至多sampled_number个节点，放入对应layer_data中；1+k类；
    # 然后针对采样到的节点，和对应的source_type，以其为target_type,进行buget更新，并删除已采样的节点；(\sum_k{k_i})类
    # 如果sampled_type大于二，则针对每个采样重复上述过程，则第二轮存在(\sum_k{k_i}+1)类sts
    # 则元路径的长度即sampled_depth1-k-\sum_k{k_i}
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
            # 由于每个关系定义了逆关系，因此该方法实际上会遍历所有类型
            for k in sampled_keys:
                layer_data[source_type][k] = [len(layer_data[source_type]), budget[source_type][k][1]]
            #
            for k in sampled_keys:
                # budget[source_type][k][1]为target_time;
                # 如source_type,为venue, 采样到venue下的某个ID，就会以venue为target_type,取出边；
                # 增加venue为出发target_type, source_type下的采样节点
                add_budget(te, k, budget[source_type][k][1], layer_data, budget)
                budget[source_type].pop(k)   
    '''
        Prepare feature, time and adjacency matrix for the sampled graph
        feature,每个类型下所有节点拼接的特征；
        times,每个类型下每个节点的时间
        texts,每个类型每个节点的文本
        indxs,每个类型每个节点在原图中的索引
        
    '''
    feature, times, indxs, texts = feature_extractor(layer_data, graph)
            
    edge_list = defaultdict(  # target_type
                        lambda: defaultdict(  # source_type
                            lambda: defaultdict(  # relation_type
                                lambda: []  # [target_id, source_id]
                                    )))
    for _type in layer_data:
        for _key in layer_data[_type]:
            _ser = layer_data[_type][_key][0]
            edge_list[_type][_type]['self'] += [[_ser, _ser]]
    '''
        Reconstruct sampled adjacancy matrix by checking whether each
        link exist in the original graph
        重建边列表，即邻接矩阵；
    '''

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
            sld  = layer_data[source_type]
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
                            Check whether each link (target_id, source_id) exist in original adjacancy matrix
                        '''
                        # 如果元关系指定目标节点ID下的源节点ID在source_layer_data中，
                        # 则取出节点ID对应的target_layer_data中对应的ser,即索引；
                        # 索引的二层列表；
                        if source_key in sld:
                            source_ser = sld[source_key][0]
                            edge_list[target_type][source_type][relation_type] += [[target_ser, source_ser]]

    return feature, times, edge_list, indxs, texts


def to_torch(feature, time, edge_list, graph):
    '''
        Transform a sampled sub-graph into pytorch Tensor
        node_dict: {node_type: <node_number, node_type_ID>} node_number is used to trace back the nodes in original graph.
        edge_dict: {edge_type: edge_type_ID}
    '''
    node_dict = {}
    node_feature = []
    node_type    = []
    node_time    = []
    edge_index   = []
    edge_type    = []
    edge_time    = []
    
    node_num = 0
    types = graph.get_types()
    # 构造节点数字典，node_num为每个类型下子图的节点数，
    # len(node_dict)作为类型的索引
    for t in types:
        node_dict[t] = [node_num, len(node_dict)]
        node_num     += len(feature[t])
    # 将所有节点取出，组成为一个无类型的node_feature、node_time列表
    # node_type为每个节点类型的掩码,为一个一层的list;
    for t in types:
        # node_feature,node_time的元素为array
        #
        node_feature += list(feature[t])
        node_time    += list(time[t])
        node_type    += [node_dict[t][1] for _ in range(len(feature[t]))]
    # meta_graph,[(target_type, source_type, r_type)],元关系列表
    # 元关系：索引字点
    # 最后增加一个自关系类型
    edge_dict = {e[2]: i for i, e in enumerate(graph.get_meta_graph())}
    edge_dict['self'] = len(edge_dict)

    for target_type in edge_list:
        for source_type in edge_list[target_type]:
            for relation_type in edge_list[target_type][source_type]:
                # ti,target_id;si,source_id，新体系的索引，每个都是从0开始；
                for ii, (ti, si) in enumerate(edge_list[target_type][source_type][relation_type]):
                    # 加入上一个类型节点数，作为新的节点索引，构成无类型的边列表
                    tid, sid = ti + node_dict[target_type][0], si + node_dict[source_type][0]
                    edge_index += [[sid, tid]]
                    # 边类型掩码列表；
                    edge_type  += [edge_dict[relation_type]]   
                    '''
                        Our time ranges from 1900 - 2020, largest span is 120.
                    '''
                    edge_time  += [node_time[tid] - node_time[sid] + 120]
    node_feature = torch.FloatTensor(node_feature)
    node_type    = torch.LongTensor(node_type)
    edge_time    = torch.LongTensor(edge_time)
    edge_index   = torch.LongTensor(edge_index).t()
    edge_type    = torch.LongTensor(edge_type)
    return node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict


class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "GPT_GNN.data" or module == 'data':
            renamed_module = "pyHGT.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


                # 手动构造邻接矩阵
                # feture: feture[type]:pd.DataFrame
    #             node_pair_list = edge_list[target_type][source_type][relation_type]
    #             node_set_index = np.expand_dims(np.array(set([pair[0] for pair in node_pair_list])),1)
    #             # 其可能的形式为[[0,0][0,1][1,1]],因此在将其构造为networkx中的图是，必须重命名，且
    #             # 是否针对每个节点构造
    #             # 针对每类type,ser必然唯一，但target_type,source_type是不同的，
    #
    #             # 抽取target_type下每个target_id的distance_feature的特征；
    #             distance_feature_list_for_target = []
    #             for set_index in node_set_index:
    #                 distance_feature = get_distance_feature(node_pair_list,
    #                                                         set_index,
    #                                                         hop_num=2,
    #                                                         feature_flags=['rw','sp'],
    #                                                         max_sprw=[4,6])
    #                 distance_feature_list_.append(distance_feature)
    #             distance_feature_dict[target_type][source_type][relation_type] = distance_feature_list
    #
    # for _type in layer_data:
    #     if len(layer_data[_type]) == 0:
    #         continue
    #         # 特定类下所有节点的索引,注意此处为在图中的索引，并不是节点的ID
    #     idxs = np.array(list(layer_data[_type].keys()))
    #     feature_type = feature[_type]
    #     distance_feature_type = distance_feature_dict[_type]
    #     if not distance_feature_type:
    #         continue
    #     else:
    #         distance_temp_list = []
    #         for source_type, source_dict in distance_feature_type.items():
    #             for relation_type, relation_source_dict in source_dict.items():
    #                 pass

# def form_adjacency_matrix_from_raw(node_pair_list):
#     """
#     从原始节点对生成致密的邻接矩阵
#     :param node_pair_list: 节点对的list;
#     :return: dense的邻接矩阵
#     """
#     relation_list = node_pair_list
#     node_set = set(np.ravel(relation_list))
#
#     node_dict = {item: idx for idx, item in enumerate(node_set)}
#     set_index = [value for key, value in node_dict.items()]
#     adj_matrix = np.zeros(len(set(np.ravel(relation_list))))
#     for target_ser, source_ser in relation_list:
#         target_index = node_dict[target_ser]
#         source_index = node_dict[source_ser]
#         adj_matrix[target_index][source_index] = 1
#
#     return adj_matrix
