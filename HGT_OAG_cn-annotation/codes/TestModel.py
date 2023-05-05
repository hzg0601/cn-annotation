# -*-  coding: utf-8 -*-
# @Time      :2021/4/1 21:21
# @Author    :huangzg28153
# @File      :TestModel.py
# @Software  :PyCharm
import torch
from Args import args,device
from pyHGT.utils import ndcg_at_k,mean_reciprocal_rank
import numpy as np
import oaglog


def test_model(test_data, model, model_flag='best'):
    """
    :param test_data: 测试数据
    :param model:model: 包含gnn,classifier
    :param model_flag: 输入模型的类型,"best"或"last"
    """
    oaglog.logger.info("%s 模型测试开始。。。" % model_flag)
    gnn, classifier = model
    with torch.no_grad():
        test_res = []
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in test_data:

            paper_rep = gnn.forward(
                                    node_feature.to(device),
                                    node_type.to(device),
                                    edge_time.to(device),
                                    edge_index.to(device),
                                    edge_type.to(device))
            res = classifier.forward(paper_rep[x_ids])
            for ai, bi in zip(ylabel, res.argsort(descending=True)):
                test_res += [(bi == ai).int().tolist()]
        test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
        oaglog.logger.info('%s Test NDCG: %.4f' % (model_flag, np.average(test_ndcg)))
        test_mrr = mean_reciprocal_rank(test_res)
        oaglog.logger.info('%s Test MRR:  %.4f' % (model_flag, np.average(test_mrr)))
        oaglog.logger.info("%s 模型测试结束。" % model_flag)

    # best_model = torch.load(os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
    # best_model.eval()
    # gnn, classifier = best_model

    # with torch.no_grad():
    #     test_res = []
    #     for _ in range(10):
    #         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = test_data
    #         paper_rep = gnn.forward(node_feature.to(device), node_type.to(device),
    #                                 edge_time.to(device), edge_index.to(device), edge_type.to(device))[x_ids]
    #         res = classifier.forward(paper_rep)
    #         for ai, bi in zip(ylabel, res.argsort(descending=True)):
    #             test_res += [(bi == ai).int().tolist()]
    #     test_ndcg = [ndcg_at_k(resi, len(resi)) for resi in test_res]
    #     print('Best Test NDCG: %.4f' % np.average(test_ndcg))
    #     test_mrr = mean_reciprocal_rank(test_res)
    #     print('Best Test MRR:  %.4f' % np.average(test_mrr))