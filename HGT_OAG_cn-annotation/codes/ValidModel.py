# -*-  coding: utf-8 -*-
# @Time      :2021/4/1 21:19
# @Author    :huangzg28153
# @File      :ValidModel.py
# @Software  :PyCharm

#     '''
#         Valid (2015 <= time <= 2016)
#     '''
#     model.eval()
#     with torch.no_grad():
#         node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel = valid_data
#         node_rep = gnn.forward(node_feature.to(device), node_type.to(device), \
#                                edge_time.to(device), edge_index.to(device), edge_type.to(device))
#         res = classifier.forward(node_rep[x_ids])
#         loss = criterion(res, ylabel.to(device))
#
#         '''
#             Calculate Valid NDCG. Update the best model based on highest NDCG score.
#         '''
#         valid_res = []
#         for ai, bi in zip(ylabel, res.argsort(descending=True)):
#             valid_res += [(bi == ai).int().tolist()]
#         valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])
#
#         if valid_ndcg > best_val:
#             best_val = valid_ndcg
#             torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
#             print('UPDATE!!!')
#
#         st = time.time()
#         print(("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f") % \
#               (epoch, (st - et), optimizer.param_groups[0]['lr'], np.average(train_losses), \
#                loss.cpu().detach().tolist(), valid_ndcg))
#         stats += [[np.average(train_losses), loss.cpu().detach().tolist()]]
#         del res, loss
#     del train_data, valid_data
#     oaglog.logger.info("本epoch验证完毕。")