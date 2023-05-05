# -*-  coding: utf-8 -*-
# @Time      :2021/4/1 21:18
# @Author    :huangzg28153
# @File      :TrainModel.py
# @Software  :PyCharm
import torch
import torch.nn as nn
import numpy as np
import time
from Args import args, device
import oaglog
import os
from pyHGT.utils import ndcg_at_k
# stats = []
# res = []
# best_val = 0
# train_step = 1500
# criterion = nn.NLLLoss()


def train_eval_model(models, train_params, train_data, valid_data, args, device, epoch):
    """

    :param models:create_model的返回值，建立模型的所需的对象
    :param train_data:n_batch个样本
    :param valid_data:1个batch个样本
    :param args:输入配置参数
    :param epoch:epoch的顺序
    :return:best_val
    """
    stats, best_val, train_step, criterion = train_params
    scheduler, model, optimizer, gnn, classifier = models

    '''
        Train (time < 2015)
    '''
    et = time.time()
    oaglog.logger.info("epoch %d 训练开始。。。" % epoch)
    model.train()
    train_losses = []
    # 清空cuda的缓存
    torch.cuda.empty_cache()
    for repeat_order in range(args.repeat):
        batch_order = 0
        # 针对train_data的每个batch
        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in train_data:
            oaglog.logger.debug("epoch %d, repeat %d, batch %d 训练开始。。" % (epoch,repeat_order,batch_order))

            node_rep = gnn.forward(
                                   node_feature.to(device),
                                   node_type.to(device),
                                   edge_time.to(device),
                                   edge_index.to(device),
                                   edge_type.to(device)
                                )
            # 预测结果
            res = classifier.forward(node_rep[x_ids])
            # 计算损失函数
            loss = criterion(res, ylabel.to(device))
            # 参数梯度缓存器置零
            optimizer.zero_grad()
            # 清空cuda缓存
            torch.cuda.empty_cache()
            # 误差反向传播
            loss.backward()
            # 裁剪并标准化梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # 更新optimizer
            optimizer.step()
            # 计算n_batch个batch训练损失
            train_losses += [loss.cpu().detach().tolist()]
            train_step += 1
            scheduler.step(train_step)
            oaglog.logger.debug("epoch %d, repeat %d, batch %d 训练结束。。" % (epoch, repeat_order, batch_order))
            batch_order += 1
            del res, loss
    oaglog.logger.info("epoch %d 训练结束" % epoch)
    '''
        Valid (2015 <= time <= 2016)
    '''
    oaglog.logger.info("epoch %d valid 开始。。。" % epoch)
    model.eval()
    # 对于valid,test不需要梯度反向传播，无需追踪梯度
    with torch.no_grad():

        for node_feature, node_type, edge_time, edge_index, edge_type, x_ids, ylabel in valid_data:
            node_rep = gnn.forward(node_feature.to(device),
                                   node_type.to(device),
                                   edge_time.to(device),
                                   edge_index.to(device),
                                   edge_type.to(device))
            # 预测结果
            res = classifier.forward(node_rep[x_ids])
            # 计算损失值
            loss = criterion(res, ylabel.to(device))

        '''
            Calculate Valid NDCG. Update the best model based on highest NDCG score.
        '''
        valid_res = []
        for ai, bi in zip(ylabel, res.argsort(descending=True)):
            # 预测正确的样本数
            valid_res += [(bi == ai).int().tolist()]
        # 计算平均ndcg_at_k值
        valid_ndcg = np.average([ndcg_at_k(resi, len(resi)) for resi in valid_res])

        if valid_ndcg > best_val:
            best_val = valid_ndcg
            torch.save(model, os.path.join(args.model_dir, args.task_name + '_' + args.conv_name))
            oaglog.logger.info('UPDATE!!!')

        st = time.time()
        #
        valid_loss = loss.cpu().detach().tolist()
        # 记录epoch、训练时长、learning rate参数、train_loss、valid_loss、valid NDCG
        oaglog.logger.info("Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid NDCG: %.4f" %
              (epoch,
               (st - et),
               optimizer.param_groups[0]['lr'],
               np.average(train_losses),
               valid_loss,
               valid_ndcg))
        # 记录平均训练损失，和损失
        stats += [[np.average(train_losses), valid_loss]]
        oaglog.logger.info("epoch %d valid 结束。" % epoch)
        del res, loss
    del train_data, valid_data

    train_params = stats, best_val, train_step, criterion
    models = scheduler, model, optimizer, gnn, classifier
    return models, train_params
