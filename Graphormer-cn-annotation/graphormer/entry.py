# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
""" 
主函数
1，wrapper.py首先构造My*Dataset类，
    该类针对每张图调用preprocess_item对每张图进行处理；
    提取原始特征、邻接矩阵、注意力矩阵、相对距离矩阵、出度、入度、最短路径特征
2，处理好的图传入colator.py中的collator函数，进行pad，
    补足为固定长度，并向量化，然后调用Batch类组装为mini-batch; 
3，data.py中的get_dataset函数构造pytorch_light的数据字典，
    其dataset字段调用wrapper.py的My*Dataset类处理原始数据；
    GraphDataModule则按照pytorch_lighting框架的数据处理类，将传入的原始数据特征
    传入Dataloader，DataLoader调用collator.py的Batch, collator进行填充；
4，Graphormer.py 针对填充好的处理后特征进行嵌入，
    加入虚拟节点，包含虚拟节点的距离偏置、原始特征
    设置注意力的边特征、相对距离偏置、节点与虚拟节点的全局（原始）偏置；
    加入in_degree_encoding\out_degree_encoding编码
    定义多头注意力层；
    定义输出映射层； 
5，调用pytorch_lighting框架进行训练； 
"""
from model import Graphormer
from data import GraphDataModule, get_dataset

from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os


def cli_main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Graphormer.add_model_specific_args(parser)
    parser = GraphDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    args.max_steps = args.tot_updates + 1
    if not args.test and not args.validate:
        print(args)
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    dm = GraphDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    if args.checkpoint_path != '':
        model = Graphormer.load_from_checkpoint(
            args.checkpoint_path,
            strict=False,
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    else:
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            weight_decay=args.weight_decay,
            ffn_dim=args.ffn_dim,
            dataset_name=dm.dataset_name,
            warmup_updates=args.warmup_updates,
            tot_updates=args.tot_updates,
            peak_lr=args.peak_lr,
            end_lr=args.end_lr,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            flag=args.flag,
            flag_m=args.flag_m,
            flag_step_size=args.flag_step_size,
        )
    if not args.test and not args.validate:
        print(model)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # ------------
    # training
    # ------------
    metric = 'valid_' + get_dataset(dm.dataset_name)['metric']
    dirpath = args.default_root_dir + f'/lightning_logs/checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor=metric,
        dirpath=dirpath,
        filename=dm.dataset_name + '-{epoch:03d}-{' + metric + ':.4f}',
        save_top_k=100,
        mode=get_dataset(dm.dataset_name)['metric_mode'],
        save_last=True,
    )
    if not args.test and not args.validate and os.path.exists(dirpath + '/last.ckpt'):
        args.resume_from_checkpoint = dirpath + '/last.ckpt'
        print('args.resume_from_checkpoint', args.resume_from_checkpoint)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(checkpoint_callback)
    trainer.callbacks.append(LearningRateMonitor(logging_interval='step'))

    if args.test:
        result = trainer.test(model, datamodule=dm)
        pprint(result)
    elif args.validate:
        result = trainer.validate(model, datamodule=dm)
        pprint(result)
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
