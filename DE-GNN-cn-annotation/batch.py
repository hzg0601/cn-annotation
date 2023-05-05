# -*-  coding: utf-8 -*-
# @Time      :2021/3/10 16:53
# @Author    :huangzg28153
# @File      :batch.py
# @Software  :PyCharm
import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
import torch_geometric
from torch_geometric.data import Data


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            else:
                self[key] = item

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_nodes_list__ = None
        self.__num_graphs__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""
        # 首先取出data_list中的所有数据的keys,
        # 再把所有keys组装为一个keys列表
        # 并声明'batch'不在key里,如果在其中则必然是已经组装好的
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys
        # Data类型并非是一个简单的dict;
        # data_list[0].__dict__.keys(),dict_keys(['x', 'edge_index', 'edge_attr',
        # 'y', 'pos', 'normal', 'face', 'set_indices'])
        batch = Batch()
        for key in data_list[0].__dict__.keys():
            # 如果该key是Data的私有成员，则我们不能访问，设其值为None
            # 单下划线用来指定私有变量，不能用from module import * 访问，其他与共有变量一样
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None
        # data_list[0].__class__, torch_geometric.data.data.Data
        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        # 初始化，并增加'batch'关键字
        for key in keys + ['batch']:
            batch[key] = []
        # slice,用以记录
        #
        # cum_sum,
        #
        # cat_dims
        #
        # num_nodes_list
        # batch.batch,用于记录每个节点来自Data的掩码；
        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        # 取出每个Data,
        # 针对每个Data中的key进行遍历
        for i, data in enumerate(data_list):
            for key in keys:
                # 如data['x'],则为一个tensor，长度为Data所包含的所有节点
                item = data[key]

                # Increase values by `cumsum` value.
                # cum,初始值为0,取最后一个进行累加
                cum = cumsum[key][-1]
                # 重新安排节点索引，在当前Data节点索引的基础上加上累积加入的节点，构成当前Data节点的索引；
                # 由于每个Data的索引自成体系，因此累加可以生效；
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Treat 0-dimensional tensors as 1-dimensional.
                if isinstance(item, Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)
                #
                batch[key].append(item)

                # Gather the size of the `cat` dimension.
                # Returns the dimension for which :obj:`value` of attribute
                # :obj:`key` will get concatenated when creating batches.
                # 返回在构造batches时，属性key的值value在拼接中会放在哪一维
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                cat_dims[key] = cat_dim
                if isinstance(item, Tensor):
                    # item.size(),返回item的size,这里取cat_dim位置的size;
                    # 边的个数，
                    size = item.size(cat_dim)
                    device = item.device
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                    device = item.device()

                slices[key].append(size + slices[key][-1])
                # Returns the incremental count to cumulatively increase the value
                # of the next attribute of :obj:`key` when creating batches.
                # 计算构造batch时key的value时下一个属性累积计数,此处为第一个Data的节点数
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch: #False
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size, ), i, dtype=torch.long,
                                       device=device))

            if hasattr(data, '__num_nodes__'): # False
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                # Returns a tensor of size :attr:`size` filled with :attr:`fill_value`.
                # full(size, fill_value, out=None, dtype=None, layout=torch.strided,
                #      device=None, requires_grad=False) -> Tensor
                # 构造batch中的节点来自哪个Data的掩码，batch.batch节点来自batch的序号；
                item = torch.full((num_nodes, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using `Batch.from_data_list()`.'))

        data_list = []
        for i in range(len(list(self.__slices__.values())[0]) - 1):
            data = self.__data_class__()

            for key in self.__slices__.keys():
                item = self[key]
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][i]
                    end = self.__slices__[key][i + 1]
                    item = item.narrow(dim, start, end - start)
                elif isinstance(item, SparseTensor):
                    for j, dim in enumerate(self.__cat_dims__[key]):
                        start = self.__slices__[key][i][j].item()
                        end = self.__slices__[key][i + 1][j].item()
                        item = item.narrow(dim, start, end - start)
                else:
                    item = item[self.__slices__[key][i]:self.
                                __slices__[key][i + 1]]
                    item = item[0] if len(item) == 1 else item

                # Decrease its value by `cumsum` value:
                cum = self.__cumsum__[key][i]
                if isinstance(item, Tensor):
                    if not isinstance(cum, int) or cum != 0:
                        item = item - cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value - cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item - cum

                data[key] = item

            if self.__num_nodes_list__[i] is not None:
                data.num_nodes = self.__num_nodes_list__[i]

            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        if self.__num_graphs__ is not None:
            return self.__num_graphs__
        return self.batch[-1].item() + 1
