import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGPooling, GCNConv
from utils import *
from ncut_loss import NCutLoss


class GAT_Block(torch.nn.Module):
    # TODO 考虑添加 BN, WN, Pooling层
    """
    GAT残差块的实现

    @param neighbor_hop: int,邻接链表跳数
    @param in_channels: int,输入特征
    @param out_channels: int,输出特征
    @param heads: int,GATmulti-head
    @param dropout: float,(0~1),dropout系数
    @param negative_slope: float,LeakyRelu系数
    @param pool_ratio: ratio表示取注意力值所占计算出来的注意力的比值，默认0.5
    """
    def __init__(self,
                 neighbor_hop=1,
                 in_channels=100,
                 out_channels=100,
                 heads=10,
                 dropout=0.5,
                 negative_slope=0.2,
                 pooling_ratio=1):
        super(GAT_Block, self).__init__()

        self.neighbor_hop = neighbor_hop

        # conv1和conv2是得Block到F(x)那两层，conv3目的是让x的列数与F(x)的列数一样
        # 给conv1、conv2添加WN层
        self.conv1 = GATConv(in_channels,
                             out_channels,
                             heads=heads,
                             dropout=dropout)
        # GAT默认cat=Ture,也就是把heads路得出的特征拼接，所以self.conv1输出特征为out_channels * heads,self.conv2的输入是self.conv1的输出，所以self.conv2的in_channels=out_channels * heads
        self.conv2 = GATConv(out_channels * heads,
                             out_channels,
                             heads=heads,
                             dropout=dropout,
                             concat=False)
        self.conv3 = GATConv(in_channels,
                             out_channels,
                             heads=heads,
                             dropout=dropout,
                             concat=False)

        self.fc = torch.nn.Linear(in_channels, out_channels)

        # 添加BN层
        self.BN1 = torch.nn.BatchNorm1d(out_channels * heads,
                                       eps=1e-05,
                                       momentum=0.1,
                                       affine=True)
        self.BN2 = torch.nn.BatchNorm1d(out_channels,
                                       eps=1e-05,
                                       momentum=0.1,
                                       affine=True)
        self.BN3 = torch.nn.BatchNorm1d(out_channels,
                                       eps=1e-05,
                                       momentum=0.1,
                                       affine=True)

        # 添加pool层,ratio表示取注意力值所占计算出来的注意力的比值，默认0.5
        self.pool = SAGPooling(out_channels, ratio=pooling_ratio)

    def forward(self, x, edge_indices):
        """
        @param x: torch.tensor,[N,feature],输入的特征
        @param edge_index: torch.tensor,[2,|edges|],邻接链表
        @param edge_indices：torch.tensor,多个edge_index
        """

        assert self.neighbor_hop in edge_indices

        edge_index = edge_indices[self.neighbor_hop]

        # 残差块第一层
        f1 = self.conv1(x, edge_index)

        # 使用BN层,BN层一般使用在激活函数之前
        # f1 = self.BN1(f1)

        # f1 = F.leaky_relu(f1)
        # 残差块第二层
        f2 = self.conv2(f1, edge_index)
        # f2 = self.BN2(f2)
        # f2 = F.leaky_relu(f2)
        # 改变x尺寸
        # xchange = self.conv3(x, edge_index)
        xchange = self.fc(x)

        # xchange = self.BN3(xchange)
        # xchange = F.leaky_relu(xchange)

        res = torch.add(f2, xchange)

        # res = F.leaky_relu(res)

        # 使用池化层
        # res, _, _, _, _, _ = self.pool(res, edge_index)

        return res


class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class DCGATEncoder(torch.nn.Module):
    """
    用三层GAT来实现dilated causal gat encoder

    @param in_channels: int,输入的特征
    @param out_channels: int,输出的特征
    @param heads: int,GATmulti-head
    @param dropout: float,(0~1),dropout系数
    @param negative_slope: float,LeakyRelu系数
    @param depth: int,GAT个数
    """
    def __init__(self,
                 in_channels=100,
                 out_channels=[64,32,16],
                 heads=10,
                 dropout=0.5,
                 negative_slope=0.2,
                 depth=3,
                 pooling_ratio=1):

        super(DCGATEncoder, self).__init__()

        assert len(out_channels) == depth

        self.depth = depth

        blocks = []
        hop = 1

        for i in range(self.depth):
            in_channels_block = in_channels if i == 0 else out_channels[i-1]
            blocks += [
                GAT_Block(hop, in_channels_block, out_channels[i], heads, dropout,
                          negative_slope, pooling_ratio)
            ]
            hop *= 3

        self.layers = ListModule(*blocks)

    def forward(self, features, edge_indexes):

        for i in range(self.depth):
            features = self.layers[i](features, edge_indexes)

        return features


class Netwrok(torch.nn.Module):
    """
    说明

    @param nclass: int, 分区个数
    @param in_channels: int, see Dilated_Causal_GAT_Encoder.py
    @param out_channels: int, see Dilated_Causal_GAT_Encoder.py
    @param heads: int, see Dilated_Causal_GAT_Encoder.py
    @param dropout: float, see Dilated_Causal_GAT_Encoder.py
    @param negative_slope: float, see Dilated_Causal_GAT_Encoder.py
    @param depth: int, see Dilated_Causal_GAT_Encoder.py
    """
    def __init__(self,
                 nclass,
                 in_channels=100,
                 out_channels=100,
                 heads=10,
                 dropout=0.5,
                 negative_slope=0.2,
                 depth=3,
                 pooling_ratio=1,
                 N=100,
                 g=10):
        super(Netwrok, self).__init__()

        # 表示模块
        self.encoder = DCGATEncoder(in_channels, out_channels[:-1], heads, dropout,
                                    negative_slope, depth, pooling_ratio)

        # 划分模块
        self.fc1 = torch.nn.Linear(out_channels[-2], out_channels[-1])
        self.fc2 = torch.nn.Linear(out_channels[-1], nclass)

        self.loss = NCutLoss(N, g)

    def forward(self, node_features, edge_indexes, A):
        """
        说明

        @param slaver_id: int 子图编号
        """
        embedding = self.encoder(node_features, edge_indexes)

        features = self.fc2(self.fc1(embedding))

        predictions = torch.nn.functional.softmax(features, dim=1)

        loss = self.loss(predictions, A)

        return loss

