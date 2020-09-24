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


class DCGCNEncoder(torch.nn.Module):
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

        super(DCGCNEncoder, self).__init__()

        self.depth = depth

        blocks = []
        hop = 1

        for i in range(self.depth):
            in_channels_block = in_channels if i == 0 else out_channels[i-1]
            blocks += [
                GCNConv(in_channels_block,out_channels[i])
            ]

        self.layers = ListModule(*blocks)

    def forward(self, features, edge_indexes):

        hop = 1

        for i in range(self.depth):
            assert hop in edge_indexes
            features = F.relu(self.layers[i](features, edge_indexes[hop]))
            hop *= 3

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
                 out_channels=[64,32,16],
                 heads=10,
                 dropout=0.5,
                 negative_slope=0.2,
                 depth=3,
                 pooling_ratio=1,
                 N=100,
                 g=10):
        super(Netwrok, self).__init__()

        # 表示模块
        self.encoder = DCGCNEncoder(in_channels, out_channels, heads, dropout,
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
