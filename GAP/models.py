import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from utils import *


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H, A):
        W = self.weight
        b = self.bias

        HW = torch.mm(H, W)
        # AHW = SparseMM.apply(A, HW)
        AHW = torch.spmm(A, HW)
        if self.bias is not None:
            return AHW + b
        else:
            return AHW

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):
    def __init__(self, gl, ll, dropout, N=100, g=10):
        super(GCN, self).__init__()
        if ll[0] != gl[-1]:
            assert 'Graph Conv Last layer and Linear first layer sizes dont match'
        # self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.graphlayers = nn.ModuleList([
            GraphConvolution(gl[i], gl[i + 1], bias=True)
            for i in range(len(gl) - 1)
        ])
        self.linlayers = nn.ModuleList(
            [nn.Linear(ll[i], ll[i + 1]) for i in range(len(ll) - 1)])
        self.loss = NewNCutLoss(N, g)

    def forward(self, H, A):
        # x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, adj)
        for idx, hidden in enumerate(self.graphlayers):
            H = F.relu(hidden(H, A))
            if idx < len(self.graphlayers) - 2:
                H = F.dropout(H, self.dropout, training=self.training)

        H_emb = H

        for idx, hidden in enumerate(self.linlayers):
            H = F.relu(hidden(H))

        # print(H)
        loss = self.loss(F.softmax(H, dim=1), A)
        return loss

    def __repr__(self):
        return str([self.graphlayers[i]
                    for i in range(len(self.graphlayers))] +
                   [self.linlayers[i] for i in range(len(self.linlayers))])


class NewNCutLoss(nn.Module):
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    def __init__(self, N, g):
        super(NewNCutLoss, self).__init__()
        self.N = N
        self.g = g

    def forward(self, Y, A):
        D = torch.sparse.sum(A, dim=1).to_dense()
        # print("D is requires_grad: ", D.requires_grad)
        Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        # print("Gamma is requires_grad: ", Gamma.requires_grad)
        YbyGamma = torch.div(Y, Gamma.t())
        # print("YbyGamma is requires_grad: ", YbyGamma.requires_grad)
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss = torch.tensor([0.], requires_grad=True).to('cuda')
        idx = A._indices()
        data = A._values()
        print("开始计算损失")
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1,
                                                                 i]]) * data[i]
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)
        return loss