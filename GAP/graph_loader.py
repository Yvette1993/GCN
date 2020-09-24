#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/17 下午1:47
# @Author  : Yingying Li

import numpy as np
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt

def input_sparse_matrix_data(data_path, number):
    '''
    # 读取外部文件的数据
    Returns a input sparse SciPy adjecency matrix data
    '''
    # 最大节点的编号
    max_value = 0
    temp_row = []
    temp_col = []
    temp_data = []

    with open(data_path) as f:
        if number == "All":
            lines = f.readlines()
        else:
            lines = f.readlines(number)
        for line in lines:
            temp = line.split(" ")
            max_value = max(max_value, int(temp[0]), int(temp[1]))
            temp_row.append(int(temp[0]))
            temp_col.append(int(temp[1]))
            temp_data.append(1)

    N = max_value + 1
    data = np.array(temp_data)
    row = np.array(temp_row)
    col = np.array(temp_col)

    A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    #print(A.toarray())

    return A
def symnormalise(M):
    """
    symmetrically normalise sparse matrix
    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2}
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))
    dhi = np.power(d, -1 / 2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)  # D half inverse i.e. D^{-1/2}


    return (DHI.dot(M)).dot(DHI)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GraphLoader:
    def __init__(self, data_path, number):
        self.path = data_path
        self.number = number

    def get_data(self):
        A = input_sparse_matrix_data(self.path, self.number)
        # Modifications
        A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
        norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
        adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to('cuda')  # SciPy to Torch sparse
        As = sparse_mx_to_torch_sparse_tensor(A).to('cuda')  # SciPy to sparse Tensor
        A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to('cuda')  # SciPy to Torch Tensor

        ###plot
        fig = plt.figure(num=1, figsize=(500, 500))
        plt.ion()

        fig1 = fig.add_subplot(1, 2, 1)
        fig1.set_title('original matrix')
        plt.imshow(A.cpu().numpy())

        # fig2 = fig.add_subplot(1, 2, 2)
        # fig2.set_title('adjacency matrix')
        # plt.imshow(adj.to_array())
        plt.show()

        return (A, As, adj)



