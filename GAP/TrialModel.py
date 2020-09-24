from __future__ import division
from __future__ import print_function
import tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from scipy import sparse
from models import *
import matplotlib.pyplot as plt
from graph_loader import GraphLoader

def Train(model, x, adj, A, optimizer):
    '''
    Training Specifications
    '''

    max_epochs = 100
    min_loss = 100
    for epoch in (range(max_epochs)):
        loss = model(x, adj)
        # loss = custom_loss(Y, A)
        print('Epoch {}:   Loss = {}'.format(epoch, loss.item()))
        if loss < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), "./trial_weights.pt")
        loss.backward()
        optimizer.step()


def Test(model, x, adj, A, *argv):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load("./trial_weights.pt"))
    Y = model(x, adj)
    node_idx = test_partition(Y)
    print(node_idx)
    if argv != ():
        if argv[0] == 'debug':
            print(
                'Normalized Cut obtained using the above partition is : {0:.3f}'
                .format(custom_loss(Y, A).item()))
    else:
        print('Normalized Cut obtained using the above partition is : {0:.3f}'.
              format(CutLoss.apply(Y, A).item()))


def main():
    '''
    Adjecency matrix and modifications
    '''
    # # A = input_matrix()
    # A = input_sparse_matrix_data("../data/random_graph_data.txt", "All")
    # # print(A.toarray())
    # # Modifications
    # A_mod = A + sp.eye(A.shape[0])  # Adding Self Loop
    # # print(A_mod.toarray())
    # norm_adj = symnormalise(A_mod)  # Normalization using D^(-1/2) A D^(-1/2)
    # adj = sparse_mx_to_torch_sparse_tensor(norm_adj).to(
    #     'cuda')  # SciPy to Torch sparse
    # As = sparse_mx_to_torch_sparse_tensor(A).to(
    #     'cuda')  # SciPy to sparse Tensor
    # A = sparse_mx_to_torch_sparse_tensor(A).to_dense().to(
    #     'cuda')  # SciPy to Torch Tensor
    graph_data  = GraphLoader("/home/lisali/GCN_Partitioning/data/random_graph_data.txt", "All")
    A, As, adj = graph_data.get_data()



    # ###plot
    # fig = plt.figure(num=1, figsize=(500, 500))
    # plt.ion()
    # fig1 = fig.add_subplot(1, 2, 1)
    # fig1.set_title('original matrix')
    # plt.imshow(As.to('cpu').toarray())
    # fig2 = fig.add_subplot(1, 2, 2)
    # fig2.set_title('adjacency matrix')
    # plt.imshow(adj.to('cpu').toarray())
    # plt.show()
    '''
    Declare Input Size and Tensor
    '''
    N = A.shape[0]
    d = 100

    torch.manual_seed(100)
    x = torch.randn(N, d)
    x = x.to('cuda')
    '''
    Model Definition
    '''
    gl = [d, 64, 16]
    ll = [16, 5]

    model = GCN(gl, ll, dropout=0.5).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-6)
    print(model)

    # check_grad(model, x, adj, A, As)

    #Train
    Train(model, x, adj, As, optimizer)

    # Test the best partition
    Test(model, x, adj, As)


if __name__ == '__main__':
    main()
