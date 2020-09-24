from __future__ import division
from __future__ import print_function
from tqdm import trange, tqdm
import time
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from scipy import sparse
from model.gcn_models import *
import matplotlib.pyplot as plt

#与loss相同的函数 ,N表示图上节点的个数，g表示分区的个数
def loss_function(Y1,Y2,A, N, g):
    # print("Y is requires_grad: ", Y.requires_grad)
    # Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    Gamma = torch.sparse.sum(A)
    # print("Gamma is requires_grad: ", Gamma.requires_grad)
    YbyGamma = torch.div(Y1, Gamma)
    # print("YbyGamma is requires_grad: ", YbyGamma.requires_grad)
    # print(Gamma)
    Y_t = (1 - Y1).t()
    loss_1 = torch.tensor([0.], requires_grad=True).to('cuda')
    idx = A._indices()
    data = A._values()
    # print(idx, data)
    # print("开始计算损失")
    for i in range(idx.shape[1]):
        loss_1 += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1,i]])
        # print(loss)
    # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)

    loss_2 = torch.tensor([0.], requires_grad=True).to('cuda')
    # i 表示第几个子图
    e = torch.tensor([N / g]).to('cuda')
    # loss_2 = torch.div(torch.max(torch.sum(Y2, 0)), e)
    for i in range(Y2.shape[1]):
            # 纵向求和, 计算属于具体一个字图的概率之和
            loss_2 += torch.pow((torch.sum(Y2, 0)[i] - e), 2)
    
    return loss_1, loss_2





#与loss相同的函数 ,N表示图上节点的个数，g表示分区的个数
def loss_function_v2(Y1,Y2,A, N, g):
    # print("Y is requires_grad: ", Y.requires_grad)
    # Gamma = torch.mm(Y.t(), D.unsqueeze(1))
    Gamma = torch.sparse.sum(A)
    # print("Gamma is requires_grad: ", Gamma.requires_grad)
    YbyGamma = torch.div(Y1, Gamma)
    # print("YbyGamma is requires_grad: ", YbyGamma.requires_grad)
    # print(Gamma)
    Y_t = (1 - Y1).t()
    loss_1 = torch.tensor([0.], requires_grad=True).to('cuda')
    idx = A._indices()
    data = A._values()
    # print(idx, data)
    # print("开始计算损失")
    for i in range(idx.shape[1]):
        loss_1 += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1,i]])
        # print(loss)
    # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)

    loss_2 = torch.tensor([0.], requires_grad=True).to('cuda')
    # i 表示第几个子图
    e = torch.tensor([N / g]).to('cuda')
        # loss_2 = torch.div(torch.max(torch.sum(Y, 0)), e)
    for i in range(Y2.shape[1]):
        # 纵向求和, 计算属于具体一个字图的概率之和
        loss_2 += torch.pow(((torch.sum(Y2, 0)[i]) / e - 1), 2)

    return loss_1, loss_2 / g





def Train(model, x, k_adj_map, original_adj, optimizer, N, nclass):
    '''
    Training Specifications
    '''

    max_epochs = 200
    min_loss = 100
    loss_list = []
    # epochs = trange(max_epochs, desc = "Train Loss")
    epochs = range(max_epochs)
    for epoch in epochs:
        model.zero_grad()
        print("第{}轮，开始加载模型".format(epoch))
        loss,loss_1,loss_2,Y = model(x, k_adj_map,original_adj)
        
        Y1 = Y.clone().detach()
        Y1.requires_grad=True
        Y1 = Y1.to('cuda')

        Y2 = Y.clone().detach()
        Y2.requires_grad=True
        Y2 = Y2.to('cuda')
        
        v = torch.ones([1]).float().to('cuda')
        # print("V :  ", v)
        # z.backward(v)
        # print("Y.grad:                      ", Y1.grad)
        #还没有修改完
        loss1, loss2 = loss_function_v2(Y1,Y2,original_adj, N, nclass)
        loss1.backward(v)
        grad1 =Y1.grad
        # print(grad1)
        grad1 = torch.reshape(grad1,(1,N*nclass))
        loss2.backward(v)
        grad2 = Y2.grad
        # print(grad2)
        grad2 = torch.reshape(grad2,(1,N*nclass))
        grad_cos = torch.cosine_similarity(grad1, grad2, dim=1)
        # alpha = 0.5
        # if grad2.mm(grad1.t()).item() >= grad1.mm(grad1.t()).item():
        #     alpha = 1
        # elif grad2.mm(grad1.t()).item() >= grad2.mm(grad2.t()).item():
        #     alpha = 0
        # else:
        #     alpha = grad2.mm((grad2 - grad1).t()).item() / (torch.norm(grad2 - grad1)**2).item()

        # print('alpha',alpha)

        # loss = alpha*loss_1+(1-alpha)*loss_2

        grad_joint = grad1 + grad2

        print('cos = {}'.format( grad_cos))

        print('cos similarity grad1 and joint = {}'.format( torch.cosine_similarity(grad1, grad_joint, dim=1)))

        print('cos similarity grad2 and joint = {}'.format( torch.cosine_similarity(grad2, grad_joint, dim=1)))

        print('grad1 norm',torch.norm(grad1))

        print('grad2 norm',torch.norm(grad2))

        print('Loss = {}, {}, {}'.format(loss.item(), loss_1.item(), loss_2.item()))

        # print('Y', Y)
        
        # epochs.set_description("Train Loss: %g" % round(loss_2.item(),4))
        # if loss < min_loss:
            # min_loss = loss.item()
            # torch.save(model.state_dict(), "./trial_weights.pt")
        loss.backward()
        # print("Y.grad:                ", Y.grad)
        optimizer.step()

        loss_list.append(loss_2.item())
        print()
        
    ShowPicture(loss_list,max_epochs)

def ShowPicture(list, epochs):
    x = np.arange(epochs)
    y = np.array(list)
    plt.plot(x, y)
    # print(plt.plot(x, y))
    plt.savefig('loss.jpg')
    plt.show()

def Test(model, x, k_adj_map, original_adj, *argv):
    '''
    Test Final Results
    '''
    model.load_state_dict(torch.load("./trial_weights.pt"))
    Y = model(x, k_adj_map)
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

    print("开始加载图数据")
    # graph = load_graph_data_file('./edges.csv')
    graph = load_graph_data_file('data/random_graph_data_1000.txt')
    hop = [1, 3, 9]
    print("开始计算 k 跳邻接信息")
    hop_adj = graph_k_adj(graph, hop)
    k_adj_map = {
        key: torch.LongTensor(value.long()).to('cuda')
        for key, value in hop_adj.items()
    }

    for key, value in hop_adj.items():
        print(key, value.size())

    # 读取图的稀疏邻接矩阵
    print("开始生成邻接矩阵")
    # A = input_sparse_matrix_data('./edges.csv')
    A = input_sparse_matrix_data('data/random_graph_data_1000.txt')
    A = sparse_mx_to_torch_sparse_tensor(A).to('cuda')

    N = graph.num_vertices()
    edges_number = graph.num_edges()
    d = 512

    torch.manual_seed(100)
    x = torch.randn(N, d,requires_grad=True)
    x = x.to('cuda')
    '''
    Model Definition
    '''
    nclass = 5

    model = Netwrok(nclass=nclass,
                  edges=edges_number,
                  in_channels=512,
                  out_channels=[256,128,64,32],
                  heads=10,
                  dropout=0.5,
                  negative_slope=0.2,
                  depth=3,
                  pooling_ratio=1,
                  N=N).to('cuda')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    # check_grad(model, x, adj, A, As)

    #Train
    print("开始训练")

    # Y = model()
    
    Train(model, x, k_adj_map, A, optimizer, N, nclass)

    # Test the best partition
    model.eval()
    # Test(model, x, k_adj_map)


if __name__ == '__main__':
    main()