import torch.nn as nn
import torch

class Cut_Balance_Loss(nn.Module):
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    def __init__(self, N, g):
        super(Cut_Balance_Loss, self).__init__()
        self.N = N
        self.g = g

    def forward(self, Y, A):
        # D = torch.sparse.sum(A, dim=1).to_dense()
        # print("Y is requires_grad: ", Y.requires_grad)
        # Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        Gamma = torch.sparse.sum(A)
        # print("Gamma is requires_grad: ", Gamma.requires_grad)
        YbyGamma = torch.div(Y, Gamma)
        # print("YbyGamma is requires_grad: ", YbyGamma.requires_grad)
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss_1 = torch.tensor([0.], requires_grad=True).to('cuda')
        idx = A._indices()
        data = A._values()
        # print(idx, data)
        # print("开始计算损失")
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss_1 += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1,i]])
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)

        loss_2 = torch.tensor([0.], requires_grad=True).to('cuda')
        # i 表示第几个子图
        e = torch.tensor([self.N/self.g]).to('cuda')
        # loss_2 = torch.div(torch.max(torch.sum(Y, 0)), e)
        for i in range(Y.shape[1]):
            # 纵向求和, 计算属于具体一个字图的概率之和
            loss_2 += torch.pow((torch.sum(Y, 0)[i] - e), 2)

        loss = loss_1 + loss_2
        return loss, loss_1, loss_2, Y






#平衡损失修改后的结果
class Cut_Balance_Loss_v2(nn.Module):
    '''
    Class for forward and backward pass for the loss function described in https://arxiv.org/abs/1903.00614

    arguments:
        Y_ij : Probability that a node i belongs to partition j
        A : sparse adjecency matrix

    Returns:
        Loss : Y/Gamma * (1 - Y)^T dot A
    '''

    def __init__(self, N, g):
        super(Cut_Balance_Loss_v2, self).__init__()
        self.N = N
        self.g = g

    def forward(self, Y, A):
        # D = torch.sparse.sum(A, dim=1).to_dense()
        # print("Y is requires_grad: ", Y.requires_grad)
        # Gamma = torch.mm(Y.t(), D.unsqueeze(1))
        Gamma = torch.sparse.sum(A)
        # print("Gamma is requires_grad: ", Gamma.requires_grad)
        YbyGamma = torch.div(Y, Gamma)
        # print("YbyGamma is requires_grad: ", YbyGamma.requires_grad)
        # print(Gamma)
        Y_t = (1 - Y).t()
        loss_1 = torch.tensor([0.], requires_grad=True).to('cuda')
        idx = A._indices()
        data = A._values()
        # print(idx, data)
        # print("开始计算损失")
        for i in range(idx.shape[1]):
            # print(YbyGamma[idx[0,i],:].dtype)
            # print(Y_t[:,idx[1,i]].dtype)
            # print(torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1, i]]) * data[i])
            loss_1 += torch.dot(YbyGamma[idx[0, i], :], Y_t[:, idx[1,i]])
            # print(loss)
        # loss = torch.sum(torch.mm(YbyGamma, Y_t) * A)

        loss_2 = torch.tensor([0.], requires_grad=True).to('cuda')
        # i 表示第几个子图
        e = torch.tensor([self.N/self.g]).to('cuda')
        # loss_2 = torch.div(torch.max(torch.sum(Y, 0)), e)
        for i in range(Y.shape[1]):
            # 纵向求和, 计算属于具体一个字图的概率之和
            loss_2 += torch.pow(((torch.sum(Y, 0)[i]) / e - 1), 2)

        loss = loss_1 + (loss_2 / self.g)

        return loss, loss_1, loss_2, Y