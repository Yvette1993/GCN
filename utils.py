import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import graph_tool
import graph_tool.all as gt
import re
from sklearn.manifold import spectral_embedding
import random


def gene_random_graph_data(vertices_count,
                           edges_count,
                           save_file_path='random_graph_data.txt'):
    '''
    Returns a random sparse graph adjecency matrix

    @param vertices_count: int, 生成顶点的个数
    @param edges_count: int, 生成边的个数
    @param save_file_path: str, 图数据文件的保存路径
    '''

    # 用于保存随机生成的边的集合，元素（i, j）表示一条边
    edge_set = set()
    # 当edge_set的size满足参数edges_count的大小时，停止随机生成。这里edges_count * 2是因为生成的边是双向，即(i, j) and (j, i)。
    while len(edge_set) < edges_count * 2:
        node1 = random.randint(0, vertices_count - 1)
        node2 = random.randint(0, vertices_count - 1)
        # 如果顶点1的编号与顶点2的编号相同，则重新生成顶点2的编号
        while node1 == node2:
            node2 = random.randint(0, vertices_count - 1)
        edge_set.add((node1, node2))
        edge_set.add((node2, node1))

    # 写入文件
    with open (save_file_path, "w") as fo:
        for (i, j) in edge_set:
            fo.write(str(i) + " " + str(j) + "\n")


def input_sparse_matrix_data(data_path, number="All"):
    '''
    # 读取外部文件的数据
    Returns a input sparse SciPy adjecency matrix data
    '''
    # 最大节点的编号
    max_value = 0
    temp_row = []
    temp_col = []
    temp_data = []

    regex = ",|，|\\s+|\t"

    with open(data_path) as f:
        if number == "All":
            lines = f.readlines()
        else:
            lines = f.readlines(number)
        for line in lines:
            temp = re.split(regex, line)
            max_value = max(max_value, int(temp[0]), int(temp[1]))
            temp_row.append(int(temp[0]))
            temp_col.append(int(temp[1]))
            temp_data.append(1)

    N = max_value + 1
    data = np.array(temp_data)
    row = np.array(temp_row)
    col = np.array(temp_col)

    A = sp.csr_matrix((data, (row, col)), shape=(N, N))

    return A


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def graph_k_adj(graph, hop):
    '''
    功能描述：根据 hop 中的跳数信息返回邻接列表
    输入参数：融合后的图，跳数
    输出参数：一个 map，key 是跳数，value 是映射关系
    '''
    g = graph.copy()
    # 返回节点数量
    ver_num = g.num_vertices()
    # 判断索引是否越界
    # print(ver_num)
    assert max(hop) <= ver_num - 1

    # 结果存储
    # hop_adj = {}
    edge_indexs = {}
    # 生成邻接的矩阵图
    for h in hop:
        edge_indexs[h] = [[], []]

    # 遍历 所有 的节点
    for ver in g.vertices():
        # a 表示返回实际的距离矩阵
        # print("*****************************", g.vertex_index[ver])
        dist = gt.shortest_distance(g, source=g.vertex(ver)).a
        # 如果说，是连通图，且 h 跳邻居存在。那么就取 h 的。
        for h in hop:

            if h in dist:
                target = h
        # 否则取最大的跳数
            else:
                # 降序排列
                ls = [i for i in range(max(hop), min(hop) - 1, -1)]
                # 判断哪个最大的跳先满足
                mask = np.isin(ls, dist).tolist()
                if True in mask:
                    # 第一个 true 就是要取值的点
                    target = ls[mask.index(True)]
                # 孤立节点 和自己距离是 0 ，和其他点的距离都无限
                else:
                    target = 0

            mask = np.where(dist == target)
            idx1 = g.vertex_index[ver]
            # a 返回的矩阵是(size,) 没有列，0 表示取第一维
            idx2_group = mask[0]
            if len(idx2_group) != 0:
                for idx2 in idx2_group:
                    edge_indexs[h][0].append(idx1)
                    edge_indexs[h][1].append(idx2)

    for h in hop:
        edge_indexs[h] = torch.from_numpy(np.array(edge_indexs[h]))

    return edge_indexs


def load_graph_data_file(load_data_path):
    '''
    Returns a graph and file formated 'xml.gz'
    '''
    regex = ",|，|\\s+|\t"
    g = gt.Graph(directed=False)

    with open(load_data_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            #文件中的数据的每一行是图上的每一条边的两个顶点
            temp = re.split(regex, line)
            g.add_edge(int(temp[0]), int(temp[1]))

    return g



def node_features(graph, k):
    '''
    功能描述：根据给出的图graph，求其给定的k维的node feature
    输入参数：需要求的node feature的图
    输出参数：图上每个节点的feature
    '''
    adj_matrix = graph_tool.spectral.adjacency(graph, weight=None, index=None)
    adj_matrix = adj_matrix.todense()
    print(adj_matrix.shape)
    node_feature = spectral_embedding(adj_matrix, k)
    return node_feature



if __name__ == '__main__':
    gene_random_graph_data(1000,
                           4000,
                           save_file_path='random_graph_data_1000.txt')