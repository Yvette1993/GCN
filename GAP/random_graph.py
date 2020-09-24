#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/21 下午2:17
# @Author  : Yingying Li

import argparse
import numpy as np
import os
from igraph import Graph, EdgeSeq
root = os.getcwd()
path = os.path.join(root, 'GCN_Partitioning/GAP/')
parser = argparse.ArgumentParser(description='Create graph matrix')
## tree params
parser.add_argument('--nodes_number', default='25', type=int, help='nodes number')
parser.add_argument('--tree_children', default='2', type=int, help='tree children number')

parser.add_argument('--graph_model', default='Tree', type=str,
                    help='graph layout model, default Tree ')
parser.add_argument('--graph_layout', default='kk', type=str,
                    help='graph layout model, default "rt", you can choice "kk","circle","sphere","drl","fr"')
parser.add_argument('--save_path', default=path, type=str, metavar='PATH', help='numpy save path')
args = parser.parse_args()


nr_vertices = args.nodes_number
v_label = list(map(str, range(nr_vertices)))
if args.graph_model == 'Tree':
    G = Graph.Tree(nr_vertices, args.tree_children) # 2 stands for children number

lay = G.layout(args.graph_layout)

position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(G) # sequence of edges
E = [e.tuple for e in G.es] # list of edges

matrix = np.zeros((args.nodes_number, args.nodes_number))

L = len(position)
Xn = [position[k][0] for k in range(L)]
Yn = [2*M-position[k][1] for k in range(L)]
Xe = []
Ye = []
for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    matrix[edge[0], edge[1]] = 1
    matrix[edge[1], edge[0]] = 1

# print("matirx", matrix)
# file_path = os.path.join(args.save_path, 'graph_matrix.bin')
# # print(file_path)
# matrix.tofile(file_path)


labels = v_label

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   ))
fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  mode='markers',
                  name='bla',
                  marker=dict(symbol='circle-dot',
                                size=18,
                                color='#6175c1',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  text=labels,
                  hoverinfo='text',
                  opacity=0.8
                  ))
def make_annotations(pos, text, font_size=10, font_color='rgb(250,250,250)'):
    L=len(pos)
    if len(text)!=L:
        raise ValueError('The lists pos and text must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=labels[k], # or replace labels with a different list for the text within the circle
                x=pos[k][0], y=2*M-position[k][1],
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations
axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

fig.update_layout(title= 'Tree with Reingold-Tilford Layout',
              annotations=make_annotations(position, v_label),
              font_size=12,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              margin=dict(l=40, r=40, b=85, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
              )
fig.show()
