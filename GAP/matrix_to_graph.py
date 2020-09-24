#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 下午5:46
# @Author  : Yingying Li


import argparse
import os
from igraph import Graph, EdgeSeq
import numpy as np
root = os.getcwd()
path = os.path.join(root, 'GCN_Partitioning/GAP/')
parser = argparse.ArgumentParser(description='  matrix to graph')
parser.add_argument('--graph_layout', default='kk', type=str,
                    help='graph layout model, default "rt", you can choice "kk","circle","sphere","drl","fr"')
parser.add_argument('--load_matrix_file', default=os.path.join(path, 'file2.npy'), type=str, metavar='PATH', help='loading matrix path')
parser.add_argument('--save_total_matrix', default=os.path.join(path, 'total.npy'), type=str, metavar='PATH', help='saving total matrix path')
args = parser.parse_args()

# random_npy

matrix = np.load(args.load_matrix_file)
nr_vertices = len(matrix[0])
v_label = list(map(str, range(nr_vertices)))
g = Graph()
g.add_vertices(nr_vertices)
nozero = matrix.nonzero()
Edges = [(idx, idy) for idx, idy in zip(nozero[0], nozero[1])]
g.add_edges(Edges)
lay = g.layout(args.graph_layout)


position = {k: lay[k] for k in range(nr_vertices)}
Y = [lay[k][1] for k in range(nr_vertices)]
M = max(Y)

es = EdgeSeq(g) # sequence of edges
E = [e.tuple for e in g.es] # list of edges

matrix = np.zeros((nr_vertices, nr_vertices))

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

if args.save_total_matrix:
    file_path = os.path.join(args.save_total_matrix, 'graph_matrix.npy')
    np.save(file_path, matrix)


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






