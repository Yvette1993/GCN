#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 下午5:46
# @Author  : Yingying Li


import argparse
import os
from igraph import Graph
import numpy as np
parser = argparse.ArgumentParser(description='  matrix to graph')
parser.add_argument('--matrix_file',default='./file.npy', type=str, metavar='PATH', help='numpy save path')
args = parser.parse_args()

# random_npy

matrix = np.load(args.matrix_file)

g = Graph(1)
g.add_vertex(len(matrix[0]))



