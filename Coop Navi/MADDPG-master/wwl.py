# -----------------------------------------------------------------------------
# This file contains the functions to compute the node embeddings and to compute
# the wasserstein distance matrix
#
# October 2019, M. Togninalli, E. Ghisu, B. Rieck
# -----------------------------------------------------------------------------
import numpy as np

from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin

import argparse
import os
import ot

import copy
from collections import defaultdict
from typing import List


####################
# Weisfeiler-Lehman
####################


def compute_wasserstein_distance(labels_1,labels_2, h, sinkhorn=False, 
                                    discrete=False, sinkhorn_lambda=1e-2):
    '''
    Generate the Wasserstein distance matrix for the graphs embedded 
    in label_sequences
    '''
    
    # Iterate over pairs of graphs
    # labels_1 = label_sequences[0]
    # labels_2 = label_sequences[1]
    # Get cost matrix
    ground_distance = 'hamming' if discrete else 'euclidean'
    costs = ot.dist(labels_1, labels_2, metric=ground_distance)

    if sinkhorn:
        mat = ot.sinkhorn(np.ones(len(labels_1))/len(labels_1), 
                            np.ones(len(labels_2))/len(labels_2), costs, sinkhorn_lambda, 
                            numItermax=50)
        dis = np.sum(np.multiply(mat, costs))
    else:
        dis = ot.emd2([], [], costs)
                    
    return dis

def compute_wl_propagation_aggregation(n_nodes, init_labels, iterations, weights, n_items):
    '''
    n_items: The number of attributes of the node
    '''
    dict_labels = {}
    dict_labels[0] = init_labels#存储每次传播的结果
    
    for it in range(iterations):
        label_before = dict_labels[it]  #传播前的状态
        label_after = []    #传播后的状态
        for node_id in range(n_nodes):
            label_node = label_before[node_id]
            label_node_new = []
            for i in range(n_items):
                label_node_new.append(label_node[i])
            for neighbor_id in range(n_nodes):
                if neighbor_id != node_id:
                    label_neighbor = label_before[neighbor_id]
                    t = (node_id,neighbor_id)
                    if node_id > neighbor_id:
                        t = (neighbor_id, node_id)
                    weight = weights[str(t)]
                    for i in range(n_items):
                        label_node_new[i] += 1 / weight * label_neighbor[i]
            label_after.append(label_node_new)
        dict_labels[it+1] = np.array(label_after)
    return dict_labels   #返回最终的传播结果



