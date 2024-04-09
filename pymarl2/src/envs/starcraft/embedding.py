# -----------------------------------------------------------------------------
# This script runs the experiments reported in the WWL paper
#
# October 2019, M. Togninalli, E. Ghisu, B. Rieck
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import argparse
import os

from utilities import read_labels, custom_grid_search_cv
from wwl import *
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from wwl import compute_wl_embeddings_discrete
import random

import json,glob
def read_dataset(path):
    data = json.load(open(path))
    edges = data['edges']
    features = data['features']
    weights = data['weight']

    return features,weights

def get_embedding():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='MUTAG', type=str, help='Provide the dataset name (MUTAG or Enzymes)',
                            choices=['MUTAG', 'ENZYMES'])
    parser.add_argument('--crossvalidation', default=False, action='store_true', help='Enable a 10-fold crossvalidation')
    parser.add_argument('--gridsearch', default=False, action='store_true', help='Enable grid search')
    parser.add_argument('--sinkhorn', default=False, action='store_true', help='Use sinkhorn approximation')
    parser.add_argument('--h', type = int, required=False, default=2, help = "(Max) number of WL iterations")

    args = parser.parse_args()
    dataset = args.dataset
    h = args.h
    sinkhorn = args.sinkhorn
    print(f'Generating results for {dataset}...')
    #---------------------------------
    # Setup
    #---------------------------------
    # Start by making directories for intermediate and final files
    data_path = os.path.join('../data', dataset)
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)
    
    for path in [output_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    #---------------------------------
    # Embeddings
    #---------------------------------
    # Load the data and generate the embeddings 
    # paths = glob.glob(os.path.join('../data/starcraft/', '*.json'))
    dir = '../data/starcraft/'
    paths = [dir +'4_graph_1.json',dir+'4_graph_2.json']
    label_sequences = []
    for path in paths:
        features,weights = read_dataset(path)
        n_nodes = len(features)
        n_items = 3
        init_labels = []
        for key in features.keys():
            init_labels.append(features[key])
        dict_labels = compute_wl_propagation_aggregation(n_nodes,init_labels,h,weights,n_items)
        label_sequences.append(dict_labels[h-1])

    
    #---------------------------------
    # Wasserstein & Kernel computations
    #---------------------------------
    # Run Wasserstein distance computation
    print('Computing the Wasserstein distances...')
    wasserstein_distances = compute_wasserstein_distance(label_sequences, h, sinkhorn=sinkhorn,discrete=False)
    return wasserstein_distances

