from __future__ import division
from __future__ import print_function

import time
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import cv2
import pickle
import optuna
import sklearn.metrics as sk_metrics

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import networkx as nx

from utils import *
from models import *


model_path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/src/pygcn/pygcn/trained_model_one_feature.pth"
path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/4Graphs/"
def test(path):
    # Initialize the model and optimizer
    
    
    lr = 0.01
    weight_decay = 5e-4
    dropout = 0.2
    nNodes = 728
    n_layers = 4
    hidden = 4
    
    # # Load data
    # adj, features, labels, idx_train, idx_val, idx_test = load_data()
    n_features = 1
    # n_features = 8
    n_class = 2

    model = GCN(nfeat=n_features, nhid=hidden, nclass=n_class, dropout=dropout, nNodes=nNodes)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Load the saved model and optimizer state dictionaries
    model, optimizer = load_model(model, optimizer, model_path)
    model.cuda()

    model.eval()
    loss_list = []
    acc_list = []
    f1_list = []
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        # Load data
        adj, features, labels = load_data(path = file_path, resample = False)
        
        features = features.cuda()
        adj = adj.cuda()
        # labels = labels.cuda()
        
        
        output = model(features, adj)
        # loss_test = F.nll_loss(output, labels)
        # weights = calculate_weights(labels)
        # loss_test = F.cross_entropy(output, labels, weight=weights)
        
        # Define BCEWithLogitsLoss with weights
        # criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        # output = output.squeeze(1).float()
        # labels = labels.float()
        # print(f"output size = {output.size()}")
        # loss_test = criterion(output, labels)

        # print(f"original label for {file_name} is {labels}, has shape {labels.shape}")
        # print(f"output for {file_name} is :{output}, has shape{output.shape}")

        # incorrect_nodes = incorrect_node(output, labels)
        # acc_test = accuracy(output, labels)
        # f1_test = f1_score(output, labels)
        # print(f"output is {output}")
        # print(f"label is {labels}")
        
        
        # loss_list.append(loss_test)
        # acc_list.append(acc_test)
        # f1_list.append(f1_test)
        
        # print("Test set results:",
        #     "loss= {:.4f}".format(loss_test.item()),
        #     "accuracy= {:.4f}".format(acc_test.item()),
        #     "f1_test= {:.4f}".format(f1_test.item()))
        visualize(file_name, adj, features, output)
        # break
                
                
test(path)