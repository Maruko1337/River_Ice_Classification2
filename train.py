from __future__ import division
from __future__ import print_function

import time
import pandas as pd
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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
# print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
output_form = {'Date': [], 'Accuracy': []}
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
model_name = "lwGCN"
area_of_interest = "lake"

num_layers = [4] # 256
test_years = ["2017", "2018","2019","2020","2021"]
to_mask = False
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.get_device_properties('cuda:0'))

for test_year in test_years:
    hidden = 4
    # torch.cuda.synchronize()
    print("Doing experiment for ", test_year, " year!", flush=True)
    # torch.cuda.synchronize()
    
    def objective(trial):
        
        # # Load data
        # adj, features, labels, idx_train, idx_val, idx_test = load_data()
        n_features = 2
        # n_features = 8
        n_class = 2

        
        # Hyperparameters to tune
        # lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        # weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        # hidden = trial.suggest_int('hidden', 4, 128)
        # dropout = trial.suggest_discrete_uniform('dropout', 0.5, 0.8, q=0.1)
        
        lr = 0.01
        weight_decay = 5e-4
        dropout = 0.2
        if area_of_interest != "lake":
            nNodes = 1181
        else:
            nNodes = 2131
        n_layers = 4
        

        # Model and optimizer
        if model_name == "lwGCN":
            model = lwGCN(nfeat=n_features,
                        nhid=hidden,
                        nclass=n_class,
                        dropout=dropout, 
                        nNodes=nNodes)
        else:
            model = GCN(nfeat=n_features,
                        nhid=hidden,
                        nclass=n_class,
                        dropout=dropout, 
                        nNodes=nNodes)
        # model = DeeperGCN(dataset="Beauhorinoise",
        #               node_feat_dim=n_features,
        #               edge_feat_dim=None,
        #               hid_dim=hidden,
        #               out_dim=n_class,
        #               num_layers=n_layers,
        #               dropout=dropout,
        #               learn_beta=True)

        # print(model)
        
        optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)

        if args.cuda:
            model.cuda()
            # features = features.cuda()
            # adj = adj.cuda()
            # labels = labels.cuda()
        #     idx_train = idx_train.cuda()
        #     idx_val = idx_val.cuda()
        #     idx_test = idx_test.cuda()
        
        

        
        
        def train(epoch, adj, features, labels):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            print(f"features {features}")
            print(f"adj:{adj}")
            output = model(features, adj)
            # print(f"output is {output}, with size {output.shape}")
            # print(f"label is {labels}, with size {labels.shape}")
            
            # loss_train = F.nll_loss(output, labels)
            
            weights = calculate_weights(labels)
            print(f"weights = {weights}")
            print(f"label = {labels}")
            print(f"output = {output}")
            
            loss_train = F.cross_entropy(output, labels, weight=weights)
            
            # Define BCEWithLogitsLoss with weights
            # criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
            # output = output.squeeze(1).float()
            # labels = labels.float()
            # print(f"output size = {output.size()}")
            # print(f"labels size = {labels.size()}")
            # loss_train = criterion(output, labels)

            acc_train = accuracy(output, labels)
            f1_train = f1_score(output, labels)
            
            optimizer.zero_grad() # added
            loss_train.backward()
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output = model(features, adj)

            # loss_val = F.nll_loss(output, labels)
            # acc_val = accuracy(output, labels)
            print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'acc_train: {:.4f}'.format(acc_train),
                'f1_train: {:.4f}'.format(f1_train),
                #   'loss_val: {:.4f}'.format(loss_val.item()),
                #   'acc_val: {:.4f}'.format(acc_val.item()),
                'time: {:.4f}s'.format(time.time() - t))
            return loss_train.item(), acc_train, f1_train

        # Function to train the model on the training set
        def train_model(path, train_files):
            train_loss_list = []
            train_acc_list = []
            train_f1_list = []
            for file_name in train_files:
                
                if not file_name.startswith(test_year):
                    file_path = os.path.join(path, file_name)
                    
                    print(f"training doing {file_path}")
                    # Load and preprocess training data
                    if to_mask:
                        adj, features, labels, mask = load_data(file_path, mask = True)
                        mask_rate = sum(mask) / len(mask) * 100
                        print(f"mask = {mask}, label rate = {mask_rate}")
                        mask = torch.tensor(mask, dtype=torch.bool)
                        print(f"label length = {len(labels)}, mask len = {len(mask)}")
                        print(f"label mask = 1: {labels[mask]}")
                    else:
                        adj, features, labels = load_data(file_path)
                    
                    if args.cuda:
                        features = features.cuda()
                        adj = adj.cuda()
                        labels = labels.cuda()

                    # Perform forward pass
                    print(f"size of feature: {features.size()}, size of adj:{adj.size()}")
                    output = model(features, adj)

                    # Calculate loss
                    # breakpoint()
                    if to_mask:
                        weights = calculate_weights(labels[mask])
                        train_loss = F.cross_entropy(output[mask], labels[mask], weight=weights)
                    
                        # Calculate accuracy and F1 score
                        train_acc = accuracy(output[mask], labels[mask])
                        train_f1 = f1_score(output[mask], labels[mask])
                    else:
                        weights = calculate_weights(labels)
                        train_loss = F.cross_entropy(output, labels, weight=weights)
                    
                        # Calculate accuracy and F1 score
                        train_acc = accuracy(output, labels)
                        train_f1 = f1_score(output, labels)
                    print(f"weight = {weights}, weight size = {len(weights)}")
                    
                    # train_loss = F.cross_entropy(output, labels, weight=weights)
                    
                    

                    train_loss_list.append(train_loss.item())
                    train_acc_list.append(train_acc)
                    train_f1_list.append(train_f1)

                

                

                    # Perform backward pass and update parameters
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()
                
            # Calculate mean validation metrics
            mean_val_loss = np.mean(train_loss_list)
            mean_val_acc = np.mean(train_acc_list)
            mean_val_f1 = np.mean(train_f1_list)
                
            return mean_val_loss, mean_val_acc, mean_val_f1

        def cross_validation(path, test_year = '2020', n_splits=4):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

            val_loss_list = []
            val_acc_list = []
            val_f1_list = []
            
            train_loss_list = []
            train_acc_list = []
            train_f1_list = []

            for train_idx, val_idx in kf.split(os.listdir(path)):
                # Split data into train and validation sets based on the fold indices
                train_files = [os.listdir(path)[i] for i in train_idx]
                val_files = [os.listdir(path)[i] for i in val_idx]

                # Train the model
                train_loss, train_acc, train_f1 = train_model(path, train_files)

                # Validate the model
                val_loss, val_acc, val_f1 = validation(path, val_files)

                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)
                val_f1_list.append(val_f1)
                
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                train_f1_list.append(train_f1)

            return val_loss_list, val_acc_list, val_f1_list, train_loss_list, train_acc_list, train_f1_list

        def validation(path, val_files,  test_year = '2020'):
            model.eval()  # Set the model to evaluation mode
            val_loss_list = []
            val_acc_list = []
            val_f1_list = []
            
            for file_name in val_files:
                if not file_name.startswith(test_year):
                    
                    file_path = os.path.join(path, file_name)
                    print(f"validation doing:{file_path}")
                    # Load and preprocess validation data
                    if to_mask:
                        adj, features, labels, mask = load_data(file_path, mask = True)
                        mask = torch.tensor(mask, dtype=torch.bool)
                    else:
                        adj, features, labels = load_data(file_path)

                    if args.cuda:
                        features = features.cuda()
                        adj = adj.cuda()
                        labels = labels.cuda()

                    # Perform forward pass
                    output = model(features, adj)
                    if to_mask:
                        masked_label = labels[mask]
                        
                        weights = calculate_weights(masked_label)
                        print(f"val weight size = {len(weights)}")
                        print(f"weight = {weights}")
                        print(f"source : {file_path}")
                        val_loss = F.cross_entropy(output[mask], labels[mask], weight=weights)

                        # Calculate accuracy and F1 score
                        val_acc = accuracy(output[mask], labels[mask])
                        val_f1 = f1_score(output[mask], labels[mask])
                    else:
                        masked_label = labels
                        
                        weights = calculate_weights(masked_label)
                        print(f"val weight size = {len(weights)}")
                        print(f"weight = {weights}")
                        print(f"source : {file_path}")
                        val_loss = F.cross_entropy(output, labels, weight=weights)

                        # Calculate accuracy and F1 score
                        val_acc = accuracy(output, labels)
                        val_f1 = f1_score(output, labels)

                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)
                    val_f1_list.append(val_f1)

            # Calculate mean validation metrics
            mean_val_loss = np.mean(val_loss_list)
            mean_val_acc = np.mean(val_acc_list)
            mean_val_f1 = np.mean(val_f1_list)

            return mean_val_loss, mean_val_acc, mean_val_f1

        def test(path, test_year="2020", last=False):
            model.eval()
            loss_list = []
            acc_list = []
            f1_list = []
            for file_name in os.listdir(path):
                if file_name.startswith(test_year):
                    file_path = os.path.join(path, file_name)
                    # Load data
                    if to_mask:
                        adj, features, labels, mask = load_data(path = file_path, mask = True)
                        mask = torch.tensor(mask, dtype=torch.bool)
                    else:
                        adj, features, labels = load_data(path = file_path)
                        
                    if args.cuda:
                        features = features.cuda()
                        adj = adj.cuda()
                        labels = labels.cuda()
                        # idx_train = idx_train.cuda()
                        # idx_val = idx_val.cuda()
                        # idx_test = idx_test.cuda()
                    
                    output = model(features, adj)
                    # loss_test = F.nll_loss(output, labels)
                    if to_mask:
                        weights = calculate_weights(labels[mask])
                        loss_test = F.cross_entropy(output[mask], labels[mask], weight=weights)
                        
                        
                        # incorrect_nodes = incorrect_node(output, labels)
                        acc_test = accuracy(output[mask], labels[mask])
                        f1_test = f1_score(output[mask], labels[mask])
                    else:
                        weights = calculate_weights(labels)
                        loss_test = F.cross_entropy(output, labels, weight=weights)
                        
                        
                        # incorrect_nodes = incorrect_node(output, labels)
                        acc_test = accuracy(output, labels)
                        f1_test = f1_score(output, labels)
                    # print(f"output is {output}")
                    # print(f"label is {labels}")
                    
                    # Define BCEWithLogitsLoss with weights
                    # criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
                    # output = output.squeeze(1).float()
                    # labels = labels.float()
                    # print(f"output size = {output.size()}")
                    # loss_test = criterion(output, labels)
            
                    # print(f"original label for {file_name} is {labels}, has shape {labels.shape}")
                    # print(f"output for {file_name} is :{output}, has shape{output.shape}")

                    
                    loss_list.append(loss_test)
                    acc_list.append(acc_test)
                    f1_list.append(f1_test)
                    
                    # print("Test set results:",
                    #     "loss= {:.4f}".format(loss_test.item()),
                    #     "accuracy= {:.4f}".format(acc_test.item()),
                    #     "f1_test= {:.4f}".format(f1_test.item()))
                    if last: 
                        visualize(file_name, adj, features, output, labels, area_of_interest, model_name)
                        output_form['Date'].append(file_name)
                        output_form['Accuracy'].append(acc_test)
                        
            loss_list = [loss.detach().cpu().numpy() for loss in loss_list]
            # acc_list = [acc.detach().cpu().numpy() for acc in acc_list]
            # f1_list = [f1.detach().cpu().numpy() for f1 in f1_list]
            return np.mean(loss_list), np.mean(acc_list), np.mean(f1_list)

        
        # Train model
        t_total = time.time()
        # path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/denser/4Graphs"
        if area_of_interest == "lake":
            path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/6Graph_label_intensity"
        else:
            path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/7relabel_graphs"
        # path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/densePreproc/4Graphs"
        # path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/6ResampledGraph/"
        # path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/4Graphs/"
        
        loss_train_list = []
        acc_train_list = []
        f1_train_list = []
        loss_val_list = []
        acc_val_list = []
        f1_val_list = []
        loss_test_list = []
        acc_test_list = []
        f1_test_list = []
        single_loss_train_list = []
        single_acc_train_list = []
        single_f1_train_list = []
        single_loss_val_list = []
        single_acc_val_list = []
        single_f1_val_list = []
        
        
        # Training process
        for epoch in range(args.epochs):
            
            
            
            # Iterate over the graph files and merge them into the merged_graph
            # for file_name in os.listdir(path):
            #     if not file_name.startswith(test_year):
            #         print(f"date is: {file_name}")
            #         file_path = os.path.join(path, file_name)
                
            #         # Load data
            #         # adj, features, labels, idx_train, idx_val, idx_test = load_data(path = file_path, resample = False)
                    
            #         # if args.cuda:
            #         #     features = features.cuda()
            #         #     adj = adj.cuda()
            #         #     labels = labels.cuda()
            #             # idx_train = idx_train.cuda()
            #             # idx_val = idx_val.cuda()
            #             # idx_test = idx_test.cuda()
            #         # print(f"date {file_name} has total node : {len(labels)}, feature size :{features.size()}, label: {labels}")
                    
                    
            #         # loss_train, acc_train, f1_train = train(epoch, adj, features, labels)
                    
            #         loss_val, acc_val, f1_val, loss_train, acc_train, f1_train = cross_validation(file_path)
                    
            #         if math.isnan(loss_train) or math.isnan(loss_val) : raise Exception("nan output -----------------------------")
            #         single_loss_train_list.append(loss_train)
            #         single_acc_train_list.append(acc_train)
            #         single_f1_train_list.append(f1_train)
                    
            #         single_loss_val_list.append(loss_val)
            #         single_acc_val_list.append(acc_val)
            #         single_f1_val_list.append(f1_val)
            #         # break

            loss_val, acc_val, f1_val, loss_train, acc_train, f1_train = cross_validation(path, test_year)
                    
            
            mean_train_loss = np.mean(loss_train)
            mean_train_acc = np.mean(acc_train)
            mean_train_f1 = np.mean(f1_train)
            
            loss_train_list.append(mean_train_loss)
            acc_train_list.append(mean_train_acc)
            f1_train_list.append(mean_train_f1)
            
            
            mean_val_loss = np.mean(loss_val)
            mean_val_acc = np.mean(acc_val)
            mean_val_f1 = np.mean(f1_val)
            
            loss_val_list.append(mean_val_loss)
            acc_val_list.append(mean_val_acc)
            f1_val_list.append(mean_val_f1)
            
            print(f"for epoch {epoch}:mean train loss = {mean_train_loss}, mean train acc = {mean_train_acc}, mean train f1 = {mean_train_f1}")
            print(f"mean val loss = {mean_val_loss}, mean val acc = {mean_val_acc}, mean val f1 = {mean_val_f1}")
            
            # validation
            
            loss_test, acc_test, f1_test = test(path, test_year)
            loss_test_list.append(loss_test)
            acc_test_list.append(acc_test)
            f1_test_list.append(f1_test)
            
            print(f"for epoch {epoch}:mean test loss= {loss_test}, mean test accuracy= {acc_test}, mean test f1 = {f1_test}")
            
            # # Convert CUDA tensors to CPU and calculate np.mean
            # loss_train_list = [lst for lst in loss_train_list]
            # acc_train_list = [lst for lst in acc_train_list]
            # f1_train_list = [lst for lst in f1_train_list]

            # loss_test_list = [loss_test for loss_test in loss_test_list]
            # acc_test_list = [acc_test for acc_test in acc_test_list]
            # f1_test_list = [f1_test for f1_test in f1_test_list]
            
            # plot the curve:
            # total_epochs = range(1, len(loss_train_list) + 1)
            # update_plots(total_epochs, loss_train_list, acc_train_list, loss_test_list, acc_test_list)
            # break

        # Evaluation process
        total_epochs = range(1, len(loss_train_list) + 1)
        

        
        plt.figure(figsize=(18, 6))
        
        # Plotting training loss
        plt.subplot(1, 3, 1)
        
        plt.plot(total_epochs, loss_train_list, 'b', label='Training loss')
        plt.plot(total_epochs, loss_test_list, 'r', label='Test loss')
        plt.plot(total_epochs, loss_val_list, 'g', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation/Testing Loss')
        plt.legend()
        
        # Plotting training accuracy
        plt.subplot(1, 3, 2)
        plt.plot(total_epochs, acc_train_list, 'b', label='Training acc')
        plt.plot(total_epochs, acc_test_list, 'r', label='Test acc')
        plt.plot(total_epochs, acc_val_list, 'g', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training/Validation/Testing Accuracy')
        plt.legend()
        
        #  Plotting training accuracy
        plt.subplot(1, 3, 3)
        plt.plot(total_epochs, f1_train_list, 'b', label='Training F1')
        plt.plot(total_epochs, f1_test_list, 'r', label='Test F1')
        plt.plot(total_epochs, f1_val_list, 'g', label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Training/Validation/Testing F1 score')
        plt.legend()
        
        
        # Save the figure
        plt.savefig(f'{area_of_interest}_{test_year}_{model_name}_Loss_acc_f1_plot.png')
        plt.close()
            
            

        end = time.time()
        save_model(model, optimizer, "trained_model_one_feature.pth")


        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        test(path, test_year, last = True)

        # Can return validation loss to minimize or validation accuracy to maximize
        
        return acc_test_list[-1]
    


    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)  # Adjust the number of trials as needed

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


df = pd.DataFrame(output_form)

# Save DataFrame to Excel file
df.to_excel(f'accuracy_results_{area_of_interest}_{model_name}.xlsx', index=False, engine='openpyxl')  # or engine='xlsxwriter'