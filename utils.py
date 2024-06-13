import numpy as np
import scipy.sparse as sp
import torch
import os
import networkx as nx
from sklearn.preprocessing import normalize
import pickle
import sys
import cv2
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt

import sklearn.metrics as sk_metrics

input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/20170107.jpg"

image = cv2.imread(input_file)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def Cora_load_data(path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/src/pygcn/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print("Cora")
    print(f"adj: {adj.size()}")
    print(f"features: {features.size()}")
    print(f"labels: {labels.size()}")
    # print(f"idx_train: {idx_train}")
    # print(f"idx_val: {idx_val}")
    # print(f"idx_test: {idx_test}")

    return adj, features, labels, idx_train, idx_val, idx_test


def get_feature(graph):
    # Initialize lists to store features
    feature_list = []
    node_count = 0
    # Iterate through nodes and extract features
    for node in graph.nodes():
        node_count += 1
        attributes = graph.nodes[node]
        # Extract feature values and append them to the list
        features = [
            attributes['pos'][0],  # X-coordinate
            attributes['pos'][1],  # Y-coordinate
            attributes['intensity'],
            attributes['contrast'],
            attributes['correlation'],
            attributes['energy'],
            attributes['homogeneity'],
            attributes['entropy'],
            attributes['dissimilarity'],
            attributes['sum_of_squares_variance']
        ]
        feature_list.append(features)
    return feature_list, node_count
    
def concat_load_data(path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/4Graphs/", dataset="beauhornoise"):
    train_graph, test_graph, val_graph = nx.Graph(), nx.Graph(), nx.Graph()  # Create an empty graph to store the merged graphs

    test_year = "2020"
    # Iterate over the graph files and merge them into the merged_graph
    for file_name in os.listdir(path):
        if file_name.startswith(test_year):
            file_path = os.path.join(path, file_name)
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
            test_graph = nx.compose(test_graph, graph)  # Merge the current graph into the merged_graph
        else:
            file_path = os.path.join(path, file_name)
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
            train_graph = nx.compose(train_graph, graph)  # Merge the current graph into the merged_graph
        
    
    
    # get features for both graphs
    train_feature_list, train_node_count = get_feature(train_graph)
    test_feature_list, test_node_count = get_feature(test_graph)
    # print(f"train node: {train_node_count}")
    # print(f"test node: {test_node_count}")
    
    # Convert NetworkX graph to adjacency matrix
    train_adj = nx.adjacency_matrix(train_graph)
    test_adj = nx.adjacency_matrix(test_graph)

    # Normalize features and adjacency matrix
    train_feature = np.array(train_feature_list)  
    train_feature = normalize(train_feature)
    train_adj = normalize(train_adj + sp.eye(train_adj.shape[0]))
    
    test_feature = np.array(test_feature_list)  
    test_feature = normalize(test_feature)
    test_adj = normalize(test_adj + sp.eye(test_adj.shape[0]))

    

    # Convert to PyTorch tensors
    train_feature = torch.FloatTensor(train_feature)
    train_adj = sp.coo_matrix(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj)
    
    test_feature = torch.FloatTensor(test_feature)
    test_adj = sp.coo_matrix(test_adj)
    test_adj = sparse_mx_to_torch_sparse_tensor(test_adj)
    
    
    # Extract labels from the graph
    train_node_labels = [train_graph.nodes[node]['label'] for node in train_graph.nodes]

    # # Fit and transform the labels using one-hot encoding
    train_labels = encode_onehot(train_node_labels)
    train_labels = torch.LongTensor(np.where(train_labels)[1])
    
    # Extract labels from the graph
    test_node_labels = [test_graph.nodes[node]['label'] for node in test_graph.nodes]

    # # Fit and transform the labels using one-hot encoding
    test_labels = encode_onehot(test_node_labels)
    test_labels = torch.LongTensor(np.where(test_labels)[1])
    
    
    # Concatenate the dense tensors along the first dimension
    adj = torch.cat((train_adj, test_adj), dim=0)
    
    features = torch.cat((train_feature, test_feature), dim=0)
    
    labels = torch.cat((train_labels, test_labels), dim=0)
    
    idx_train = range(int(0.8 * train_node_count))
    idx_val = range(int(0.8 * train_node_count), train_node_count)
    idx_test = range(train_node_count, train_node_count + test_node_count)
    
    

    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    
    # print("Beauhornoise")
    # print(f"adj: {adj.size()}, train adj: {train_adj}, test adj: {test_adj}")
    # print(f"features: {features.size()}, train features: {train_feature}, test features:{test_feature}")
    # print(f"labels: {labels.size()}, train label: {train_labels}, test label: {test_labels}")
    # print(f"idx_train: {idx_train}")
    # print(f"idx_val: {idx_val}")
    # print(f"idx_test: {idx_test}")
    
    return adj, features, labels, idx_train, idx_val, idx_test

def save_graph(graph_data, folder_path, filename):
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(graph_data, f)
        
def add_k_nearest_edges(graph, node, k, adj):
    distances = {}
    for n in graph.nodes():
        if n != node:
            distances[n] = euclidean(graph.nodes[node]['pos'], graph.nodes[n]['pos'])
    sorted_nodes = sorted(distances.items(), key=lambda x: x[1])
    k_nearest = [n for n, _ in sorted_nodes[:k]]
    for neighbor in k_nearest:
        graph.add_edge(node, neighbor)
        neighbor_idx = list(graph.nodes()).index(neighbor)
        new_idx = list(graph.nodes()).index(node)
        adj[new_idx, neighbor_idx] = 1
        adj[neighbor_idx, new_idx] = 1
    return graph, adj

def resample_data(path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/4Graphs/20190109", k = 3, resample = True, draw = False, dataset="beauhornoise"):
    """Load citation network dataset"""
    print('Loading {} dataset...'.format(dataset))
    
    with open(path, 'rb') as f:
        G = pickle.load(f)
    
    # Initialize lists to store features
    feature_list = []
    node_count = 0
    # Iterate through nodes and extract features
    for node in G.nodes():
        node_count += 1
        attributes = G.nodes[node]
        # Extract feature values and append them to the list
        features = [
            attributes['pos'][0],  # X-coordinate
            attributes['pos'][1],  # Y-coordinate
            attributes['intensity'],
            attributes['contrast'],
            attributes['correlation'],
            attributes['energy'],
            attributes['homogeneity'],
            attributes['entropy'],
            attributes['dissimilarity'],
            attributes['sum_of_squares_variance']
        ]
        feature_list.append(features)
    
    # Convert NetworkX graph to adjacency matrix
    adj = nx.adjacency_matrix(G)

    # Normalize features and adjacency matrix
    features = np.array(feature_list)  
    norm_features = normalize(features)
    norm_adj = normalize(adj + sp.eye(adj.shape[0]))

    
    

    # Convert to PyTorch tensors
    norm_features = torch.FloatTensor(norm_features)
    norm_adj = sp.coo_matrix(norm_adj)
    norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
    
    
    features_resampled = norm_features
    adj_resampled = norm_adj

    # Extract labels from the graph
    node_labels = [G.nodes[node]['label'] for node in G.nodes]
    # print(f"before one hot {node_labels}")
    # # Fit and transform the labels using one-hot encoding
    labels = encode_onehot(node_labels)
    # print(f"after one hot:{labels}")
    
    labels = torch.LongTensor(np.where(labels)[1])
    
    # print(f"labels {labels}")
    labels_resampled = labels
    
   
    
    if resample:
        print("start smote")
        
        # Use SMOTE for oversampling
        smote = SMOTE(k_neighbors=4, random_state=42)
        features_resampled, labels_resampled = smote.fit_resample(features, labels)
        
        
        # Create an empty graph
        G_resampled = nx.Graph()
        
        
        largest_node_id = max(G.nodes) + 1
        
        # Iterate over nodes and add features as node attributes
        for node, fts in enumerate(features_resampled):
            attributes = {
                'contrast': fts[3],
                'correlation': fts[4],
                'energy': fts[5],
                'homogeneity': fts[6],
                'entropy': fts[7],
                'dissimilarity': fts[8],
                'sum_of_squares_variance': fts[9]
            }
            # Add node to the graph with the attributes dictionary
            G_resampled.add_node(node, pos=(fts[0], fts[1]), intensity = fts[2], **attributes)
        
        original_positions = nx.get_node_attributes(G, 'pos')
        # print(f"original pos = {original_positions}")
        
        
        
        
        
        

        # add edge to the new nodes with nearest original nodes
        pos_resampled = nx.get_node_attributes(G_resampled, 'pos')  # Assuming the first two columns contain "pos" features
        # print(f"resampled position = {pos_resampled}")
        
        
        # Extract the keys corresponding to the newly added nodes
        new_node_keys = list(pos_resampled.keys())[len(original_positions):]

        # Now new_node_keys contains the keys of the newly added nodes

        # Use these keys to get the positions of the newly added nodes
        new_pos = {k: pos_resampled[k] for k in new_node_keys}
        
        distances = cdist(np.array(list(new_pos.values())), np.array(list(original_positions.values()))) # original_positions contain positions of original nodes
        # print(f"distance = {distances}")
        
        nearest_indices = np.argmin(distances, axis=1)
        # print(f"nearest indices = {nearest_indices}, size: {nearest_indices.shape}")

        # Combine the nearest indices with node IDs
        nearest_node_ids = [list(original_positions.keys())[i] for i in nearest_indices]
        # print(f"Nearest node IDs with indices: {nearest_node_ids}, size: {len(nearest_node_ids)}")
        
        # Step 3: Update the graph structure
        # Create a new adjacency matrix for the oversampled graph
        num_nodes = len(features_resampled)
        old_nnodes = len(features)
        added_nodes = len(nearest_indices)
        # print(f"number of resampled node = {num_nodes}")
        # print(f"number of original node = {old_nnodes}")
        # print(f"number of added node = {added_nodes}")
        
        adj_resampled = np.zeros((old_nnodes + added_nodes, old_nnodes + added_nodes))
        # print(f"initial adj resampled size: {adj_resampled.shape}")
        G_combined = nx.Graph()
        
        G_combined.add_nodes_from(G.nodes(data=True))
        G_combined.add_edges_from(G.edges())

        
        # Add resampled nodes and edges to the combined graph
        for nearest_idx in range(added_nodes):
            i = nearest_idx + old_nnodes
            
            attributes = {
                'contrast': features_resampled[i][3],
                'correlation': features_resampled[i][4],
                'energy': features_resampled[i][5],
                'homogeneity': features_resampled[i][6],
                'entropy': features_resampled[i][7],
                'dissimilarity': features_resampled[i][8],
                'sum_of_squares_variance': features_resampled[i][9]
            }
            new_node_ID = nearest_idx + largest_node_id + 1
            # print(f"node {new_node_ID} has position ({features_resampled[i][0]}, {features_resampled[i][1]})")
            # Add node to the graph with the attributes dictionary
            # print(f"label resampled {labels_resampled}")
            if labels_resampled[i] == 1:
                label = "ice"
            else:
                label = "water"
            G_combined.add_node(new_node_ID, pos=(features_resampled[i][0], features_resampled[i][1]), label = label, intensity = features_resampled[i][2], **attributes)
            
            # G_combined.add_edge(new_node_ID, node_id)  # Add edge to nearest original node
            G_combined, adj_resampled = add_k_nearest_edges(G_combined, new_node_ID, k, adj_resampled)
            
            
            # old_node_idx = list(G.nodes()).index(node_id)
            # adj_resampled[old_node_idx, i] = 1
            # adj_resampled[i, old_node_idx] = 1
            # print(f"added edge : ({new_node_ID}, {nearest_idx})")
            
        # print(f"adj size: {adj.size()}")
        # print(f"largest + resampled = {largest_node_id + len(nearest_indices)}")
        # Iterate over the adjacency matrix to add edges
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):

                if adj[i, j] == 1:  # If there is an edge between nodes i and j
                 
                    adj_resampled[i, j] = 1
                    adj_resampled[j, i] = 1
        

        # Iterate over nodes in G, adding them to G_combined along with attributes
        for node, attr in G.nodes(data=True):
            G_combined.add_node(node, **attr)  # Add node with its attributes

    
        if draw:
            # Create a color map based on node labels
            color_map = []
            # print(f"label resampled: {labels_resampled}")
            for node in G_combined.nodes():
                idx = list(G_combined.nodes()).index(node)
                if labels_resampled[idx] == 0:
                    if idx >= old_nnodes:
                        color_map.append('red')
                        # print("new ice colored red")
                    else:
                        color_map.append('green')
                        # print("old ice colored green")
                else:  # Assuming label 'water'
                    if idx >= old_nnodes:
                        color_map.append('yellow')
                        # print("new water colored yellow")
                    else:
                        color_map.append('lightblue')
                        # print("old water colored lightblue")
            
            # Draw the graph
            pos_resampled = nx.get_node_attributes(G_combined, 'pos')
            
            # print(f"resampled position  = {pos_resampled}")
            # print(f"original position length = {len(original_positions)}")
            pos_resampled = {node: (x, image.shape[0] - y) for node, (x, y) in pos_resampled.items()}
            
            
            # Assuming G_combined is your graph and pos_resampled is your position dictionary
            missing_nodes = [node for node in G_combined.nodes() if node not in pos_resampled]
            if missing_nodes:
                print(f"Missing positions for nodes: {missing_nodes}")
                print(len(missing_nodes))
                # Handle missing positions here, such as assigning default positions or removing nodes from the graph
        
            nx.draw(G_combined, pos_resampled, with_labels=False, node_color=color_map, node_size=1, node_shape='.')
            # Save the plot as a PNG image
            plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/graphs/resampled.png')
            # print(f"resampled position = {pos_resampled}")
            plt.close()
            
            original_positions = {node: (x, image.shape[0] - y) for node, (x, y) in original_positions.items()}
            
            nx.draw(G, original_positions, with_labels=False, node_color=color_map[:old_nnodes], node_size=2)
            # Save the plot as a PNG image
            plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/graphs/original.png')
            # print(f"original position = {original_positions}")
            plt.close()

    
    # Split indices for training, validation, and testing
    n_label = len(labels_resampled)
    n_train = int(n_label * 0.8)
    n_test = n_label - n_train
    
    n_val = int(n_test/2)
    n_test = n_val
    idx_train = range(n_train)
    idx_val = range(n_train, n_train + n_val)
    idx_test = range(n_train + n_val, n_train + n_val + n_test)
    
    # idx_train = range(140)
    # idx_val = range(200, 500)
    # idx_test = range(500, 1500)
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    
    # print("Beauhornoise")
    # print(f"adj: {adj_resampled.shape}")
    # print(f"features: {features_resampled.shape}")
    # print(f"labels: {labels_resampled.shape}")
    
    num_edges = G.number_of_edges()
    # print(f"Number of edges in the original graph: {num_edges}")
    num_edges_resampled = G_combined.number_of_edges()
    # print(f"Number of edges in the resampled graph: {num_edges_resampled}")

    # print(f"features resampled: {features_resampled}")
    # print(f"original features: {features}")
    
    # Count the number of ice nodes
    ice_count_ori = sum(1 for label in labels if label == 0)
    ice_count_resample = sum(1 for label in labels_resampled if label == 0)

    # Calculate the total number of nodes
    total_nodes_resample = len(labels_resampled)
    total_nodes_ori = len(labels)

    # Calculate the percentage of ice nodes
    ice_percentage_resample = (ice_count_resample / total_nodes_resample) * 100
    ice_percentage_ori = (ice_count_ori / total_nodes_ori) * 100

    # Print the percentage of ice nodes
    # print("Percentage of resampled ice nodes: {:.2f}%".format(ice_percentage_resample))
    # print("Percentage of original ice nodes: {:.2f}%".format(ice_percentage_ori))
    # print(f"idx_train: {idx_train}")
    # print(f"idx_val: {idx_val}")
    # print(f"idx_test: {idx_test}")
    
##################    
    if resample:
        # Normalize features and adjacency matrix
        features = np.array(feature_list)  
        features_resampled = normalize(features_resampled)
        
        
        adj_resampled = normalize(adj_resampled + sp.eye(adj_resampled.shape[0]))


        # Convert to PyTorch tensors
        features_resampled = torch.FloatTensor(features_resampled)
        adj_resampled = sp.coo_matrix(adj_resampled)
        # print(f"2adj resampled size: {adj_resampled.shape}")
        adj_resampled = sparse_mx_to_torch_sparse_tensor(adj_resampled)
        


        # labels_resampled = torch.LongTensor(np.where(labels_resampled)[1])
        labels_resampled = torch.tensor(labels_resampled)
        
        filename = os.path.basename(path)
        save_path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/6ResampledGraph/"
        save_graph(G_combined, save_path, filename)
        # print(f"graph for {filename} is saved to {save_path}")
    ###############
        
    
    
    return adj_resampled, features_resampled, labels_resampled, idx_train, idx_val, idx_test

def load_data(path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/4Graphs/20190109", k = 3, mask = False, resample = False, draw = False, dataset="beauhornoise"):
    """Load citation network dataset"""
    # print('Loading {} dataset...'.format(dataset))
    print(f"path = {path}")
    
    with open(path, 'rb') as f:
        G = pickle.load(f)
    
    # Initialize lists to store features
    feature_list = []
    if mask: mask = []
    node_count = 0
    # Iterate through nodes and extract features
    for node in G.nodes():
        node_count += 1
        attributes = G.nodes[node]
        # print(attributes)
        # Extract feature values and append them to the list
        if mask: mask.append(attributes['mask'])
        features = [
            # attributes['pos'][0],  # X-coordinate
            # attributes['pos'][1],  # Y-coordinate
            attributes['intensity'],
            attributes['std']
            # attributes['intensity'], 
            # attributes['b4'],
            # attributes['b5']
            # attributes['contrast'],
            # attributes['correlation'],
            # attributes['energy'],
            # attributes['homogeneity'],
            # attributes['entropy'],
            # attributes['dissimilarity'],
            # attributes['sum_of_squares_variance']
        ]
        feature_list.append(features)
    
    # Convert NetworkX graph to adjacency matrix
    adj = nx.adjacency_matrix(G)

    # Normalize features and adjacency matrix
    features = np.array(feature_list)  
    norm_features = features
    # norm_features = normalize(features)
    norm_adj = normalize(adj + sp.eye(adj.shape[0]))

    
    

    # Convert to PyTorch tensors
    norm_features = torch.FloatTensor(norm_features)
    norm_adj = sp.coo_matrix(norm_adj)
    norm_adj = sparse_mx_to_torch_sparse_tensor(norm_adj)
    
    labels = None
    
    if 'label' in G.nodes[node]:
        # Extract labels from the graph
        node_labels = [G.nodes[node]['label'] for node in G.nodes]
        # print(f"node labels: {node_labels}")
        

        # # Fit and transform the labels using one-hot encoding
        labels = encode_onehot(node_labels)
        # print(f"after onehot labels:{labels}")
        
        labels = torch.LongTensor(np.where(labels)[1])
        # print(f"labels{labels}") 
        
        
        # Count the number of ice nodes
        ice_count_ori = sum(1 for label in labels if label == 0)

        # Calculate the total number of nodes
        total_nodes_ori = len(labels)

        # Calculate the percentage of ice nodes
        ice_percentage_ori = (ice_count_ori / total_nodes_ori) * 100

        # Print the percentage of ice nodes
        # print("Percentage of ice nodes: {:.2f}%".format(ice_percentage_ori))
        
        # Split indices for training, validation, and testing
        n_label = len(labels)
        n_train = int(n_label * 0.8)
        n_test = n_label - n_train
        
        n_val = int(n_test/2)
        n_test = n_val
        idx_train = range(n_train)
        idx_val = range(n_train, n_train + n_val)
        idx_test = range(n_train + n_val, n_train + n_val + n_test)
        
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    
    # print("Beauhornoise")
    # print(f"adj: {norm_adj.shape}")
    # print(f"features: {norm_features.shape}")
    # print(f"labels: {labels.shape}")
    if mask:return norm_adj, norm_features, labels, mask
    
    
    return norm_adj, norm_features, labels


def save_model(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    # raise Exception("division by zero in normalization")
        
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# def accuracy(output, labels):
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)

def incorrect_node(output, labels):
    if len(output.shape) == 1:
        preds = torch.round(torch.sigmoid(output))  # For binary classification
    else:
        preds = output.argmax(dim=-1)
    correct = (preds == labels).sum().item()  # Count the number of correct predictions
    total = labels.size(0)  # Total number of samples
    

    # Compare predicted labels with true labels and count incorrect predictions
    incorrect = (preds != labels).item()
    
    print(f"incorrect = {incorrect}")

    return incorrect
    
def accuracy(output, labels):
    if len(output.shape) == 1:
        preds = torch.round(torch.sigmoid(output))  # For binary classification
    else:
        preds = output.argmax(dim=-1)
    correct = (preds == labels).sum().item()  # Count the number of correct predictions
    total = labels.size(0)  # Total number of samples
    accuracy = correct / total
    return accuracy

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# def calculate_weights(labels):
#     # Count the number of samples for each class
#     class_counts = torch.bincount(labels)
#     total_samples = torch.sum(class_counts)

#     # Calculate inverse class frequencies
#     class_weights = 1.0 / class_counts.float()

#     # Normalize weights (optional, depending on your application)
#     class_weights /= torch.sum(class_weights)  # Normalize so that sum equals 1

#     # Map class weights back to the original labels
#     # weights_for_labels = class_weights[labels]

#     print(f"Class weights: {class_weights}")
#     # print(f"Weights for labels: {weights_for_labels}")

#     return class_weights

def calculate_weights(labels):
    # Calculate class frequencies
    class_counts = torch.bincount(labels)

    if len(class_counts) == 1:
        # If there is only one type, assign equal weights to both types
        return torch.tensor([0.5, 0.5]).cuda()
    # Inverse class frequencies
    class_weights = 1.0 / class_counts.float()

    # Normalize weights
    class_weights /= class_weights.sum()

    return class_weights


def f1_score(output, target):
    if len(output.shape) == 1:
        pred = torch.round(torch.sigmoid(output))  # For binary classification
    else:
        pred = output.argmax(dim=1, keepdim=True)
    # print(f"predict = {pred}")
    f1 = sk_metrics.f1_score(target.cpu(), pred.detach().cpu().numpy())
    return f1

# def accuracy_to_form(file_name, adj, features, output, labels, area_of_interest, model_name):
    
    
    
def visualize(date, adj, features, output, labels = None, area_of_interest = "lake", model_name = "lwGCN"):
    # Move tensors from GPU to CPU
    adj = adj.cpu()
    features = features.cpu()
    
    if labels is not None: labels = labels.cpu()
    
    output = output.cpu()
    
    # adj = adj.numpy()
    # features = features.numpy()
    # labels = labels.numpy()
    if area_of_interest == "lake":
        path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/6Graph_label_intensity/" + date
    else:
        path = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/7relabel_graphs/" + date
    with open(path, 'rb') as f:
        G_ori = pickle.load(f)
    
    node_list = list(G_ori.nodes())
    # Create a graph
    G = nx.Graph()
    
    ori_pos = nx.get_node_attributes(G_ori, 'pos')
    # print(f"original position {ori_pos}")
    
    # Add nodes to the graph and set their positions based on longitude and latitude
    for i in range(len(features)):
        node_id = node_list[i]
        G.add_node(node_id)

    # Add edges to the graph based on the adjacency matrix
    for i in range(len(adj)):
        for j in range(i + 1, len(adj[i])):
            if adj[i, j] == 1:
                G.add_edge(i, j)
                
    # print(labels)
    # Color nodes based on labels
    if len(output.shape) == 1:
        prob = torch.round(torch.sigmoid(output))  # For binary classification
    else:
        prob = output.argmax(dim=-1)
        
    # prob =  labels.max(1)[1].type_as(labels)
    # print(prob)
    # print(prob.shape)
    bi_labels = torch.where(prob > 0.5, torch.tensor(1), torch.tensor(0))
    # print(f"bi_labels = {bi_labels}")
    node_colors = ['yellow' if label == 1 else 'blue' for label in bi_labels]
    correct = 0
    total = len(bi_labels)
    if labels is not None: 
        for i, node in enumerate(node_colors):
            if labels[i] != bi_labels[i]:
                if labels[i] == 1:
                    node_colors[i] = 'green'
                else:
                    node_colors[i] = 'red'
            else:
                correct += 1 

    # Get node positions
    # pos = nx.get_node_attributes(G, 'pos')
    if area_of_interest == "lake":
        input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/lake_stlawrence/0jpg_sar_image/20170112_Band1.jpg"
    else:
        input_file = "/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/sar_jpg/20170107.jpg"

    image = cv2.imread(input_file)
    # Flip y-coordinates to correct orientation
    pos = {node: (x, image.shape[0] - y) for node, (x, y) in ori_pos.items()}
    
    plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100))
    # Plot the graph
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=2)
    # Save the plot as a PNG image
    if area_of_interest == "lake":
        save_path = '/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/outPygcn/lake_relabel/new_' + date + '.png'
    else:
        save_path = f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/outPygcn/{model_name}_{date}.png'
    plt.savefig(save_path)
    print(f"output of model {model_name} in {area_of_interest} region of image {date} saved to {save_path}---------------------------------------------------")
    plt.close()  # Close the plot to prevent it from being displayed
    return (correct / total) * 100
            
            
def update_plots(epoch, train_losses, train_acc, test_losses, test_acc):
    plt.figure(figsize=(12, 6))
    
    # Plotting training loss
    plt.subplot(1, 2, 1)
    
    plt.plot(epoch, train_losses, 'b', label='Training loss')
    plt.plot(epoch, test_losses, 'r', label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Testing Loss')
    plt.legend()
    
    # Plotting training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch, train_acc, 'b', label='Training acc')
    plt.plot(epoch, test_acc, 'r', label='Test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training/Testing Accuracy')
    plt.legend()
    
    epoch_len = len(epoch)
    
    # Save the figure
    plt.savefig(f'/home/maruko/projects/def-ka3scott/maruko/seaiceClass/output/graphs/pygcn/training_metrics_epoch_{epoch_len+1}.png')
    plt.close()
      

# Save the current stdout
original_stdout = sys.stdout

# Specify the file path where you want to save the printed output
output_file_path = "20170217GraphInfo.txt"

# Open the file in write mode
with open(output_file_path, "w") as f:
    # Redirect stdout to the file
    sys.stdout = f
    
    Cadj, Cfeatures, Clabels, Cidx_train, Cidx_val, Cidx_test = Cora_load_data()
    path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/4Graphs/"
    test_year = "2020"
    for file_name in os.listdir(path):
        if file_name.startswith(test_year):
            print(f"date: {file_name}")
            graph_path = path + file_name
            adj, features, labels, idx_train, idx_val, idx_test = resample_data(path=graph_path, draw=True)
            
    # path="/home/maruko/projects/def-ka3scott/maruko/seaiceClass/data/6ResampledGraph/"
    # test_year = "2020"
    # for file_name in os.listdir(path):
    #     if not file_name.startswith(test_year):
    #         print(f"date: {file_name}")
    #         graph_path = path + file_name
    #         adj, features, labels, idx_train, idx_val, idx_test = load_data(path=graph_path, draw=True)
            
    
    # adj, features, labels, idx_train, idx_val, idx_test = load_data(draw=True)
    
    def check_same_type(obj1, obj2):
        return type(obj1) == type(obj2)

    print(check_same_type(Cadj, adj))  
    print(check_same_type(Cfeatures, features))  
    print(check_same_type(Clabels, labels))  
    print(check_same_type(Cidx_train, idx_train))  
    print(check_same_type(Cidx_val, idx_val))  
    print(check_same_type(Cidx_test, idx_test))  



    
    # Now all printed output will be written to the file
    print("Printed output will be saved to printed_output.txt")

    # Restore the original stdout
    sys.stdout = original_stdout

