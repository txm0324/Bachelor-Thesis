from gnn_class import DrugNetworkDataset # for import gnn_class of the Script gnn_class.py
from sklearn.model_selection import KFold # for the Cross Validation
import pandas as pd # read Dataframe 
import gseapy as gp # for retrieving pathway information

# Create with the GNN_Class a train and test sample with 3-Cross Validation


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset
import random
import numpy as np

#call pytorch lightning functions
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning import Trainer, seed_everything

from torch_geometric.data import DataLoader as GeoDataLoader

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling, global_max_pool as gmp, global_mean_pool as gap

from torch_geometric.data import Batch

# -----------------------------------
# Step 0: Load and Prepare DataFrames
# -----------------------------------

print("\n" + "="*31)
print("Step 0: Download and Preprocess")
print("="*31)

# Load the full GDSC dataset (FPKM + AUC values for all drugs)
gdsc_dataset = pd.read_csv('./data/TUGDA/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)

# Extract gene and drug columns:
# - First 1780 columns correspond to gene expression data
# - Remaining columns represent drug AUC values
gene_list = gdsc_dataset.columns[0:1780]
drug_list = gdsc_dataset.columns[1780:] 

# Retrieve KEGG pathways using gseapy
kegg_gmt = gp.parser.get_library('KEGG_2021_Human', organism='Human', min_size=3, max_size=2000)
pathway_list = list(kegg_gmt.keys())

# Gene Expression Data (FPKM values)
expression_data = gdsc_dataset.iloc[:, :1780]

# Response Data: Combine log_IC50 values from 3-fold cross-validation test sets
response_1 = pd.read_csv("./data/TUGDA/cl_y_test_o_k1.csv", index_col=0)
response_2 = pd.read_csv("./data/TUGDA/cl_y_test_o_k2.csv", index_col=0)
response_3 = pd.read_csv("./data/TUGDA/cl_y_test_o_k3.csv", index_col=0)
response_data = pd.concat([response_1, response_2, response_3], axis=0, ignore_index=False)

# Sort both datasets by index to ensure alignment
expression_data = expression_data.sort_index()
response_data = response_data.sort_index()

# Remove duplicate indices (keep first occurrence) to avoid conflicts during merging
expression_data = expression_data[~expression_data.index.duplicated(keep='first')]
labels_df = response_data[~response_data.index.duplicated(keep='first')] 

print(labels_df)

# Convert labels_df in long_format and keep original index
labels_long = labels_df.stack(dropna=False).reset_index()
labels_long.columns = ['cell_line', 'drug', 'response']
labels_long = labels_long.set_index('cell_line')

# --------------------------
# Step 1: 3-Cross-Validation
# --------------------------

print("\n" + "="*26)
print("Step 1: 3-Cross-Validation")
print("="*26)

print("Generating training and test data...")

kf = KFold(n_splits=3, shuffle=True, random_state=42)
all_cell_lines = labels_df.index.tolist()
drug_list = labels_df.columns.tolist()

train_datasets = []
test_datasets = []

# KFold-Loop
for fold, (train_idx, test_idx) in enumerate(kf.split(all_cell_lines), 1):
    print(f"\n--- Fold {fold} ---")
    
    # Get cell-lines for this fold
    train_cell_lines = [all_cell_lines[i] for i in train_idx]
    test_cell_lines = [all_cell_lines[i] for i in test_idx]

    # Subset Labels for this fold
    train_labels_long = labels_long[labels_long.index.isin(train_cell_lines)]
    test_labels_long = labels_long[labels_long.index.isin(test_cell_lines)]

    print("Start building Train Dataset")
    # Run class for train data
    train_dataset = DrugNetworkDataset(
        root="./results/Network/",
        drug_list=drug_list,
        gene_list=gene_list,
        pathway_list=pathway_list,
        labels_df=train_labels_long,
        expression_data=expression_data
    ) # Output for each fold: 536 cell_lines --> [536, 200]

    print("Start building Test Dataset")
    # Run class for test data
    test_dataset = DrugNetworkDataset(
        root="./results/Network/",
        drug_list=drug_list,
        gene_list=gene_list,
        pathway_list=pathway_list,
        labels_df=test_labels_long,
        expression_data=expression_data
    ) # Output for each fold: 268 cell_lines --> [268, 200]

    # Save as lists
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)

class GNNEncoder(nn.Module):
    def __init__(self, feature_size=3, embedding_size=512, output_dim=256):
        super(GNNEncoder, self).__init__()

        # Block 1
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform1 = nn.Linear(embedding_size * 3, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)

        # Block 2
        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform2 = nn.Linear(embedding_size * 3, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)

        # Final projection
        self.linear1 = nn.Linear(embedding_size * 2, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Block 1
        x = self.conv1(x, edge_index)
        x = F.relu(self.head_transform1(x))
        x = self.bn1(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Block 2
        x = self.conv2(x, edge_index)
        x = F.relu(self.head_transform2(x))
        x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Combine
        x = x1 + x2

        # Final layers
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        graph_embedding = self.linear2(x)

        return graph_embedding

# ----------------------------------
# Step 2: Quick Ckeck with results
# ----------------------------------

if __name__ == "__main__":
   
    data = train_dataset[0]  # a graph-base representation of one-cell-line (drug_graphs, ic50_tensor)

    drug_graphs, ic50_tensor = data

    # Example for first drug of this cell-line
    data_drug = drug_graphs[0]

    print("- Cell Line Name:", data_drug.cell_line)

    print("\n--- First Drug-Graph ---")
    print("- Drug Name:", data_drug.drug)

    print("\n- Edge Index (COO format):") # tensor([a,b], [c,d]): node a is conntected to node b and node c is conntected to node d
    print(data_drug.edge_index.t())
    print("Shape:", data_drug.edge_index.t().shape) # Tensor of shape [num_edges, 2]

    print("\n- Node Features (x):") # Gene expression values of gene_x with Cell line and is_gene, is_pathway
    print(data_drug.x)
    print("Shape:", data_drug.x.shape) # Tensor of shape [num_nodes, num_node_features]

    nodes = data_drug.nodes
    print("\n- Beispiele für Nodes:")
    for i in range(min(10, len(nodes))):
        node_name = nodes[i]
        feature_value = data_drug.x[i][0].item() # Just take the first feature (gene expression)
        print(f"  - Index {i}: {node_name} → Feature: {feature_value:.4f}")

    print("\n- Label (IC50-Werte für alle Drugs):") # log_IC50 value for each drug-cell line combination
    print(ic50_tensor)  # shape: [num_tasks]
    print("Shape:", ic50_tensor.shape)


    ''' Single
    data = train_dataset[0] # graph-base representations of drug-cell line pairs (200 * 1780)

    print("- Drug Name:", data.drug)
    print("- Cell Line Name:", data.cell_line)

    print("\n- Edge Index (COO format):") # tensor([a,b], [c,d]): node a is conntected to node b and node c is conntected to node d
    print(data.edge_index.t())
    print(data.edge_index.t().shape) # Tensor of shape [num_edges, 2]

    print("\n- Node Features (x):") # Gene expression values of gene_x with Cell line and is_gene, is_pathway
    print(data.x)
    print(data.x.shape) # Tensor of shape [num_nodes, num_node_features]

    nodes = data.nodes

    for i in range(10):
        node_name = nodes[i]
        feature_value = data.x[i][0].item()  # Just take the first feature (gene expression)
        print(f"- Index {i}: {node_name} → Feature: {feature_value:.4f}")

    print("\n- Label (y):") # log_IC50 value for the drug-cell line combination
    print(data.y)
    print(data.y.shape) # Tensor of shape [1]
    ''' 