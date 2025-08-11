from gnn_class import DrugNetworkDataset # for import gnn_class of the Script gnn_class.py
from sklearn.model_selection import KFold # for the Cross Validation
import pandas as pd # read Dataframe 

# Create with the GNN_Class a train and test sample with 3-Cross Validation


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