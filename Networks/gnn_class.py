import pandas as pd # read Dataframe 
from torch_geometric.data import Dataset, Data # for creating graph-based datasets
import os # for file and directory operations 
import numpy as np # for numerical computations with arrays 
import gseapy as gp # for retrieving pathway information


# Creating a Custom Dataset in Pytorch Geometric 

# -----------------------------------
# Step 0: Load and Prepare DataFrames
# -----------------------------------

print("\n" + "="*31)
print("Step 0: Download and Preprocess")
print("="*31)

# Load the full GDSC dataset (FPKM + AUC values for all drugs)
gdsc_dataset = pd.read_csv('/sybig/home/tmu/TUGDA/data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)

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
response_1 = pd.read_csv("/sybig/home/tmu/TUGDA/data/cl_y_test_o_k1.csv", index_col=0)
response_2 = pd.read_csv("/sybig/home/tmu/TUGDA/data/cl_y_test_o_k2.csv", index_col=0)
response_3 = pd.read_csv("/sybig/home/tmu/TUGDA/data/cl_y_test_o_k3.csv", index_col=0)
response_data = pd.concat([response_1, response_2, response_3], axis=0, ignore_index=False)

# Sort both datasets by index to ensure alignment
expression_data = expression_data.sort_index()
response_data = response_data.sort_index()

# Remove duplicate indices (keep first occurrence) to avoid conflicts during merging
expression_data = expression_data[~expression_data.index.duplicated(keep='first')]
labels_df = response_data[~response_data.index.duplicated(keep='first')] 

print("\n" + "Done!")

###################
### GNN Dataset ###
###################

print("\n" + "="*33)
print("Step 1: GNN Dataset & Quick Check")
print("="*33)

class DrugNetworkDataset(Dataset):
    def __init__(self, root, drug_list, gene_list, pathway_list, labels_df, expression_data, transform=None, pre_transform=None):

        """
        A custom PyTorch Geometric Dataset for drug-cell line interaction graphs.

        Parameters:
            - root = Where the datase4t should be stored. This folder is split 
            - into raw_dir (downloaded datset) and processed_dir (processed data)

            - drug_list: List of drugs (tasks) (200)
            - gene_list: Lisz of genes sample (1780)
            - pathway_list: List of pathways from KEGG 
            - labels_df: response data (log_IC50)
            - expression_data: Gene expression values (preprocessed according to Mourragui et al. (2020): library-size using TMM, log-transformed, gene-level-mean-centering and standardization)
        """

        # Define all files that the dataset needs
        self.drug_list = drug_list
        self.gene_list = gene_list
        self.pathway_list = pathway_list
        self.labels_df = labels_df
        self.expression_data = expression_data

        # Get all combination of Drug + Cell Line
        self.samples = [
            (drug, cell_line) 
            for drug in self.drug_list 
            for cell_line in self.labels_df.index
        ]

        # Define custom raw_dir
        self.custom_raw_dir = os.path.join(root, 'drug_matrices_csv')

        super(DrugNetworkDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If these files exist in raw_dir, the download is not triggered """
        return [f"{drug}_matrix.csv" for drug in self.drug_list]

    @property
    def raw_dir(self):
        return self.custom_raw_dir

    @property
    def processed_file_names(self):
        """ If these files exist in processed_dir, processing is skipped """
        return ['placeholder.pt'] # not implemented 

    def download(self):
        pass 

    def _load_adjacency_matrix(self, drug):
        """ Loads the adjacency matrix for every drug """
        csv_path = os.path.join(self.raw_dir, f"{drug}_matrix.csv")
        df = pd.read_csv(csv_path, index_col=0)
        return df

    def _get_node_features(self, nodes, cell_line):
        x = []
        for node in nodes:
             # If the node is a gene, get its expression value for the given cell line
            if node in self.gene_list:
                expr_value = self.expression_data.loc[cell_line, node]
                x.append([float(expr_value)]) # Append as a single-element list 
            elif node in self.pathway_list:
                # If the node is a pathway, use a default feature value of 0.0
                x.append([0.0])
            else:
                x.append([0.0])
        return torch.tensor(x, dtype=torch.float)
    
    # Qucik check fuction for debugging
    def get_node_name_by_index(self, idx, node_index):
        """ Returns the name of the node at the specified index for the given sample """
        drug, cell_line = self.samples[idx]
        
        adj_df = self._load_adjacency_matrix(drug)
        nodes = adj_df.columns.tolist()
        
        if node_index < len(nodes):
            return nodes[node_index]
        else:
            raise IndexError(f"Node index {node_index} out of range for this graph.")

    def len(self):
        """ 
        Returns the total number of samples (drug-cell line combinations)
        useful for classes such as datasets for machine learning so that they are compatible with len() and for loops 
        """
        return len(self.samples)

    def get(self, idx):
        """
        In standard PyG datasets, get() loads preprocessed data saved via process(); 
        here the preprocessing step is skipped and construct each drug–cell line graph directly in get(), 
        as preprocessing wouldn't reduce computation time
        """
        drug, cell_line = self.samples[idx]

        # Load adjacency matric for each drug 
        adj_df = self._load_adjacency_matrix(drug)
        nodes = adj_df.columns.tolist()
        adj_matrix = adj_df.values
        # Build edge_index in COO format
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
        # Get node_features
        x = self._get_node_features(nodes, cell_line)
        # Get label info (log_IC50 value for each drug-cell line combination)
        y = torch.tensor([self.labels_df.loc[cell_line, drug]], dtype=torch.float)

        # Create data object
        data = Data(x=x, edge_index=edge_index, y=y, drug=drug, cell_line=cell_line)

        # Attach node names for debugging purposes
        data.nodes = nodes

        return data

# Run class
dataset = DrugNetworkDataset(
    root="./results/Network/",
    drug_list=drug_list,
    gene_list=gene_list,
    pathway_list=pathway_list,
    labels_df=labels_df, 
    expression_data=expression_data
)

# Qucik Ckeck with results
data = dataset[0] # graph-base representations of drug-cell line pairs (200 * 1780)

print("Drug Name:", data.drug)
print("Cell Line Name:", data.cell_line)

print("Edge Index (COO format):") # tensor([a,b], [c,d]): node a is conntected to node b and node c is conntected to node d
print(data.edge_index.t())
print(data.edge_index.t().shape) # Tensor of shape [num_edges, 2]

print("\nNode Features (x):") # Gene expression values of gene_x with Cell line
print(data.x)
print(data.x.shape) # Tensor of shape [num_nodes, num_node_features]

nodes = data.nodes

for i in range(10):
    node_name = nodes[i]
    feature_value = data.x[i].item()
    print(f"Index {i}: {node_name} → Feature: {feature_value:.4f}")

print("\nLabel (y):") # log_IC50 value for the drug-cell line combination
print(data.y)
print(data.y.shape) # Tensor of shape [1]