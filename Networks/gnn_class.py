import pandas as pd # read Dataframe 
import torch
from torch_geometric.data import Dataset, Data # for creating graph-based datasets
import os # for file and directory operations 
import numpy as np # for numerical computations with arrays 
import gseapy as gp # for retrieving pathway information
from tqdm import tqdm 

# Creating a Custom Dataset in Pytorch Geometric 

###################
### GNN Dataset ###
###################

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

        # Define Catch to save graphs 
        self.graph_cache = {}

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
        return [f'drug_{drug}_cellline_{cell_line}.pt' for drug, cell_line in self.samples]

    def download(self):
        pass 

    def _load_adjacency_matrix(self, drug):
        """ Loads the adjacency matrix for every drug """

        # Check if the graph for this drug is already cached to avoid reloading
        if drug in self.graph_cache:
            return self.graph_cache[drug]

        # Load the adjacency matrix from the CSV file
        csv_path = os.path.join(self.raw_dir, f"{drug}_matrix.csv")
        df = pd.read_csv(csv_path, index_col=0)

        # Extract node names
        nodes = df.columns.tolist()

        # Convert the DataFrame to a NumPy array representing the adjacency matrix
        adj_matrix = df.values

        # Get the indices of non-zero elements in the adjacency matrix to construct edges
        # This returns a 2D array where each row corresponds to source and target node indices
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)

        # Store the processed graph data (nodes and edges) in a dictionary
        graph_data = {
            "nodes": nodes,
            "edge_index": edge_index
        }

        # Cache the graph data for this drug to avoid redundant computation
        self.graph_cache[drug] = graph_data

        return graph_data

    def _get_node_features(self, nodes, cell_line):
        """ 
        This will return a matrix / 2d array othe shape [Number of edges, Edge Feature size]
        Each Feature represent: [Expression value, is_gene, is_pathway]
        """
        x = []
        for node in nodes:
            if node in self.gene_list:
                 # If the node is a gene, get its expression value for the given cell line and type [expr, is_gene, is_pathway]
                expr_value = self.expression_data.loc[cell_line, node]
                x.append([float(expr_value), 1.0, 0.0])  # [expr, is_gene, is_pathway]
            elif node in self.pathway_list:
                # If the node is a pathway, use a default feature value of 0.0
                x.append([0.0, 0.0, 1.0])  # [expr_dummy, is_gene, is_pathway]
            else:
                # Unknown node
                x.append([0.0, 0.0, 0.0])  # [expr_dummy, is_gene, is_pathway]
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
        drug, cell_line = self.samples[idx]

        # Load adjacency matric for each drug 
        graph_data = self._load_adjacency_matrix(drug)
        nodes = graph_data["nodes"]

        # Get edge_index in COO format
        edge_index = graph_data["edge_index"]

        # Get Node Features 
        x = self._get_node_features(nodes, cell_line)

        # Get label info (log_IC50 value for each drug-cell line combination)
        y = torch.tensor([self.labels_df.loc[cell_line, drug]], dtype=torch.float)

        # Create data object
        data = Data(x=x, edge_index=edge_index, y=y, drug=drug, cell_line=cell_line)

        # Attach node names for debugging purposes
        data.nodes = nodes

        return data

    def process(self):
        """ Processing the dataset by converting each sample into a graph data object and saving it to disk """
        pass
    
        """ 
        When you want to save each drug, include the follwoing code instead of pass, BUT ATTENTION! 
        You will save 805 * 200 Combination (- 24340 NaN labels in labels_df), each file has a memory of 896KB

        print(f"Processing dataset: {len(self.samples)} graphs to save")
        # Iterate over all samples ((drug, cell_line) pair)
        for idx, (drug, cell_line) in tqdm(enumerate(self.samples), total=len(self.samples), desc="Processing Graphs"):
            filename = f'drug_{drug}_cellline_{cell_line}.pt'
            pt_path = os.path.join(self.processed_dir, filename)

            # Only process and save if the file doesn't already exist
            if not os.path.exists(pt_path):
                data = self.get(idx)
                torch.save(data, pt_path)
        """