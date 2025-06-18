import pandas as pd # read Dataframe 
from collections import defaultdict
from tqdm import tqdm # Processing steps 
import gseapy as gp # to get the pathway levels 

import os # for Directory
import numpy as np # for matrix

# to build sparse matrix
import scipy
from scipy.sparse import csr_matrix

################################
# Create Drug-Specific Network #
################################

# For each drug, nodes at the gene and pathway levels are used to built a network. 

print("\n" + "="*31)
print("Step 1: Download and Preprocess")
print("="*31)

# ------------------------------------
# Step 0: Download biological networks
# ------------------------------------

print("- Import DGI (Drug Gene Interactions)")  
# columns: gene_name, drug_name and database (BioGrid, ChEMBL, CTD, DGIdb, PharmaGKB)
# Code from 'Networks.ipynb' Point 1: Drug-Gene-Interactions
dgi_df = pd.read_csv("./data/DGI_Final.csv", index_col=0)

print("- Import PPI-Template (String Database)")
# columns: source, target
# Code from 'Networks.ipynb' Point 3: Protein-Protein-Network
ppi_df_STRING = pd.read_csv("./data/final_PPI_String.csv")
ppi_df_IntAct = pd.read_csv("./data/final_PPI_IntAct.csv")

ppi_df = pd.concat([ppi_df_STRING, ppi_df_IntAct], ignore_index=True)
ppi_df.drop_duplicates(subset=['source', 'target'], inplace=True)

# ppi_df = pd.read_csv("/sybig/home/tmu/Schreibtisch/Thesis/Networks/results/PPI/final_PPI_String.csv", index_col=0)

# Threshold Test
ppi_df = ppi_df[ppi_df['combined_score'] > 500]

# ------------------
# Step 1: Preprocess 
# ------------------

# cell-line and drug sample dataset of TUGDA (GDSC)
# gdsc_dataset = pd.read_csv('/Users/tm03/Desktop/TUGDA_1/data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)
gdsc_dataset = pd.read_csv('/sybig/home/tmu/TUGDA/data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)
gene_list = gdsc_dataset.columns[0:1780]
drug_list = gdsc_dataset.columns[1780:]

# Number of genes in PPI
ppi_genes = list(set(ppi_df['source'].to_list() + ppi_df['target'].to_list()))
print('- PPI Genes:', len(ppi_genes)) 

# Number of common genes between PPI and TUGDA to have the same samples
common_genes = list(set(ppi_genes).intersection(set(gene_list)))
print("- Common Genes (PPI and TUGDA):", len(common_genes))

drug_target_info = dgi_df[dgi_df['drug_name'].isin(drug_list)]
drug_target_info = drug_target_info[drug_target_info['gene_name'].isin(common_genes)]
drug_target_info = drug_target_info[["gene_name", "drug_name"]]
print("Final Dataframe 'drug_target_info")
print(drug_target_info.head())

# --------------------
# Step 2: Create Edges 
# --------------------

print("\n" + "="*20)
print("Step 2: Create Edges")
print("="*20)
print("Level 1 Edges: Targets")

# For each drug: get direct targets and their PPI neighbors as indirect targets 
print("- Define Direct Targets")
# Map each drug to its direct targets 
drug_to_direct_targets = (
    drug_target_info
    .query('drug_name in @drug_list')
    .groupby('drug_name')['gene_name']
    .unique()
    .to_dict()
)

# Mapping from Gen <--> PPI Partner (bidrectional)
ppi_genes = ppi_df[['source', 'target']].values.tolist()
gene_to_partners = defaultdict(set)

for source, target in ppi_genes:
    gene_to_partners[source].add(target)
    gene_to_partners[target].add(source)

total_direct_targets = sum(len(targets) for targets in drug_to_direct_targets.values())
print(f"Number of direct targets: {total_direct_targets}") # 23,063

print("- Define Indirect Targets")

# Collect all edges (drug -> direct target -> indirect target)
all_edges = []
drugs2target_edges = defaultdict(list)

for drug, direct_targets in tqdm(drug_to_direct_targets.items(), desc="Processing drugs"):
    # here just for 176 instead of 200 (TUGDA) because just for 176 Drugs Drug-Gene-Interactions was found
    for direct_target in direct_targets:
        if direct_target not in gene_list:
            continue # Just genes in TUGDA sample
        # Level 1: direct edge: Drug -> Target 
        all_edges.append((drug, direct_target, drug)) # second drug to indicate which drug this interaction belongs to

        # indirect targets via PPI 
        indirect_targets = gene_to_partners.get(direct_target, set())
        for partner in list(indirect_targets): # Threshold, Downstream
            if partner in gene_list:
            # Level 2: indirect edge: Target -> Partner 
                all_edges.append((direct_target, partner, drug)) # drug to indicate which drug this interaction belongs to
# Delete duplicates
lvl1_edges_df = pd.DataFrame(all_edges, columns = ["source", "target", "drug"]).drop_duplicates()
print(f"- All direct/indirect gene pairs: {lvl1_edges_df.shape[0]:,}") # 31,717,030 without classification, 253,693 with classfication

print("Define Connections")

# only keep edges that also occur in the PPI 
# Create a list of all valid PPI edges as a tuple 

ppi_set = set()
# is always saved in the same format
# For each pair (source, target) you ensure that it is always saved in the same format: always as (smaller_gen, larger_gen), alphabetical
for source, target in ppi_df[['source', 'target']].values:
    source = str(source)
    target = str(target)
    if source <= target:
        ppi_set.add((source, target))
    else:
        ppi_set.add((target, source))

# Filter
# e.g. tuple(sorted(['TP53', 'BRCA1'])) → ('BRCA1', 'TP53'), just same them one time instead of each time (a,b)/(b,a) is included 
sources = lvl1_edges_df['source'].values
targets = lvl1_edges_df['target'].values

mask = [tuple(sorted([source, target])) in ppi_set for source, target in zip(sources, targets)]
lvl1_edges_df = lvl1_edges_df[mask].reset_index(drop=True)

print(f"- Edges after PPI filtering: {lvl1_edges_df.shape[0]:,}")
print("Final Dataframe 'lv1_edges_df'")
print(lvl1_edges_df.head())

print("Level 2: Pathways")

all_lvl2_edges = []

# get pathways by library gseapy
# here I used KEGG_2021_Human, yon can also use 'GO_Biological_Process_2023', 'GO_Cellular_Component_2023' 'GO_Molecular_Function_2023', 'KEGG_2021_Human', 'Reactome_2022'
# gp.get_library_name()
kegg_gmt = gp.parser.get_library('KEGG_2021_Human', organism='Human', min_size=3, max_size=2000)

print("- Number of Pathways:", len(kegg_gmt))

# Number of genes 
kegg_genes = set(gene for genes in kegg_gmt.values() for gene in genes)
network_genes = set(lvl1_edges_df[["source", "target"]].values.ravel())
print("- Network genes:", len(network_genes))
filtered_kegg_genes = network_genes.intersection(kegg_genes)
print("- Filtered Genes (in network and KEGG):", len(filtered_kegg_genes))

# Create DataFrame with (Pathway, Gene), only for genes in the network
kegg_df_data = []
for term, genes in kegg_gmt.items():
    for gene in genes:
        if gene in filtered_kegg_genes:
            kegg_df_data.append((term, gene))

for drug in drug_list:   
    # Create DataFrame with (Pathway, Gene) for each drug, only for genes in the network
    kegg_df_data = []
    for term, genes in kegg_gmt.items():
        for gene in genes:
            if gene in filtered_kegg_genes:
                kegg_df_data.append((term, gene))

    # Add drug column to indicate which drug this interaction belongs to
    kegg_df = pd.DataFrame(kegg_df_data, columns=["source", "target"]).drop_duplicates()
    kegg_df['drug'] = drug
    
    # Speichern
    all_lvl2_edges.append(kegg_df)

# Merge Pathway Dataframes
lvl2_edges_df = pd.concat(all_lvl2_edges, axis=0).reset_index(drop=True)
print("Final Dataframe 'lv2_edges_df'")
print(lvl2_edges_df.head())

print(f"- Pathway edges added: {kegg_df.shape[0]:,}")
final_network = pd.concat([
    lvl1_edges_df,
    lvl2_edges_df,
], axis=0).drop_duplicates().reset_index(drop=True)

print(f"- Final network contains {final_network.shape[0]:,} unique edges.")
print(final_network.head())

# ------------------------------------------
# Step 3: Create binary matrix for each drug
# ------------------------------------------

# Directory to save the matrices
output_dir = "./results/Network"
npz_output_dir = os.path.join(output_dir, "drug_sparse_matrices_npz")
os.makedirs(npz_output_dir, exist_ok=True)

print("\n" + "="*31)
print("Step 3: Create adjacency matrix")
print("="*31)

# For each drug in drug_list
for drug in tqdm(drug_list, desc="Drug Processing"):  

    # Define involving edges and nodes of the current drug
    drug_mask = final_network['drug'] == drug
    drug_edges = final_network[drug_mask]
    nodes = pd.concat([drug_edges['source'], drug_edges['target']]).unique()
    
    # Mapping Gene -> Index for quick access 
    node_index = {node: idx for idx, node in enumerate(nodes)}
    dimension = len(nodes)

    # Initialize an empty binary adjacency matrix
    binary_matrix = np.zeros((dimension, dimension))

    # add self-loops (node with itself is a interaction - diagonal elements)
    np.fill_diagonal(binary_matrix, 1)

    # Create for each drug its binary matrix
    for _, row in drug_edges.iterrows(): 
        i = node_index[row["source"]]
        j = node_index[row["target"]]
        binary_matrix[i, j] = 1
        binary_matrix[j, i] = 1 # Symmetric update for undirected graph 
    
    # Save matrix as Sparse Matrix Representations (Storage-efficient, Charge-optimized, ML-compatible)
    '''
    Input : 
    10  20  0  0  0  0
    0  30  0  4  0  0
    0   0 50 60 70  0
    0   0  0  0  0 80

    Output :  
    A = [10 20 30 4 50 60 70 80], # vector of the non-zero elements 
    IA = [0 2 4 7 8] #  IA[0] = 0. IA[1] = IA[0] + no of non-zero elements in row 0
    JA = [0 1 1 3 2 3 4 5] # column indices of elements in A
    '''
    sparse_matrix = csr_matrix(binary_matrix)

    # Save complete data including metadata (gene names) for analysis and interpretation
    np.savez(
        os.path.join(npz_output_dir, f"{drug}_full.npz"),
        data=sparse_matrix.data, # non-zero values
        indices=sparse_matrix.indices, # column indices of these values
        indptr=sparse_matrix.indptr, # pointer to when a new row begins
        shape=sparse_matrix.shape, # dimensions of the matrix
        genes_pathways=nodes  # List of genes and pathways that occur in the matrix
    )


# --------------------------------------
# Step 4: Graph-Representation (Example)
# --------------------------------------

# import networkx as nx
# import matplotlib.pyplot as plt
# import pandas as pd

# print("Step 4: Graph (First Drug): First 100 Edges")

# example_drug = drug_list[0]
# drug_network = final_network[final_network['drug'] == example_drug]

# G = nx.Graph()

# # Füge Kanten hinzu
# for _, row in drug_network.head(100).iterrows():
#     source = row['source']
#     target = row['target']
#     G.add_edge(source, target)

# pos = nx.spring_layout(G, k=0.5, seed=42)

# node_colors = []
# for node in G.nodes:
#     if node in common_genes:
#         node_colors.append('lightblue') # Gene
#     elif node in kegg_gmt.keys():
#         node_colors.append('lightgreen') # Pathways
#     else:
#         node_colors.append('gray') # Drug

# # Plot
# plt.figure(figsize=(12, 8))
# nx.draw(G, pos, with_labels=True, node_size=300, node_color=node_colors,
#         font_size=10, edge_color='gray', alpha=0.8, width=1.2)
# plt.title(f"Drug-Specific Network for '{example_drug}'")
# plt.axis("off")
# plt.tight_layout()
# plt.show()
