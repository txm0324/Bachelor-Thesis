import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm


# Define output directory
output_dir = "./results/Network"

# Load the GDSC dataset (contains gene expression data and drug response values) to define the drug names
gdsc_dataset = pd.read_csv('/sybig/home/tmu/TUGDA/data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)
drug_list = gdsc_dataset.columns[1780:]

# Create a directory to store CSV files for each drug's matrix
csv_output_dir = os.path.join(output_dir, "drug_matrices_csv")
os.makedirs(csv_output_dir, exist_ok=True)


for drug in tqdm(drug_list, desc="Drug Processing"):  

    # Construct the path to the .npz file that contains the sparse matrix for this drug
    file_path = os.path.join(output_dir, "drug_sparse_matrices_npz", f"{drug}_full.npz")

    # Load the saved sparse matrix components from the .npz file
    with np.load(file_path, allow_pickle=True) as data_file:
        loaded_data = data_file['data']         # non-zero values
        loaded_indices = data_file['indices']   # column indices of these values
        loaded_indptr = data_file['indptr']     # pointer to when a new row begins
        loaded_shape = data_file['shape']       # dimensions of the matrix
        loaded_genes_pathways = data_file['genes_pathways']  # Gene/pathway labels

    # Reconstruct the sparse matrix using CSR (Compressed Sparse Row) format
    reconstructed_sparse = csr_matrix((loaded_data, loaded_indices, loaded_indptr), shape=loaded_shape)

    # Convert the sparse matrix to a dense NumPy array for easier manipulation and readability
    reconstructed_dense = reconstructed_sparse.toarray()

    # Create a DataFrame from the dense array
    # Rows and columns correspond to genes/pathways
    df_reconstructed = pd.DataFrame(
        reconstructed_dense,
        index=loaded_genes_pathways,      
        columns=loaded_genes_pathways     
    ).astype(int)  # Store values as integers for cleaner display

    # Save the DataFrame as a CSV file for persistent storage
    csv_path = os.path.join(csv_output_dir, f"{drug}_matrix.csv")
    df_reconstructed.to_csv(csv_path, index=True)  # Keep index (gene/pathway names)
