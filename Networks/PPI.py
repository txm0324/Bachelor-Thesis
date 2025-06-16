import pandas as pd

# Create a Dataframe of a Protein-Protein-Network

# Load STRING Protein-Protein Interaction (PPI) data and mapping file
ppi = pd.read_csv("raw_data/PPI/9606.protein.links.v12.0.txt.gz", sep=" ")
gene_name = pd.read_csv("raw_data/PPI/protein2gene.csv", sep='\t')

# Remove species prefix "9606." from ENSP IDs in PPI dataset
ppi["protein1"] = ppi["protein1"].str.replace("9606.", "", regex=False)
ppi["protein2"] = ppi["protein2"].str.replace("9606.", "", regex=False)

# Rename columns for clarity
gene_name = gene_name.rename(columns={"protein_ENSP": "protein", "gene name": "gene"})

# Map protein1 (source) to gene names using the gene_name lookup table
ppi = ppi.merge(gene_name, how="left", left_on="protein1", right_on="protein")
ppi = ppi.rename(columns={"gene": "source"})
ppi = ppi.drop(columns=["protein"])  # Remove redundant column after merge

# Map protein2 (target) to gene names using the same lookup table
ppi = ppi.merge(gene_name, how="left", left_on="protein2", right_on="protein")
ppi = ppi.rename(columns={"gene": "target"})
ppi = ppi.drop(columns=["protein"])  # Remove redundant column after merge

# Keep only relevant columns: source and target gene names
ppi = ppi[["source", "target"]]

# Filter out rows where either source or target gene is missing ("Not Found")
mask = (ppi['source'] != 'Not Found') & (ppi['target'] != 'Not Found')
final_ppi_string = ppi[mask]


print("Final Dataframe:")
print(final_ppi_string.head())

"""
Comment on mapping process:

- To convert ENSP IDs to gene symbols, we used the `idconverter.py` script from: https://github.com/yafeng/idconverter 

Steps to generate the mapping file (protein2gene.csv):
1. Extract all unique ENSP IDs from both 'protein1' and 'protein2' columns: 
   - Run this code snippet to save the unique IDs:
     ppi = pd.read_csv("raw_data/PPI/9606.protein.links.v12.0.txt.gz", sep=" ")
     ppi["protein1"] = ppi["protein1"].str.replace("9606.", "", regex=False)
     ppi["protein2"] = ppi["protein2"].str.replace("9606.", "", regex=False)
     ensp_series = pd.concat([ppi['protein1'], ppi['protein2']], ignore_index=True)
     unique_ensp = ensp_series.unique()
     ensp_df = pd.DataFrame(unique_ensp, columns=['ENSP IDs'])
     ensp_df.to_csv('raw_data/PPI/unique_ENSP_IDs.csv', index=False)

2. Use idconverter.py to map IDs:
   - Command example: python idconverter.py --input unique_ENSP_IDs.csv --output protein2gene.csv --n 1
   - The flag `--n 1` indicates that the ENSP IDs are in the first data column

3. Fix compatibility if needed:
   - If the script uses Python 2 syntax, update print statements like:
     print "Warning..." → print("Warning...")
     print "Example..." → print("Example...")

Notes:
- During mapping, some ENSP IDs could not be converted to gene names ("Not Found")
    - For protein1/source: ~2647 interactions were lost due to "Not found"
    - For protein2/target: ~2647 interactions were lost due to "Not found"
    - After filtering both ends, ~2647 unique interactions were removed entirely
"""

# Optional: Save final cleaned PPI network to CSV (its already uploaded under Networks/data/final_PPI_Sting)
# final_ppi_string.to_csv("./results/PPI/final_PPI_String.csv", index=False)