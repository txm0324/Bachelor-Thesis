import pandas as pd # for Dataframe
import re # for regular expressions (RegEx)

# Create a Dataframe of a Protein-Protein-Network

# -----------------------
# Step 1: STRING Database 
# -----------------------

print("\n" + "="*38)
print("Step 1: Download and Preprocess STRING")
print("="*38)

# Load STRING Protein-Protein Interaction (PPI) data and mapping file
ppi_string = pd.read_csv("raw_data/PPI/9606.protein.links.v12.0.txt.gz", sep=" ")
gene_name_string = pd.read_csv("raw_data/PPI/protein2gene_STRING.csv", sep='\t')

# Remove species prefix "9606." from ENSP IDs in PPI dataset
ppi_string["protein1"] = ppi_string["protein1"].str.replace("9606.", "", regex=False)
ppi_string["protein2"] = ppi_string["protein2"].str.replace("9606.", "", regex=False)

# Rename columns for clarity
gene_name_string = gene_name_string.rename(columns={"protein_ENSP": "protein", "gene name": "gene"})

# Map protein1 (source) to gene names using the gene_name lookup table
ppi_string = ppi_string.merge(gene_name_string, how="left", left_on="protein1", right_on="protein")
ppi_string = ppi_string.rename(columns={"gene": "source"})
ppi_string = ppi_string.drop(columns=["protein"])  # Remove redundant column after merge

# Map protein2 (target) to gene names using the same lookup table
ppi_string = ppi_string.merge(gene_name_string, how="left", left_on="protein2", right_on="protein")
ppi_string = ppi_string.rename(columns={"gene": "target"})
ppi_string = ppi_string.drop(columns=["protein"])  # Remove redundant column after merge

# Keep only relevant columns: source and target gene names
ppi_string = ppi_string[["source", "target", "combined_score"]]

# Filter out rows where either source or target gene is missing ("Not Found")
mask = (ppi_string['source'] != 'Not Found') & (ppi_string['target'] != 'Not Found')
final_ppi_string = ppi_string[mask]

print("Final Dataframe - STRING:")
print(final_ppi_string.head(10))

# -----------------------
# Step 2: IntAct Database 
# -----------------------

print("\n" + "="*38)
print("Step 2: Download and Preprocess IntAct")
print("="*38)

# Functios to preprocess files
def extract_uniprotkb(text):
    """
    Extracts the UniProtKB identifier from a string.

    Parameters:
    - text: A string containing the UniProtKB ID.

    Returns:
    - string: The extracted UniProtKB ID or the original text if no ID is found.
    """
    match = re.search(r'uniprotkb:([A-Z0-9]+)', text)
    return match.group(1) if match else text

def extract_intact_miscore(text):
    """
    Extracts the Confidence value(s) (MI-Score) from a string.

    Parameters:
    - text: A string containing the MI-Score.

    Returns:
    - string: The extracted Confidence value(s) or the original text if no ID is found.
    """
    match = re.search(r'intact-miscore:([0-9.]+)', text)
    return float(match.group(1)) if match else None

def process_df(df):
    """
    Processes a DataFrame by extracting relevant information and applying functions to specific columns.

    Parameters:
    - df: A pandas DataFrame containing columns 'Protein', 'Interactor', 'Confidence value(s)', 'Database'.

    Returns:
    - pd.DataFrame: A processed DataFrame with additional columns for taxa and names.
    """
    df['Protein'] = df['#ID(s) interactor A'].apply(extract_uniprotkb)
    df['Interactor'] = df['ID(s) interactor B'].apply(extract_uniprotkb)
    df['intact_miscore'] = df['Confidence value(s)'].apply(extract_intact_miscore)
    df['Database'] = 'IntAct'
    return df

# Load STRING Protein-Protein Interaction (PPI) data and mapping file
try:
    ppi_intact = pd.read_csv("/Users/tm03/Desktop/Uni/Softwarepraktikum/Softwarepraktikum_local/Encoding/PPI/Aktuell/human.txt", sep="\t")
except FileNotFoundError:
    raise FileNotFoundError(
        f"The file '{"raw_data/PPI/human.txt"}' was not found.\n"
        "Please download the file from the following website:\n"
        "https://www.ebi.ac.uk/intact/interactomes\n\n"
        "→ Choose: *Homo sapiens* as the organism\n"
        "→ Format: *MITAB 2.7*\n"
        "Then save the file as 'human.txt' in the folder 'raw_data/PPI/'"
    )

gene_name_intact = pd.read_csv("raw_data/PPI/protein2gene_IntAct.csv", sep='\t')

# Run preprocess functions
ppi_intact = process_df(ppi_intact)
selected_columns = ['Protein', 'Interactor', 'intact_miscore', 'Database']
ppi_intact = ppi_intact[selected_columns]
ppi_intact.rename(columns={"intact_miscore" : "combined_score"}, inplace=True)

# Drop rows with missing value
ppi_intact = ppi_intact[~(ppi_intact[['Protein', 'Interactor']] == '-').any(axis=1)]

# Rename columns for clarity
gene_name_intact = gene_name_intact.rename(columns={'UniProt_ID': 'Protein_ID', 'gene name': 'Gene_Name'})

# Map protein1 (source) to gene names using the gene_name lookup table
ppi_intact = ppi_intact.merge(gene_name_intact, left_on='Protein', right_on='Protein_ID', how='left')
ppi_intact = ppi_intact.rename(columns={'Gene_Name': 'source'})
ppi_intact = ppi_intact.drop(columns='Protein_ID')

# Map protein2 (target) to gene names using the same lookup table
ppi_intact = ppi_intact.merge(gene_name_intact, left_on='Interactor', right_on='Protein_ID', how='left')
ppi_intact = ppi_intact.rename(columns={'Gene_Name': 'target'})
ppi_intact = ppi_intact.drop(columns='Protein_ID')

# Keep only relevant columns: source, target gene names and the score as threshold
ppi_intact = ppi_intact[["source", "target", "combined_score"]]
ppi_intact['combined_score'] = ppi_intact['combined_score'] * 1000

# Filter out rows where either source or target gene is missing ("Not Found")
mask = (ppi_intact['source'] != 'Not Found') & (ppi_intact['target'] != 'Not Found')
final_ppi_intact = ppi_intact[mask]

print("Final Dataframe - IntAct:")
print(final_ppi_intact.head(10))
"""
Comment on mapping process:

- To convert ENSP IDs to gene symbols, we used the `idconverter.py` script from: https://github.com/yafeng/idconverter 

Steps to generate the mapping file (protein2gene.csv):
1. Extract all unique ENSP IDs from both 'protein1' and 'protein2' columns: 
   - Run this code snippet to save the unique IDs (STRING):
     ppi = pd.read_csv("raw_data/PPI/9606.protein.links.v12.0.txt.gz", sep=" ")
     ppi["protein1"] = ppi["protein1"].str.replace("9606.", "", regex=False)
     ppi["protein2"] = ppi["protein2"].str.replace("9606.", "", regex=False)
     ensp_series = pd.concat([ppi['protein1'], ppi['protein2']], ignore_index=True)
     unique_ensp = ensp_series.unique()
     ensp_df = pd.DataFrame(unique_ensp, columns=['ENSP IDs'])
     ensp_df.to_csv('raw_data/PPI/unique_ENSP_IDs.csv', index=False)
   - Run this code snippet to save the unique IDs (IntAct):
     unique_uniprot_ids = pd.concat([IntAct_df_cleaned['Protein'], IntAct_df_cleaned['Interactor']]).dropna().unique()
     unique_uniprot_list = unique_uniprot_ids.tolist()
     filtered_uniprot_list = [uid for uid in unique_uniprot_list if not uid.startswith('intact:')]
     filtered_uniprot_ids = np.array(filtered_uniprot_list)
     unique_uniprot_df = pd.DataFrame({'UniProt_ID': filtered_uniprot_ids})
     unique_uniprot_df.to_csv('unique_uniprot_ids.csv', index=False)

2. Use idconverter.py to map IDs:
   - Command example: python idconverter.py --input unique_ENSP_IDs.csv --output protein2gene.csv --n 1
   - The flag `--n 1` indicates that the ENSP IDs are in the first data column

3. Fix compatibility if needed:
   - If the script uses Python 2 syntax, update print statements like:
     print "Warning..." → print("Warning...")
     print "Example..." → print("Example...")

Notes:
- During mapping, some ENSP IDs could not be converted to gene names ("Not Found")
    - For protein1/source: 2647 interactions were lost due to "Not found"
    - For protein2/target: 2647 interactions were lost due to "Not found"
    - After filtering both ends, 2647 unique interactions were removed entirely
"""

# Optional: Save final cleaned PPI network to CSV (its already uploaded under Networks/data/final_PPI_Sting)
# final_ppi_string.to_csv("./data/final_PPI_String.csv", index=False)
# final_ppi_string.to_csv("./data/final_PPI_IntAct.csv", index=False)