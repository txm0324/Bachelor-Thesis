import pandas as pd # read Dataframe 
import requests # to get UniProt Id
import numpy as np # for numerical computations with arrays 

from concurrent.futures import ThreadPoolExecutor, as_completed # Import modules for parallel processing using thread pools
# ChEMBL web client to query bioactivity data from the ChEMBL database
from chembl_webresource_client.new_client import new_client  # type: ignore
import time # Time module for measuring execution duration
from tqdm import tqdm # tqdm for progress bars in loops and iterations

# Create a Dataframe of Drug-Gene-Interactions

# -----------------------------------
# Step 0: Load and Prepare DataFrames
# -----------------------------------

# Load the full GDSC dataset (FPKM + AUC values for all drugs)
gdsc_dataset = pd.read_csv('/sybig/home/tmu/TUGDA/data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)
# gdsc_dataset = pd.read_csv('/Users/tm03/Desktop/TUGDA_1/data/GDSCDA_fpkm_AUC_all_drugs.zip', index_col=0)

# Extract gene and drug columns:
# - First 1780 columns correspond to gene expression data
# - Remaining columns represent drug AUC values
gene_list = gdsc_dataset.columns[0:1780]
drug_list = gdsc_dataset.columns[1780:]

# ----------------------------------
# Step 1: Get Drug-Gene-Interactions
# ----------------------------------

# -----
# DGIDB
# -----

# to convert the same output as in dgidb
drug_list_upper = set(drug_list.str.upper())

# Load the DGIDB dataset 
dgidb_df = pd.read_csv("raw_data/interactions.tsv", sep="\t")

# Filters DGIdb drug-gene interactions matching the gene and drug list of TUGDA
dgidb_filtered = dgidb_df[dgidb_df['gene_name'].isin(gene_list) & dgidb_df['drug_name'].isin(drug_list_upper)].copy()

# Map original drug name to uppercase
drug_name_map = {drug.upper(): drug for drug in drug_list}
dgidb_filtered['drug_name'] = dgidb_filtered['drug_name'].str.upper().map(drug_name_map)
dgidb_final = dgidb_filtered[['gene_name', 'drug_name']].drop_duplicates() # remove duplicates (n=337)

print("- DGIDB: Done")

# ------
# ChEMBL
# ------

### Preprocess

# Load any file in folder /data_TUGDA directory (e.g. cl_y_test_o_k1.csv)
# extract all drug names (expected: 200)
# convert drug names to ChEMBL IDs using PubChem's ID exchange service: https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi
''' 
drug_list_df = drug_list.tolist()
drug_list_df = pd.DataFrame(drug_list)
drug_list_df.to_csv('./data/DGI/ChEMBL/drug_list.csv', header=False, index=False)
'''

# Load the output file from PubChem's ID Exchange: maps drug names to ChEMBL IDs
drug2chembl = pd.read_csv('raw_data/drug2chembl.txt', sep='\t', names=['drug_name', 'chembl_id'], header=None)

# Identify duplicate entries caused by drug name synonyms or multiple ChEMBL IDs (can happen due to synonyms and records keys in ChEMBL)
duplicates = drug2chembl[drug2chembl.duplicated('drug_name', keep=False)]

# Manually resolve duplicates for specific drugs by selecting the correct ChEMBL ID
allowed_pairs = {
    'AZD6738': 'CHEMBL4285417',
    'BMS-345541': 'CHEMBL249697',
    'EPZ5676': 'CHEMBL3414626',
    'Linsitinib': 'CHEMBL1091644',
    'Luminespib': 'CHEMBL252164',
    'NVP-ADW742': 'CHEMBL399021',
    'OSI-027': 'CHEMBL3120215',
    'Obatoclax Mesylate': 'CHEMBL2107358',
}

# Keep only the manually verified ChEMBL ID for each listed drug
def keep_entry(row):
    drug = row['drug_name']
    chembl = row['chembl_id']
    if drug in allowed_pairs:
        return chembl == allowed_pairs[drug]
    return True 

drug2chembl = drug2chembl[drug2chembl.apply(keep_entry, axis=1)]

# Identify drugs without assigned ChEMBL ID 
non_found_chembl = drug2chembl[drug2chembl['chembl_id'].isna()]

# Manually search for ChEMBL IDs or known synonyms (based on external sources or manual curation)
non_found_chembl_search = [
    ["Oxaliplatin", "CHEMBL414804"], # directly in ChEMBL
    ["Nutlin-3a (-)", "CHEMBL191334"], # directly in ChEMBL
    ["Cisplatin", "CHEMBL11359"], # directly in ChEMBL
    ["BPD-00008900", None],
    ["BDP-00009066", None],
    ["JAK1_8709", None],
    ["IRAK4_4710", None],
    ["Podophyllotoxin bromide", "CHEMBL61"],
    ["Sinularin", "CHEMBL488193"], # synonym: Flexibilide
    ["VSP34_8731", None],
    ["KRAS (G12C) Inhibitor-12", None], 
    ["ERK_2440", None],
    ["Mirin", "CHEMBL570841"],
    ["Picolinici-acid", "CHEMBL72628"], # synonym: 2-PICOLINIC ACID
    ["JAK_8517", None],
    ["ERK_6604", None],
    ["PAK_5339", None],
    ["TAF1_5496", None],
    ["IGF1R_3801", None],
    ["CDK9_5576", None],
    ["CDK9_5038", None],
    ["ULK1_4989", None],
    ["IAP_5620", None],
    ["Eg5_9814", None],
    ["Cetuximab", "CHEMBL1201577"], # directly in ChEMBL
    ["Bleomycin", "CHEMBL403664"], # directly in ChEMBL
    ["Bleomycin (50 uM)", None]
]

# 18 drugs still lack a valid ChEMBL ID

# Integrates curated ChEMBL matches, de-duplicates existing mappings, excludes unresolved drugs
new_entries = pd.DataFrame(non_found_chembl_search, columns=['drug_name', 'chembl_id'])
new_entries = new_entries[new_entries['chembl_id'].notna()]
drug2chembl = drug2chembl[~drug2chembl['chembl_id'].isin(new_entries['chembl_id'])]
drug2chembl = pd.concat([drug2chembl, new_entries], ignore_index=True)
drug2chembl = drug2chembl.dropna(subset=['chembl_id'])

### get targets 

def fetch_targets_for_id(chembl_id, retries=3, delay=2):
    """
    Takes a list of ChEMBL IDs and returns a DataFrame with organism, pref_name, target_chembl_id,
    target_type, UniProt accession and chembl_id via the ChEMBL API.

    Filters:
    - organism: only Homo sapiens
    - target_type: must contain 'Protein'
    
    Returns:
    - pd.DataFrame: columns [organism, pref_name, target_chembl_id, target_type, uniprot_id, chembl_id]
    """
    bioactivities_api = new_client.activity
    targets_api = new_client.target

    for attempt in range(retries):
        try:
            bioactivities = bioactivities_api.filter(molecule_chembl_id=chembl_id).only("target_chembl_id")
            bioactivities_df = pd.DataFrame.from_records(bioactivities)

            if bioactivities_df.empty:
                print(f"No targets found for {chembl_id}")
                return pd.DataFrame([{
                    "target_chembl_id": None,
                    "pref_name": None,
                    "organism": None,
                    "target_type": None,
                    "uniprot_id": None,
                    "chembl_id": chembl_id
                }])

            target_ids = bioactivities_df["target_chembl_id"].drop_duplicates().tolist()
            targets = targets_api.filter(target_chembl_id__in=target_ids).only(
                "target_chembl_id", "pref_name", "organism", "target_type", "target_components"
            )

            records = []
            for target in targets:
                # Just targets with organism: Homo sapiens and target_type: PROTEIN
                if target.get("organism") != "Homo sapiens":
                    continue
                if " PROTEIN" not in target.get("target_type", "").upper():
                    continue

                components = target.get("target_components", [])
                uniprot_ids = [comp.get("accession") for comp in components if comp.get("accession")]

                for uniprot_id in uniprot_ids:
                    records.append({
                        "target_chembl_id": target.get("target_chembl_id"),
                        "pref_name": target.get("pref_name"),
                        "organism": target.get("organism"),
                        "target_type": target.get("target_type"),
                        "uniprot_id": uniprot_id,
                        "chembl_id": chembl_id
                    })

            if not records:
                print(f"No hits found for targets in Homo sapiens with {chembl_id}")
                return pd.DataFrame([{
                    "target_chembl_id": None,
                    "pref_name": None,
                    "organism": None,
                    "target_type": None,
                    "uniprot_id": None,
                    "chembl_id": chembl_id
                }])

            return pd.DataFrame(records)

        except Exception as e:
            print(f"[{chembl_id}] Error during attempt {attempt + 1}/{retries}: {e}")
            time.sleep(delay)

    print(f"[{chembl_id}] All attempts failed.")
    return pd.DataFrame([{
        "target_chembl_id": None,
        "pref_name": None,
        "organism": None,
        "target_type": None,
        "uniprot_id": None,
        "chembl_id": chembl_id
    }])

def get_targets_parallel(chembl_ids, max_workers=8):
    """
    Parallelizes the retrieval of targets for a list of ChEMBL IDs.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_targets_for_id, chembl_id): chembl_id for chembl_id in chembl_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc="- Parallel processing of ChEMBL IDs"):
            try:
                df = future.result()
                if not df.empty:
                    results.append(df)
            except Exception as e:
                print(f"[FUTURE ERROR] Error while processing a thread: {e}")

    if results:
        all_targets_df = pd.concat(results, ignore_index=True)
        return all_targets_df.drop_duplicates()
    else:
        print("No data collected.")
        return pd.DataFrame()

targets_df = get_targets_parallel(drug2chembl["chembl_id"].tolist())

"""
No targets found for 
- CHEMBL1421 (Dasatinib), wondering becuase there are over 3000 bioactivites (maybe to much to catch them all)
- CHEMBL1201577 (Cetuximab), no data available for compound 

No hits found for targets in Homo sapiens with
- CHEMBL2349416 (Pyridostatin), just target_type: cellline (11, but only 1 with Homo sapiens as Target Organism)
- CHEMBL1969416, just target_type: cellline (55) 
- CHEMBL924 (Zoledronic acid anhydrous), remark: there are SINGLE PROTEIN as target_type
- CHEMBL399907 (Elephantin), just target_type: cellline (3)
- CHEMBL488193 (Flexibilide), just target_type: cellline (2)
"""
# new format: drug_name, pref_name, chembl_id (rename to drug_chembl_id), target_chembl_id, uniprot_id (rename too target_uniprot_id), organism, target_type
targets_df = targets_df.rename(columns={
    'chembl_id': 'drug_chembl_id',
    'uniprot_id': 'target_uniprot_id'
})

targets_df = targets_df.merge(
    drug2chembl[['chembl_id', 'drug_name']],
    left_on='drug_chembl_id',
    right_on='chembl_id',
    how='left'
)

targets_df = targets_df[['drug_name', 'pref_name', 'drug_chembl_id', 'target_chembl_id', 'target_uniprot_id', 'organism', 'target_type']]


### get gene_name

# get unique UniProt IDs fromt the DataFrame (fewer API calls by avoiding duplicates and take unique genes)
unique_uniprot_ids = targets_df['target_uniprot_id'].dropna().unique()

def fetch_gene_name(uniprot_id):
    """
    Takes a list of UniProt IDs and returns a DataFrame with the first hint of the gene_name via the ChEMBL API.

    Returns:
    - pd.DataFrame: columns [gene_name]
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            genes = data.get('genes')
            if genes and 'geneName' in genes[0]:
                return genes[0]['geneName'].get('value')
    except:
        pass
    return None

# Map UniProt IDs to gene names using the UniProt API
uniprot_to_gene = {}
for uniprot_id in tqdm(unique_uniprot_ids, desc="- Loading gene names from UniProt"):
    gene_name = fetch_gene_name(uniprot_id)
    uniprot_to_gene[uniprot_id] = gene_name
    time.sleep(0.5)  # Respect API rate limits

# Add gene name column to the DataFrame by mapping UniProt IDs
targets_df['gene_name'] = targets_df['target_uniprot_id'].map(uniprot_to_gene)

# Filters ChEMBL drug-gene interactions matching the gene and drug list of TUGDA
targets_df = targets_df[['drug_name', 'gene_name']]
targets_df = targets_df[targets_df['gene_name'].isin(gene_list)]
targets_df = targets_df[targets_df['drug_name'].isin(drug_list)]
chembl_final = targets_df

print("- ChEMBL: Done")

# -------
# BioGrid
# -------

# Download BIOGRID-ALL-4.4.244.tab3.txt (https://downloads.thebiogrid.org/File/BioGRID/Release-Archive/BIOGRID-4.4.246/BIOGRID-CHEMICALS-4.4.246.chemtab.zip)
BioGrid_df = pd.read_csv("raw_data/BIOGRID-CHEMICALS-4.4.246.chemtab.zip", compression='zip', sep="\t", low_memory=False) 
# low_memory=False disables chunk processing to properly infer data types, which uses more memory but avoids dtype warnings

# Filter BioGRID data to include only interactions involving relevant drugs and genes
# Matches are based on official names and known synonyms

# Preprocessing: Split column as list (pipe-separated)
BioGrid_df['Chemical Name List'] = BioGrid_df['Chemical Name'].str.split('|')
BioGrid_df['Chemical Synonyms List'] = BioGrid_df['Chemical Synonyms'].str.split('|')
BioGrid_df['Official Symbol List'] = BioGrid_df['Official Symbol'].str.split('|')
BioGrid_df['Synonyms List'] = BioGrid_df['Synonyms'].str.split('|')

# Expand the lists: each list entry becomes a separate row (one-to-many explosion)
df_exploded_drugs = BioGrid_df.explode('Chemical Name List').explode('Chemical Synonyms List')
BioGrid_exploded_all = df_exploded_drugs.explode('Official Symbol List').explode('Synonyms List')

# Mapping 
# Match, if hits for gene and drug in the corresponding columns
drug_set = set(drug_list)
gene_set = set(gene_list)

# Create boolean masks for rows that match known drugs or their synonyms
# Drug with Chemical Name List and Chemical Synonyms List
drug_mask = (
    BioGrid_exploded_all['Chemical Name List'].isin(drug_set) |
    BioGrid_exploded_all['Chemical Synonyms List'].isin(drug_set)
)

# Create boolean masks for rows that match known drugs or their synonyms
# Gene with Official Symbol List and and Synonyms List
gene_mask = (
    BioGrid_exploded_all['Official Symbol List'].isin(gene_set) |
    BioGrid_exploded_all['Synonyms List'].isin(gene_set)
)

# Filter rows where both drug and gene match
BioGrid_final = BioGrid_exploded_all[drug_mask & gene_mask].copy()


# Issue: Rows may contain multiple matches; we need clear 1:1 mapping for gene–drug pairs
# Solution: Determine the exact matching name (original or synonym) for each gene and drug

# Resolve drug name: prefer official name, fallback to synonym
BioGrid_final['drug_name'] = np.where(
    BioGrid_final['Chemical Name List'].isin(drug_set),
    BioGrid_final['Chemical Name List'],
    np.where(
        BioGrid_final['Chemical Synonyms List'].isin(drug_set),
        BioGrid_final['Chemical Synonyms List'],
        None
    )
)

# Resolve gene name: prefer official symbol, fallback to synonym
BioGrid_final['gene_name'] = np.where(
    BioGrid_final['Official Symbol List'].isin(gene_set),
    BioGrid_final['Official Symbol List'],
    np.where(
        BioGrid_final['Synonyms List'].isin(gene_set),
        BioGrid_final['Synonyms List'],
        None
    )
)

BioGrid_final = BioGrid_final[['gene_name', 'drug_name']].drop_duplicates()

print("- BioGrid: Done")

# ---------
# PharmaGKB
# ---------

# Download clinicalAnnotations.zip (https://www.pharmgkb.org/downloads)

'''
If this file an zip_file run: 
# Download relationship file from zip file
with ZipFile("raw_data/relationships.zip") as file: 
    file.extract("relationships.tsv", path="raw_data")
# Delete zip file
os.remove("raw_data/relationships.zip")
''' 

# Load file and filter relevant columns
Pharma_relationships = pd.read_csv("raw_data/relationships.tsv", sep='\t') 
Pharma_relationships = Pharma_relationships[['Entity1_name', 'Entity2_name']]
Pharma_relationships['Entity2_name'] = Pharma_relationships['Entity2_name'].str.capitalize()

# Filters PharmaGKB drug-gene interactions matching the gene and drug list of TUGDA
Pharma_relationships = Pharma_relationships.rename(columns={'Entity1_name': 'gene_name', 'Entity2_name': 'drug_name'})
Pharma_relationships = Pharma_relationships[Pharma_relationships['gene_name'].isin(gene_list)]
Pharma_final = Pharma_relationships[Pharma_relationships['drug_name'].isin(drug_list)]
Pharma_final = Pharma_final.drop_duplicates()

print("- PharmaGKB: Done")

# ---
# CTD
# ---

# Download the file from the "Chemical-gene-interactions" section at: https://ctdbase.org/downloads/
# Define all column names as specified under the "Fields" section of the Chemical-Gene Interactions file
column_names = [
    "ChemicalName",
    "ChemicalID",
    "CasRN",
    "GeneSymbol",
    "GeneID",
    "GeneForms",
    "Organism",
    "OrganismID",
    "Interaction",
    "InteractionActions",
    "PubMedIDs"
]

# Load the CTD chemical-gene interaction data, skipping comment lines and assigning column names
CTD = pd.read_csv("raw_data/CTD_chem_gene_ixns.csv.gz",comment="#",names=column_names)

# Rename columns for clarity and consistency with downstream analysis
CTD = CTD.rename(columns={'GeneSymbol': 'gene_name', 'ChemicalName': 'drug_name'})

# Filters CTD drug-gene interactions for human-specific entries matching the gene and drug list of TUGDA
# returning unique drug-gene pairs
CTD = CTD[['drug_name','gene_name', 'Organism']]
CTD = CTD[CTD['Organism'] == 'Homo sapiens']
CTD = CTD[CTD['gene_name'].isin(gene_list)]
CTD = CTD[CTD['drug_name'].isin(drug_list)]
CTD_final = CTD[['drug_name','gene_name']].drop_duplicates()

print("- CTD: Done")

# ------------------------------------
# Step 2: Merge all databases together
# ------------------------------------

# Define a dictionary mapping database names to their respective DataFrames
dfs = {
    "DGIdb": dgidb_final,
    "ChEMBL": chembl_final,
    "BioGrid": BioGrid_final,
    "PharmaGKB": Pharma_final,
    "CTD": CTD_final
}

# Add a 'database' column to each DataFrame and store in a list
df_list = []
for name, df in dfs.items():
    df_copy = df.copy()
    df_copy["database"] = name
    df_list.append(df_copy)

# Concatenate all DataFrames into one 
final_all_DGI = pd.concat(df_list, ignore_index=True)

# Combine all duplicates with the same gene_name and drug_name into one entry
final_all_DGI = (
    final_all_DGI
    .groupby(["gene_name", "drug_name"]) # remove the duplicates so that each unique entry (gene_name, drug_name) entry occurs only once
    .agg({
        "database": lambda x: ", ".join(sorted(set(x))), # in the "database" column, list all participating databases separated by commas
    })
    .reset_index()
)


# ---------------------------------
# Step 3: Overview of interactions
# ---------------------------------

# Final check for consistency between DGI data and TUGDA input

# 1.) How many interactions per database?
print("="*45)
print("1.) How many interactions per database?\n")
for name, df in dfs.items():
    print(f"   - {name:}: {len(df):} pairs")

# 2.) Total number of drug-gene Interactions 
print("\n" + "="*45)
print(f"2.) Total Drug–Gene Interactions: {len(final_all_DGI)}")
print("="*45)

# 3.) Overlapping between the databases
# Filter rows where multiple databases contributed the same interaction in 'database'
combi = final_all_DGI[final_all_DGI['database'].str.contains(',')]
counts = combi['database'].value_counts()
print(f"3.) Overlapping Drug–Gene Interactions: {counts.sum()}")
print("="*45)

# # 4.) Check overlap with TUGDA Input (genes and drugs)
print("4.) Overlap with TUGDA Input\n")

# Drugs: Match between DGI and TUGDA
drugs_final_dgi = set(final_all_DGI['drug_name'].dropna().unique())
drugs_TUGDA = set(drug_list)
drugs_common = drugs_final_dgi & drugs_TUGDA

percent_drugs = len(drugs_common) / len(drugs_TUGDA) * 100 if drugs_TUGDA else 0
print(f"   - Drugs: {len(drugs_common)} of {len(drugs_TUGDA)} matched ({percent_drugs:.2f}%)") # 176 of 200 common genes (88%)

# Genes: Match between DGI and TUGDA
genes_final_dgi = set(final_all_DGI['gene_name'].dropna().unique())
genes_TUGDA = set(gene_list)
genes_common = genes_final_dgi & genes_TUGDA
percent_genes = len(genes_common) / len(genes_TUGDA) * 100 if genes_TUGDA else 0
print(f"   - Genes: {len(genes_common)} of {len(genes_TUGDA)} matched ({percent_genes:.2f}%)") # 1636 of 1780 common genes (91,91%)
print("="*45 )

print("Final Dataframe:")
print(final_all_DGI.head())
print("="*45 )