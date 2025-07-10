import os # for file and directory operations 
import requests # for downloading files from URLs

# Script to download external data sets of biological/chemical interactions 

# ---------------------------------
# Step 1: Define URLs for Databases
# ---------------------------------

# List of file URLs to relevant databases containing drug-gene interaction data and Protein-Protein-Interaction Network (PPI)
urls = [
    "https://dgidb.org/data/latest/interactions.tsv", # DGIdb: Drug Gene Interaction Database
    "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.246/BIOGRID-CHEMICALS-4.4.246.chemtab.zip", # BioGrid: Protein interactions including small molecules
    "https://api.pharmgkb.org/v1/download/file/data/relationships.zip", # PharmGKB: Pharmacogenomics knowledge base
    "https://ctdbase.org/reports/CTD_chem_gene_ixns.csv.gz", # CTD: Comparative Toxicogenomics Database (chemical-gene interactions)
    "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz" # STRING: Template network (the gene names have to be converted from ensembl IDs to gene symbols)
]

# ------------------------------------------------
# Step 2: Create Local Directory for Raw Downloads
# ------------------------------------------------

# Set up folder structure
base_path = "./Networks"
raw_data_path = os.path.join(base_path, "raw_data")
ppi_path = os.path.join(raw_data_path, "PPI")

# Create folder
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(ppi_path, exist_ok=True)  

# ------------------------------------
# Step 3: Download Files from Each URL
# ------------------------------------

print("\n" + "="*23)
print("Download Files from URL")
print("="*23)

# Download each file
for url in urls:
    try:
        # Extract the file name from the URL
        filename = os.path.basename(url)
        file_path = os.path.join(raw_data_path, filename)

        # Decide target path based on URL
        if "stringdb" in url.lower():
            file_path = os.path.join(ppi_path, filename)
        else:
            file_path = os.path.join(raw_data_path, filename)

        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Save the file to disk
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"- Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")