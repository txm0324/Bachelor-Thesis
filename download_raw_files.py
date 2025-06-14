import os
import requests

# List of file URLs
urls = [
    "https://dgidb.org/data/latest/interactions.tsv", # DGIdb
    "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.246/BIOGRID-CHEMICALS-4.4.246.chemtab.zip", # BioGrid
    "https://api.pharmgkb.org/v1/download/file/data/relationships.zip", # PharmaGKB
    # CTD to add
    # ChEMBL to add
]

base_path = "./Networks"
raw_data_path = os.path.join(base_path, "raw_data")
os.makedirs(raw_data_path, exist_ok=True)

# Download each file
for url in urls:
    try:
        # Extract the file name from the URL
        filename = os.path.basename(url)
        file_path = os.path.join(raw_data_path, filename)

        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Save the file to disk
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")