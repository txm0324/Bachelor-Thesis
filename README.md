# Bachelor Thesis Project: Optimizing the prediction of oncological drug responses by integrating biological network information into the deep learning framework [TUGDA](https://github.com/CSB5/TUGDA)
- token: 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

This repository contains the code implementation for my Bachelor Thesis titled "Optimizing the Prediction of Oncological Drug Responses by Integrating Biological Network Information into the Deep Learning Framework TUGDA." The project focuses on evaluating the performance of TUGDA in predicting drug responses based on omics data, with the aim of improving its accuracy through the integration of biological extensions. To achieve this, interactions such as drug–gene, drug–pathway, and protein–protein interactions are incorporated into the model architecture. 

## Prerequisites 

Before running the code, please ensure you have the following installed: @ToDo

- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/txm0324/Bachelor-Thesis.git
   cd Bachelor-Thesis
2. Install required dependencies
   ```bash
   pip install -r requirements.txt

## Project Pipeline 
The project follows this sequential workflow: 

1. Network Construction
   ```bash
   Network_Construction.py: Builds the initial graph structure, including direct and indirect targets and their pathway connections, using databases such as ChEMBL and BioGrid
2. Matrix Construction
   ```bash
   build_matrix.py: Creates the adjacency matrix representation of the network for each drug
3. Data Preparation
   ```bash
   Train_Test_Data.py: Splits the data into training and testing sets
4. Model implentation 
   ```bash
   Extension_TUGDA_MTL.py: Extension of TUGDA implementation with additional features

## How to Run
Execute the scripts in the following order: 
- python3 Network_Construction.py
- python3 build_matrix.py
- python3 Extension_TUGDA_MTL.py
  
