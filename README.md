# Bachelor Thesis Project: [Your Thesis Title]
- token: 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

This repository contains the code implementation for my Bachelor Thesis titled "[Your Thesis Title]". The project focuses on [briefly describe your thesis topic in 1-2 sentences].

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
   Network_Construction.py: Builds the initial graph structure
2. Matrix Construction
   ```bash
   build_matrix.py: Creates for each drug the adjadency representation of the network
3. Data Preparation
   ```bash
   Train_Test_Data.py: Splite data into training and testings sets
4. Model implentation 
   ```bash
   Extension_TUGDA_MTL.py: Extension of TUGDA Implementation with additional features

## How to Run
Execute the scripts in the following order: 
- python3 Network_Construction.py
- python3 build_matrix.py
- python3 Train_Test_Data.py
- python3 Extension_TUGDA_MTL.py
  
