# Federated Learning for Mobile Network Performance Estimation
Carlos Cano Garrido, UMA 2025

## Description
This repository contains the code for my final degree project, adapting a public GitHub repository ([original repo](https://github.com/adap/flower/tree/f34d6e8d42864cfdfc4b0c4d582ce0eed07dbdef/examples/xgboost-comprehensive)) to estimate LTE Advanced cell-level throughput using XGBoost in a **federated learning** setup.  

The original model was a **classification model**, and this repository adapts it to a **regression task** for the dataset containing 36 sites (`dataset_con_site_id.xlsx`). It includes a **federated model**, a **centralized model**, and a **performance evaluation script** for federated learning.

## Repository Structure

```
TFG_XGBoost_Federated/
 │ 
 ├── Federated_model/ # Federated learning scripts
 │ ├── __init__.py 
 │ ├── client5G.py/ 
 │ ├── server5G.py/ 
 │ └── task5G.py/ 
 │ 
 │ 
 ├── dataset_con_site_id.xlsx/ 
 │ 
 ├── Federated_performance.py/  # Evaluation script for federated model
 │ 
 ├── pyproject.toml/  # Project dependencies and federated model configurations
 │ 
 └── xgboost_centralizado.py # Centralized XGBoost model
````

## Environment

Make sure you have Conda and Python 3.10 installed. Then:

```bash
conda create -n flwr-xgb python=3.10 -y
conda activate flwr-xgb
pip install -e .
````

## Usage

### Run the federated model
This will save the test indices for evaluation:

```bash
flwr run .
````

First, run the federated version to save the indices of the test data points, ensuring that the same samples are used for evaluation in both centralized and federated cases.

## Project Report
The full project report is available in the [docs folder](docs/Memoria_TFG_Carlos_Cano_Garrido.pdf).
