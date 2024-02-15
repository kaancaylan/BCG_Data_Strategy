# BCG_Data_Strategy README

## Overview
This project aims to analyze customer churn with the use of a dynamic churn definition and XGBoost.

## File Structure
- `main.py`: Contains the main script for model training and prediction.
- `dataloader.py`: Includes functions for data loading and preprocessing.

## Configuration
- `config.ini`: Configuration file with parameters for data preprocessing and model training.

## Data Files
Both the data files below are not included in the repository and are required for the model. 
- `transactions_dataset.csv`: Data file containing transaction information. 
- `sales_client_relationship_dataset.xlsx`: Data file containing client relationship with sales. 

## Environment Setup
To set up a virtual environment and install dependencies, follow these steps:

1. Install Python 3 if it is not already installed on your system.
2. Open a terminal or command prompt.
3. Navigate to the project directory.
4. Run the following commands:
   - To create a virtual environment: `python3 -m venv venv`
   - To activate the virtual environment:
     - On Windows: `venv\Scripts\activate`
     - On MacOS/Linux: `source venv/bin/activate`
5. Once the virtual environment is activated, install the required packages: `pip install -r requirements.txt`

## Usage
1. Ensure the data files are located in a `data` folder, created at the root of this repository.
2. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On MacOS/Linux: `source venv/bin/activate`
3. Run `main.py` to predict the churn in clients.
