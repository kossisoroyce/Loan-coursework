"""
Script to create a clean, comprehensive Jupyter notebook for loan approval coursework
"""

import json
from pathlib import Path

def create_cell(cell_type, source, execution_count=None):
    """Create a notebook cell"""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }
    
    if cell_type == "code":
        cell["execution_count"] = execution_count
        cell["outputs"] = []
    
    return cell

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Cell 1: Title and Introduction
notebook["cells"].append(create_cell("markdown", [
    "# Loan Approval Coursework - Complete Machine Learning Analysis\n",
    "\n",
    "**Module:** Data Mining & Machine Learning  \n",
    "**Date:** November 2025\n",
    "\n",
    "---\n",
    "\n",
    "## Project Overview\n",
    "\n",
    "This notebook implements a complete machine learning pipeline for loan approval prediction:\n",
    "\n",
    "### Part A: Classification (Loan Approval Status)\n",
    "- **Objective:** Predict whether a loan application will be Approved or Declined\n",
    "- **Models:** Naive Bayes, Logistic Regression, Random Forest\n",
    "- **Success Criteria:** High Recall and Precision for \"Declined\" class\n",
    "\n",
    "### Part B: Regression (Maximum Loan Amount)\n",
    "- **Objective:** Estimate the maximum loan amount for approved applications\n",
    "- **Models:** Decision Tree Regression (DT1: numeric, DT2: all features)\n",
    "- **Success Criteria:** High R² to explain variance in loan amounts\n",
    "\n",
    "---"
]))

# Cell 2: Import Libraries
notebook["cells"].append(create_cell("code", [
    "# Import required libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from pathlib import Path\n",
    "\n",
    "# Classification models\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "# Regression models\n",
    "from sklearn.tree import DecisionTreeRegressor, plot_tree\n",
    "\n",
    "# Model evaluation and preprocessing\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.metrics import (\n",
    "    confusion_matrix, accuracy_score, recall_score, \n",
    "    precision_score, f1_score, roc_auc_score,\n",
    "    mean_squared_error, mean_absolute_error, r2_score\n",
    ")\n",
    "from sklearn.preprocessing import RobustScaler\n",
    "\n",
    "# Settings\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "np.random.seed(42)\n",
    "\n",
    "# Configure visualization\n",
    "plt.style.use('seaborn-v0_8-darkgrid')\n",
    "sns.set_palette(\"husl\")\n",
    "\n",
    "print(\"✓ All libraries imported successfully\")\n",
    "print(f\"✓ Random seed set to 42 for reproducibility\")"
], 1))

# Cell 3: Load Data
notebook["cells"].append(create_cell("markdown", [
    "## 1. Data Loading & Initial Exploration\n",
    "\n",
    "Load the raw dataset and perform initial inspection."
]))

notebook["cells"].append(create_cell("code", [
    "# Load the dataset\n",
    "DATA_PATH = Path('../data/loan_approval_data.csv')\n",
    "\n",
    "df_raw = pd.read_csv(DATA_PATH, low_memory=False)\n",
    "\n",
    "print(f\"Dataset loaded successfully!\")\n",
    "print(f\"Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns\")\n",
    "print(f\"\\nFirst few rows:\")\n",
    "df_raw.head()"
], 2))

# Cell 4: Standardize column names
notebook["cells"].append(create_cell("code", [
    "# Standardize column names for consistency\n",
    "rename_map = {\n",
    "    'id': 'ID',\n",
    "    'age': 'Age',\n",
    "    'Sex': 'Sex',\n",
    "    'Education_Qualifications': 'Education Qualifications',\n",
    "    'income': 'Income',\n",
    "    'home_ownership': 'Home Ownership',\n",
    "    'emplyment_length': 'Employment Length',\n",
    "    'loan_intent': 'Loan Intent',\n",
    "    'loan_amount': 'Loan Amount',\n",
    "    'loan_interest_rate': 'Loan Interest Rate',\n",
    "    'loan_income_ratio': 'Loan-to-Income Ratio (LTI)',\n",
    "    'payment_default_on_file': 'Payment Default on File',\n",
    "    'credit_history_length': 'Credit History Length',\n",
    "    'loan_approval_status': 'Loan Approval Status',\n",
    "    'max_allowed_loan': 'Maximum Loan Amount',\n",
    "    'Credit_Application_Acceptance': 'Credit Application Acceptance'\n",
    "}\n",
    "\n",
    "df = df_raw.rename(columns=rename_map)\n",
    "\n",
    "print(\"Column names standardized:\")\n",
    "for i, col in enumerate(df.columns, 1):\n",
    "    print(f\"{i:2d}. {col}\")"
], 3))

# Cell 5: Initial data quality check
notebook["cells"].append(create_cell("code", [
    "# Initial data quality assessment\n",
    "print(\"=\"*70)\n",
    "print(\"INITIAL DATA QUALITY ASSESSMENT\")\n",
    "print(\"=\"*70)\n",
    "\n",
    "print(f\"\\n1. Dataset shape: {df.shape}\")\n",
    "\n",
    "print(f\"\\n2. Data types:\")\n",
    "print(df.dtypes)\n",
    "\n",
    "print(f\"\\n3. Missing values:\")\n",
    "missing = df.isnull().sum()\n",
    "print(missing[missing > 0])\n",
    "print(f\"Total missing values: {missing.sum()}\")\n",
    "\n",
    "print(f\"\\n4. Duplicate rows: {df.duplicated().sum()}\")\n",
    "\n",
    "print(f\"\\n5. Target variable distribution (Loan Approval Status):\")\n",
    "print(df['Loan Approval Status'].value_counts(dropna=False))"
], 4))

# Save the notebook
output_path = Path(__file__).parent.parent / 'notebooks' / 'loan_approval_complete.ipynb'
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Notebook created: {output_path}")
print(f"✓ Total cells: {len(notebook['cells'])}")
