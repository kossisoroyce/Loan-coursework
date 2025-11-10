"""
Add comprehensive cells to the loan approval notebook
"""

import json
from pathlib import Path

def add_cells_to_notebook(notebook_path):
    """Add all analysis cells to the notebook"""
    
    # Load existing notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    def create_cell(cell_type, source, execution_count=None):
        """Helper to create a cell"""
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        }
        if cell_type == "code":
            cell["execution_count"] = execution_count
            cell["outputs"] = []
        return cell
    
    # Starting from cell 6, add new cells
    new_cells = []
    
    # SECTION: DATA CLEANING
    new_cells.append(create_cell("markdown", [
        "---\n",
        "\n",
        "## 2. Task 3: Data Cleaning & Preprocessing\n",
        "\n",
        "Implementing all cleaning decisions from Task 1 and Task 3(a).\n",
        "\n",
        "### Cleaning Actions Summary:\n",
        "1. Drop unwanted columns (ID, Sex, Age, Credit Application Acceptance)\n",
        "2. Remove whitespace from categorical variables\n",
        "3. Format Loan Intent categories (add spaces)\n",
        "4. Encode Payment Default on File as binary (Y=1, N=0)\n",
        "5. Standardize Loan Approval Status labels\n",
        "6. Handle missing values\n",
        "7. Remove negative Maximum Loan Amount values\n",
        "8. Cap Maximum Loan Amount at £500,000"
    ]))
    
    # CELL: Drop columns
    new_cells.append(create_cell("code", [
        "# STEP 1: Drop unwanted columns\n",
        "print(\"=\"*70)\n",
        "print(\"STEP 1: DROPPING UNWANTED COLUMNS\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "columns_to_drop = ['ID', 'Sex', 'Age', 'Credit Application Acceptance']\n",
        "\n",
        "print(f\"\\nColumns to drop: {columns_to_drop}\")\n",
        "df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])\n",
        "\n",
        "print(f\"✓ Columns dropped\")\n",
        "print(f\"✓ New shape: {df.shape}\")\n",
        "print(f\"\\nRemaining columns ({len(df.columns)}):\")\n",
        "for i, col in enumerate(df.columns, 1):\n",
        "    print(f\"  {i:2d}. {col}\")"
    ], None))
    
    # CELL: Clean Education Qualifications
    new_cells.append(create_cell("code", [
        "# STEP 2: Clean Education Qualifications - Remove whitespace\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 2: CLEANING EDUCATION QUALIFICATIONS\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nBefore cleaning:\")\n",
        "print(df['Education Qualifications'].value_counts())\n",
        "\n",
        "df['Education Qualifications'] = df['Education Qualifications'].str.strip()\n",
        "\n",
        "print(\"\\nAfter cleaning:\")\n",
        "print(df['Education Qualifications'].value_counts())\n",
        "print(\"\\n✓ Whitespace removed\")"
    ], None))
    
    # CELL: Clean Home Ownership
    new_cells.append(create_cell("code", [
        "# STEP 3: Clean Home Ownership\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 3: CLEANING HOME OWNERSHIP\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nBefore cleaning:\")\n",
        "print(df['Home Ownership'].value_counts())\n",
        "\n",
        "df['Home Ownership'] = df['Home Ownership'].str.strip()\n",
        "\n",
        "print(\"\\nAfter cleaning:\")\n",
        "print(df['Home Ownership'].value_counts())\n",
        "print(\"\\n✓ Minor category imbalance preserved - provides signal\")"
    ], None))
    
    # CELL: Clean Loan Intent
    new_cells.append(create_cell("code", [
        "# STEP 4: Clean Loan Intent - Format categories\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 4: CLEANING LOAN INTENT\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nBefore cleaning:\")\n",
        "print(df['Loan Intent'].value_counts())\n",
        "\n",
        "# Add spaces to concatenated words\n",
        "df['Loan Intent'] = df['Loan Intent'].str.replace('HOMEIMPROVEMENT', 'Home Improvement', regex=False)\n",
        "df['Loan Intent'] = df['Loan Intent'].str.replace('DEBTCONSOLIDATION', 'Debt Consolidation', regex=False)\n",
        "df['Loan Intent'] = df['Loan Intent'].str.title()\n",
        "\n",
        "print(\"\\nAfter cleaning:\")\n",
        "print(df['Loan Intent'].value_counts())\n",
        "print(\"\\n✓ Categories formatted for readability\")"
    ], None))
    
    # CELL: Encode Payment Default
    new_cells.append(create_cell("code", [
        "# STEP 5: Encode Payment Default on File as binary\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 5: ENCODING PAYMENT DEFAULT ON FILE\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nBefore encoding:\")\n",
        "print(df['Payment Default on File'].value_counts(dropna=False))\n",
        "\n",
        "# Standardize to Y/N first\n",
        "df['Payment Default on File'] = df['Payment Default on File'].str.strip().str.upper()\n",
        "df['Payment Default on File'] = df['Payment Default on File'].replace({'YES': 'Y', 'NO': 'N'})\n",
        "\n",
        "# Encode as binary\n",
        "df['Payment Default on File'] = df['Payment Default on File'].map({'Y': 1, 'N': 0})\n",
        "\n",
        "print(\"\\nAfter encoding:\")\n",
        "print(df['Payment Default on File'].value_counts(dropna=False))\n",
        "print(\"\\n✓ Encoded as binary: Y=1, N=0\")"
    ], None))
    
    # CELL: Clean Loan Approval Status
    new_cells.append(create_cell("code", [
        "# STEP 6: Clean Loan Approval Status - Standardize labels\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 6: CLEANING LOAN APPROVAL STATUS (TARGET VARIABLE)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nBefore cleaning:\")\n",
        "print(df['Loan Approval Status'].value_counts(dropna=False))\n",
        "print(f\"Missing values: {df['Loan Approval Status'].isna().sum()}\")\n",
        "\n",
        "# Standardize all variations\n",
        "df['Loan Approval Status'] = df['Loan Approval Status'].str.strip()\n",
        "df['Loan Approval Status'] = df['Loan Approval Status'].replace({\n",
        "    'Reject': 'Declined', 'reject': 'Declined', 'REJECT': 'Declined', 'DECLINED': 'Declined',\n",
        "    'Accept': 'Approved', 'APPROVED': 'Approved', 'ACCEPT': 'Approved'\n",
        "})\n",
        "\n",
        "# Drop rows with missing target\n",
        "rows_before = len(df)\n",
        "df = df.dropna(subset=['Loan Approval Status'])\n",
        "rows_dropped = rows_before - len(df)\n",
        "\n",
        "print(f\"\\n✓ Dropped {rows_dropped} row(s) with missing target\")\n",
        "print(\"\\nAfter cleaning:\")\n",
        "print(df['Loan Approval Status'].value_counts())\n",
        "print(f\"\\n✓ Standardized to 2 classes: Approved and Declined\")"
    ], None))
    
    # CELL: Clean Maximum Loan Amount
    new_cells.append(create_cell("code", [
        "# STEP 7: Clean Maximum Loan Amount\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 7: CLEANING MAXIMUM LOAN AMOUNT\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nBefore cleaning:\")\n",
        "print(df['Maximum Loan Amount'].describe())\n",
        "\n",
        "# Check for negative values\n",
        "negative_count = (df['Maximum Loan Amount'] < 0).sum()\n",
        "print(f\"\\nNegative values found: {negative_count}\")\n",
        "\n",
        "# Remove negatives\n",
        "df = df[df['Maximum Loan Amount'] >= 0]\n",
        "print(f\"✓ Removed {negative_count} rows with negative values\")\n",
        "\n",
        "# Cap at £500,000\n",
        "above_cap = (df['Maximum Loan Amount'] > 500000).sum()\n",
        "df['Maximum Loan Amount'] = df['Maximum Loan Amount'].clip(upper=500000)\n",
        "print(f\"✓ Capped {above_cap} values at £500,000\")\n",
        "\n",
        "print(\"\\nAfter cleaning:\")\n",
        "print(df['Maximum Loan Amount'].describe())"
    ], None))
    
    # CELL: Handle missing values
    new_cells.append(create_cell("code", [
        "# STEP 8: Handle remaining missing values\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"STEP 8: HANDLING MISSING VALUES\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\nMissing values by column:\")\n",
        "missing = df.isnull().sum()\n",
        "print(missing[missing > 0])\n",
        "\n",
        "# Impute numerical columns with median\n",
        "numerical_cols = ['Income', 'Employment Length', 'Loan Amount', 'Loan Interest Rate',\n",
        "                  'Loan-to-Income Ratio (LTI)', 'Credit History Length', 'Maximum Loan Amount']\n",
        "\n",
        "for col in numerical_cols:\n",
        "    missing_count = df[col].isnull().sum()\n",
        "    if missing_count > 0:\n",
        "        median_val = df[col].median()\n",
        "        df[col] = df[col].fillna(median_val)\n",
        "        print(f\"✓ Imputed {missing_count} values in {col} with median ({median_val:.2f})\")\n",
        "\n",
        "# Impute categorical columns with mode\n",
        "categorical_cols = ['Education Qualifications', 'Home Ownership', 'Loan Intent']\n",
        "\n",
        "for col in categorical_cols:\n",
        "    missing_count = df[col].isnull().sum()\n",
        "    if missing_count > 0:\n",
        "        mode_val = df[col].mode()[0]\n",
        "        df[col] = df[col].fillna(mode_val)\n",
        "        print(f\"✓ Imputed {missing_count} values in {col} with mode ({mode_val})\")\n",
        "\n",
        "# Handle Payment Default if any missing after encoding\n",
        "if df['Payment Default on File'].isnull().sum() > 0:\n",
        "    mode_val = df['Payment Default on File'].mode()[0]\n",
        "    df['Payment Default on File'] = df['Payment Default on File'].fillna(mode_val)\n",
        "    print(f\"✓ Imputed Payment Default on File with mode ({mode_val})\")\n",
        "\n",
        "print(f\"\\nTotal missing values after imputation: {df.isnull().sum().sum()}\")"
    ], None))
    
    # CELL: Final data quality summary
    new_cells.append(create_cell("code", [
        "# STEP 9: Final Data Quality Summary\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"FINAL DATA CLEANING SUMMARY\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(f\"\\n1. Final dataset shape: {df.shape}\")\n",
        "print(f\"   Rows: {df.shape[0]:,}\")\n",
        "print(f\"   Columns: {df.shape[1]}\")\n",
        "\n",
        "print(f\"\\n2. Data quality metrics:\")\n",
        "print(f\"   Missing values: {df.isnull().sum().sum()}\")\n",
        "print(f\"   Duplicate rows: {df.duplicated().sum()}\")\n",
        "\n",
        "print(f\"\\n3. Target variable distribution:\")\n",
        "print(df['Loan Approval Status'].value_counts())\n",
        "approval_rate = (df['Loan Approval Status'] == 'Approved').sum() / len(df) * 100\n",
        "print(f\"\\n   Approval rate: {approval_rate:.2f}%\")\n",
        "print(f\"   Decline rate: {100-approval_rate:.2f}%\")\n",
        "\n",
        "print(f\"\\n4. Cleaned columns ({len(df.columns)}):\")\n",
        "for i, col in enumerate(df.columns, 1):\n",
        "    print(f\"   {i:2d}. {col}\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✓ DATA CLEANING COMPLETE - READY FOR MODELING\")\n",
        "print(\"=\"*70)"
    ], None))
    
    # CELL: Save cleaned data
    new_cells.append(create_cell("code", [
        "# Save cleaned dataset\n",
        "output_path = '../data/loan_approval_data_cleaned.csv'\n",
        "df.to_csv(output_path, index=False)\n",
        "print(f\"✓ Cleaned dataset saved to: {output_path}\")\n",
        "print(f\"✓ Shape: {df.shape}\")\n",
        "print(f\"✓ Ready for Task 4 modeling\")"
    ], None))
    
    # Add all new cells to the notebook
    notebook["cells"].extend(new_cells)
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Added {len(new_cells)} cells to notebook")
    print(f"✓ Total cells now: {len(notebook['cells'])}")

if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / 'notebooks' / 'loan_approval_complete.ipynb'
    add_cells_to_notebook(notebook_path)
    print(f"✓ Notebook updated successfully!")
