"""
Data Cleaning Script for Loan Approval Coursework
Based on Task 1 decisions and Task 3 mitigation strategies
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_loan_data(input_path, output_path):
    """
    Clean the loan approval dataset according to coursework specifications
    """
    
    print("="*70)
    print("LOAN APPROVAL DATA CLEANING")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Initial shape: {df.shape}")
    
    # Standardize column names
    print("\n2. Standardizing column names...")
    rename_map = {
        'id': 'ID',
        'age': 'Age',
        'Sex': 'Sex',
        'Education_Qualifications': 'Education Qualifications',
        'income': 'Income',
        'home_ownership': 'Home Ownership',
        'emplyment_length': 'Employment Length',  # Note: typo in original
        'loan_intent': 'Loan Intent',
        'loan_amount': 'Loan Amount',
        'loan_interest_rate': 'Loan Interest Rate',
        'loan_income_ratio': 'Loan-to-Income Ratio (LTI)',
        'payment_default_on_file': 'Payment Default on File',
        'credit_history_length': 'Credit History Length',
        'loan_approval_status': 'Loan Approval Status',
        'max_allowed_loan': 'Maximum Loan Amount',
        'Credit_Application_Acceptance': 'Credit Application Acceptance'
    }
    df = df.rename(columns=rename_map)
    
    # STEP 1: Drop unwanted columns
    print("\n3. Dropping unwanted columns...")
    columns_to_drop = ['ID', 'Sex', 'Age', 'Credit Application Acceptance']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print(f"   Dropped: {columns_to_drop}")
    print(f"   Shape after drop: {df.shape}")
    
    # STEP 2: Clean Education Qualifications - remove whitespace
    print("\n4. Cleaning Education Qualifications...")
    df['Education Qualifications'] = df['Education Qualifications'].str.strip()
    print("   ✓ Whitespace removed")
    
    # STEP 3: Clean Home Ownership - remove whitespace
    print("\n5. Cleaning Home Ownership...")
    df['Home Ownership'] = df['Home Ownership'].str.strip()
    print("   ✓ Minor category imbalance preserved")
    
    # STEP 4: Clean Loan Intent - format categories
    print("\n6. Cleaning Loan Intent...")
    df['Loan Intent'] = df['Loan Intent'].str.replace('HOMEIMPROVEMENT', 'Home Improvement', regex=False)
    df['Loan Intent'] = df['Loan Intent'].str.replace('DEBTCONSOLIDATION', 'Debt Consolidation', regex=False)
    df['Loan Intent'] = df['Loan Intent'].str.title()
    print("   ✓ Categories formatted for readability")
    
    # STEP 5: Encode Payment Default on File as binary
    print("\n7. Encoding Payment Default on File...")
    df['Payment Default on File'] = df['Payment Default on File'].str.strip().str.upper()
    df['Payment Default on File'] = df['Payment Default on File'].replace({'YES': 'Y', 'NO': 'N'})
    df['Payment Default on File'] = df['Payment Default on File'].map({'Y': 1, 'N': 0})
    print("   ✓ Encoded as binary: Y=1, N=0")
    
    # STEP 6: Clean Loan Approval Status
    print("\n8. Cleaning Loan Approval Status...")
    df['Loan Approval Status'] = df['Loan Approval Status'].str.strip()
    
    # Standardize all variations to "Approved" or "Declined"
    df['Loan Approval Status'] = df['Loan Approval Status'].replace({
        'Reject': 'Declined',
        'reject': 'Declined',
        'REJECT': 'Declined',
        'DECLINED': 'Declined',
        'Accept': 'Approved',
        'APPROVED': 'Approved',
        'ACCEPT': 'Approved'
    })
    
    rows_before = len(df)
    df = df.dropna(subset=['Loan Approval Status'])
    rows_dropped = rows_before - len(df)
    print(f"   ✓ Standardized status values")
    print(f"   ✓ Dropped {rows_dropped} row(s) with missing values")
    
    # STEP 7: Clean Maximum Loan Amount
    print("\n9. Cleaning Maximum Loan Amount...")
    negative_count = (df['Maximum Loan Amount'] < 0).sum()
    print(f"   Negative values found: {negative_count}")
    df = df[df['Maximum Loan Amount'] >= 0]
    
    above_cap = (df['Maximum Loan Amount'] > 500000).sum()
    df['Maximum Loan Amount'] = df['Maximum Loan Amount'].clip(upper=500000)
    print(f"   ✓ Capped {above_cap} values to £500,000")
    
    # STEP 8: Handle remaining missing values
    print("\n10. Handling remaining missing values...")
    print(f"   Missing values by column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # For numerical columns with missing values, impute with median (robust to outliers)
    numerical_cols = ['Income', 'Employment Length', 'Loan Amount', 'Loan Interest Rate',
                      'Loan-to-Income Ratio (LTI)', 'Credit History Length', 'Maximum Loan Amount']
    
    for col in numerical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"   ✓ Imputed {missing_count} values in {col} with median")
    
    # For categorical columns, impute with mode
    categorical_cols = ['Education Qualifications', 'Home Ownership', 'Loan Intent']
    
    for col in categorical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"   ✓ Imputed {missing_count} values in {col} with mode")
    
    # Check if Payment Default on File has missing values after encoding
    if df['Payment Default on File'].isnull().sum() > 0:
        # This means there were values that weren't Y or N
        # Let's impute with mode (most common: 0 or 1)
        mode_val = df['Payment Default on File'].mode()[0]
        df['Payment Default on File'] = df['Payment Default on File'].fillna(mode_val)
        print(f"   ✓ Imputed Payment Default on File with mode")
    
    missing_after = df.isnull().sum().sum()
    print(f"\n   Total missing values after imputation: {missing_after}")
    
    # Final summary
    print("\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    print(f"Final shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    print("\nClass distribution (Loan Approval Status):")
    print(df['Loan Approval Status'].value_counts())
    
    # Save cleaned data
    print(f"\n11. Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print("✓ Cleaning complete!")
    
    return df

if __name__ == "__main__":
    # Define paths
    data_dir = Path(__file__).parent.parent / 'data'
    input_file = data_dir / 'loan_approval_data.csv'
    output_file = data_dir / 'loan_approval_data_cleaned.csv'
    
    # Run cleaning
    df_cleaned = clean_loan_data(input_file, output_file)
    
    print("\n" + "="*70)
    print("Ready for modeling!")
    print("="*70)
