"""
Model Training Script for Loan Approval Coursework
Trains classification and regression models on cleaned data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Classification models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression models
from sklearn.tree import DecisionTreeRegressor

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, 
    precision_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_cleaned_data():
    """Load the cleaned dataset"""
    data_dir = Path(__file__).parent.parent / 'data'
    df = pd.read_csv(data_dir / 'loan_approval_data_cleaned.csv')
    print(f"✓ Loaded cleaned data: {df.shape}")
    return df


def train_classification_models(df):
    """
    PART A: Train classification models for Loan Approval Status
    """
    print("\n" + "="*70)
    print("PART A: CLASSIFICATION - LOAN APPROVAL STATUS PREDICTION")
    print("="*70)
    
    # Select categorical features for classification
    categorical_features = [
        'Education Qualifications',
        'Home Ownership',
        'Loan Intent',
        'Payment Default on File'
    ]
    
    # Prepare features and target
    X = df[categorical_features].copy()
    y = df['Loan Approval Status'].copy()
    
    # One-hot encode categorical features (except Payment Default which is already binary)
    X_encoded = pd.get_dummies(X, columns=['Education Qualifications', 'Home Ownership', 'Loan Intent'], 
                                drop_first=True)
    
    print(f"\nFeature matrix shape: {X_encoded.shape}")
    print(f"Features: {list(X_encoded.columns)[:5]}... (showing first 5)")
    
    # Stratified train-test split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))
    
    # Train models
    print("\n" + "-"*70)
    print("Training Classification Models...")
    print("-"*70)
    
    models = {
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Metrics (focusing on "Declined" class per success criteria)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label='Declined')
        precision = precision_score(y_test, y_pred, pos_label='Declined')
        f1 = f1_score(y_test, y_pred, pos_label='Declined')
        
        # AUC-ROC (convert to binary: Declined=1)
        y_binary = (y_test == 'Declined').astype(int)
        declined_idx = list(model.classes_).index('Declined')
        auc_roc = roc_auc_score(y_binary, y_proba[:, declined_idx])
        
        results[name] = {
            'Accuracy': accuracy,
            'Recall (Declined)': recall,
            'Precision (Declined)': precision,
            'F1-Score (Declined)': f1,
            'AUC-ROC': auc_roc
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Recall (Declined): {recall:.4f}")
        print(f"  Precision (Declined): {precision:.4f}")
        print(f"  F1-Score (Declined): {f1:.4f}")
        print(f"  AUC-ROC: {auc_roc:.4f}")
    
    print("\n" + "="*70)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("="*70)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4))
    
    return results


def train_regression_models(df):
    """
    PART B: Train regression models for Maximum Loan Amount
    """
    print("\n" + "="*70)
    print("PART B: REGRESSION - MAXIMUM LOAN AMOUNT PREDICTION")
    print("="*70)
    
    # Filter for approved loans only
    df_approved = df[df['Loan Approval Status'] == 'Approved'].copy()
    print(f"\nFiltered for approved loans: {df_approved.shape}")
    
    # Numerical features for Model 1
    numeric_features = [
        'Income', 'Employment Length', 'Loan Amount', 
        'Loan Interest Rate', 'Loan-to-Income Ratio (LTI)', 
        'Credit History Length'
    ]
    
    # Target
    y_reg = df_approved['Maximum Loan Amount'].copy()
    
    # Model 1 (DT1): Numeric features only
    X_dt1 = df_approved[numeric_features].copy()
    
    # Model 2 (DT2): All features (numeric + categorical)
    categorical_features = [
        'Education Qualifications', 'Home Ownership', 
        'Loan Intent', 'Payment Default on File'
    ]
    
    X_dt2_cat = pd.get_dummies(df_approved[categorical_features], 
                               columns=['Education Qualifications', 'Home Ownership', 'Loan Intent'],
                               drop_first=True)
    X_dt2 = pd.concat([df_approved[numeric_features], X_dt2_cat], axis=1)
    
    print(f"\nDT1 (Numeric only): {X_dt1.shape}")
    print(f"DT2 (All features): {X_dt2.shape}")
    
    # Train-test split (80:20) with same random state for consistency
    X_train_dt1, X_test_dt1, y_train_dt1, y_test_dt1 = train_test_split(
        X_dt1, y_reg, test_size=0.2, random_state=42
    )
    
    X_train_dt2, X_test_dt2, y_train_dt2, y_test_dt2 = train_test_split(
        X_dt2, y_reg, test_size=0.2, random_state=42
    )
    
    # Train models
    print("\n" + "-"*70)
    print("Training Regression Models...")
    print("-"*70)
    
    # Model 1
    print("\nTraining DT1 (Numeric features only)...")
    dt1 = DecisionTreeRegressor(random_state=42)
    dt1.fit(X_train_dt1, y_train_dt1)
    y_pred_dt1 = dt1.predict(X_test_dt1)
    
    mse_dt1 = mean_squared_error(y_test_dt1, y_pred_dt1)
    mae_dt1 = mean_absolute_error(y_test_dt1, y_pred_dt1)
    r2_dt1 = r2_score(y_test_dt1, y_pred_dt1)
    
    print(f"  MSE: {mse_dt1:,.2f}")
    print(f"  MAE: {mae_dt1:,.2f}")
    print(f"  R²: {r2_dt1:.4f}")
    
    # Model 2
    print("\nTraining DT2 (All features)...")
    dt2 = DecisionTreeRegressor(random_state=42)
    dt2.fit(X_train_dt2, y_train_dt2)
    y_pred_dt2 = dt2.predict(X_test_dt2)
    
    mse_dt2 = mean_squared_error(y_test_dt2, y_pred_dt2)
    mae_dt2 = mean_absolute_error(y_test_dt2, y_pred_dt2)
    r2_dt2 = r2_score(y_test_dt2, y_pred_dt2)
    
    print(f"  MSE: {mse_dt2:,.2f}")
    print(f"  MAE: {mae_dt2:,.2f}")
    print(f"  R²: {r2_dt2:.4f}")
    
    print("\n" + "="*70)
    print("REGRESSION RESULTS SUMMARY")
    print("="*70)
    
    results = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'R²'],
        'DT1 (Numeric)': [mse_dt1, mae_dt1, r2_dt1],
        'DT2 (All Features)': [mse_dt2, mae_dt2, r2_dt2]
    })
    print(results.to_string(index=False))
    
    # Determine best model based on R²
    if r2_dt2 > r2_dt1:
        print(f"\n✓ Best Model: DT2 (R² = {r2_dt2:.4f})")
        print("  Categorical features add predictive value")
    else:
        print(f"\n✓ Best Model: DT1 (R² = {r2_dt1:.4f})")
        print("  Simpler model with comparable performance")
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("LOAN APPROVAL COURSEWORK - MODEL TRAINING")
    print("="*70)
    
    # Load cleaned data
    df = load_cleaned_data()
    
    # Part A: Classification
    classification_results = train_classification_models(df)
    
    # Part B: Regression
    regression_results = train_regression_models(df)
    
    print("\n" + "="*70)
    print("✓ MODEL TRAINING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review model performance metrics")
    print("2. Perform hyperparameter tuning (GridSearchCV)")
    print("3. Generate confusion matrices and visualizations")
    print("4. Complete model evaluation and interpretation")
    print("5. Document findings in coursework report")
