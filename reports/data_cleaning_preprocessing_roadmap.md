# Data Cleaning & Preprocessing Roadmap
**Loan Approval Coursework Project**

---

## 📋 Phase 1: Variable Selection

### **Columns to DROP** (5 columns)
```python
columns_to_drop = [
    'id',                              # Not relevant to objective
    'age',                             # May introduce bias
    'Sex',                             # May introduce bias  
    'Credit_Application_Acceptance'    # May introduce bias (duplicate of target)
]
```

### **Columns to RETAIN** (12 columns)
- Education Qualifications
- Income
- Home Ownership
- Employment Length
- Loan Intent
- Loan Amount
- Loan Interest Rate
- Loan-to-Income Ratio (LTI)
- Payment Default on File
- Credit History Length
- **Loan Approval Status** (TARGET)
- Maximum Loan Amount

---

## 🔧 Phase 2: Data Quality Issues & Solutions

### **Summary of Issues Found**

| Issue | Records Affected | Impact | Action |
|-------|------------------|---------|--------|
| Typo: "emplyment_length" | All rows | Column naming | Rename |
| Negative Maximum Loan Amount | 3 (0.01%) | Data errors | Drop rows |
| Impossible Employment Length (>50 years) | ~10 records | Data errors | Cap/filter |
| Whitespace in Education Qualifications | ~1,000 | Duplicate categories | Strip whitespace |
| Loan Intent formatting | All 6 categories | Readability | Add spaces, title case |
| Payment Default inconsistencies | 16 (0.03%) | 4 variants + 5 NaN | Standardize & encode |
| Loan Approval Status inconsistencies | 167 (0.28%) | 8 variants + 1 NaN | Standardize to 2 classes |
| Missing Loan Interest Rate | 11 (0.02%) | Missing values | Median imputation |

**Total Data Loss:** <0.01% (4-15 rows maximum)

---

## 🛠️ Phase 3: Step-by-Step Cleaning Roadmap

### **STEP 1: Load Data**
```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/loan_approval_data.csv', low_memory=False)
print(f"Initial dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
```

---

### **STEP 2: Drop Irrelevant Columns**
```python
# Drop columns that introduce bias or are not relevant
columns_to_drop = ['id', 'age', 'Sex', 'Credit_Application_Acceptance']
df = df.drop(columns=columns_to_drop)

print(f"After dropping columns: {df.shape}")
```

**Justification:** Removes bias-inducing variables and identifiers

---

### **STEP 3: Fix Column Name Typo**
```python
# Rename misspelled column
df = df.rename(columns={
    'emplyment_length': 'Employment Length',
    'income': 'Income',
    'home_ownership': 'Home Ownership',
    'loan_intent': 'Loan Intent',
    'loan_amount': 'Loan Amount',
    'loan_interest_rate': 'Loan Interest Rate',
    'loan_income_ratio': 'Loan-to-Income Ratio (LTI)',
    'payment_default_on_file': 'Payment Default on File',
    'credit_history_length': 'Credit History Length',
    'loan_approval_status': 'Loan Approval Status',
    'max_allowed_loan': 'Maximum Loan Amount',
    'Education_Qualifications': 'Education Qualifications'
})

print("Column names standardized")
```

**Justification:** Consistent naming convention improves code readability

---

### **STEP 4: Remove Data Entry Errors**

#### 4a. Remove Negative Maximum Loan Amounts
```python
# Check negative values
print(f"Negative Maximum Loan Amount: {(df['Maximum Loan Amount'] < 0).sum()} records")

# Remove (only 3 records = 0.01%)
df = df[df['Maximum Loan Amount'] >= 0].copy()
print(f"After removing negatives: {df.shape[0]:,} rows")
```

**Justification:** Negative loan amounts are impossible - clear data entry errors

#### 4b. Cap Impossible Employment Length
```python
# Check for impossible values
print(f"Max employment length: {df['Employment Length'].max()} years")
print(f"Records with >50 years: {(df['Employment Length'] > 50).sum()}")

# Cap at 50 years (reasonable career limit)
df = df[df['Employment Length'] <= 50].copy()
print(f"After employment filter: {df.shape[0]:,} rows")
```

**Justification:** 150 years employment is impossible; 50 years is a reasonable upper limit

---

### **STEP 5: Clean Categorical Variables**

#### 5a. Education Qualifications - Remove Whitespace
```python
# Before cleaning
print("Before:")
print(df['Education Qualifications'].value_counts())

# Strip trailing/leading whitespace
df['Education Qualifications'] = df['Education Qualifications'].str.strip()

# After cleaning
print("\nAfter:")
print(df['Education Qualifications'].value_counts())
```

**Justification:** Merges "Unknown " and "Unknown" into single category

#### 5b. Loan Intent - Format for Readability
```python
# Before
print("Before:", df['Loan Intent'].unique())

# Add spaces and proper casing
df['Loan Intent'] = df['Loan Intent'].replace({
    'HOMEIMPROVEMENT': 'Home Improvement',
    'DEBTCONSOLIDATION': 'Debt Consolidation',
    'EDUCATION': 'Education',
    'MEDICAL': 'Medical',
    'VENTURE': 'Venture',
    'PERSONAL': 'Personal'
})

# After
print("After:", df['Loan Intent'].unique())
```

**Justification:** Improves readability in reports and visualizations

#### 5c. Payment Default on File - Standardize & Encode
```python
# Before
print("Before:")
print(df['Payment Default on File'].value_counts(dropna=False))

# Standardize to Y/N first
df['Payment Default on File'] = df['Payment Default on File'].replace({
    'YES': 'Y',
    'NO': 'N'
})

# Fill missing with mode (most common = 'N')
mode_value = df['Payment Default on File'].mode()[0]
df['Payment Default on File'].fillna(mode_value, inplace=True)

# Binary encode (Y=1, N=0)
df['Payment Default on File'] = df['Payment Default on File'].map({'Y': 1, 'N': 0})

# After
print("\nAfter:")
print(df['Payment Default on File'].value_counts())
```

**Justification:** Binary encoding required for ML models; mode imputation appropriate for <0.1% missing

---

### **STEP 6: Standardize Target Variable**

```python
# Before
print("Before:")
print(df['Loan Approval Status'].value_counts(dropna=False))

# Standardize to 2 classes
df['Loan Approval Status'] = df['Loan Approval Status'].str.upper().replace({
    'APPROVED': 'Approved',
    'ACCEPT': 'Approved',
    'DECLINED': 'Declined',
    'REJECT': 'Declined'
})

# Proper case
df['Loan Approval Status'] = df['Loan Approval Status'].str.title()

# Drop the 1 row with missing value
df = df[df['Loan Approval Status'].notna()].copy()

# After
print("\nAfter:")
print(df['Loan Approval Status'].value_counts())
```

**Justification:** Binary classification requires exactly 2 classes; 1 missing row = negligible loss

---

### **STEP 7: Handle Missing Numerical Values**

```python
# Check missing values
retained_vars = [
    'Education Qualifications', 'Income', 'Home Ownership', 'Employment Length',
    'Loan Intent', 'Loan Amount', 'Loan Interest Rate', 'Loan-to-Income Ratio (LTI)',
    'Payment Default on File', 'Credit History Length', 'Loan Approval Status',
    'Maximum Loan Amount'
]

print("Missing values in retained variables:")
print(df[retained_vars].isnull().sum())

# Impute Loan Interest Rate with median (robust to outliers)
median_rate = df['Loan Interest Rate'].median()
df['Loan Interest Rate'].fillna(median_rate, inplace=True)

print(f"\nImputed 11 missing loan interest rates with median: {median_rate:.2f}%")
```

**Justification:** Median imputation preserves central tendency without outlier bias

---

### **STEP 8: Final Verification**

```python
# Check for remaining issues
print("="*70)
print("FINAL DATA QUALITY CHECK")
print("="*70)
print(f"Final shape: {df.shape}")
print(f"\nMissing values:")
print(df[retained_vars].isnull().sum())
print(f"\nDuplicate rows: {df[retained_vars].duplicated().sum()}")
print(f"\nTarget distribution:")
print(df['Loan Approval Status'].value_counts())
```

---

## 🔬 Phase 4: Feature Engineering & Encoding

### **STEP 9: Encode Categorical Variables**

```python
# One-hot encode categorical features for modeling
categorical_features = [
    'Education Qualifications',
    'Home Ownership',
    'Loan Intent'
    # Note: 'Payment Default on File' already encoded as 0/1
]

# Create encoded features
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

print(f"After encoding: {df_encoded.shape[1]} features")
print(f"New feature names: {df_encoded.columns.tolist()}")
```

**Justification:** 
- One-hot encoding converts categories to numerical format
- `drop_first=True` prevents multicollinearity (dummy variable trap)

---

### **STEP 10: Prepare for Modeling**

```python
# Separate features and target
X = df_encoded.drop(columns=['Loan Approval Status', 'Maximum Loan Amount'])
y = df_encoded['Loan Approval Status']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"\nFeature columns: {X.columns.tolist()}")
```

---

## 📊 Phase 5: Scaling Strategy (Algorithm-Specific)

### **IMPORTANT: Apply Scaling ONLY to Specific Models**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize scaler
scaler = StandardScaler()

# FOR NAIVE BAYES & LOGISTIC REGRESSION ONLY:
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train distance-based models with scaled data
nb_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)

# FOR RANDOM FOREST: Use UNSCALED data
rf_model.fit(X_train, y_train)  # NO SCALING
```

### **Scaling Decision Matrix**

| Algorithm | Needs Scaling? | Scaler | Reason |
|-----------|---------------|---------|---------|
| **Naive Bayes** | ✅ Yes | StandardScaler | Assumes normal distribution |
| **Logistic Regression** | ✅ Yes | StandardScaler | Gradient-based optimization |
| **Random Forest** | ❌ No | None | Tree-based (split rules) |
| **Decision Trees** | ❌ No | None | Tree-based (split rules) |

**Why NOT Robust Scaler?**
- Your "outliers" (£1.9M income, high loan amounts) are **real data**, not errors
- They provide valuable predictive information
- StandardScaler preserves these relationships
- RobustScaler would compress important variance

---

## 📝 Complete Cleaning Summary Table for Report

| Variable | Issue | Mitigation | Justification |
|----------|-------|------------|---------------|
| **ID, Age, Sex, Credit Application** | Introduce bias/noise | Drop 4 columns | Not relevant to loan approval prediction |
| **Employment Length** | Typo + impossible values (150 years) | Rename + filter ≤50 years | Removes data entry errors |
| **Education Qualifications** | Trailing whitespace | `.str.strip()` | Merges duplicate categories |
| **Loan Intent** | No spaces (e.g., HOMEIMPROVEMENT) | Add spaces, title case | Improves readability |
| **Payment Default on File** | 4 variants (Y/N/YES/NO) + 5 missing | Standardize to 1/0, fill with mode | Binary encoding for models |
| **Loan Interest Rate** | 11 missing values (0.02%) | Median imputation | Robust to outliers |
| **Loan Approval Status** | 8 variants + 1 missing | Standardize to Approved/Declined, drop 1 row | Binary classification target |
| **Maximum Loan Amount** | 3 negative values (0.01%) | Drop rows | Data entry errors |
| **Categorical variables** | Need numerical format | One-hot encoding with `drop_first=True` | ML models require numerical input |
| **Numerical features** | Different scales | StandardScaler for NB/LR only | Distance-based algorithms require scaling |

**Total Data Loss:** 4-15 rows (0.007%-0.026% of 58,645 records) ✅ **Negligible impact**

---

## ✅ Final Checklist

- [x] Drop bias-inducing variables (ID, Age, Sex, Credit Application Acceptance)
- [x] Fix column name typo (emplyment → Employment)
- [x] Remove data entry errors (negative values, impossible employment length)
- [x] Clean categorical variables (whitespace, formatting)
- [x] Standardize target variable to 2 classes
- [x] Handle missing values (median for rates, mode for defaults)
- [x] Encode categorical features (one-hot encoding)
- [x] Apply scaling appropriately (only for NB/LR, not RF/DT)
- [x] Verify no missing values remain
- [x] Document all cleaning steps with justifications

---

## 📊 Expected Final Dataset

```
Before cleaning: 58,645 rows × 16 columns
After cleaning:  ~58,630 rows × 12 retained columns
After encoding:  ~58,630 rows × 20+ features (with one-hot encoding)

Target balance: ~85% Approved, ~15% Declined
Total data loss: <0.03% (excellent preservation)
```

---

## 🎯 Key Takeaways

1. **Minimal Data Loss:** Only losing ~15 rows (0.026%) ensures robust model training
2. **Bias Removal:** Dropping Age, Sex removes potential discrimination
3. **Quality Over Quantity:** Removing data errors improves model reliability
4. **Algorithm-Specific Approach:** Different models need different preprocessing
5. **Documentation:** Every decision justified with data-driven reasoning

---

## 📚 References for Report

- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

---

**Document Created:** November 6, 2025  
**Project:** Loan Approval Coursework - Machine Learning Analysis
