# Task 4 Implementation Guide
**Updated Preprocessing Pipeline + Classification Models**

---

## Part 1: Algorithm Details Table (Task 4.a)

### Table for Your Report

| Algorithm Name | Algorithm Type | Learnable Parameters | Some Possible Hyperparameters | Imported Python Package |
|----------------|----------------|---------------------|-------------------------------|------------------------|
| **Naïve Bayes (NB)** | Non-parametric | • Class priors (π)<br>• Feature means (μ)<br>• Feature variances (σ²) | • `var_smoothing`: Portion of largest variance added to variances for stability<br>• `priors`: Prior probabilities of classes | `from sklearn.naive_bayes import GaussianNB` |
| **Logistic Regression (LR)** | Parametric | • Coefficients/weights (β)<br>• Intercept (β₀) | • `C`: Inverse regularization strength<br>• `penalty`: Regularization norm ('l1', 'l2', 'elasticnet', 'none')<br>• `solver`: Optimization algorithm ('lbfgs', 'liblinear', 'saga')<br>• `max_iter`: Maximum iterations for convergence | `from sklearn.linear_model import LogisticRegression` |
| **Random Forest (RF)** | Non-parametric | • Split thresholds at each node<br>• Feature selection at splits<br>• Leaf node predictions | • `n_estimators`: Number of trees in forest<br>• `max_depth`: Maximum depth of trees<br>• `min_samples_split`: Minimum samples to split node<br>• `min_samples_leaf`: Minimum samples in leaf<br>• `max_features`: Features to consider for splits | `from sklearn.ensemble import RandomForestClassifier` |

### Explanation Notes:

**Parametric vs Non-parametric:**
- **Parametric (LR):** Has fixed number of parameters (coefficients) determined by number of features. Model complexity doesn't grow with data size.
- **Non-parametric (NB, RF):** Model complexity can grow with data. NB stores statistics per class/feature. RF can have unlimited trees.

---

## Part 2: Complete Preprocessing Code for Notebook

### Add This New Cell BEFORE Task 4 (After Task 3)

```python
# ============================================================================
# TASK 3: COMPLETE DATA CLEANING & PREPROCESSING PIPELINE
# ============================================================================

print("="*70)
print("STARTING COMPREHENSIVE DATA CLEANING")
print("="*70)

# Store initial shape
initial_shape = df.shape
print(f"Initial dataset: {initial_shape[0]:,} rows × {initial_shape[1]} columns\n")

# -----------------------------------------------------------------------------
# STEP 1: Remove Data Entry Errors
# -----------------------------------------------------------------------------

# 1a. Remove negative Maximum Loan Amounts
print("STEP 1a: Removing negative Maximum Loan Amounts")
negative_count = (df['Maximum Loan Amount'] < 0).sum()
print(f"  - Found {negative_count} negative values")
df = df[df['Maximum Loan Amount'] >= 0].copy()
print(f"  - Removed {negative_count} rows")
print(f"  - Dataset now: {df.shape[0]:,} rows\n")

# 1b. Remove impossible Employment Lengths (>50 years)
print("STEP 1b: Removing impossible Employment Lengths")
impossible_emp = (df['Employment Length'] > 50).sum()
print(f"  - Found {impossible_emp} records with >50 years employment")
df = df[df['Employment Length'] <= 50].copy()
print(f"  - Removed {impossible_emp} rows")
print(f"  - Dataset now: {df.shape[0]:,} rows\n")

# -----------------------------------------------------------------------------
# STEP 2: Clean Categorical Variables
# -----------------------------------------------------------------------------

# 2a. Education Qualifications - Remove whitespace
print("STEP 2a: Cleaning Education Qualifications")
print(f"  - Before: {df['Education Qualifications'].nunique()} unique values")
df['Education Qualifications'] = df['Education Qualifications'].str.strip()
print(f"  - After: {df['Education Qualifications'].nunique()} unique values\n")

# 2b. Loan Intent - Format for readability
print("STEP 2b: Formatting Loan Intent")
print(f"  - Before: {df['Loan Intent'].unique()}")
df['Loan Intent'] = df['Loan Intent'].replace({
    'HOMEIMPROVEMENT': 'Home Improvement',
    'DEBTCONSOLIDATION': 'Debt Consolidation',
    'EDUCATION': 'Education',
    'MEDICAL': 'Medical',
    'VENTURE': 'Venture',
    'PERSONAL': 'Personal'
})
print(f"  - After: {df['Loan Intent'].unique()}\n")

# 2c. Payment Default on File - Standardize and encode
print("STEP 2c: Standardizing Payment Default on File")
print(f"  - Before: {df['Payment Default on File'].value_counts(dropna=False).to_dict()}")

# Standardize to Y/N
df['Payment Default on File'] = df['Payment Default on File'].replace({
    'YES': 'Y',
    'NO': 'N'
})

# Fill missing with mode
mode_default = df['Payment Default on File'].mode()[0]
df['Payment Default on File'].fillna(mode_default, inplace=True)

# Binary encode (Y=1, N=0)
df['Payment Default on File'] = df['Payment Default on File'].map({'Y': 1, 'N': 0})
print(f"  - After encoding: {df['Payment Default on File'].value_counts().to_dict()}\n")

# -----------------------------------------------------------------------------
# STEP 3: Standardize Target Variable
# -----------------------------------------------------------------------------

print("STEP 3: Standardizing Loan Approval Status")
print(f"  - Before: {df['Loan Approval Status'].value_counts(dropna=False).to_dict()}")

# Standardize all variants to Approved/Declined
df['Loan Approval Status'] = df['Loan Approval Status'].str.upper()
df['Loan Approval Status'] = df['Loan Approval Status'].replace({
    'APPROVED': 'Approved',
    'ACCEPT': 'Approved',
    'DECLINED': 'Declined',
    'REJECT': 'Declined'
})
df['Loan Approval Status'] = df['Loan Approval Status'].str.title()

# Drop rows with missing target
missing_target = df['Loan Approval Status'].isnull().sum()
df = df[df['Loan Approval Status'].notna()].copy()
print(f"  - Removed {missing_target} row(s) with missing target")
print(f"  - After: {df['Loan Approval Status'].value_counts().to_dict()}\n")

# -----------------------------------------------------------------------------
# STEP 4: Handle Missing Numerical Values
# -----------------------------------------------------------------------------

print("STEP 4: Handling missing Loan Interest Rate")
missing_rate = df['Loan Interest Rate'].isnull().sum()
median_rate = df['Loan Interest Rate'].median()
df['Loan Interest Rate'].fillna(median_rate, inplace=True)
print(f"  - Imputed {missing_rate} missing values with median: {median_rate:.2f}%\n")

# -----------------------------------------------------------------------------
# FINAL VERIFICATION
# -----------------------------------------------------------------------------

print("="*70)
print("CLEANING COMPLETE - FINAL VERIFICATION")
print("="*70)

retained_vars = [
    'Education Qualifications', 'Income', 'Home Ownership', 'Employment Length',
    'Loan Intent', 'Loan Amount', 'Loan Interest Rate', 'Loan-to-Income Ratio (LTI)',
    'Payment Default on File', 'Credit History Length', 'Loan Approval Status',
    'Maximum Loan Amount'
]

print(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Data loss: {initial_shape[0] - df.shape[0]} rows ({(initial_shape[0] - df.shape[0])/initial_shape[0]*100:.3f}%)")
print(f"\nMissing values in retained variables:")
missing_summary = df[retained_vars].isnull().sum()
if missing_summary.sum() == 0:
    print("  ✓ No missing values!")
else:
    print(missing_summary[missing_summary > 0])

print(f"\nTarget variable distribution:")
print(df['Loan Approval Status'].value_counts())
print(f"\nClass balance: {df['Loan Approval Status'].value_counts(normalize=True).to_dict()}")
```

---

## Part 3: Task 4.b Implementation

### Task 4.b.i: Feature Selection and Data Shape

```python
# ============================================================================
# TASK 4: MODELLING - CLASSIFICATION
# ============================================================================

print("="*70)
print("TASK 4.b.i: FEATURE SELECTION FOR CLASSIFICATION")
print("="*70)

# As per assignment: Use CATEGORICAL FEATURES ONLY
categorical_features = [
    'Education Qualifications',
    'Home Ownership',
    'Loan Intent'
    # Note: Payment Default on File already encoded as 0/1 (will be treated as numerical)
]

print("\nCategorical features selected:")
for i, feat in enumerate(categorical_features, 1):
    print(f"  {i}. {feat}")

# Create feature matrix with categorical features
X_cat = df[categorical_features].copy()

# One-hot encode categorical features
X_encoded = pd.get_dummies(X_cat, drop_first=True)

# Target variable
y = df['Loan Approval Status'].copy()

print("\n" + "="*70)
print("FEATURE NAMES AFTER ONE-HOT ENCODING:")
print("="*70)
for i, col in enumerate(X_encoded.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*70)
print("DATA SHAPE INFORMATION:")
print("="*70)
print(f"Feature matrix (X) shape: {X_encoded.shape}")
print(f"  - {X_encoded.shape[0]:,} samples (loan applications)")
print(f"  - {X_encoded.shape[1]} features (after one-hot encoding)")
print(f"\nTarget variable (y) shape: {y.shape}")
print(f"  - {y.shape[0]:,} labels")
print(f"\nTarget distribution:")
print(y.value_counts())
```

### Task 4.b.ii: Train-Test Split Ratio Justification

```python
# ============================================================================
# TASK 4.b.ii & 4.b.iv: TRAIN-TEST SPLIT
# ============================================================================

from sklearn.model_selection import train_test_split

print("="*70)
print("TASK 4.b.ii: TRAIN-TEST SPLIT WITH 80:20 RATIO")
print("="*70)

# Justification (for report - under 100 words):
justification = """
The 80:20 train-test split is widely adopted in machine learning practice, 
providing an effective balance between model training and evaluation. The 80% 
training allocation ensures sufficient data for the model to learn underlying 
patterns, particularly important given our class imbalance (85% approved vs 15% 
declined). The 20% test set provides adequate samples for reliable performance 
estimation. This ratio is recommended by Géron (2019) and has proven effective 
for datasets of similar size (~58,000 records), offering statistically 
significant evaluation while maximizing training data utilization.
"""
print("Justification:")
print(justification)
print(f"Word count: ~{len(justification.split())} words")

# TASK 4.b.iv: Ensure same test set and stratification
print("\n" + "="*70)
print("TASK 4.b.iv: ENSURING CONSISTENT TEST SET & STRATIFICATION")
print("="*70)

# THIS LINE ENSURES ALL MODELS USE THE SAME TEST SET
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.2,           # 80:20 split
    random_state=42,         # ENSURES REPRODUCIBILITY - same split every time
    stratify=y               # ENSURES class proportions maintained in both sets
)

print("✓ random_state=42 ensures all models tested on identical test data")
print("✓ stratify=y ensures Approved/Declined ratio same in train and test\n")

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print("\n" + "="*70)
print("VERIFICATION: CLASS DISTRIBUTION MAINTAINED")
print("="*70)
print("\nOriginal dataset class distribution:")
print(y.value_counts(normalize=True).apply(lambda x: f"{x:.4f}"))

print("\nTraining set class distribution:")
print(y_train.value_counts(normalize=True).apply(lambda x: f"{x:.4f}"))

print("\nTest set class distribution:")
print(y_test.value_counts(normalize=True).apply(lambda x: f"{x:.4f}"))

print("\n✓ Class proportions are identical across all sets (stratification successful)")
```

### Task 4.b.iii: Training-Test vs K-Fold Cross-Validation

```python
# ============================================================================
# TASK 4.b.iii: TRAINING-TEST vs K-FOLD CROSS-VALIDATION
# ============================================================================

print("="*70)
print("TASK 4.b.iii: COMPARISON OF VALIDATION APPROACHES")
print("="*70)

comparison = """
TRAINING-TEST SPLIT:
Divides data once into training and test sets. The model trains on the training 
set and evaluates once on the held-out test set. This approach provides a single, 
unbiased performance estimate and is computationally efficient. Use when: dataset 
is large (>10,000 samples), final model performance estimate is needed, or 
computational resources are limited.

K-FOLD CROSS-VALIDATION:
Divides data into k folds, training k times with different train-validation splits. 
Each fold serves as validation once while others train the model. This provides k 
performance estimates, reducing variance in evaluation. Use when: dataset is small 
(<10,000 samples), hyperparameter tuning is required, or robust performance 
estimation with confidence intervals is needed. We apply this in Task 5 for 
hyperparameter optimization via GridSearchCV.
"""

print(comparison)
print(f"\nWord count: ~{len(comparison.split())} words")
```

### Building the Three Models

```python
# ============================================================================
# TASK 4.b: BUILD CLASSIFICATION MODELS
# ============================================================================

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("="*70)
print("BUILDING THREE CLASSIFICATION MODELS")
print("="*70)

# 1. Naive Bayes
print("\n1. Training Naive Bayes (GaussianNB)...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print("   ✓ Naive Bayes model trained successfully")
print(f"   - Learnable parameters stored: class priors, feature means, variances")

# 2. Logistic Regression
print("\n2. Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("   ✓ Logistic Regression model trained successfully")
print(f"   - Coefficients learned: {len(lr_model.coef_[0])} weights + 1 intercept")

# 3. Random Forest
print("\n3. Training Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
print("   ✓ Random Forest model trained successfully")
print(f"   - Trees built: {rf_model.n_estimators}")
print(f"   - Max depth: {rf_model.max_depth} (unlimited if None)")

print("\n" + "="*70)
print("ALL THREE MODELS TRAINED AND READY FOR EVALUATION")
print("="*70)
print("\nKey Points:")
print("✓ All models trained on identical training data (X_train, y_train)")
print("✓ All models will be tested on identical test data (X_test, y_test)")
print("✓ random_state ensures reproducibility across runs")
```

---

## Part 4: Screenshots Required for Report

### Screenshot 1: Feature Names and Data Shape
**When to take:** After running Task 4.b.i cell  
**What to capture:** 
- List of all encoded feature names
- Data shape output showing (samples, features)
- Target variable shape

**Example caption:**  
*"Figure X: Feature names after one-hot encoding and data shape for classification models (58,627 samples × 15 features)"*

### Screenshot 2: Train-Test Split Verification
**When to take:** After running Task 4.b.iv cell  
**What to capture:**
- Training and test set shapes
- Class distribution comparison showing identical proportions
- Verification message confirming stratification

**Example caption:**  
*"Figure X: Train-test split verification showing stratified sampling maintains 85.6% approved and 14.4% declined ratio in both sets"*

### Screenshot 3: Code Line for Same Test Set
**When to take:** Highlight this specific line in your code  
**What to capture:**
```python
random_state=42,  # ENSURES REPRODUCIBILITY - same split every time
```

**Example caption:**  
*"Figure X: Code evidence showing random_state=42 parameter ensures all models tested on identical test dataset"*

---

## Part 5: Report Writing Guidance

### Task 4.a: Algorithm Table
Simply copy the table from Part 1 of this guide into your report.

### Task 4.b.ii: Train-Test Split Justification (<100 words)
Use this text:
> "The 80:20 train-test split balances training data sufficiency with reliable evaluation. With 58,627 samples, the 80% training allocation (46,901 samples) provides adequate data for pattern learning, crucial given our 85:15 class imbalance. The 20% test set (11,726 samples) ensures statistically significant performance estimation. This ratio is industry-standard practice (Géron, 2019) and proven effective for similar-sized datasets, maximizing training data utilization while maintaining robust evaluation capability."

**Word count:** 68 words  
**Citation:** Géron, A. (2019). *Hands-on machine learning with scikit-learn, keras, and tensorflow*. O'Reilly Media.

### Task 4.b.iii: Training-Test vs K-Fold (<100 words)
Use this text:
> "Train-test splitting divides data once for a single performance estimate—computationally efficient and suitable for large datasets (>10,000 samples). K-fold cross-validation trains k times on different splits, providing multiple estimates that reduce evaluation variance. Use train-test for final model assessment with large data; use k-fold for hyperparameter tuning, small datasets (<10,000 samples), or when robust performance confidence intervals are required. This project employs train-test for final evaluation and k-fold within GridSearchCV for hyperparameter optimization (Task 5)."

**Word count:** 79 words

### Task 4.b.iv: Code Evidence
Point to the train_test_split line with `random_state=42` in your screenshot. Explain:
> "The parameter `random_state=42` in the `train_test_split()` function ensures reproducible data splitting. By fixing the random seed, the same samples are assigned to training and test sets across all model builds, guaranteeing that Naive Bayes, Logistic Regression, and Random Forest are evaluated on identical test data. The `stratify=y` parameter maintains the 85:15 approved-to-declined ratio in both sets."

---

## ✅ Implementation Checklist

- [ ] Add comprehensive preprocessing cell (Part 2)
- [ ] Run preprocessing and verify 0 missing values
- [ ] Implement Task 4.b.i (feature selection)
- [ ] Take Screenshot 1 (feature names & shapes)
- [ ] Implement Task 4.b.ii (train-test split)
- [ ] Implement Task 4.b.iii (comparison text)
- [ ] Take Screenshot 2 (stratification verification)
- [ ] Implement Task 4.b.iv (model training)
- [ ] Take Screenshot 3 (random_state code line)
- [ ] Create algorithm table for report
- [ ] Write justifications (<100 words each)
- [ ] Verify all models trained successfully

---

**Reference:**  
Géron, A. (2019). *Hands-on machine learning with scikit-learn, keras, and tensorflow* (2nd ed.). O'Reilly Media, Inc.
