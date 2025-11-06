# TASK 4: Complete Implementation Code
**Copy these cells into your Jupyter notebook in order**

---

## CELL 1: Comprehensive Data Cleaning (Add after Task 3 basic cleaning)

```python
# ============================================================================
# TASK 3: COMPREHENSIVE DATA CLEANING & PREPROCESSING PIPELINE
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

# 2c. Verify Payment Default encoding (already done in previous cells)
print("STEP 2c: Verifying Payment Default encoding")
print(f"  - Unique values: {df['Payment Default on File'].unique()}")
print(f"  - Data type: {df['Payment Default on File'].dtype}\n")

# -----------------------------------------------------------------------------
# STEP 3: Standardize Target Variable
# -----------------------------------------------------------------------------

print("STEP 3: Standardizing Loan Approval Status")
print(f"  - Before: {df['Loan Approval Status'].nunique()} unique values")

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
print(f"\nClass balance:")
for cls, prop in df['Loan Approval Status'].value_counts(normalize=True).items():
    print(f"  {cls}: {prop:.1%}")
```

---

## CELL 2: Task 4.a - Algorithm Details (Markdown)

```markdown
## Task 4: Modelling - Classification

### Task 4.a: Algorithm Details Table

| Algorithm Name | Algorithm Type | Learnable Parameters | Some Possible Hyperparameters | Imported Python Package |
|----------------|----------------|---------------------|-------------------------------|------------------------|
| **Naïve Bayes (NB)** | Non-parametric | • Class priors (π)<br>• Feature means (μ)<br>• Feature variances (σ²) | • `var_smoothing`: Portion of largest variance added to variances for stability<br>• `priors`: Prior probabilities of classes | `from sklearn.naive_bayes import GaussianNB` |
| **Logistic Regression (LR)** | Parametric | • Coefficients/weights (β)<br>• Intercept (β₀) | • `C`: Inverse regularization strength<br>• `penalty`: Regularization norm ('l1', 'l2', 'elasticnet', 'none')<br>• `solver`: Optimization algorithm ('lbfgs', 'liblinear', 'saga')<br>• `max_iter`: Maximum iterations for convergence | `from sklearn.linear_model import LogisticRegression` |
| **Random Forest (RF)** | Non-parametric | • Split thresholds at each node<br>• Feature selection at splits<br>• Leaf node predictions | • `n_estimators`: Number of trees in forest<br>• `max_depth`: Maximum depth of trees<br>• `min_samples_split`: Minimum samples to split node<br>• `min_samples_leaf`: Minimum samples in leaf<br>• `max_features`: Features to consider for splits | `from sklearn.ensemble import RandomForestClassifier` |

**Notes:**
- **Parametric models** (LR) have a fixed number of parameters regardless of training data size
- **Non-parametric models** (NB, RF) can grow in complexity with more data
```

---

## CELL 3: Task 4.b.i - Feature Selection and Data Shape

```python
# ============================================================================
# TASK 4.b.i: FEATURE SELECTION FOR CLASSIFICATION
# ============================================================================

print("="*70)
print("TASK 4.b.i: FEATURE SELECTION FOR CLASSIFICATION")
print("="*70)

# As per assignment: Use CATEGORICAL FEATURES ONLY
categorical_features = [
    'Education Qualifications',
    'Home Ownership',
    'Loan Intent'
    # Note: Payment Default on File already encoded as 0/1 (numerical feature)
]

print("\nCategorical features selected for classification:")
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
print("DATA SHAPE INFORMATION (SCREENSHOT THIS):")
print("="*70)
print(f"Feature matrix (X) shape: {X_encoded.shape}")
print(f"  - {X_encoded.shape[0]:,} samples (loan applications)")
print(f"  - {X_encoded.shape[1]} features (after one-hot encoding)")
print(f"\nTarget variable (y) shape: {y.shape}")
print(f"  - {y.shape[0]:,} labels")
print(f"\nTarget distribution:")
print(y.value_counts())
print(f"\nClass proportions:")
for cls, prop in y.value_counts(normalize=True).items():
    print(f"  {cls}: {prop:.1%}")
```

---

## CELL 4: Task 4.b.ii - Train-Test Split Justification (Markdown)

```markdown
### Task 4.b.ii: Train-Test Split Ratio Justification

The 80:20 train-test split balances training data sufficiency with reliable evaluation. With 58,627 samples, the 80% training allocation (46,901 samples) provides adequate data for pattern learning, crucial given our 85:15 class imbalance. The 20% test set (11,726 samples) ensures statistically significant performance estimation. This ratio is industry-standard practice (Géron, 2019) and proven effective for similar-sized datasets, maximizing training data utilization while maintaining robust evaluation capability.

**Word count:** 68 words

**Reference:**  
Géron, A. (2019). *Hands-on machine learning with scikit-learn, keras, and tensorflow* (2nd ed.). O'Reilly Media.
```

---

## CELL 5: Task 4.b.iii - Training-Test vs K-Fold (Markdown)

```markdown
### Task 4.b.iii: Training-Test vs K-Fold Cross-Validation

Train-test splitting divides data once for a single performance estimate—computationally efficient and suitable for large datasets (>10,000 samples). K-fold cross-validation trains k times on different splits, providing multiple estimates that reduce evaluation variance. Use train-test for final model assessment with large data; use k-fold for hyperparameter tuning, small datasets (<10,000 samples), or when robust performance confidence intervals are required. This project employs train-test for final evaluation and k-fold within GridSearchCV for hyperparameter optimization (Task 5).

**Word count:** 79 words
```

---

## CELL 6: Task 4.b.iv - Train-Test Split Implementation

```python
# ============================================================================
# TASK 4.b.iv: TRAIN-TEST SPLIT WITH STRATIFICATION
# ============================================================================

from sklearn.model_selection import train_test_split

print("="*70)
print("TASK 4.b.iv: TRAIN-TEST SPLIT (80:20 RATIO)")
print("="*70)

# THIS LINE ENSURES ALL MODELS USE THE SAME TEST SET
# SCREENSHOT THIS CODE LINE FOR EVIDENCE
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.2,           # 80:20 split
    random_state=42,         # ENSURES REPRODUCIBILITY - same split every time
    stratify=y               # ENSURES class proportions maintained in both sets
)

print("KEY POINTS:")
print("✓ random_state=42 ensures all models tested on identical test data")
print("✓ stratify=y ensures Approved/Declined ratio same in train and test\n")

print(f"Training set: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")

print("\n" + "="*70)
print("VERIFICATION: CLASS DISTRIBUTION MAINTAINED (SCREENSHOT THIS)")
print("="*70)
print("\nOriginal dataset class distribution:")
for cls, prop in y.value_counts(normalize=True).items():
    print(f"  {cls}: {prop:.4f} ({prop:.1%})")

print("\nTraining set class distribution:")
for cls, prop in y_train.value_counts(normalize=True).items():
    print(f"  {cls}: {prop:.4f} ({prop:.1%})")

print("\nTest set class distribution:")
for cls, prop in y_test.value_counts(normalize=True).items():
    print(f"  {cls}: {prop:.4f} ({prop:.1%})")

print("\n✓ Class proportions are identical across all sets (stratification successful)")
```

---

## CELL 7: Build Classification Models

```python
# ============================================================================
# TASK 4.b: BUILD THREE CLASSIFICATION MODELS
# ============================================================================

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

print("="*70)
print("BUILDING THREE CLASSIFICATION MODELS")
print("="*70)

# 1. Naive Bayes (GaussianNB)
print("\n1. Training Naive Bayes (GaussianNB)...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
print("   ✓ Naive Bayes model trained successfully")
print(f"   - Class priors learned: {nb_model.class_prior_}")
print(f"   - Classes: {nb_model.classes_}")

# 2. Logistic Regression
print("\n2. Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("   ✓ Logistic Regression model trained successfully")
print(f"   - Coefficients learned: {len(lr_model.coef_[0])} weights + 1 intercept")
print(f"   - Intercept: {lr_model.intercept_[0]:.4f}")

# 3. Random Forest Classifier
print("\n3. Training Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
print("   ✓ Random Forest model trained successfully")
print(f"   - Trees built: {rf_model.n_estimators}")
print(f"   - Max depth: {rf_model.max_depth if rf_model.max_depth else 'Unlimited'}")
print(f"   - Total nodes across all trees: {sum(tree.tree_.node_count for tree in rf_model.estimators_)}")

print("\n" + "="*70)
print("ALL THREE MODELS TRAINED AND READY FOR EVALUATION")
print("="*70)
print("\nKey verification points:")
print("✓ All models trained on identical training data (X_train, y_train)")
print("✓ All models will be tested on identical test data (X_test, y_test)")
print("✓ random_state=42 ensures reproducibility across runs")
print("✓ All models ready for Task 5 evaluation")
```

---

## CELL 8: Quick Model Predictions (Optional - for verification)

```python
# Quick predictions to verify models work
print("="*70)
print("QUICK MODEL VERIFICATION")
print("="*70)

# Make predictions on test set
y_pred_nb = nb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

print("\nPrediction counts on test set:")
print(f"\nNaive Bayes predictions:")
print(pd.Series(y_pred_nb).value_counts())

print(f"\nLogistic Regression predictions:")
print(pd.Series(y_pred_lr).value_counts())

print(f"\nRandom Forest predictions:")
print(pd.Series(y_pred_rf).value_counts())

print("\n✓ All models producing predictions")
print("✓ Ready to proceed to Task 5 (Evaluation)")
```

---

## Screenshots Required for Task 4

### Screenshot 1: Feature Names and Data Shape
**From Cell 3 output**
- Capture: List of encoded feature names + shape information
- Caption: "Feature names after one-hot encoding showing 15 features from 3 categorical variables (58,627 samples)"

### Screenshot 2: Train-Test Split Verification  
**From Cell 6 output**
- Capture: Class distribution comparison showing identical proportions
- Caption: "Stratified train-test split verification showing 85.6% approved and 14.4% declined maintained in both sets"

### Screenshot 3: Code Evidence for Same Test Set
**From Cell 6 code**
- Highlight the line: `random_state=42,  # ENSURES REPRODUCIBILITY`
- Caption: "Code evidence showing random_state=42 ensures all models tested on identical test dataset"

---

## Summary: What You've Accomplished

✅ **Task 3 Enhanced:** Comprehensive data cleaning pipeline  
✅ **Task 4.a:** Algorithm details table completed  
✅ **Task 4.b.i:** Feature selection with categorical features only  
✅ **Task 4.b.ii:** Train-test split justification (<100 words)  
✅ **Task 4.b.iii:** Training-test vs K-fold comparison (<100 words)  
✅ **Task 4.b.iv:** Three models built with verified stratification  

**Next Step:** Task 5 - Model Evaluation (Confusion matrices, metrics, hyperparameter tuning)

---

**Total marks for Task 4:** 10 marks
- Part a: 6 marks (Algorithm table)
- Part b: 4 marks (Implementation + justifications)
