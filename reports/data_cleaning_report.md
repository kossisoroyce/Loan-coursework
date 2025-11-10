# Data Cleaning Report - Loan Approval Coursework

## Executive Summary

Data cleaning has been successfully completed for the loan approval dataset. The cleaned dataset is now ready for machine learning model training.

## Cleaning Actions Performed

### 1. Column Removal (Task 1 Decisions)

The following columns were **DROPPED** as per Task 1 analysis:

| Column | Justification |
|--------|---------------|
| **ID** | Not relevant to target variable, may cause noise |
| **Sex** | Does not contain information relevant to status prediction |
| **Age** | Constant in most cases (age 40), minimal variation to learn from |
| **Credit Application Acceptance** | Duplicate of Loan Approval Status, introduces noise |

**Result:** Dataset reduced from 16 columns to 12 columns

### 2. Columns Retained (Task 1 Decisions)

The following columns were **RETAINED** for modeling:

| Column | Type | Justification |
|--------|------|---------------|
| **Education Qualifications** | Categorical | Predicts social and financial stability |
| **Income** | Numerical | Aids in predicting loan-to-income ratio |
| **Home Ownership** | Categorical | Determines collateral and monthly liabilities |
| **Employment Length** | Numerical | Indicates income stability |
| **Loan Intent** | Categorical | Determines potential return |
| **Loan Amount** | Numerical | Central variable for loan approval status |
| **Loan Interest Rate** | Numerical | Necessary for profit prediction |
| **Loan-to-Income Ratio (LTI)** | Numerical | Predicts loan burden on income |
| **Payment Default on File** | Binary | Predicts probability of loan repayment |
| **Credit History Length** | Numerical | Predicts credit and risk behavior |
| **Loan Approval Status** | Categorical (Target) | Output variable for classification |
| **Maximum Loan Amount** | Numerical | Target for regression modeling |

### 3. Data Quality Issues Resolved (Task 3)

#### Issue 1: Education Qualifications - Whitespace

**Problem:** Values had trailing/leading whitespace (e.g., "Unknown" vs "Unknown  ")

**Mitigation:** Removed all whitespace using `.str.strip()`

**Justification:** Prevents duplication of class labels

#### Issue 2: Loan Intent - Formatting

**Problem:** No spaces between words in some categories (e.g., "HOMEIMPROVEMENT", "DEBTCONSOLIDATION")

**Mitigation:** 
- Replaced "HOMEIMPROVEMENT" → "Home Improvement"
- Replaced "DEBTCONSOLIDATION" → "Debt Consolidation"
- Applied `.str.title()` for consistent capitalization

**Justification:** Improves readability and consistency

#### Issue 3: Payment Default on File - Encoding

**Problem:** Text values "Y" and "N" need to be encoded for modeling

**Mitigation:** Encoded as binary (Y=1, N=0)

**Justification:** Simple and interpretable for machine learning algorithms

#### Issue 4: Loan Approval Status - Inconsistency

**Problem:** 
- Inconsistent values: "Approved", "Declined", "Reject", "Accept", "APPROVED", "DECLINED"
- 1 row with missing value

**Mitigation:**
- Standardized all variations to "Approved" or "Declined"
- Dropped 1 row with missing value

**Justification:** Ensures consistent target variable labels

#### Issue 5: Maximum Loan Amount - Negative Values

**Problem:** 
- 3 rows with negative loan amounts
- 60 values above £1,000,000

**Mitigation:**
- Dropped 3 rows with negative values
- Capped values at £500,000

**Justification:** 
- Negative values introduce noise and are incorrect
- Capping large outliers prevents model from learning unrealistic patterns

#### Issue 6: Missing Values

**Problem:** 
- 11 missing values in Loan Interest Rate
- 5 missing values in Payment Default on File
- 1 missing value in Loan Approval Status

**Mitigation:**
- Numerical features: Imputed with median (robust to outliers)
- Categorical features: Imputed with mode (most frequent value)
- Dropped row with missing target variable

**Justification:** 
- Median is resistant to outliers (better than mean)
- Mode preserves the most common pattern in categorical data
- Cannot train on rows with missing target

### 4. Scaling Considerations

**Decision:** Scaling will be applied **during modeling** as needed

**Approach:** RobustScaler (uses median and IQR)

**Justification:**
- Resistant to outliers in Income, Loan Amount, Employment Length
- However, for classification with categorical features, scaling is not required
- For Decision Tree regression, scaling is not required (scale-invariant)
- Only needed if distance-based algorithms are used (KNN, SVM, Neural Networks)

## Final Dataset Statistics

### Dataset Dimensions
- **Initial shape:** (58,645 rows × 16 columns)
- **Final shape:** (58,641 rows × 12 columns)
- **Rows dropped:** 4 (3 negative Maximum Loan Amount + 1 missing target)

### Data Quality Metrics
- **Missing values:** 0 ✓
- **Duplicate rows:** 4 (acceptable - different customers may have identical features)

### Class Distribution (Loan Approval Status)
- **Approved:** 50,291 (85.8%)
- **Declined:** 8,350 (14.2%)

**Note:** Class imbalance exists - will use stratified train-test split

## Files Generated

1. **Cleaned Dataset:** `data/loan_approval_data_cleaned.csv`
2. **Cleaning Script:** `scripts/data_cleaning.py`
3. **This Report:** `reports/data_cleaning_report.md`

## Next Steps

### For Classification (Part A)
1. Select categorical features for one-hot encoding
2. Use stratified train-test split (80:20)
3. Train models: Naive Bayes, Logistic Regression, Random Forest
4. Evaluate with focus on Recall and Precision for "Declined" class

### For Regression (Part B)
1. Filter dataset for "Approved" loans only
2. Select numerical features for Model 1
3. Select all features for Model 2
4. Train Decision Tree regressors
5. Evaluate with R² score

## Conclusion

The dataset has been thoroughly cleaned according to the specifications in Task 1 and Task 3. All data quality issues have been resolved, and the dataset is now ready for machine learning model development.

**Status:** ✓ READY FOR MODELING

---

*Generated: Data Cleaning Phase Complete*
*Next Phase: Model Training and Evaluation*
