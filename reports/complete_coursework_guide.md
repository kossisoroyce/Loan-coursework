# Complete Coursework Guide - Loan Approval Analysis

This guide helps you complete the report using outputs from the notebook.

---

## PART A: CLASSIFICATION - LOAN APPROVAL STATUS PREDICTION

### Task 1: Domain Understanding (6 marks)

Use the table you already have in the coursework description. Your justifications are already listed.

---

### Task 2: Data Understanding (3 marks)

**Instructions:**
1. Run cells in the notebook up to "Task 2: Data Understanding"
2. Screenshot the output of `df_retained.describe(include='all')`
3. Screenshot the bar chart showing Loan Approval Status distribution
4. Use the guide in `reports/task2_data_understanding.md`

---

### Task 3: Data Preparation (16 marks)

#### Task 3.a: Issues and Mitigations Table (8 marks)

| Variable Name | Issue Description | Proposed Mitigation | Justification |
|---------------|-------------------|---------------------|---------------|
| Loan Approval Status | Multiple spellings (Approved, APPROVED, Accept, Declined, DECLINED, Reject) and 1 NaN | Standardize to 'Approved' and 'Declined'; drop NaN rows | Ensures consistency and prevents model confusion from duplicate categories |
| Payment Default on File | Inconsistent values (Y, N, YES, NO) and 5 missing values | Standardize to 'Y' and 'N'; impute missing with mode | Binary standardization required for encoding; mode imputation preserves class distribution |
| Loan Interest Rate | 11 missing values | Impute with median | Median is robust to outliers in financial data; maintains distribution shape |

#### Task 3.b: Evidence of Implementation (8 marks)

**Instructions:**
1. Run the cleaning cells in the notebook
2. Take "before" and "after" screenshots for each issue:
   - Loan Approval Status: Before/after value counts
   - Payment Default on File: Before/after value counts
   - Loan Interest Rate: Before/after missing count
3. Annotate each screenshot clearly (e.g., "Figure 1: Before cleaning - Loan Approval Status")

---

### Task 4: Modelling (10 marks)

#### Task 4.a: Algorithm Details Table (6 marks)

| Algorithm Name | Algorithm Type | Learnable Parameters | Possible Hyperparameters | Python Package |
|----------------|----------------|----------------------|--------------------------|----------------|
| NB (Naive Bayes) | Non-parametric | Class priors, feature means, feature variances | var_smoothing | sklearn.naive_bayes.GaussianNB |
| LR (Logistic Regression) | Parametric | Coefficients (weights), intercept | C, penalty, solver, max_iter | sklearn.linear_model.LogisticRegression |
| RF (Random Forest) | Non-parametric | Split rules at each node | n_estimators, max_depth, min_samples_split, min_samples_leaf | sklearn.ensemble.RandomForestClassifier |

#### Task 4.b: Training Setup (4 marks)

**i. Feature names screenshot:**
Run the cell that prints feature names and shape. Screenshot the output.

**ii. Train-test split justification (< 100 words):**
"An 80:20 split is widely recommended for moderate datasets as it provides sufficient training data (80%) for the model to learn patterns while retaining enough test data (20%) for reliable performance evaluation (Géron, 2019). This ratio balances model learning capacity with evaluation robustness."

**iii. Training-test vs K-fold cross-validation (< 100 words):**
"Training-test split is simpler and faster, suitable when you have sufficient data and want a single holdout evaluation. K-fold cross-validation divides data into K subsets, training K times and averaging results, providing more robust estimates but requiring more computation. Use training-test for initial development and large datasets; use K-fold when data is limited or when you need confidence intervals on performance metrics."

**iv. Code evidence:**
Screenshot the line: `random_state=42, stratify=y`
- `random_state=42` ensures reproducibility (same split every run)
- `stratify=y` ensures same class proportions in train and test

---

### Task 5: Evaluation (30 marks)

#### Task 5.a: Confusion Matrices (3 marks)

**Instructions:**
Screenshot the three confusion matrices from notebook outputs:
- Naive Bayes confusion matrix
- Logistic Regression confusion matrix
- Random Forest confusion matrix

#### Task 5.b: Performance Metrics (15 marks)

| Metrics | USE or DO NOT USE | Justification | Model Name | Test Score |
|---------|-------------------|---------------|------------|------------|
| Accuracy | DO NOT USE | With 85.6% approved and 14.4% declined, a model predicting "Approved" for all cases would achieve 85.6% accuracy, making it misleading for imbalanced data | NB | [from notebook] |
| | | | LR | [from notebook] |
| | | | RF | [from notebook] |
| Recall | **USE** | Measures the proportion of actual "Declined" applications correctly identified. High recall means fewer declined loans are misclassified as approved, directly addressing the success criteria's goal of detecting rejections | NB | [from notebook] |
| | | | LR | [from notebook] |
| | | | RF | [from notebook] |
| Precision | **USE** | Measures the proportion of "Declined" predictions that are actually declined. High precision means when the model predicts "Declined," it's usually correct, reducing false alarms | NB | [from notebook] |
| | | | LR | [from notebook] |
| | | | RF | [from notebook] |
| F-Score | **USE** | Harmonic mean of recall and precision, balancing both metrics. Ensures the model doesn't sacrifice one for the other, aligning with the need for both high detection rate and accuracy | NB | [from notebook] |
| | | | LR | [from notebook] |
| | | | RF | [from notebook] |
| AUC-ROC | **USE** | Measures the model's ability to discriminate between approved and declined across all threshold values, providing overall assessment of classification quality independent of class imbalance | NB | [from notebook] |
| | | | LR | [from notebook] |
| | | | RF | [from notebook] |

**Instructions:** Fill in the Test Score column with values from your notebook output.

#### Task 5.c: Best Model Selection (2 marks)

"Based on the evaluation metrics, **[Model Name]** is the best performer with Recall of **[X.XX]** and Precision of **[X.XX]** for the Declined class. This model satisfies the success criteria by achieving a high detection rate of declined applications (**[XX]%** of actual declines caught) while maintaining prediction accuracy (**[XX]%** of declined predictions are correct), effectively balancing risk mitigation with operational efficiency."

#### Task 5.d: Model Fit Assessment (3 marks)

**Instructions:**
1. Screenshot the train vs test accuracy comparison from notebook
2. Interpret:
   - If train ≈ test (within 2-3%): Good fit
   - If train >> test (>5% difference): Overfitting
   - If both are low (<70%): Underfitting

#### Task 5.e: Hyperparameter Tuning (5 marks)

**i. K-folds used:**
"5-fold cross-validation was used (screenshot the code output showing cv=5)"

**ii. Hyperparameters comparison table:**

| Hyperparameter | Original Value | Tuned Value |
|----------------|----------------|-------------|
| n_estimators | 100 (default) | [from notebook] |
| max_depth | None (default) | [from notebook] |
| min_samples_split | 2 (default) | [from notebook] |
| min_samples_leaf | 1 (default) | [from notebook] |

**iii. Confusion matrices:**
Screenshot both the original and tuned confusion matrices side-by-side.

**iv. Metric scores comparison:**

| Metric | Original RF | Tuned RF |
|--------|-------------|----------|
| Recall | [from notebook] | [from notebook] |
| Precision | [from notebook] | [from notebook] |
| F-Score | [from notebook] | [from notebook] |

**v. Tuning impact (< 50 words):**
"Tuning [improved/maintained/reduced] the model's Recall from [X.XX] to [X.XX], indicating [better/similar/worse] ability to detect declined applications. The precision [increased/decreased] from [X.XX] to [X.XX], showing [interpretation]. Overall, hyperparameter tuning [enhanced/had minimal impact on] the model's alignment with success criteria."

#### Task 5.f: Research Question Answer & Critique (2 marks)

**Answer to Research Question A:**
"Yes, the machine learning models demonstrate potential to automate loan approval decisions. The best-performing model achieves [XX]% recall and [XX]% precision for declined applications, indicating reliable identification of high-risk cases while maintaining reasonable accuracy."

**Critique (< 100 words):**
"The model has limitations: (1) It uses only categorical features, potentially missing important numerical predictors like income or loan amount; (2) Class imbalance (85% approved) may bias predictions toward approval; (3) The model explains correlation, not causation, and may perpetuate historical biases. [Model name] likely outperformed others because [non-parametric nature handles categorical data well / ensemble averaging reduces variance / etc.]."

**Ethical issues:**
"Automated loan decisions may discriminate against protected groups if training data contains historical biases. Lack of transparency (especially in RF) makes it difficult for applicants to understand rejections. Over-reliance on algorithms may overlook individual circumstances requiring human judgment."

---

## PART B: REGRESSION - MAXIMUM LOAN AMOUNT PREDICTION

### Task 1: Domain Understanding (2 marks)

**Instructions:**
1. Screenshot the output showing filtered approved loans dimensions
2. Screenshot the list of features for regression

---

### Task 2: Data Understanding (5 marks)

**Instructions:**
1. Run the cell that generates distribution plots
2. Screenshot the entire figure showing distributions of all numerical features
3. Briefly describe each plot (e.g., "Income shows right skew with outliers")

---

### Task 3: Data Preprocessing (5 marks)

#### Task 3.a: Scaling Assessment (2 marks)

**Evidence:** Screenshot the scale check output showing min, max, mean, std for each feature.

**Recommendation (< 50 words):**
"Scaling is NOT required for Decision Tree regressors because they use rule-based splits, not distance calculations. Features operate on vastly different scales (Income: £4,200-£1,900,000 vs LTI: 0-3), but trees handle this naturally by evaluating thresholds independently for each feature."

#### Task 3.b: General Scaling Approach (3 marks - < 150 words)

"For regression modelling, scaling depends on the algorithm. For distance-based methods (Linear Regression, KNN, SVM), scale **both input features AND the target variable** if using standardization, but only **input features** if using normalization (Min-Max scaling), since the target should remain in its original interpretable units (Müller & Guido, 2016). For tree-based methods (Decision Trees, Random Forests), scaling is unnecessary as they use split thresholds. Always fit the scaler on training data only and transform both train and test sets to prevent data leakage."

---

### Task 4: Modelling (7 marks)

#### Task 4.a: Decision Tree Benefits (2 marks - < 50 words)

"Decision Trees provide interpretable decision rules that financial analysts can explain to stakeholders and regulators. They capture non-linear relationships between income, loan amount, and maximum lending without requiring feature engineering. The tree structure visualizes which factors most influence lending decisions, supporting transparent financial practices."

#### Task 4.b: Model Setup (5 marks)

**i. Reproducibility code:**
Screenshot the line: `random_state=42`

**ii. Dimensions and features:**
Screenshot outputs showing:
- DT1 training/test shapes and feature list (6 numeric features)
- DT2 training/test shapes and feature list (numeric + encoded categorical)

---

### Task 5: Evaluation (16 marks)

#### Task 5.a: Performance Metrics (6 marks)

| Metrics | USE or DO NOT USE | Justification | Model Name | Test Score |
|---------|-------------------|---------------|------------|------------|
| MSE | DO NOT USE | Mean Squared Error penalizes large errors but the squared units (£²) are not intuitive for financial interpretation | DT1 | [from notebook] |
| | | | DT2 | [from notebook] |
| MAE | DO NOT USE | Mean Absolute Error is interpretable (average £ error) but doesn't indicate how much variance in maximum loan amounts is explained by features | DT1 | [from notebook] |
| | | | DT2 | [from notebook] |
| R-Square | **USE** | R² directly measures the proportion of variance in maximum loan amounts explained by the input features, addressing the success criteria's emphasis on explanatory power. Values closer to 1.0 indicate better feature relevance | DT1 | [from notebook] |
| | | | DT2 | [from notebook] |

#### Task 5.b: R-Square Caveats (2 marks)

"R² can be misleading if: (1) Additional features artificially inflate the score without genuine predictive value; (2) High R² doesn't guarantee unbiased predictions (systematic over/under-estimation); (3) R² near 1.0 may signal overfitting if training and test scores diverge significantly; (4) It doesn't validate whether residuals meet assumptions (normality, homoscedasticity)."

#### Task 5.c: Best Model Selection (2 marks)

**Instructions:**
Based on notebook R² scores, complete:

"The best regression model is **[DT1/DT2]** with an R² score of **[X.XX]**, meaning it explains **[XX]%** of the variance in maximum loan amounts. This [satisfies/partially satisfies] the success criteria because [the features effectively capture lending patterns / additional categorical features improve predictions / numeric features alone are sufficient]."

#### Task 5.d: Pruning Analysis (4 marks)

**Instructions:**
1. Screenshot the pruned decision tree visualization
2. Screenshot the performance comparison (original vs pruned R²)

**Description (< 75 words):**
"The pruned tree (max_depth=4) shows [key splitting features at top levels]. Pre-pruning [reduced/maintained] R² from [X.XX] to [X.XX]. This represents [a trade-off of X% accuracy for interpretability / minimal performance loss / improved generalization]. The advantage is ease of explanation to stakeholders; the disadvantage is [slightly lower predictive power / potential underfitting]."

#### Task 5.e: Client Prediction (2 marks)

**Instructions:**
Screenshot the prediction output for client 60256.

"Using the pruned [DT1/DT2] model, the predicted maximum loan amount for client 60256 is **£[X,XXX.XX]**."

---

## FINAL CHECKLIST

### Before Submission:
- [ ] All screenshots are outputs only (no code visible)
- [ ] Each screenshot has a clear caption/label
- [ ] All tables are filled with actual values from notebook
- [ ] Word count limits respected for each section
- [ ] Report does not exceed 23 pages
- [ ] Font is Arial size 10, single-spaced
- [ ] Margins are 1 inch minimum
- [ ] .ipynb notebook file is ready for separate submission
- [ ] All citations use Harvard style

### Screenshot Requirements:
- Statistical descriptions (Task 2)
- Target variable distribution chart
- Before/after cleaning outputs (Task 3b)
- Feature names and shapes
- All confusion matrices (3 for original, 1 for tuned)
- Metrics tables (must match notebook outputs)
- Train vs test accuracy comparison
- Hyperparameter comparison
- Distribution plots for regression
- Scaling analysis output
- Dimensions of DT1 and DT2
- Pruned tree visualization
- Performance before/after pruning
- Client 60256 prediction

---

## Tips for Your Student

1. **Run the notebook sequentially** from top to bottom
2. **Save outputs** after each cell execution before taking screenshots
3. **Crop screenshots** to show only relevant output
4. **Label everything** clearly (Figure 1, Table 1, etc.)
5. **Be concise** - markers value precision over verbosity
6. **Verify numbers** - ensure report values match notebook outputs exactly
7. **Proofread** - check for typos and formatting consistency

Good luck!
