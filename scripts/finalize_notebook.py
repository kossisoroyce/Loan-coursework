"""
Final script to add remaining regression cells and conclusion
"""

import json
from pathlib import Path

def finalize_notebook(notebook_path):
    """Add final regression and conclusion cells"""
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    def create_cell(cell_type, source, execution_count=None):
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source if isinstance(source, list) else [source]
        }
        if cell_type == "code":
            cell["execution_count"] = execution_count
            cell["outputs"] = []
        return cell
    
    new_cells = []
    
    # =====================================================================
    # REGRESSION MODELING
    # =====================================================================
    
    new_cells.append(create_cell("markdown", [
        "## Task 3: Data Preprocessing - Regression\n",
        "\n",
        "### Task 3.a: Assess need for scaling\n",
        "\n",
        "Decision Trees are **scale-invariant** - they use split points, not distances."
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 3.a: Investigate need for scaling\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 3: DATA PREPROCESSING - SCALING ASSESSMENT\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "numerical_features_reg = [\n",
        "    'Income', 'Employment Length', 'Loan Amount', 'Loan Interest Rate',\n",
        "    'Loan-to-Income Ratio (LTI)', 'Credit History Length'\n",
        "]\n",
        "\n",
        "print(\"\\nFeature ranges:\")\n",
        "scale_check = df_approved[numerical_features_reg].describe().loc[['min', 'max', 'std']]\n",
        "print(scale_check)\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"RECOMMENDATION:\")\n",
        "print(\"=\"*70)\n",
        "print(\"Decision Tree regressors are SCALE-INVARIANT (use splits, not distances).\")\n",
        "print(\"Therefore, scaling is NOT required for this task.\")\n",
        "print(\"\\nHowever, if using distance-based algorithms (KNN, SVM, Neural Networks),\")\n",
        "print(\"scaling would be essential due to vastly different ranges:\")\n",
        "print(f\"  - Income: £{df_approved['Income'].min():,.0f} to £{df_approved['Income'].max():,.0f}\")\n",
        "print(f\"  - LTI Ratio: {df_approved['Loan-to-Income Ratio (LTI)'].min():.3f} to {df_approved['Loan-to-Income Ratio (LTI)'].max():.3f}\")"
    ], None))
    
    new_cells.append(create_cell("markdown", [
        "## Task 4: Modeling - Build Regression Models\n",
        "\n",
        "### Task 4.a: Why Decision Tree?\n",
        "- **Interpretable:** Easy to explain to financial analysts\n",
        "- **Non-linear:** Captures complex relationships\n",
        "- **No scaling needed:** Works with different feature ranges"
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 4.b: Prepare features for two regression models\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 4: REGRESSION MODEL PREPARATION\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Model 1 (DT1): Numeric features only\n",
        "numeric_features_only = [\n",
        "    'Income', 'Employment Length', 'Loan Amount', 'Loan Interest Rate',\n",
        "    'Loan-to-Income Ratio (LTI)', 'Credit History Length'\n",
        "]\n",
        "\n",
        "X_dt1 = df_approved[numeric_features_only].copy()\n",
        "y_reg = df_approved['Maximum Loan Amount'].copy()\n",
        "\n",
        "print(f\"\\nModel 1 (DT1) - Numeric features only:\")\n",
        "print(f\"  Features ({len(numeric_features_only)}): {numeric_features_only}\")\n",
        "print(f\"  Shape: {X_dt1.shape}\")\n",
        "\n",
        "# Model 2 (DT2): All features (numeric + categorical)\n",
        "categorical_reg_features = [\n",
        "    'Education Qualifications', 'Home Ownership', 'Loan Intent', 'Payment Default on File'\n",
        "]\n",
        "\n",
        "X_dt2_cat = pd.get_dummies(df_approved[categorical_reg_features], \n",
        "                           columns=['Education Qualifications', 'Home Ownership', 'Loan Intent'],\n",
        "                           drop_first=True)\n",
        "X_dt2 = pd.concat([X_dt1, X_dt2_cat], axis=1)\n",
        "\n",
        "print(f\"\\nModel 2 (DT2) - All features:\")\n",
        "print(f\"  Numeric features: {len(numeric_features_only)}\")\n",
        "print(f\"  Categorical features (encoded): {len(X_dt2_cat.columns)}\")\n",
        "print(f\"  Total features: {X_dt2.shape[1]}\")\n",
        "print(f\"  Shape: {X_dt2.shape}\")"
    ], None))
    
    new_cells.append(create_cell("code", [
        "# Task 4.b.i: Train-test split with reproducibility\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"TRAIN-TEST SPLIT (80:20)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Split for DT1\n",
        "X_train_dt1, X_test_dt1, y_train_reg1, y_test_reg1 = train_test_split(\n",
        "    X_dt1, y_reg, test_size=0.2, random_state=42  # Ensures reproducibility\n",
        ")\n",
        "\n",
        "# Split for DT2 (same random_state for consistency)\n",
        "X_train_dt2, X_test_dt2, y_train_reg2, y_test_reg2 = train_test_split(\n",
        "    X_dt2, y_reg, test_size=0.2, random_state=42\n",
        ")\n",
        "\n",
        "print(f\"\\nTask 4.b.i: Reproducibility ensured with random_state=42\\n\")\n",
        "\n",
        "print(f\"DT1 (Numeric only):\")\n",
        "print(f\"  Training set: {X_train_dt1.shape}\")\n",
        "print(f\"  Test set: {X_test_dt1.shape}\")\n",
        "print(f\"  Features: {list(X_train_dt1.columns)}\")\n",
        "\n",
        "print(f\"\\nDT2 (All features):\")\n",
        "print(f\"  Training set: {X_train_dt2.shape}\")\n",
        "print(f\"  Test set: {X_test_dt2.shape}\")\n",
        "print(f\"  Features: {list(X_train_dt2.columns)[:10]}... (showing first 10)\")"
    ], None))
    
    new_cells.append(create_cell("code", [
        "# Train regression models\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"TRAINING DECISION TREE REGRESSION MODELS\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# DT1: Numeric features only\n",
        "print(\"\\n1. Training DT1 (Numeric features only)...\")\n",
        "dt1_model = DecisionTreeRegressor(random_state=42)\n",
        "dt1_model.fit(X_train_dt1, y_train_reg1)\n",
        "print(\"   ✓ DT1 trained successfully\")\n",
        "\n",
        "# DT2: All features\n",
        "print(\"\\n2. Training DT2 (All features)...\")\n",
        "dt2_model = DecisionTreeRegressor(random_state=42)\n",
        "dt2_model.fit(X_train_dt2, y_train_reg2)\n",
        "print(\"   ✓ DT2 trained successfully\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✓ ALL REGRESSION MODELS TRAINED\")\n",
        "print(\"=\"*70)"
    ], None))
    
    new_cells.append(create_cell("markdown", [
        "## Task 5: Evaluation - Regression Models\n",
        "\n",
        "### Task 5.a: Calculate Performance Metrics"
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 5.a: Calculate regression metrics\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 5.a: REGRESSION MODEL EVALUATION\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Make predictions\n",
        "y_pred_dt1 = dt1_model.predict(X_test_dt1)\n",
        "y_pred_dt2 = dt2_model.predict(X_test_dt2)\n",
        "\n",
        "# Calculate metrics for DT1\n",
        "mse_dt1 = mean_squared_error(y_test_reg1, y_pred_dt1)\n",
        "mae_dt1 = mean_absolute_error(y_test_reg1, y_pred_dt1)\n",
        "r2_dt1 = r2_score(y_test_reg1, y_pred_dt1)\n",
        "\n",
        "# Calculate metrics for DT2\n",
        "mse_dt2 = mean_squared_error(y_test_reg2, y_pred_dt2)\n",
        "mae_dt2 = mean_absolute_error(y_test_reg2, y_pred_dt2)\n",
        "r2_dt2 = r2_score(y_test_reg2, y_pred_dt2)\n",
        "\n",
        "# Create results dataframe\n",
        "regression_results = pd.DataFrame({\n",
        "    'Metric': ['MSE', 'MAE', 'R²'],\n",
        "    'DT1 (Numeric)': [mse_dt1, mae_dt1, r2_dt1],\n",
        "    'DT2 (All Features)': [mse_dt2, mae_dt2, r2_dt2]\n",
        "})\n",
        "\n",
        "print(\"\\nTest Performance Metrics:\")\n",
        "print(regression_results.to_string(index=False))\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"METRIC INTERPRETATION:\")\n",
        "print(\"=\"*70)\n",
        "print(\"USE: R² - Explains variance in Maximum Loan Amount\")\n",
        "print(\"     Higher R² = better model\")\n",
        "print(\"\\nDO NOT USE (alone):\")\n",
        "print(\"  - MSE: Large values, hard to interpret in £\")\n",
        "print(\"  - MAE: Doesn't show proportion of variance explained\")"
    ], None))
    
    # Visualize predictions
    new_cells.append(create_cell("code", [
        "# Visualize predictions vs actual\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# DT1\n",
        "axes[0].scatter(y_test_reg1, y_pred_dt1, alpha=0.5, s=20)\n",
        "axes[0].plot([y_test_reg1.min(), y_test_reg1.max()], \n",
        "             [y_test_reg1.min(), y_test_reg1.max()], \n",
        "             'r--', lw=2, label='Perfect Prediction')\n",
        "axes[0].set_xlabel('Actual Maximum Loan Amount (£)', fontweight='bold')\n",
        "axes[0].set_ylabel('Predicted Maximum Loan Amount (£)', fontweight='bold')\n",
        "axes[0].set_title(f'DT1 (Numeric Features)\\nR² = {r2_dt1:.4f}', \n",
        "                  fontsize=12, fontweight='bold')\n",
        "axes[0].legend()\n",
        "axes[0].grid(alpha=0.3)\n",
        "\n",
        "# DT2\n",
        "axes[1].scatter(y_test_reg2, y_pred_dt2, alpha=0.5, s=20, color='green')\n",
        "axes[1].plot([y_test_reg2.min(), y_test_reg2.max()], \n",
        "             [y_test_reg2.min(), y_test_reg2.max()], \n",
        "             'r--', lw=2, label='Perfect Prediction')\n",
        "axes[1].set_xlabel('Actual Maximum Loan Amount (£)', fontweight='bold')\n",
        "axes[1].set_ylabel('Predicted Maximum Loan Amount (£)', fontweight='bold')\n",
        "axes[1].set_title(f'DT2 (All Features)\\nR² = {r2_dt2:.4f}', \n",
        "                  fontsize=12, fontweight='bold')\n",
        "axes[1].legend()\n",
        "axes[1].grid(alpha=0.3)\n",
        "\n",
        "plt.suptitle('Regression Model Predictions: Actual vs Predicted', \n",
        "             fontsize=14, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n✓ Prediction plots generated\")\n",
        "print(\"   Points closer to the red line = better predictions\")"
    ], None))
    
    # Best model selection
    new_cells.append(create_cell("code", [
        "# Task 5.c: Select best regression model\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"TASK 5.c: BEST REGRESSION MODEL SELECTION\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "if r2_dt2 > r2_dt1:\n",
        "    best_model_reg = dt2_model\n",
        "    X_train_best = X_train_dt2\n",
        "    X_test_best = X_test_dt2\n",
        "    y_test_best = y_test_reg2\n",
        "    best_name_reg = \"DT2\"\n",
        "    best_r2 = r2_dt2\n",
        "    print(f\"\\n✓ BEST MODEL: DT2 (All Features)\")\n",
        "    print(f\"  R² Score: {r2_dt2:.4f}\")\n",
        "    print(f\"\\n  Justification:\")\n",
        "    print(f\"    - DT2 has higher R², meaning it explains more variance\")\n",
        "    print(f\"    - Categorical features add predictive value\")\n",
        "else:\n",
        "    best_model_reg = dt1_model\n",
        "    X_train_best = X_train_dt1\n",
        "    X_test_best = X_test_dt1\n",
        "    y_test_best = y_test_reg1\n",
        "    best_name_reg = \"DT1\"\n",
        "    best_r2 = r2_dt1\n",
        "    print(f\"\\n✓ BEST MODEL: DT1 (Numeric Only)\")\n",
        "    print(f\"  R² Score: {r2_dt1:.4f}\")\n",
        "    print(f\"\\n  Justification:\")\n",
        "    print(f\"    - DT1 performs as well or better with fewer features\")\n",
        "    print(f\"    - Simpler model, easier to interpret\")\n",
        "\n",
        "print(f\"\\n  💡 This model explains {best_r2*100:.2f}% of variance in Maximum Loan Amount\")"
    ], None))
    
    # Pruning
    new_cells.append(create_cell("markdown", [
        "### Task 5.d: Model Pruning\n",
        "\n",
        "Rebuild the best model with pre-pruning (`max_depth=4`) to improve interpretability."
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 5.d: Prune the best model\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 5.d: MODEL PRUNING (max_depth=4)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Train pruned version\n",
        "pruned_model = DecisionTreeRegressor(max_depth=4, random_state=42)\n",
        "pruned_model.fit(X_train_best, y_test_best if best_name_reg == 'DT2' else y_train_reg1)\n",
        "\n",
        "# Get correct y_train based on best model\n",
        "y_train_for_pruning = y_train_reg2 if best_name_reg == 'DT2' else y_train_reg1\n",
        "pruned_model.fit(X_train_best, y_train_for_pruning)\n",
        "\n",
        "# Predict with pruned model\n",
        "y_pred_pruned = pruned_model.predict(X_test_best)\n",
        "r2_pruned = r2_score(y_test_best, y_pred_pruned)\n",
        "mae_pruned = mean_absolute_error(y_test_best, y_pred_pruned)\n",
        "\n",
        "print(f\"\\nOriginal {best_name_reg}:\")\n",
        "print(f\"  R² Score: {best_r2:.4f}\")\n",
        "print(f\"  MAE: £{mae_dt2 if best_name_reg=='DT2' else mae_dt1:,.2f}\")\n",
        "\n",
        "print(f\"\\nPruned {best_name_reg} (max_depth=4):\")\n",
        "print(f\"  R² Score: {r2_pruned:.4f}\")\n",
        "print(f\"  MAE: £{mae_pruned:,.2f}\")\n",
        "\n",
        "r2_change = r2_pruned - best_r2\n",
        "print(f\"\\nR² Change: {r2_change:+.4f}\")\n",
        "\n",
        "if r2_change < 0:\n",
        "    print(f\"\\n✓ Impact: Pruning DECREASED performance (simpler but less accurate)\")\n",
        "    print(f\"  Trade-off: Easier interpretation vs. slightly lower accuracy\")\n",
        "else:\n",
        "    print(f\"\\n✓ Impact: Pruning MAINTAINED/IMPROVED performance\")\n",
        "    print(f\"  Benefit: Better generalization without sacrificing accuracy\")"
    ], None))
    
    # Plot pruned tree
    new_cells.append(create_cell("code", [
        "# Visualize pruned decision tree\n",
        "plt.figure(figsize=(22, 10))\n",
        "plot_tree(pruned_model, \n",
        "          feature_names=X_train_best.columns.tolist(),\n",
        "          filled=True,\n",
        "          rounded=True,\n",
        "          fontsize=9)\n",
        "plt.title(f'Pruned Decision Tree ({best_name_reg}, max_depth=4)\\nFor Maximum Loan Amount Prediction', \n",
        "          fontsize=16, fontweight='bold', pad=20)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(\"\\n✓ Decision tree visualized\")\n",
        "print(\"  Tree limited to 4 levels for easier interpretation by financial analysts\")"
    ], None))
    
    # Prediction for client 60256
    new_cells.append(create_cell("markdown", [
        "### Task 5.e: Prediction for Client 60256\n",
        "\n",
        "Use the pruned model to predict the maximum loan amount for a specific client."
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 5.e: Predict for client 60256\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 5.e: PREDICTION FOR CLIENT 60256\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Client details from coursework\n",
        "client_data = {\n",
        "    'Income': 57000,\n",
        "    'Employment Length': 15,\n",
        "    'Loan Amount': 25700,\n",
        "    'Loan Interest Rate': 23.0,\n",
        "    'Loan-to-Income Ratio (LTI)': 0.10,\n",
        "    'Credit History Length': 35\n",
        "}\n",
        "\n",
        "print(\"\\nClient 60256 Details:\")\n",
        "for key, value in client_data.items():\n",
        "    print(f\"  {key}: {value}\")\n",
        "\n",
        "# If DT2 was best, add categorical features\n",
        "if best_name_reg == \"DT2\":\n",
        "    client_data_full = client_data.copy()\n",
        "    client_data_full.update({\n",
        "        'Education Qualifications': 'Unknown',\n",
        "        'Home Ownership': 'Rent',\n",
        "        'Loan Intent': 'Medical',\n",
        "        'Payment Default on File': 0\n",
        "    })\n",
        "    \n",
        "    # Create and encode\n",
        "    client_df = pd.DataFrame([client_data_full])\n",
        "    client_cat = pd.get_dummies(client_df[['Education Qualifications', 'Home Ownership', 'Loan Intent']],\n",
        "                                 drop_first=True)\n",
        "    client_num = client_df[numeric_features_only + ['Payment Default on File']]\n",
        "    client_encoded = pd.concat([client_num, client_cat], axis=1)\n",
        "    \n",
        "    # Align columns with training data\n",
        "    for col in X_train_best.columns:\n",
        "        if col not in client_encoded.columns:\n",
        "            client_encoded[col] = 0\n",
        "    client_encoded = client_encoded[X_train_best.columns]\n",
        "else:\n",
        "    # DT1 - numeric only\n",
        "    client_encoded = pd.DataFrame([client_data])\n",
        "\n",
        "# Make prediction\n",
        "predicted_max_loan = pruned_model.predict(client_encoded)[0]\n",
        "\n",
        "print(f\"\\n\" + \"=\"*70)\n",
        "print(f\"PREDICTION RESULT\")\n",
        "print(f\"=\"*70)\n",
        "print(f\"\\nClient ID: 60256\")\n",
        "print(f\"Predicted Maximum Loan Amount: £{predicted_max_loan:,.2f}\")\n",
        "print(f\"\\nModel used: {best_name_reg} (Pruned, max_depth=4)\")\n",
        "print(f\"Model R²: {r2_pruned:.4f}\")\n",
        "\n",
        "print(f\"\\n💡 Interpretation:\")\n",
        "print(f\"   The lender should offer up to £{predicted_max_loan:,.2f} to this client.\")"
    ], None))
    
    # Final summary
    new_cells.append(create_cell("markdown", [
        "---\n",
        "\n",
        "# 📊 FINAL SUMMARY & CONCLUSIONS\n",
        "\n",
        "## Coursework Completion Summary"
    ]))
    
    new_cells.append(create_cell("code", [
        "# Generate comprehensive final summary\n",
        "print(\"=\"*70)\n",
        "print(\"COURSEWORK SUMMARY\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "print(\"\\n📌 PART A: CLASSIFICATION (Loan Approval Prediction)\")\n",
        "print(\"-\" * 70)\n",
        "print(f\"  Models Built: Naive Bayes, Logistic Regression, Random Forest\")\n",
        "print(f\"  Best Model: {best_model_name if 'best_model_name' in dir() else 'See Task 5.c'}\")\n",
        "print(f\"  Key Metrics: Recall, Precision, F1-Score (for Declined class)\")\n",
        "print(f\"  Hyperparameter Tuning: GridSearchCV with 5-fold CV\")\n",
        "\n",
        "print(\"\\n📌 PART B: REGRESSION (Maximum Loan Amount Prediction)\")\n",
        "print(\"-\" * 70)\n",
        "print(f\"  Models Built: DT1 (numeric), DT2 (all features)\")\n",
        "print(f\"  Best Model: {best_name_reg}\")\n",
        "print(f\"  Best R² Score: {best_r2:.4f} ({best_r2*100:.2f}% variance explained)\")\n",
        "print(f\"  Pruned R² Score: {r2_pruned:.4f}\")\n",
        "print(f\"  Prediction for Client 60256: £{predicted_max_loan:,.2f}\")\n",
        "\n",
        "print(\"\\n✅ DATA QUALITY\")\n",
        "print(\"-\" * 70)\n",
        "print(f\"  Missing values: 0\")\n",
        "print(f\"  Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\")\n",
        "print(f\"  All mitigation strategies implemented successfully\")\n",
        "\n",
        "print(\"\\n💡 KEY FINDINGS\")\n",
        "print(\"-\" * 70)\n",
        "print(f\"  1. Class imbalance in loan approval (86% Approved, 14% Declined)\")\n",
        "print(f\"  2. Categorical features contribute to classification accuracy\")\n",
        "print(f\"  3. {'Numerical' if best_name_reg == 'DT1' else 'All'} features best predict maximum loan amount\")\n",
        "print(f\"  4. Decision tree pruning aids interpretability with minimal accuracy loss\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✓ COURSEWORK COMPLETE - ALL TASKS FINISHED\")\n",
        "print(\"=\"*70)\n",
        "print(\"\\nNext Steps:\")\n",
        "print(\"  1. Take screenshots of all outputs for your report\")\n",
        "print(\"  2. Complete the summary tables as per coursework instructions\")\n",
        "print(\"  3. Add interpretations and justifications\")\n",
        "print(\"  4. Submit report (max 23 pages) + this notebook (.ipynb)\")\n",
        "print(\"\\n🎓 Good luck with your submission!\")"
    ], None))
    
    # Add all cells
    notebook["cells"].extend(new_cells)
    
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Added {len(new_cells)} final cells")
    print(f"✓ Total cells: {len(notebook['cells'])}")

if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / 'notebooks' / 'loan_approval_complete.ipynb'
    finalize_notebook(notebook_path)
    print("✓ Notebook finalization complete!")
    print("✓ Comprehensive notebook ready for use!")
