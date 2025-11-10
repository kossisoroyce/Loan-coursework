"""
Add comprehensive modeling and performance analysis cells to the notebook
"""

import json
from pathlib import Path

def add_modeling_cells(notebook_path):
    """Add all modeling cells to the notebook"""
    
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
    
    new_cells = []
    
    # =====================================================================
    # PART A: CLASSIFICATION
    # =====================================================================
    
    new_cells.append(create_cell("markdown", [
        "---\n",
        "\n",
        "# PART A: CLASSIFICATION - LOAN APPROVAL STATUS PREDICTION\n",
        "\n",
        "## Task 4: Model Building\n",
        "\n",
        "### Objective\n",
        "Build classification models to predict Loan Approval Status (Approved vs Declined)\n",
        "\n",
        "### Models to Build\n",
        "1. **Naive Bayes (NB)** - Non-parametric, probabilistic classifier\n",
        "2. **Logistic Regression (LR)** - Parametric, linear classifier\n",
        "3. **Random Forest (RF)** - Non-parametric, ensemble classifier\n",
        "\n",
        "### Success Criteria (from coursework)\n",
        "> \"The model should aim to predict the 'Reject' status of subjects for as many as possible to decrease the risk of future defaulted loan payments. However, the model should demonstrate that its high 'Reject' prediction rate is mainly due to a larger portion of correctly detected (predicted) rejected loan applications.\"\n",
        "\n",
        "**Translation:** Prioritize **high Recall** and **high Precision** for the \"Declined\" class."
    ]))
    
    # CELL: Task 2 - Data Understanding
    new_cells.append(create_cell("markdown", [
        "## Task 2: Data Understanding\n",
        "\n",
        "Before building models, let's understand our features and target variable."
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 2: Statistical description of retained variables\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 2: DATA UNDERSTANDING FOR CLASSIFICATION\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Variables for classification\n",
        "classification_vars = [\n",
        "    'Education Qualifications', 'Income', 'Home Ownership', 'Employment Length',\n",
        "    'Loan Intent', 'Loan Amount', 'Loan Interest Rate', 'Loan-to-Income Ratio (LTI)',\n",
        "    'Payment Default on File', 'Credit History Length', 'Loan Approval Status'\n",
        "]\n",
        "\n",
        "print(\"\\nStatistical Summary:\")\n",
        "print(df[classification_vars].describe(include='all'))\n",
        "\n",
        "print(\"\\nVariable Types:\")\n",
        "print(df[classification_vars].dtypes)"
    ], None))
    
    # CELL: Visualize target distribution
    new_cells.append(create_cell("code", [
        "# Visualize target variable distribution\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# Count plot\n",
        "target_counts = df['Loan Approval Status'].value_counts()\n",
        "axes[0].bar(target_counts.index, target_counts.values, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')\n",
        "axes[0].set_title('Loan Approval Status Distribution', fontsize=14, fontweight='bold')\n",
        "axes[0].set_xlabel('Status')\n",
        "axes[0].set_ylabel('Count')\n",
        "axes[0].grid(axis='y', alpha=0.3)\n",
        "for i, v in enumerate(target_counts.values):\n",
        "    axes[0].text(i, v + 500, f'{v:,}\\n({v/len(df)*100:.1f}%)', ha='center', fontweight='bold')\n",
        "\n",
        "# Pie chart\n",
        "axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', \n",
        "            colors=['#2ecc71', '#e74c3c'], startangle=90, textprops={'fontweight': 'bold', 'fontsize': 12})\n",
        "axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "print(f\"\\n⚠️  Class Imbalance Detected:\")\n",
        "print(f\"   - Approved: {(target_counts['Approved']/len(df)*100):.2f}%\")\n",
        "print(f\"   - Declined: {(target_counts['Declined']/len(df)*100):.2f}%\")\n",
        "print(f\"\\n💡 Strategy: Use stratified train-test split to maintain class proportions\")"
    ], None))
    
    # CELL: Feature selection
    new_cells.append(create_cell("markdown", [
        "## Task 4: Feature Selection & Preparation\n",
        "\n",
        "For this coursework, we'll use **categorical features only** (as per Task 4.b)."
    ]))
    
    new_cells.append(create_cell("code", [
        "# Task 4.b: Select categorical features for classification\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 4: FEATURE SELECTION FOR CLASSIFICATION\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Categorical features (Payment Default is already binary encoded)\n",
        "categorical_features = [\n",
        "    'Education Qualifications',\n",
        "    'Home Ownership',\n",
        "    'Loan Intent',\n",
        "    'Payment Default on File'\n",
        "]\n",
        "\n",
        "print(f\"\\nSelected features ({len(categorical_features)}):\")\n",
        "for i, feat in enumerate(categorical_features, 1):\n",
        "    print(f\"  {i}. {feat}\")\n",
        "\n",
        "# Prepare feature matrix\n",
        "X = df[categorical_features].copy()\n",
        "y = df['Loan Approval Status'].copy()\n",
        "\n",
        "print(f\"\\nFeature matrix shape: {X.shape}\")\n",
        "print(f\"Target variable shape: {y.shape}\")\n",
        "\n",
        "# One-hot encode categorical features (except Payment Default which is already binary)\n",
        "encode_cols = ['Education Qualifications', 'Home Ownership', 'Loan Intent']\n",
        "X_encoded = pd.get_dummies(X, columns=encode_cols, drop_first=True)\n",
        "\n",
        "print(f\"\\nAfter one-hot encoding: {X_encoded.shape}\")\n",
        "print(f\"\\nEncoded features ({len(X_encoded.columns)}):\")\n",
        "for i, col in enumerate(X_encoded.columns, 1):\n",
        "    print(f\"  {i:2d}. {col}\")"
    ], None))
    
    # CELL: Train-test split
    new_cells.append(create_cell("code", [
        "# Task 4.b.ii & 4.b.iv: Train-test split with stratification\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"TRAIN-TEST SPLIT (80:20 with Stratification)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Stratified split to maintain class proportions\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X_encoded, \n",
        "    y, \n",
        "    test_size=0.2,      # 80:20 split\n",
        "    random_state=42,    # Ensures reproducibility\n",
        "    stratify=y          # Maintains class proportions\n",
        ")\n",
        "\n",
        "print(f\"\\nTraining set: {X_train.shape}\")\n",
        "print(f\"Test set: {X_test.shape}\")\n",
        "\n",
        "print(f\"\\nClass distribution in training set:\")\n",
        "train_dist = y_train.value_counts(normalize=True) * 100\n",
        "for label, pct in train_dist.items():\n",
        "    print(f\"  {label}: {pct:.2f}%\")\n",
        "\n",
        "print(f\"\\nClass distribution in test set:\")\n",
        "test_dist = y_test.value_counts(normalize=True) * 100\n",
        "for label, pct in test_dist.items():\n",
        "    print(f\"  {label}: {pct:.2f}%\")\n",
        "\n",
        "print(f\"\\n✓ Stratification successful: Train and test have same class proportions\")\n",
        "print(f\"✓ Reproducibility ensured: random_state=42\")"
    ], None))
    
    # CELL: Train models
    new_cells.append(create_cell("code", [
        "# Build and train classification models\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"TRAINING CLASSIFICATION MODELS\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# 1. Naive Bayes\n",
        "print(\"\\n1. Training Naive Bayes...\")\n",
        "nb_model = GaussianNB()\n",
        "nb_model.fit(X_train, y_train)\n",
        "print(\"   ✓ Naive Bayes trained\")\n",
        "\n",
        "# 2. Logistic Regression\n",
        "print(\"\\n2. Training Logistic Regression...\")\n",
        "lr_model = LogisticRegression(random_state=42, max_iter=1000)\n",
        "lr_model.fit(X_train, y_train)\n",
        "print(\"   ✓ Logistic Regression trained\")\n",
        "\n",
        "# 3. Random Forest\n",
        "print(\"\\n3. Training Random Forest...\")\n",
        "rf_model = RandomForestClassifier(random_state=42, n_estimators=100)\n",
        "rf_model.fit(X_train, y_train)\n",
        "print(\"   ✓ Random Forest trained\")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"✓ ALL MODELS TRAINED SUCCESSFULLY\")\n",
        "print(\"=\"*70)"
    ], None))
    
    # CELL: Evaluation section
    new_cells.append(create_cell("markdown", [
        "---\n",
        "\n",
        "## Task 5: Model Evaluation\n",
        "\n",
        "### Task 5.a: Confusion Matrices\n",
        "\n",
        "Generate confusion matrices for all three models on the test set."
    ]))
    
    # CELL: Confusion matrices
    new_cells.append(create_cell("code", [
        "# Task 5.a: Generate predictions and confusion matrices\n",
        "print(\"=\"*70)\n",
        "print(\"TASK 5.a: CONFUSION MATRICES (TEST SET)\")\n",
        "print(\"=\"*70)\n",
        "\n",
        "# Make predictions\n",
        "y_pred_nb = nb_model.predict(X_test)\n",
        "y_pred_lr = lr_model.predict(X_test)\n",
        "y_pred_rf = rf_model.predict(X_test)\n",
        "\n",
        "# Generate confusion matrices\n",
        "cm_nb = confusion_matrix(y_test, y_pred_nb, labels=['Approved', 'Declined'])\n",
        "cm_lr = confusion_matrix(y_test, y_pred_lr, labels=['Approved', 'Declined'])\n",
        "cm_rf = confusion_matrix(y_test, y_pred_rf, labels=['Approved', 'Declined'])\n",
        "\n",
        "# Visualize confusion matrices\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "cms = [cm_nb, cm_lr, cm_rf]\n",
        "titles = ['Naive Bayes', 'Logistic Regression', 'Random Forest']\n",
        "\n",
        "for ax, cm, title in zip(axes, cms, titles):\n",
        "    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, \n",
        "                xticklabels=['Approved', 'Declined'], \n",
        "                yticklabels=['Approved', 'Declined'],\n",
        "                cbar_kws={'label': 'Count'})\n",
        "    ax.set_title(f'{title}\\nConfusion Matrix', fontsize=14, fontweight='bold')\n",
        "    ax.set_xlabel('Predicted Label', fontweight='bold')\n",
        "    ax.set_ylabel('True Label', fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Print matrices\n",
        "print(\"\\nNaive Bayes - Confusion Matrix:\")\n",
        "print(f\"                 Predicted\")\n",
        "print(f\"              Approved  Declined\")\n",
        "print(f\"Actual Approved    {cm_nb[0,0]:5d}    {cm_nb[0,1]:5d}\")\n",
        "print(f\"       Declined    {cm_nb[1,0]:5d}    {cm_nb[1,1]:5d}\")\n",
        "\n",
        "print(\"\\nLogistic Regression - Confusion Matrix:\")\n",
        "print(f\"                 Predicted\")\n",
        "print(f\"              Approved  Declined\")\n",
        "print(f\"Actual Approved    {cm_lr[0,0]:5d}    {cm_lr[0,1]:5d}\")\n",
        "print(f\"       Declined    {cm_lr[1,0]:5d}    {cm_lr[1,1]:5d}\")\n",
        "\n",
        "print(\"\\nRandom Forest - Confusion Matrix:\")\n",
        "print(f\"                 Predicted\")\n",
        "print(f\"              Approved  Declined\")\n",
        "print(f\"Actual Approved    {cm_rf[0,0]:5d}    {cm_rf[0,1]:5d}\")\n",
        "print(f\"       Declined    {cm_rf[1,0]:5d}    {cm_rf[1,1]:5d}\")"
    ], None))
    
    # Add all new cells
    notebook["cells"].extend(new_cells)
    
    # Save updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✓ Added {len(new_cells)} modeling cells to notebook")
    print(f"✓ Total cells now: {len(notebook['cells'])}")

if __name__ == "__main__":
    notebook_path = Path(__file__).parent.parent / 'notebooks' / 'loan_approval_complete.ipynb'
    add_modeling_cells(notebook_path)
    print(f"✓ Notebook updated with modeling cells!")
