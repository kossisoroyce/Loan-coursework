# 🏦 Loan Approval Prediction - Machine Learning Coursework

A comprehensive machine learning project for predicting loan approval status using classification algorithms and maximum loan amount using regression models.

## 📊 Project Overview

This project implements a complete machine learning pipeline for loan approval prediction, including:
- **Part A**: Classification models for predicting loan approval/decline status
- **Part B**: Regression models for predicting maximum loan amounts

### Dataset
- **58,627 loan applications** with 16 original features
- **12 retained features** after variable selection
- **Binary target**: Approved (85.6%) vs Declined (14.4%)

## 🎯 Key Features

### Data Cleaning & Preprocessing
- Comprehensive data quality checks
- Removal of data entry errors (negative values, impossible employment lengths)
- Categorical variable standardization
- One-hot encoding for machine learning compatibility
- Zero missing values after cleaning

### Classification Models (Part A)
Three algorithms implemented and compared:
1. **Naïve Bayes (GaussianNB)** - Pattern-based probability predictions
2. **Logistic Regression** - Weighted scoring system
3. **Random Forest** - Ensemble of 100 decision trees

### Regression Models (Part B)
Decision tree regressors for maximum loan amount prediction:
- **DT1**: Using numeric features only
- **DT2**: Using all features (numeric + categorical)

## 📁 Project Structure

```
loan-approval-coursework/
├── data/
│   └── loan_approval_data.csv           # Dataset (4.6MB)
├── notebooks/
│   └── loan_approval_analysis.ipynb     # Main Jupyter notebook
├── reports/
│   ├── task2_data_understanding.md      # Data exploration guide
│   ├── task3_data_cleaning_summary.md   # Cleaning methodology
│   ├── task4_beginner_friendly_guide.md # Non-technical explanation
│   ├── task4_implementation_guide.md    # Technical implementation
│   ├── data_cleaning_preprocessing_roadmap.md
│   └── complete_coursework_guide.md     # Master guide
├── outputs/                              # Generated visualizations
├── TASK4_COMPLETE_CODE.md               # Ready-to-use code cells
├── TASK4_QUICK_START.md                 # 15-minute setup guide
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/loan-approval-coursework.git
cd loan-approval-coursework
```

2. **Create virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook notebooks/loan_approval_analysis.ipynb
```

5. **Run all cells**
- Click "Kernel" → "Restart & Run All"
- Wait for execution (2-3 minutes)

## 📚 Documentation

### For Beginners
- **[Task 4 Beginner Guide](reports/task4_beginner_friendly_guide.md)** - Non-technical explanations with real-world analogies
- **[Quick Start Guide](TASK4_QUICK_START.md)** - 15-minute implementation guide

### For Implementation
- **[Complete Code](TASK4_COMPLETE_CODE.md)** - Ready-to-copy code cells
- **[Technical Implementation](reports/task4_implementation_guide.md)** - Detailed technical guide
- **[Preprocessing Roadmap](reports/data_cleaning_preprocessing_roadmap.md)** - Data cleaning strategy

### For Understanding
- **[Data Understanding](reports/task2_data_understanding.md)** - Dataset characteristics
- **[Cleaning Summary](reports/task3_data_cleaning_summary.md)** - Data quality issues and solutions

## 🎓 Machine Learning Techniques

### Algorithms Used
| Algorithm | Type | Use Case | Package |
|-----------|------|----------|---------|
| Naïve Bayes | Non-parametric | Fast probability-based classification | `sklearn.naive_bayes.GaussianNB` |
| Logistic Regression | Parametric | Interpretable binary classification | `sklearn.linear_model.LogisticRegression` |
| Random Forest | Non-parametric | High-accuracy ensemble classifier | `sklearn.ensemble.RandomForestClassifier` |
| Decision Tree | Non-parametric | Regression for loan amounts | `sklearn.tree.DecisionTreeRegressor` |

### Evaluation Metrics
- **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Regression**: R², MSE, MAE

### Hyperparameter Tuning
- GridSearchCV with 5-fold cross-validation
- Stratified sampling to maintain class balance

## 📊 Key Results

### Data Quality
- **Initial dataset**: 58,645 rows
- **After cleaning**: 58,627 rows (0.031% data loss)
- **Missing values**: 0 (100% complete)

### Feature Engineering
- **Original features**: 16
- **Retained features**: 12
- **After encoding**: 15 features for classification

### Model Performance
- All models trained successfully
- Stratified 80:20 train-test split
- Reproducible results (random_state=42)

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib & seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📖 Learning Outcomes

This project demonstrates:
✅ Data cleaning and preprocessing pipelines  
✅ Feature selection and engineering  
✅ Multiple classification algorithms  
✅ Model evaluation and comparison  
✅ Hyperparameter optimization  
✅ Regression modeling  
✅ Statistical analysis and reporting  

## 🔍 Assignment Tasks Completed

- ✅ **Task 1**: Variable selection with justifications
- ✅ **Task 2**: Statistical descriptions and distributions
- ✅ **Task 3**: Data cleaning and preprocessing
- ✅ **Task 4**: Three classification models built
- ✅ **Task 5**: Model evaluation and hyperparameter tuning
- ✅ **Part B**: Regression models for loan amount prediction

## 📝 License

This is a coursework project for educational purposes.

## 👤 Author

**Your Name**  
Machine Learning Coursework - 2025

## 🙏 Acknowledgments

- Scikit-learn documentation and community
- Course materials and instructors
- Géron, A. (2019). *Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow*

---

⭐ **Star this repository if you find it helpful!**

📧 **Questions?** Check the documentation files or open an issue.
