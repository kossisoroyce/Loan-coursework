# Student Instructions: Step-by-Step Guide

## Before You Start

**Time needed:** 3-4 hours total
- Running notebook: 5-10 minutes
- Taking screenshots: 30-45 minutes  
- Writing report: 2-3 hours

**What you need:**
- Jupyter Notebook installed
- Python 3.8 or higher
- Microsoft Word or similar for report
- This completed notebook

---

## Step 1: Setup and Run (15 minutes)

### 1.1 Install Dependencies
Open terminal/command prompt in the coursework folder:

```bash
# Check Python version (must be 3.8+)
python3 --version

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# OR use the requirements file
pip install -r requirements.txt
```

### 1.2 Open Jupyter Notebook
```bash
jupyter notebook notebooks/loan_approval_analysis.ipynb
```

A browser window will open with the notebook.

### 1.3 Run All Cells
In Jupyter:
1. Click **Kernel** menu → **Restart & Run All**
2. Wait for all cells to complete (2-3 minutes)
3. Check for any error messages (there shouldn't be any)
4. **Save the notebook**: File → Save and Checkpoint

---

## Step 2: Take Screenshots (45 minutes)

### Golden Rules:
✅ Screenshot outputs ONLY (no code visible)
✅ Make sure text is readable
✅ Crop to show only relevant content
✅ Name files descriptively (e.g., "task2_statistics.png")

### Screenshots Needed:

#### Part A: Classification

**Task 2: Data Understanding (2 screenshots)**
- [ ] Statistical description table (`describe()` output)
- [ ] Bar chart of Loan Approval Status distribution

**Task 3: Data Cleaning (6 screenshots - before/after pairs)**
- [ ] Loan Approval Status - before cleaning
- [ ] Loan Approval Status - after cleaning
- [ ] Payment Default on File - before cleaning
- [ ] Payment Default on File - after cleaning
- [ ] Loan Interest Rate missing values - before
- [ ] Loan Interest Rate missing values - after

**Task 4: Modelling (2 screenshots)**
- [ ] List of feature names and data shape
- [ ] Code line showing `random_state=42, stratify=y`

**Task 5: Evaluation (8+ screenshots)**
- [ ] Naive Bayes confusion matrix
- [ ] Logistic Regression confusion matrix
- [ ] Random Forest confusion matrix
- [ ] Performance metrics table (all 3 models)
- [ ] Train vs test accuracy comparison
- [ ] Random Forest confusion matrix BEFORE tuning
- [ ] Random Forest confusion matrix AFTER tuning
- [ ] Performance metrics comparison (original vs tuned)

#### Part B: Regression

**Task 1: Domain Understanding (2 screenshots)**
- [ ] Dataset dimensions for approved loans
- [ ] List of features for regression

**Task 2: Data Understanding (1 screenshot)**
- [ ] Distribution plots (all 7 subplots)

**Task 3: Data Preprocessing (1 screenshot)**
- [ ] Scale analysis output (min, max, mean, std)

**Task 4: Modelling (2 screenshots)**
- [ ] DT1 dimensions and feature list
- [ ] DT2 dimensions and feature list

**Task 5: Evaluation (4 screenshots)**
- [ ] Performance metrics table (DT1 and DT2)
- [ ] Pruned tree visualization
- [ ] Performance before/after pruning
- [ ] Client 60256 prediction output

**Total: ~25 screenshots**

Save all screenshots in the `outputs/` folder.

---

## Step 3: Fill in Tables (30 minutes)

Open `reports/complete_coursework_guide.md` and:

1. **Copy the pre-made tables** into your Word document
2. **Fill in actual values** from your notebook outputs:
   - Test scores for NB, LR, RF
   - Hyperparameter values
   - DT1 and DT2 metrics
   - R² scores

3. **Keep the justifications** provided (or modify slightly to match your results)

---

## Step 4: Write Your Report (2-3 hours)

### Report Structure (use task numbers as headers):

```
Title Page
---------
[Student ID, Course, Date]

PART A: CLASSIFICATION

Task 1 – Domain Understanding
[Use the table from coursework description]

Task 2 – Data Understanding  
[Paste 2 screenshots with brief descriptions]

Task 3 – Data Preparation
Task 3.a: [Issues table with 3 rows]
Task 3.b: [6 screenshots with annotations]

Task 4 – Modelling
Task 4.a: [Algorithm details table]
Task 4.b: 
  i. [Screenshot of features]
  ii. [80:20 split justification - 100 words]
  iii. [Train-test vs K-fold - 100 words]
  iv. [Screenshot of random_state line]

Task 5 – Evaluation
Task 5.a: [3 confusion matrices]
Task 5.b: [Metrics table with USE/DO NOT USE]
Task 5.c: [Best model selection - 50 words]
Task 5.d: [Overfitting check with screenshot]
Task 5.e: 
  i-iv. [Tuning comparison]
  v. [Impact statement - 50 words]
Task 5.f: [Answer research question + critique - 100 words]

PART B: REGRESSION

Task 1 – Domain Understanding
[2 screenshots]

Task 2 – Data Understanding
[Distribution plots screenshot]

Task 3 – Data Preprocessing
Task 3.a: [Scaling recommendation - 50 words]
Task 3.b: [General scaling approach - 150 words]

Task 4 – Modelling
Task 4.a: [DT benefits - 50 words]
Task 4.b: [Dimensions and features - 2 screenshots]

Task 5 – Evaluation
Task 5.a: [Metrics table]
Task 5.b: [R² caveats - 50 words]
Task 5.c: [Best model - 50 words]
Task 5.d: [Pruned tree + performance - 75 words]
Task 5.e: [Client prediction screenshot]
```

### Formatting Checklist:
- [ ] Font: Arial 10pt
- [ ] Spacing: Single-spaced
- [ ] Margins: 1 inch on all sides
- [ ] Page numbers included
- [ ] All screenshots have captions (Figure 1, Figure 2, etc.)
- [ ] No code visible in any screenshot
- [ ] Word counts respected
- [ ] Maximum 23 pages

---

## Step 5: Final Checks (15 minutes)

### Before Submission:

1. **Numbers Match**
   - [ ] All scores in report match notebook outputs exactly
   - [ ] No placeholder values like "[from notebook]" remain

2. **Completeness**
   - [ ] All tasks answered
   - [ ] All screenshots included
   - [ ] All tables filled in
   - [ ] Research questions answered

3. **Quality**
   - [ ] Proofread for typos
   - [ ] Screenshots are clear and readable
   - [ ] Captions are descriptive
   - [ ] Citations use Harvard style (if any)

4. **Files Ready**
   - [ ] Report PDF/Word file
   - [ ] Jupyter notebook file (.ipynb)
   - [ ] Both files named correctly per submission guidelines

---

## Submission Checklist

- [ ] Report exported as PDF (recommended) or Word
- [ ] Notebook saved with all outputs visible
- [ ] File names follow submission requirements
- [ ] Uploaded to correct submission portal
- [ ] Submitted before deadline
- [ ] Confirmation email received

---

## Common Mistakes to Avoid

❌ **Including code in the report** → Only show outputs
❌ **Missing screenshots** → Check you have all ~25
❌ **Generic justifications** → Use the specific ones provided in the guide
❌ **Wrong metrics focus** → Part A focuses on Declined class, Part B on R²
❌ **Exceeding page limit** → Keep concise, remove unnecessary text
❌ **Forgetting notebook submission** → Submit both report AND .ipynb file

---

## Quick Help

**Issue:** Jupyter won't start
→ Try: `pip install --upgrade jupyter`

**Issue:** Plots not showing
→ Run this in first cell: `%matplotlib inline`

**Issue:** Numbers don't match guide
→ That's OK! Use YOUR numbers from YOUR notebook

**Issue:** Can't take screenshot
→ Windows: Windows+Shift+S | Mac: Command+Shift+4

**Issue:** Report too long
→ Remove explanatory text, keep only answers and screenshots

---

## Time Management Tips

**Day 1 (1 hour):**
- Setup environment
- Run notebook
- Take all screenshots

**Day 2 (2 hours):**
- Create report document
- Fill in all tables
- Paste screenshots with captions

**Day 3 (1 hour):**
- Write text sections
- Proofread
- Final checks
- Submit

---

## Success Tips

1. **Start early** - Don't leave it until the last day
2. **Follow the guide** - Everything you need is provided
3. **Be concise** - Markers prefer clarity over length
4. **Verify numbers** - Double-check all values match your outputs
5. **Save often** - Save both report and notebook frequently

---

**Good luck! You've got this! 🎓**

The hard work (coding) is already done. Now it's just about documenting your results clearly and professionally.
