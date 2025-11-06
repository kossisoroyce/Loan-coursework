# Task (2) – Data Understanding: Producing Your Experimental Design

## Experimental Design Overview

Our experimental approach follows a structured path to understand the loan approval dataset before building predictive models. We start by examining what data we actually have—checking the shape, distribution, and quality of the features we've retained from Task 1. This isn't just box-ticking; it's about spotting potential problems early (like missing values or inconsistent categories) and understanding the natural patterns in our data.

The design addresses two key questions:
1. **What does our data look like?** We need summary statistics to understand typical values, ranges, and any extreme outliers that might affect our models.
2. **Is our target variable balanced?** Since we're predicting loan approval status, we need to know if approvals and declines are roughly equal or if one dominates—this will guide our choice of evaluation metrics later.

By documenting these characteristics now, we create a baseline understanding that helps us make informed decisions about data cleaning (Task 3) and model selection (Task 4). It's the foundation that ensures our machine learning isn't built on misunderstood or overlooked data issues.

---

## 2.1 Basic Statistical Description

**What to do:**
1. Open `notebooks/loan_approval_analysis.ipynb`
2. Run the Task 2 cell that prints `df_retained.describe(include='all')`
3. Take a screenshot of the output table (no code visible)
4. Paste it below with a clear caption

### What the Statistics Tell Us

Looking at the summary output, several patterns emerge that will shape our modelling approach:

**Financial Features:**
- **Income** averages around $64,000, but there's huge variation—from just $4,200 up to $1.9 million. This wide range suggests we have everyone from low earners to very wealthy applicants.
- **Loan amounts** requested typically sit near $62,000, though again we see some extreme values. Most people borrow amounts roughly in line with their annual income.
- **Interest rates** hover around 12%, ranging from 0% (perhaps promotional offers) to 35% (high-risk lending).
- **Loan-to-Income ratios** average 0.64, meaning most people borrow less than their yearly salary—a sensible lending practice.

**Credit and Employment Features:**
- **Employment length** averages about 7 years, with some brand-new workers (0 years) and some long-term employees (up to 60 years—possibly data entry errors or very senior workers).
- **Credit history length** averages 17 years, suggesting most applicants have established credit records. The range from 0 to 73 years is interesting and worth investigating.

**Categorical Features:**
- **Education Qualifications** has six categories, but "Unknown" dominates—this might be a data quality issue we need to address.
- **Home Ownership** shows the expected categories: Rent, Mortgage, and Own, reflecting typical housing situations.
- **Loan Intent** varies across purposes like Education and Medical, each potentially carrying different approval rates.
- **Payment Default on File** is binary (Yes/No), a straightforward creditworthiness indicator.

**The Target Variable:**
- **Loan Approval Status** shows some inconsistency—we have variations like "Approved" and "APPROVED" that need standardizing before we can model effectively.

> **Evidence placeholder:** Paste your `describe()` screenshot here.  
> **Caption example:** "Figure 1: Summary statistics for retained variables showing central tendencies and ranges"

---

## 2.2 Understanding Variable Types

Knowing what type of data each variable contains is crucial because different algorithms handle different data types in different ways. Here's how we classify each retained variable:

### Numerical Variables

These are measurements or counts that can be ordered and calculated:

**Continuous (can take any value in a range):**
- **Income** – Any positive amount in pounds
- **Loan Amount** – Exact loan size requested
- **Loan Interest Rate** – Percentage rates with decimals
- **Loan-to-Income Ratio** – Calculated ratios between 0 and beyond 1
- **Maximum Loan Amount** – The amount a lender is willing to offer

**Discrete (whole numbers only):**
- **Employment Length** – Years worked (you can't work 7.5 years in this dataset)
- **Credit History Length** – Years on record, counted in whole years

### Categorical Variables

These are labels or categories without a natural numeric order:

**Nominal (no ranking among categories):**
- **Education Qualifications** – Degree names like "Graduate" or "High School"
- **Home Ownership** – Status like Rent, Mortgage, or Own
- **Loan Intent** – Purpose categories such as Education, Medical, Personal
- **Loan Approval Status (our target)** – Outcomes: Approved, Declined, etc.

**Binary (only two options):**
- **Payment Default on File** – Either "Yes" or "No"

### Why This Matters

Understanding these types helps us decide:
- Which variables need encoding (converting categories to numbers)
- Whether we need scaling (making numeric ranges comparable)
- Which algorithms are suitable (some work better with certain data types)

## 2.3 Target Variable Distribution: The Class Imbalance Problem

**What to do:**
1. Run the bar chart cell that plots `df['Loan Approval Status'].value_counts()`
2. Screenshot only the chart (no code)
3. Paste it below with a caption

### What We Discovered

The distribution chart reveals a significant imbalance in our target variable—and this finding has major implications for how we'll evaluate our models later:

- **Approved applications:** ~85.6% (50,210 records)
- **Declined applications:** ~14.1% (8,268 records)  
- **Other categories:** A handful of records with variants like "Reject" or "Accept" that we'll need to clean up

### Why This Matters

This imbalance isn't surprising—most lending institutions approve more loans than they decline. However, it creates a modelling challenge. If we built a naive model that simply predicted "Approved" for every single application, it would achieve 85.6% accuracy without learning anything useful!

This is why we'll need to focus on metrics like **Recall** and **Precision** for the minority class (Declined) when we evaluate our models in Task 5. We care about correctly identifying the declined applications, not just achieving high overall accuracy.

The imbalance also means we should:
- Use **stratified sampling** when splitting our data (ensuring train and test sets have the same proportions)
- Consider the **confusion matrix** carefully (not just accuracy scores)
- Potentially use techniques like **class weighting** if our model struggles with the minority class

> **Evidence placeholder:** Paste your bar chart screenshot here.  
> **Caption example:** "Figure 2: Distribution of Loan Approval Status showing class imbalance (85.6% approved vs 14.1% declined)"

---

## 2.4 Quick Reminders

When documenting your findings:
- Screenshot outputs only (no code visible in any image)
- Add clear, descriptive captions under each figure
- Save your notebook after running all cells to preserve the outputs
- Number your figures sequentially (Figure 1, Figure 2, etc.)
