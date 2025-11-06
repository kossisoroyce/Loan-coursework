# Task 3: Data Preparation - Quick Summary

## Issues Found and How They Were Fixed

### Issue 1: Inconsistent Loan Approval Status Labels
**Problem:** The target variable had multiple spellings (Approved, APPROVED, Accept, Declined, DECLINED, Reject) plus 1 missing value.

**Solution:** Standardized all variations to just two categories:
- "Approved" (includes: Approved, APPROVED, Accept)
- "Declined" (includes: Declined, DECLINED, Reject)
- Dropped the 1 row with missing value

**Why this works:** Machine learning models need consistent category names. Having "Approved" and "APPROVED" as separate categories would confuse the model.

---

### Issue 2: Inconsistent Payment Default Values
**Problem:** Binary variable had four spellings (Y, N, YES, NO) plus 5 missing values.

**Solution:** 
- Standardized to: Y and N only
- Filled missing values with "N" (the most common value)

**Why this works:** Binary variables should have exactly two values. Mode imputation (using the most frequent value) is appropriate when missing data is minimal (< 0.01%).

---

### Issue 3: Missing Loan Interest Rates
**Problem:** 11 records had missing interest rates.

**Solution:** Filled missing values with the median interest rate.

**Why this works:** Median is better than mean for financial data because it's not affected by extreme outliers (like the 35% maximum rate). This preserves the central tendency without introducing bias.

---

## Evidence Requirements for Report

For each issue, include TWO screenshots:

1. **"Before" screenshot** showing:
   - The problem (e.g., multiple category spellings)
   - Count of missing values
   - Value counts showing inconsistencies

2. **"After" screenshot** showing:
   - Cleaned categories
   - Zero missing values (or explained why some remain)
   - Confirmation the issue is resolved

**Annotation tip:** Add text labels like "Before cleaning: 8 unique categories" and "After cleaning: 2 categories" to make it crystal clear what was fixed.
