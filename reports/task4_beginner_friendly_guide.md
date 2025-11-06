# Task 4: Beginner-Friendly Implementation Guide
**Building Loan Approval Prediction Models - Explained Simply**

---

## 🎯 What Are We Actually Doing?

Imagine you work at a bank and need to decide which loan applications to approve. Instead of manually reviewing thousands of applications, we're teaching a computer to make these predictions by showing it examples of past decisions.

**Our Goal:** Build 3 different "prediction machines" (models) and see which one is best at predicting loan approvals.

---

## Part 1: Understanding the Three Algorithms (Task 4.a)

Think of algorithms as different "thinking styles" for making predictions. We're using three different styles:

### 🔍 1. Naïve Bayes (NB) - "The Pattern Matcher"

**Simple Explanation:**  
This is like a detective who looks at clues and calculates probabilities. It asks: "Among all approved loans, what percentage had a college degree? What about declined loans?" Then it uses these patterns to predict new applications.

**Real-World Example:**  
Like spam filters in your email - they learn that emails with certain words are usually spam, then use that pattern to identify new spam.

**How It Works:**
- Looks at each feature separately (education, home ownership, etc.)
- Calculates the chances (probability) of approval for each feature
- Combines these chances to make a final prediction
- Called "Naïve" because it assumes features don't affect each other (which isn't always true, but often works!)

**What It Learns:**
- Average values for each feature (like average income of approved vs declined)
- How spread out the values are
- How common each outcome is (85% approved vs 15% declined)

**What You Can Adjust (Hyperparameters):**
- `var_smoothing`: How much wiggle room to give when calculating probabilities (helps with rare cases)

**Python Package:**  
`from sklearn.naive_bayes import GaussianNB`

---

### 📊 2. Logistic Regression (LR) - "The Score Calculator"

**Simple Explanation:**  
This creates a scoring system, like a credit score. It assigns points to each feature (positive or negative) and adds them up. High total score = approved, low score = declined.

**Real-World Example:**  
Like a university admission system where you get points for GPA (+50), SAT scores (+40), extracurriculars (+10). Total above 100 = admitted.

**How It Works:**
- Assigns a weight (importance score) to each feature
- Multiplies your feature values by these weights
- Adds everything up to get a final score
- Converts the score to a probability (0% to 100% chance of approval)

**What It Learns:**
- The weight (coefficient) for each feature
  - Example: +0.5 for "College Degree" means it increases approval chances
  - Example: -0.3 for "Has Past Default" means it decreases approval chances
- A starting point (intercept) before adding feature scores

**What You Can Adjust (Hyperparameters):**
- `C`: How strictly to follow the patterns (lower = stricter, higher = more flexible)
- `penalty`: Whether to prevent the model from relying too heavily on any one feature
- `max_iter`: How many attempts to try finding the best weights

**Python Package:**  
`from sklearn.linear_model import LogisticRegression`

---

### 🌳 3. Random Forest (RF) - "The Committee Decision"

**Simple Explanation:**  
Instead of one prediction method, this creates 100 different "decision trees" (like flowcharts) and lets them vote. Majority vote wins!

**Real-World Example:**  
Like asking 100 loan officers to each review an application using slightly different criteria, then going with whatever most of them decide.

**How It Works:**
- Builds many decision trees (default: 100 trees)
- Each tree asks a series of yes/no questions:
  - "Is income above £50,000?" → Yes → Go right, No → Go left
  - "Does applicant own home?" → Yes → Go right, No → Go left
  - Keep asking questions until reaching a decision
- All trees vote, majority wins

**What It Learns:**
- Which questions to ask at each step
- What order to ask questions
- Where to split values (e.g., "income above £45,000" vs "above £60,000")

**What You Can Adjust (Hyperparameters):**
- `n_estimators`: How many trees to build (more trees = more accurate but slower)
- `max_depth`: How many questions each tree can ask (deeper = more complex)
- `min_samples_split`: Minimum applications needed before asking another question

**Python Package:**  
`from sklearn.ensemble import RandomForestClassifier`

---

## 📋 Quick Comparison: Which Algorithm When?

| Algorithm | Strengths | When to Use | Type |
|-----------|-----------|-------------|------|
| **Naïve Bayes** | Fast, works with small data | Quick predictions, simple patterns | Non-parametric* |
| **Logistic Regression** | Easy to interpret scores | Need to explain why decisions were made | Parametric* |
| **Random Forest** | Very accurate, handles complex patterns | Want best accuracy, lots of data | Non-parametric* |

**\*What does this mean?**
- **Parametric** (LR): Fixed number of things to learn (one weight per feature). Like a fixed-size form.
- **Non-parametric** (NB, RF): Can learn more as data grows. Like an expandable folder.

---

## Part 2: Algorithm Details Table (Copy This Into Your Report)

| Algorithm Name | Algorithm Type | What It Learns (Learnable Parameters) | What You Can Adjust (Hyperparameters) | Python Code |
|----------------|----------------|---------------------------------------|--------------------------------------|-------------|
| **Naïve Bayes (NB)** | Non-parametric | • How common each outcome is<br>• Average values for each feature<br>• How spread out values are | • `var_smoothing`: Smoothing for rare cases<br>• `priors`: Starting probabilities | `from sklearn.naive_bayes import GaussianNB` |
| **Logistic Regression (LR)** | Parametric | • Points (weights) for each feature<br>• Starting score (intercept) | • `C`: Flexibility level<br>• `penalty`: Type of restriction<br>• `solver`: Calculation method<br>• `max_iter`: Maximum attempts | `from sklearn.linear_model import LogisticRegression` |
| **Random Forest (RF)** | Non-parametric | • Which questions to ask<br>• Where to split values<br>• Final predictions at end of tree | • `n_estimators`: Number of trees<br>• `max_depth`: Tree depth<br>• `min_samples_split`: Minimum to split<br>• `max_features`: Features per split | `from sklearn.ensemble import RandomForestClassifier` |

---

## Part 3: Preparing the Data (What the Code Does)

### Step 1: Clean the Data 🧹

Think of this like preparing ingredients before cooking:

**What We're Doing:**
1. **Remove bad data entries**
   - 3 loans with negative amounts (impossible!)
   - People claiming 150 years of work experience (also impossible!)
   
2. **Fix messy text**
   - "Unknown " (with space) → "Unknown" (without space)
   - "HOMEIMPROVEMENT" → "Home Improvement" (more readable)
   
3. **Standardize the target**
   - "Approved", "APPROVED", "Accept" → all become "Approved"
   - "Declined", "DECLINED", "Reject" → all become "Declined"

**Result:** Clean data with no errors or missing values!

---

### Step 2: Choose Features to Use 🎯

**Remember:** The assignment says use **categorical features only** (features with categories, not numbers)

We're using these 3:
1. **Education Qualifications** (Unknown, High School, College, etc.)
2. **Home Ownership** (Rent, Mortgage, Own)
3. **Loan Intent** (Education, Medical, Personal, etc.)

**Why only these?**  
The assignment specifically asks for categorical features. We're saving numerical features (like income) for later tasks.

---

### Step 3: Convert Categories to Numbers 🔢

**The Problem:**  
Computers can't understand text like "College" or "Rent". They need numbers.

**The Solution: One-Hot Encoding**

This creates yes/no columns for each category:

**Before:**
```
Education: High School
Education: College
Education: College
```

**After:**
```
Education_High School: 1,  Education_College: 0
Education_High School: 0,  Education_College: 1
Education_High School: 0,  Education_College: 1
```

Each category gets its own column with 1 (yes) or 0 (no).

**Result:** 15 number columns that computers can use for predictions!

---

### Step 4: Split Into Training and Testing 📊

**The Analogy:**  
Imagine teaching a student for an exam. You give them practice problems (training set) and then test them on new problems they haven't seen (test set).

**What We Do:**
- **80% Training Set** (46,901 applications): The model learns from these
- **20% Test Set** (11,726 applications): We test accuracy on these

**Important Points:**

1. **Why 80:20?**
   - 80% gives enough examples to learn from
   - 20% gives enough examples to test accurately
   - This is the industry standard (like using inches and feet in construction)

2. **Random State = 42**
   - This is like setting a "bookmark" 
   - Ensures all 3 models practice and test on the EXACT same data
   - Makes fair comparisons possible
   - (42 is just a number; could be any number, but we stick with one)

3. **Stratification**
   - Makes sure training and test sets have same balance
   - Original: 85.6% approved, 14.4% declined
   - Training: 85.6% approved, 14.4% declined ✓
   - Test: 85.6% approved, 14.4% declined ✓
   - Like making sure both groups have same mix of people

---

## Part 4: Training the Models 🎓

### What "Training" Means

**The Analogy:**  
Like teaching someone to ride a bike by letting them practice. The more examples they see, the better they get.

**What Happens:**
1. Show the model the training data (46,901 examples)
2. Model looks for patterns
3. Model adjusts its internal settings
4. Repeat until patterns are learned

**For Each Algorithm:**

**Naïve Bayes:**
- Calculates: "Among approved loans, 40% had college degrees"
- Calculates: "Among declined loans, 25% had college degrees"
- Uses these percentages to predict new applications

**Logistic Regression:**
- Figures out: "College degree is worth +0.5 points"
- Figures out: "Owning home is worth +0.3 points"
- Creates scoring formula for new applications

**Random Forest:**
- Builds 100 decision trees
- Each tree learns slightly different patterns
- All trees vote on new applications

---

## Part 5: Understanding the Justifications

### Why 80:20 Split? (Task 4.b.ii)

**In Simple Terms:**

We need to split our data because:
1. **Training needs lots of examples** - Like studying from many practice problems
2. **Testing needs enough examples** - Like having enough test questions to check if you really learned

With 58,627 loan applications:
- 80% (46,901) for training = Enough to learn good patterns
- 20% (11,726) for testing = Enough to know if predictions are accurate

**Why not 90:10?** Test set would be too small (not reliable)  
**Why not 60:40?** Training set would have too few examples to learn well  
**80:20 is the "Goldilocks" split** - just right!

---

### Training-Test vs K-Fold Cross-Validation (Task 4.b.iii)

**Two Different Testing Methods:**

#### Method 1: Training-Test Split (What We're Using)

**The Analogy:**  
Study a textbook, then take one final exam.

**How It Works:**
1. Split data once into training (80%) and test (20%)
2. Train model on training set
3. Test once on test set
4. Done!

**Pros:** Fast, simple, works great with lots of data  
**Cons:** Only one test score (what if we got lucky/unlucky?)

**Use When:**
- You have lots of data (>10,000 examples) ✓ (We have 58,627!)
- You want final performance score
- Time/computer power is limited

---

#### Method 2: K-Fold Cross-Validation

**The Analogy:**  
Study a textbook, take 5 different exams on different chapters, average your scores.

**How It Works:**
1. Split data into 5 parts (folds)
2. Train on 4 parts, test on 1 part → Get score #1
3. Train on different 4 parts, test on remaining part → Get score #2
4. Repeat 5 times
5. Average all 5 scores → More reliable estimate!

**Pros:** More reliable, averages out luck  
**Cons:** Slower (trains 5 models instead of 1)

**Use When:**
- You have small data (<10,000 examples)
- You're testing different settings (hyperparameter tuning) ✓ (We'll use this in Task 5!)
- You want confident performance estimates

---

**Our Approach:**
- **Task 4:** Use train-test split for building models
- **Task 5:** Use k-fold for testing different settings (hyperparameter tuning)

Best of both worlds!

---

## Part 6: What the Code Output Means

### When You Run the Code, You'll See:

```
Final dataset: 58,627 rows
Data loss: 18 rows (0.031%)
Missing values: 0 ✓

Feature matrix shape: (58,627 samples × 15 features)
Target variable shape: (58,627 labels)

Training set: 46,901 samples × 15 features
Test set: 11,726 samples × 15 features

✓ Naive Bayes model trained successfully
✓ Logistic Regression model trained successfully
✓ Random Forest model trained successfully
```

**What This Means:**

| What You See | What It Means |
|--------------|---------------|
| **58,627 rows** | Total loan applications after cleaning |
| **Data loss: 0.031%** | Only lost 18 bad records (excellent!) |
| **Missing values: 0** | All gaps filled in |
| **15 features** | 3 categories converted to 15 yes/no columns |
| **46,901 training** | Examples the models learn from (80%) |
| **11,726 test** | Examples we use to check accuracy (20%) |
| **Models trained** | All 3 prediction machines ready to use! |

---

## 📸 Screenshots You Need (Task 4 Requirements)

### Screenshot 1: Feature Names and Data Shape
**What to capture:** The list showing all 15 feature names and the shape (58,627 × 15)

**Why:** Proves you're using the correct features and data size

**Caption example:**  
*"Feature names after converting categories to numbers, showing 15 features from 3 categorical variables (58,627 loan applications)"*

---

### Screenshot 2: Train-Test Split Verification
**What to capture:** The comparison showing 85.6% approved in original, training, and test sets

**Why:** Proves the split maintains the same balance

**Caption example:**  
*"Verification that stratified sampling maintains 85.6% approved and 14.4% declined ratio in both training and test sets"*

---

### Screenshot 3: Code Line for Random State
**What to capture:** The code line with `random_state=42`

**Why:** Proves all models use the same test data

**Caption example:**  
*"Code showing random_state=42 ensures all three models tested on identical test dataset for fair comparison"*

---

## ✅ Final Checklist (Before Moving to Task 5)

- [ ] I understand what each algorithm does (in simple terms)
- [ ] I copied the algorithm table into my report
- [ ] My code ran without errors
- [ ] I have 58,627 clean rows with 0 missing values
- [ ] I have exactly 2 target classes (Approved, Declined)
- [ ] I have 15 features after encoding
- [ ] Training and test sets have same proportions (85.6% / 14.4%)
- [ ] All 3 models trained successfully
- [ ] I took all 3 required screenshots
- [ ] I wrote my justifications (under 100 words each)

---

## 🎓 Key Takeaways (Remember These!)

1. **Three algorithms = three different "thinking styles"** for predictions
   - Naïve Bayes: Pattern matcher using probabilities
   - Logistic Regression: Score calculator  
   - Random Forest: Committee of decision trees

2. **We use categorical features only** (as per assignment)
   - Education, Home Ownership, Loan Intent
   - Converted to 15 yes/no columns

3. **80:20 split is standard** and works well with our large dataset

4. **random_state=42 is crucial** - ensures fair comparison between models

5. **Stratification maintains balance** - keeps 85.6% / 14.4% ratio in both sets

---

## 🚀 You're Ready!

You now understand:
- ✅ What each algorithm does (in plain English)
- ✅ Why we chose these approaches
- ✅ What the code does at each step
- ✅ What the output means

**Next:** Task 5 will evaluate which model is best using confusion matrices and accuracy metrics!

---

**Questions to Ask Yourself:**
- Can I explain each algorithm to a friend who doesn't know machine learning?
- Do I understand why we split 80:20?
- Do I know what one-hot encoding does?
- Can I explain what random_state=42 guarantees?

If yes to all → You're ready! 🎉
