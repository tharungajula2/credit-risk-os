---
title: "Employee Retention & Performance Analytics — Attrition Modeling with Logistic Regression, Decision Trees, and Random Forests"
date: 2026-03-17
tags:
  - employee-retention
  - attrition-modeling
  - human-resources
  - classification
  - logistic-regression
  - decision-trees
  - random-forest
  - grid-search
  - feature-engineering
  - model-evaluation
  - data-science
  - supervised-learning
  - binary-classification
  - ROC-AUC
  - precision
  - recall
cluster: Phase 7 — Applied Machine Learning Projects
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_full_pd_model]]"
  - "[[2_monitoring_model]]"
  - "[[3_lgd_ead_model_rewritten]]"
  - "[[4_ecl_cecl_stress_testing_rewritten]]"
  - "[[5_bank_churn_neural_networks_masterclass]]"
---

---

# Employee Retention & Performance Analytics — Attrition Modeling with Logistic Regression, Decision Trees, and Random Forests

> This note is my full technical record of how I use an **employee attrition project** to understand core data-science ideas from first principles: binary classification, exploratory data analysis, preprocessing, logistic regression, decision trees, random forests, feature engineering, model evaluation, and practical business interpretation.
>
> I use this project as a beginner-friendly anchor to learn both the **math** and the **code** behind predictive modeling. Even though this is an HR attrition problem rather than a credit-risk problem, the workflow is extremely useful for me because the same ideas appear again in PD models, churn models, fraud models, early-warning systems, collections prioritization, and model monitoring.

---

## The Project at a Glance

**Dataset:** Employee retention dataset loaded from `Salfort_HR_comma_sep.csv`

**Raw dataset shape:** `14,999 rows × 10 columns`

**After duplicate removal:** `11,991 rows × 10 columns`

**Objective:** Predict whether an employee will **leave the company**

**Target variable:** `left`

- `0` = employee stayed
- `1` = employee left

**Why this project matters to me:**

This is a very strong beginner project because it teaches the full supervised-learning pipeline on a realistic business problem:

- how to define a binary target
- how to inspect columns and clean data
- how to think about duplicates and outliers
- how logistic regression converts inputs into probabilities
- how tree-based models capture non-linear patterns
- how feature engineering changes model performance
- how to compare models using **precision**, **recall**, **F1**, **accuracy**, and **ROC-AUC**
- how to explain model results in business language

**The notebook pipeline:**

```text
Raw employee dataset
        │
        ▼
Understand columns and target variable
        │
        ▼
EDA + data-quality checks
        │
        ▼
Remove duplicates
        │
        ▼
Model 1: Logistic regression
        │
        ▼
Model 2: Decision tree + random forest
        │
        ▼
Feature engineering (`overworked`)
        │
        ▼
Rebuild tree-based models
        │
        ▼
Compare performance and interpret business meaning
```

---

## Part 1: What the Business Problem Actually Is

### The Concept

An **attrition model** predicts the probability that an employee will leave.

That matters because attrition is expensive. When employees leave, the organization can lose:

- productivity
- team continuity
- domain knowledge
- recruiting and onboarding cost
- manager time
- morale and stability in teams

If I can identify which employees are at high risk of leaving, the business can intervene with actions such as:

- workload redesign
- compensation review
- manager support
- promotion pathways
- role rotation
- retention planning

### The Data-Science Framing

This is a **binary classification** problem.

For each employee, I observe a feature vector:

```text
x = [satisfaction, evaluation, number of projects, hours worked, tenure, ...]
```

The model estimates:

```text
P(left = 1 | x)
```

That probability can then be converted into business actions such as:

- high-risk employee
- medium-risk employee
- low-risk employee

### Why This Matters for Interviews

This project lets me explain basic but very important concepts clearly:

- what a target variable is
- what features are
- how classification differs from regression
- why probability estimates matter
- why high accuracy can still be misleading
- how model choice depends on the decision problem

These are core ideas across many analytics and quant interviews.

---

## Part 2: Understanding the Dataset Properly

### Raw Columns

The notebook starts with these columns:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_montly_hours`
- `time_spend_company`
- `Work_accident`
- `left`
- `promotion_last_5years`
- `Department`
- `salary`

The notebook then renames some columns for clarity:

- `average_montly_hours` → `average_monthly_hours`
- `time_spend_company` → `tenure`
- `Work_accident` → `work_accident`
- `Department` → `department`

### What Each Variable Means

#### Continuous / numeric variables

- `satisfaction_level`: employee satisfaction score, usually on a 0 to 1 scale
- `last_evaluation`: last performance evaluation score
- `number_project`: how many projects the employee handled
- `average_monthly_hours`: average working hours per month
- `tenure`: number of years spent in the company

#### Binary indicator variables

- `work_accident`: whether the employee had a work accident
- `promotion_last_5years`: whether the employee was promoted recently
- `left`: whether the employee left the company

#### Categorical variables

- `department`: employee function or department
- `salary`: salary band such as low, medium, high

### Basic Data Quality Checks

The notebook shows:

- **no missing values**
- **14,999 rows**
- **10 columns**
- **3,008 duplicate rows**

So one of the first modeling decisions is to remove duplicates.

### Why Duplicate Handling Matters

If duplicate rows represent repeated identical records rather than real repeated events, keeping them can distort:

- class proportions
- correlations
- split quality
- model fit
- validation results

The notebook removes duplicates and keeps the first occurrence:

```python
df1 = df0.drop_duplicates(keep='first')
```

After this step, the working dataset has:

```text
14,999 - 3,008 = 11,991 rows
```

That is an important cleaning step.

---

## Part 3: Exploratory Data Analysis (EDA)

EDA is where I try to understand the problem before jumping into modeling.

The notebook explores:

- descriptive statistics
- a correlation heatmap for key numeric variables
- attrition counts by department
- outliers in `tenure`

### The Correlation Heatmap

The heatmap uses these variables:

- `satisfaction_level`
- `last_evaluation`
- `number_project`
- `average_monthly_hours`
- `tenure`

The main purpose is not to prove causality. It is simply to check:

- which numeric variables move together
- whether some variables may be redundant
- whether there may be multicollinearity issues in linear models

### Department-Level Attrition Plot

The notebook uses a stacked bar chart of `department` against `left`.

This helps answer a practical question:

> Are some departments showing more employee exits than others?

That matters because a model is useful, but business intervention usually happens at the team or department level.

### Outlier Check on Tenure

The notebook computes the interquartile range for `tenure`:

```python
percentile25 = df1['tenure'].quantile(0.25)
percentile75 = df1['tenure'].quantile(0.75)
iqr = percentile75 - percentile25

lower_limit = percentile25 - 1.5 * iqr
upper_limit = percentile75 + 1.5 * iqr
```

It finds:

- **lower limit:** `1.5`
- **upper limit:** `5.5`
- **rows flagged as tenure outliers:** `824`

### What This Means

Any tenure below 1.5 years or above 5.5 years is treated as an outlier by the IQR rule.

For a general data-science interview, I should know two things:

1. The IQR rule is a **simple statistical rule**, not a business truth.
2. Long-tenure employees are not necessarily “bad data.” They may actually be meaningful.

So outlier removal should be justified carefully. In this notebook, outlier filtering is used only for the logistic regression branch.

---

## Part 4: Logistic Regression — The First Baseline Model

## The Core Idea

Logistic regression is a classification model that estimates the probability of class 1.

If I denote the employee feature vector by \(x\), the model computes a linear score:

```text
z = β0 + β1x1 + β2x2 + ... + βkxk
```

Then it pushes that score through the **sigmoid** function:

```text
p = 1 / (1 + e^(-z))
```

This ensures the output is always between 0 and 1.

### Why the Sigmoid Function Matters

A plain linear model can output anything from negative infinity to positive infinity.

That is not valid for a probability.

The sigmoid fixes this by converting any real number into a value in `(0, 1)`.

### The Log-Odds Interpretation

Logistic regression can also be written as:

```text
log(p / (1 - p)) = β0 + β1x1 + ... + βkxk
```

This is called the **logit** form.

It tells me that each coefficient changes the **log-odds** of attrition, holding other variables fixed.

For interviews, I do not need to go too deep into coefficient interpretation here, because the notebook uses many dummy variables and the final best model is not logistic regression. But I should still understand the equation and the idea of probability mapping.

---

## Part 5: Preparing Data for Logistic Regression

### One-Hot Encoding

The notebook converts categorical variables into dummy variables:

```python
df_enc = pd.get_dummies(
    df1,
    prefix=['salary', 'dept'],
    columns=['salary', 'department'],
    drop_first=False
)
```

This creates indicator columns such as:

- `salary_high`
- `salary_low`
- `salary_medium`
- `dept_IT`
- `dept_sales`
- `dept_support`
- and others

### Important Modeling Note

Because `drop_first=False`, the notebook keeps **all** dummy columns for each category.

That is not ideal for a classical unregularized logistic regression because it creates perfect multicollinearity inside each categorical group.

In `scikit-learn`, logistic regression usually still runs because regularization is applied by default. But from a modeling point of view, a cleaner setup would usually be:

```python
pd.get_dummies(..., drop_first=True)
```

or use an explicit encoding pipeline.

### Tenure Outlier Filtering

The logistic regression branch removes tenure outliers:

```python
df_logreg = df_enc[
    (df_enc['tenure'] >= lower_limit) &
    (df_enc['tenure'] <= upper_limit)
]
```

This leaves a dataset where the class distribution is:

- `left = 0`: about **83.15%**
- `left = 1`: about **16.85%**

So the minority class is employees who leave.

### Feature Matrix and Target

The notebook defines:

```python
y = df_logreg['left']
X = df_logreg[[...selected features...]]
```

Then it splits the data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Part 6: Logistic Regression Results and What They Mean

The notebook fits:

```python
log_clf = LogisticRegression(random_state=42, max_iter=500)
```

Then it predicts on the test set and evaluates the results.

### Test Classification Report

The notebook reports approximately:

- **accuracy:** `0.82`
- **precision for leavers:** `0.49`
- **recall for leavers:** `0.26`
- **F1 for leavers:** `0.34`

### What This Tells Me

This model is not strong at identifying employees who will leave.

Even though the overall accuracy is 82%, the recall for the “leave” class is only 26%.

That means:

> Out of all employees who actually left, the model correctly identified only about one quarter.

This is a classic lesson in imbalanced classification:

- **accuracy can look acceptable**
- but minority-class detection can still be poor

### Why Logistic Regression Underperformed Here

There are several likely reasons:

1. **Class imbalance** makes it harder to identify leavers.
2. The relationship between predictors and attrition is probably **non-linear**.
3. Interactions may matter, such as:
   - high evaluation + too many projects
   - long hours + no promotion
   - low satisfaction + high workload
4. Outlier filtering may remove some informative cases.
5. Logistic regression assumes a more structured decision boundary than tree models.

### Why I Still Keep This Model

I still want logistic regression in the project because it teaches me:

- probability-based classification
- the sigmoid function
- odds and log-odds
- linear decision boundaries
- the difference between a baseline model and a better model

That is very useful for interviews.

---

## Part 7: Decision Trees — Learning Non-Linear Rules

## The Core Idea

A decision tree predicts the target by splitting the data step by step.

For example, a tree might ask questions like:

```text
Is satisfaction_level < 0.47?
Is number_project >= 6?
Is average_monthly_hours > 210?
```

Each split tries to separate leavers from non-leavers more clearly.

### The Math Behind Tree Splits

A classification tree usually chooses splits that reduce impurity.

A common impurity measure is **Gini impurity**:

```text
Gini = 1 - Σ p_k^2
```

For binary classification:

```text
Gini = 1 - (p0^2 + p1^2)
```

Where:

- `p0` = proportion of class 0 in the node
- `p1` = proportion of class 1 in the node

The tree looks for splits that produce child nodes with lower impurity.

### Why Trees Are Useful Here

Trees can naturally capture:

- non-linear patterns
- thresholds
- interactions
- mixed variable types after encoding

That is why they often work better than logistic regression on problems like attrition.

---

## Part 8: Decision Tree Model Setup

The notebook creates a clean three-way split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=0
)
```

This effectively gives:

- training subset for fitting candidate models
- validation subset for model comparison
- test subset for final evaluation

That is a stronger workflow than fitting directly on all data and checking only one final score.

### Hyperparameter Tuning with Grid Search

The tree model is tuned with:

- `max_depth`
- `min_samples_leaf`
- `min_samples_split`

and evaluated with multiple metrics:

- accuracy
- precision
- recall
- F1
- ROC-AUC

The notebook refits on the model with the best **ROC-AUC**.

### Best Decision Tree Parameters

The best decision-tree settings are:

```text
max_depth = 4
min_samples_leaf = 1
min_samples_split = 2
```

### Decision Tree Performance

The notebook reports:

- **cross-validation AUC:** `0.9704`
- **validation AUC:** `0.952`
- **validation precision:** `0.924`
- **validation recall:** `0.920`
- **validation F1:** `0.922`
- **validation accuracy:** `0.974`

This is already far better than the logistic regression baseline.

---

## Part 9: Random Forest — A More Powerful Tree Ensemble

## The Core Idea

A random forest is an **ensemble** of many decision trees.

Instead of relying on one tree, it builds many trees on different bootstrap samples and combines their predictions.

For classification, the forest typically predicts by majority vote, and probabilities are obtained by averaging class probabilities across trees.

### Why Ensembles Help

A single decision tree is easy to understand, but it can be unstable.

A random forest reduces variance by averaging across many trees.

This usually improves:

- generalization
- robustness
- predictive accuracy

### Best Random Forest Parameters

The best random-forest settings in the first tree-based branch are:

```text
max_depth = 5
max_features = 1.0
max_samples = 0.7
min_samples_leaf = 1
min_samples_split = 3
n_estimators = 500
```

### Random Forest Performance

The notebook reports:

#### Cross-validation

- **CV AUC:** `0.9796`
- **CV precision:** `0.9436`
- **CV recall:** `0.9230`
- **CV F1:** `0.9331`
- **CV accuracy:** `0.9780`

#### Validation

- **validation AUC:** `0.954`
- **validation precision:** `0.955`
- **validation recall:** `0.917`
- **validation F1:** `0.936`
- **validation accuracy:** `0.979`

#### Final test set

- **test AUC:** `0.955`
- **test precision:** `0.961`
- **test recall:** `0.917`
- **test F1:** `0.938`
- **test accuracy:** `0.980`

### What This Means

This is the strongest branch of the notebook.

The random forest performs very well across all major metrics and clearly outperforms the logistic regression baseline.

That suggests the attrition problem contains:

- non-linear effects
- interaction effects
- threshold behavior

which tree ensembles are capturing more effectively than logistic regression.

---

## Part 10: Feature Engineering — Creating the `overworked` Variable

The notebook then creates a new engineered feature called `overworked`.

### What the Notebook Does

It starts with:

```python
df3['overworked'] = df3['average_monthly_hours']
```

and then converts it to:

```python
df3['overworked'] = (df3['overworked'] > 175).astype(int)
```

So the variable becomes:

- `1` if average monthly hours > 175
- `0` otherwise

### Important Correction

One notebook comment says:

```text
Define `overworked` as working > 175 hrs/week
```

That wording is incorrect.

The column being used is **average monthly hours**, so the correct interpretation is:

> `overworked = 1` if average monthly hours > 175

not “per week.”

### Why This Feature Makes Sense

This is a simple business-driven feature.

Instead of letting the model work only with raw hours, I create a threshold variable that asks:

> Is this employee working unusually high hours?

That can sometimes make patterns easier for the model to learn and easier for people to interpret.

### Another Important Modeling Choice

The notebook also drops `satisfaction_level` before this branch:

```python
df3 = df1.drop('satisfaction_level', axis=1)
```

This is a strong choice because satisfaction is often one of the most predictive variables in attrition problems.

So the second branch is not just “adding `overworked`.” It is also **removing a very informative original variable**.

That makes the comparison especially important.

---

## Part 11: Tree Models After Feature Engineering

After creating `overworked` and dropping `average_monthly_hours`, the notebook rebuilds tree-based models.

### Decision Tree 2

Best parameters:

```text
max_depth = 6
min_samples_leaf = 1
min_samples_split = 4
```

Performance:

- **CV AUC:** `0.9535`
- **validation AUC:** `0.942`
- **validation precision:** `0.883`
- **validation recall:** `0.907`
- **validation F1:** `0.895`
- **validation accuracy:** `0.965`

### Random Forest 2

Best parameters:

```text
max_depth = None
max_features = 1.0
max_samples = 0.7
min_samples_leaf = 3
min_samples_split = 2
n_estimators = 300
```

Performance:

- **CV AUC:** `0.9657`
- **test AUC:** `0.935`
- **test precision:** `0.898`
- **test recall:** `0.889`
- **test F1:** `0.894`
- **test accuracy:** `0.965`

### What I Learn from This

The engineered-feature branch performs **worse** than the earlier random-forest branch.

That teaches a very important lesson:

> Feature engineering is not automatically helpful.

In this project:

- the original variables, especially `satisfaction_level`, already carried very strong predictive information
- the `overworked` threshold simplified one part of the story
- but the simplification did **not** beat the richer original information

This is a great interview point because it shows I understand that feature engineering should be tested empirically, not assumed to help.

---

## Part 12: Comparing the Modeling Approaches Clearly

| Model | Main idea | Strengths | Weaknesses | Key result |
|---|---|---|---|---|
| Logistic regression | Linear model with sigmoid probability mapping | Interpretable baseline, teaches log-odds | Weak on non-linear patterns, poor minority-class recall here | Accuracy 0.82, recall for leavers 0.26 |
| Decision tree | Sequential rule-based splits | Captures non-linearity and interactions, easy to visualize | Can overfit, unstable alone | Validation AUC about 0.952 |
| Random forest | Ensemble of many trees | Strong predictive power, robust, handles interactions well | Less interpretable than single tree | Best branch test AUC 0.955 |
| Feature-engineered random forest | Tree ensemble with `overworked` feature and dropped satisfaction | More business-oriented feature definition | Lost signal from original variables | Test AUC 0.935, worse than original RF |

### My Final Modeling Conclusion

The strongest model in this notebook is the **first random forest**, before the `overworked` feature branch.

That is the model I would treat as the best predictive model from this notebook.

---

## Part 13: The Most Important Math I Should Know for Interviews

I do not need PhD-level math here, but I should know the core formulas and what they mean.

### 1. Logistic Regression Probability

```text
p = 1 / (1 + e^(-z))
```

Where:

```text
z = β0 + β1x1 + ... + βkxk
```

This converts a linear score into a probability.

### 2. Log-Odds

```text
log(p / (1 - p)) = β0 + β1x1 + ... + βkxk
```

This is the linear structure underneath logistic regression.

### 3. Gini Impurity for Trees

```text
Gini = 1 - Σ p_k^2
```

Lower Gini means the node is purer.

### 4. Precision

```text
Precision = TP / (TP + FP)
```

Of the employees I predicted would leave, how many actually left?

### 5. Recall

```text
Recall = TP / (TP + FN)
```

Of the employees who actually left, how many did I successfully identify?

### 6. F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This balances precision and recall.

### 7. Accuracy

```text
Accuracy = (TP + TN) / Total
```

Useful, but can be misleading when classes are imbalanced.

### 8. ROC-AUC

ROC-AUC measures how well the model ranks positives above negatives across thresholds.

A higher AUC means better separation between leavers and non-leavers.

---

## Part 14: Code Patterns I Should Understand Properly

### Logistic Regression Flow

```python
df_enc = pd.get_dummies(df1, columns=['salary', 'department'])
X = df_enc.drop('left', axis=1)
y = df_enc['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

log_clf = LogisticRegression(max_iter=500, random_state=42)
log_clf.fit(X_train, y_train)

y_pred = log_clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Decision Tree with Grid Search

```python
tree = DecisionTreeClassifier(random_state=0)

cv_params = {
    'max_depth': [4, 6, 8, None],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 4, 6]
}

tree_grid = GridSearchCV(
    tree,
    cv_params,
    scoring={'accuracy', 'precision', 'recall', 'f1', 'roc_auc'},
    refit='roc_auc',
    cv=4
)

tree_grid.fit(X_tr, y_tr)
```

### Random Forest with Grid Search

```python
rf = RandomForestClassifier(random_state=0)

cv_params = {
    'max_depth': [3, 5, None],
    'max_features': [1.0],
    'max_samples': [0.7, 1.0],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 3, 4],
    'n_estimators': [300, 500]
}

rf_grid = GridSearchCV(
    rf,
    cv_params,
    scoring={'accuracy', 'precision', 'recall', 'f1', 'roc_auc'},
    refit='roc_auc',
    cv=4
)

rf_grid.fit(X_tr, y_tr)
```

### Feature Engineering Pattern

```python
df3 = df1.drop('satisfaction_level', axis=1)
df3['overworked'] = (df3['average_monthly_hours'] > 175).astype(int)
df3 = df3.drop('average_monthly_hours', axis=1)
```

That is a very common interview pattern:

- start with raw features
- create an interpretable business feature
- compare old and new performance

---

## Part 15: What This Project Teaches Me About Practical Analytics Work

This project is useful because it is not only about getting a score.

It also teaches how practical analytics work is actually done:

### 1. Start with a clear target

The target here is simple:

```text
Will the employee leave?
```

If the target is unclear, the whole modeling project becomes weak.

### 2. Data quality matters before modeling

Checking:

- nulls
- duplicates
- column names
- class balance
- outliers

is part of the real work.

### 3. A baseline model matters

Even if logistic regression is not the winner, it is still useful because it gives me a reference point.

### 4. Better models are not always linear

The random forest wins because the problem is not purely linear.

### 5. Feature engineering should be tested, not assumed

The `overworked` feature was a reasonable idea, but it did not outperform the original richer setup.

### 6. Metrics must match the business goal

If the goal is to identify at-risk employees, then recall and precision matter much more than headline accuracy alone.

---

## Part 16: How I Would Explain This in an Interview

A good concise explanation would be:

> I built an employee attrition classification workflow starting with data cleaning, duplicate removal, exploratory analysis, and baseline logistic regression. I then compared tree-based models using stratified train-validation-test splits and `GridSearchCV` with ROC-AUC as the refit metric. The best-performing model was a random forest with a test AUC of about 0.955 and strong precision and recall. I also tested feature engineering by creating an `overworked` indicator from average monthly hours, but that branch performed worse after dropping `satisfaction_level`, which taught me that feature engineering should be validated empirically rather than assumed to improve performance.

That answer shows:

- business framing
- modeling pipeline
- metric awareness
- comparison logic
- practical judgment

---

## Part 17: How This Connects to Banking and Risk Analytics

Even though this is an HR project, the modeling logic transfers directly to banking work.

### Shared concepts with risk modeling

- binary target definition
- feature engineering
- train / validation / test design
- imbalanced classification
- model selection
- precision / recall tradeoff
- business thresholding
- monitoring drift and performance later

### Examples of similar banking use cases

The same workflow structure can be used for:

- churn prediction for bank customers
- delinquency prediction
- collections prioritization
- fraud suspicion flags
- early-warning systems
- marketing response models
- complaint escalation models

So this project still strengthens my foundation for quant and analytics interviews in banking.

---

## Part 18: Limitations and How I Would Improve the Notebook

This notebook is strong for learning, but I should also know how to improve it.

### 1. Use a pipeline object

A production version should put preprocessing and modeling into a reproducible pipeline.

### 2. Handle categorical encoding more cleanly

For linear models, I would usually prefer `drop_first=True` or `OneHotEncoder`.

### 3. Consider probability calibration

If decisions depend on probability thresholds, I would check whether predicted probabilities are well calibrated.

### 4. Tune the classification threshold

The notebook mostly compares default thresholds. In practice, I would tune the threshold based on business cost.

### 5. Add feature importance or SHAP

For tree-based models, I would want:

- feature importance
- permutation importance
- SHAP values

to understand the strongest drivers of attrition.

### 6. Be careful with outlier removal

Long tenure may be an informative segment rather than bad data.

### 7. Consider fairness and governance

In real workforce analytics, feature usage and downstream actions must be handled carefully from an ethics and governance perspective.

---

## Part 19: The Key Lessons I Want to Retain

### Technical lessons

- logistic regression is a strong baseline but may miss non-linear relationships
- decision trees capture thresholds and interactions
- random forests usually improve stability and predictive power
- ROC-AUC is very useful for comparing classification models
- precision and recall matter more than accuracy when the minority class matters
- feature engineering can help, but only when tested properly

### Practical lessons

- data cleaning is part of modeling, not a separate afterthought
- duplicate handling can materially change the dataset
- business interpretation matters as much as model performance
- a simpler engineered feature is not always better than a strong original feature
- a model should be judged by how it supports decisions, not only by one metric

---

## Part 20: Final Summary

This project helps me understand binary classification from multiple angles.

I start with a realistic attrition problem, clean the data, inspect duplicates and outliers, build a logistic regression baseline, move to decision trees and random forests, engineer an `overworked` variable, and compare what happens to model performance.

The most important result is that the **first random forest** is the strongest model in the notebook, with test performance around:

- **AUC:** `0.955`
- **precision:** `0.961`
- **recall:** `0.917`
- **F1:** `0.938`
- **accuracy:** `0.980`

That makes this notebook a very good masterclass project for learning:

- classification
- preprocessing
- logistic regression
- tree-based methods
- model comparison
- feature engineering
- practical business interpretation

It also becomes a strong bridge between beginner data science and more advanced applied analytics work.

---
