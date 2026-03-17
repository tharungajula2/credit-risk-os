---
title: "Socio-Economic Household Classification — Rural vs Urban Modeling with Logistic Regression, Trees, XGBoost, PCA, and Ensembles"
date: 2026-03-17
tags:
  - socio-economic-classification
  - household-analytics
  - rural-vs-urban
  - classification
  - logistic-regression
  - decision-tree
  - random-forest
  - xgboost
  - pca
  - smote
  - feature-engineering
  - model-evaluation
  - data-science
  - supervised-learning
cluster: "04 — Applied Machine Learning Projects"
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_full_pd_model]]"
  - "[[2_monitoring_model]]"
  - "[[3_lgd_ead_model_rewritten]]"
  - "[[4_ecl_cecl_stress_testing_rewritten]]"
  - "[[5_bank_churn_neural_networks_masterclass]]"
  - "[[6_employee_retention_tree_models_masterclass]]"
---

---

# Socio-Economic Household Classification — Rural vs Urban Modeling with Logistic Regression, Trees, XGBoost, PCA, and Ensembles

> This note is my full technical record of how I use a **socio-economic household dataset** to understand classification from first principles.
>
> I use this project to learn how to go from raw tabular data to a full modeling pipeline: data understanding, cleaning, missing-value treatment, outlier handling, feature engineering, train-test split, scaling, encoding, baseline models, tree models, boosting, dimensionality reduction, class imbalance handling, and final model selection.
>
> Even though this project is not a credit-risk model, it is still extremely useful for me because the same workflow appears in PD modeling, churn prediction, fraud detection, customer segmentation, collections prioritization, early-warning systems, and model monitoring.

---

## The Project at a Glance

**Dataset:** Household-level socio-economic dataset

**Raw dataset shape:** `176,661 rows × 38 columns`

**After dropping clearly irrelevant columns:** `176,661 rows × 27 columns`

**After further cleanup of redundant fields:** `176,661 rows × 23 columns`

**After duplicate removal:** `119,104 rows × 23 columns`

**After dropping rows where all major income fields were missing:** `118,885 rows × 23 columns`

**Target variable:** `REGION_TYPE`

- `Rural`
- `Urban`

**Main objective:** Predict whether a household is **Rural** or **Urban** using socio-economic, demographic, and income-related features.

**Final train-test feature matrix after preprocessing:** `109 features`

**Why this project matters to me:**

This is a very useful beginner project because it teaches me how classification works when:

- the target is categorical
- the data is messy
- several variables are categorical
- numeric features have outliers
- the classes are imbalanced
- accuracy alone is not enough
- feature engineering can improve performance
- tree-based models and boosting often outperform simple linear baselines

---

## The Full Pipeline I Built

```text
Raw household dataset
        │
        ▼
Understand columns and target
        │
        ▼
Drop irrelevant identifiers and weights
        │
        ▼
EDA + data quality checks
        │
        ▼
Remove duplicates
        │
        ▼
Replace sentinel values (-99) with missing values
        │
        ▼
Drop rows with all key income fields missing
        │
        ▼
Cap outliers + prevent negative income values
        │
        ▼
Engineer new features
        │
        ▼
Train-test split
        │
        ▼
Scale numerical features + one-hot encode categoricals
        │
        ▼
Baseline models: Logistic Regression, Decision Tree
        │
        ▼
Advanced models: ANN, Random Forest, XGBoost
        │
        ▼
Model refinement: tuning, top-feature subset, SMOTE, PCA, ensemble
        │
        ▼
Final XGBoost model + feature importance interpretation
```

---

## Part 1: What the Business Problem Actually Is

At the highest level, this is a **binary classification** problem.

For each household, I observe a vector of features such as:

- occupation group
- education group
- gender group
- household size group
- total income
- income from wages
- income from pension
- income from government transfers
- income from self-production
- income from business profit
- income from rent
- month slot
- state

Using those inputs, I want a model that estimates:

```text
P(REGION_TYPE = Urban | x)
```

or equivalently:

```text
P(REGION_TYPE = Rural | x)
```

depending on how I encode the target.

### Why this matters analytically

A project like this teaches me how socio-economic structure appears in the data:

- rural households may have stronger links to self-production, agricultural occupations, and government transfers
- urban households may have stronger wage income, different education profiles, different occupation mixes, and different household-size patterns

Even when I later work on credit problems, the same modeling logic still matters because I often classify:

- good vs bad accounts
- churn vs retain
- fraud vs non-fraud
- delinquent vs current
- default vs non-default
- high-risk vs low-risk segments

So this project is a classification masterclass in a non-credit setting.

---

## Part 2: Understanding the Data

From the notebook, the dataset starts with **38 columns**. The variables fall into a few broad groups.

### 1. Identifier / sampling / survey administration fields

These include fields such as:

- `HH_ID`
- `HH_WEIGHT_MS`
- `HH_WEIGHT_FOR_COUNTRY_MS`
- `HH_WEIGHT_FOR_STATE_MS`
- `HH_NON_RESPONSE_MS`
- `HH_NON_RESPONSE_FOR_COUNTRY_MS`
- `HH_NON_RESPONSE_FOR_STATE_MS`
- `HR`
- `DISTRICT`
- `STRATUM`
- `PSU_ID`

These are useful for survey administration, weighting, or sampling design, but they are not directly useful as standard predictors in a straightforward classification model like this notebook.

### 2. Demographic / grouping fields

Examples include:

- `STATE`
- `MONTH_SLOT`
- `AGE_GROUP`
- `OCCUPATION_GROUP`
- `EDUCATION_GROUP`
- `GENDER_GROUP`
- `SIZE_GROUP`

These are categorical features and need encoding before most ML models can use them.

### 3. Income-related fields

Examples include:

- `TOTAL_INCOME`
- `INCOME_OF_ALL_MEMBERS_FROM_ALL_SOURCES`
- `INCOME_OF_ALL_MEMBERS_FROM_WAGES`
- `INCOME_OF_ALL_MEMBERS_FROM_PENSION`
- `INCOME_OF_ALL_MEMBERS_FROM_DIVIDEND`
- `INCOME_OF_ALL_MEMBERS_FROM_INTEREST`
- `INCOME_OF_ALL_MEMBERS_FROM_FD_PF_INSURANCE`
- `INCOME_OF_HOUSEHOLD_FROM_ALL_SOURCES`
- `INCOME_OF_HOUSEHOLD_FROM_RENT`
- `INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION`
- `INCOME_OF_HOUSEHOLD_FROM_PRIVATE_TRANSFERS`
- `INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS`
- `INCOME_OF_HOUSEHOLD_FROM_BUSINESS_PROFIT`
- `INCOME_OF_HOUSEHOLD_FROM_SALE_OF_ASSET`
- `INCOME_OF_HOUSEHOLD_FROM_GAMBLING`

These are numeric features and form the core economic signal in the project.

### 4. Target variable

```text
REGION_TYPE ∈ {Rural, Urban}
```

This is the label the model tries to predict.

---

## Part 3: Dropping Irrelevant Columns

The first structural cleanup step is to remove columns that are not appropriate as direct model inputs for this objective.

### Columns dropped first

```python
cols_to_drop = [
    'HH_ID',
    'HH_WEIGHT_MS',
    'HH_WEIGHT_FOR_COUNTRY_MS',
    'HH_WEIGHT_FOR_STATE_MS',
    'HH_NON_RESPONSE_MS',
    'HH_NON_RESPONSE_FOR_COUNTRY_MS',
    'HH_NON_RESPONSE_FOR_STATE_MS',
    'HR',
    'DISTRICT',
    'STRATUM',
    'PSU_ID'
]
```

### Additional columns dropped later

```python
cols_to_drop_for_sure = [
    'MONTH',
    'FAMILY_SHIFTED',
    'RESPONSE_STATUS',
    'REASON_FOR_NON_RESPONSE'
]
```

### Why this step matters

This is one of the first practical things I should always think about in a real project:

- Is a field genuinely predictive, or is it just administrative?
- Is a field redundant?
- Is a field unstable or unavailable at prediction time?
- Is a field likely to leak information or encode survey mechanics rather than household behavior?

That mindset matters just as much in credit-risk work. For example, in a lending model I should not blindly use all columns just because they are present in the raw extract.

---

## Part 4: Exploratory Data Analysis and Data Quality Thinking

The notebook performs univariate, bivariate, and multivariate analysis across numerical and categorical columns.

Even before modeling, I can already form useful hypotheses:

- income variables should help distinguish rural from urban households
- occupation categories should be powerful predictors
- education groups likely correlate with region type
- household size may matter
- state and month may also contain signal

### What EDA is really doing for me

EDA is not just plotting for the sake of plotting. It helps me answer:

- What is the target balance?
- Which variables are skewed?
- Which variables have outliers?
- Which categorical groups dominate?
- Which pairs of variables are strongly related?
- Which columns may be redundant?
- Which transformations may be necessary before modeling?

This is exactly the same thought process I would use in a PD scorecard build before binning and modeling.

---

## Part 5: Duplicate Removal

One of the most important data-quality findings in the notebook is this:

```text
Duplicate rows found: 57,557
```

After dropping duplicates, the dataset shrinks from:

```text
176,661 → 119,104 rows
```

### Why this matters

Duplicates can distort:

- class frequencies
- feature distributions
- train-test results
- model confidence
- feature importance
- business interpretation

If duplicate rows are truly repeated records and not meaningful repeated observations, keeping them can make the model look more certain than it really is.

### Practical lesson

Whenever I work with a tabular dataset, I should always check:

```python
df.duplicated().sum()
```

This is basic, but it matters a lot.

---

## Part 6: Missing Values and Sentinel Values

The notebook checks for missing values and also discovers that missingness is encoded using `-99` in some numerical fields.

### Why `-99` is important

Many real-world datasets do not store missing values as true `NaN`. Instead, they use placeholders such as:

- `-99`
- `999`
- `"Unknown"`
- `"NA"`
- `"Missing"`

If I do not convert these correctly, the model may think `-99` is a real income value, which would badly distort training.

### The notebook fix

```python
df.replace(-99, np.nan, inplace=True)
df.replace('-99', np.nan, inplace=True)
```

Then rows are dropped only when **all major income variables are missing**.

This keeps as much information as possible while still removing rows that are effectively unusable for an income-driven classification problem.

### Practical lesson

In real work, I should always ask:

- Is this really a value, or is it a missing code?
- Is missingness random, structural, or business-driven?
- Should I impute, drop, or create missing indicators?

---

## Part 7: Outlier Handling and Income Cleaning

Income variables are naturally skewed and often contain extreme values.

The notebook caps selected income variables at the **1st and 99th percentiles**.

### Why that is done

If a few very extreme values dominate the scale, they can distort:

- summary statistics
- distance-based reasoning
- linear models
- neural-network training
- visualization quality

Capping helps reduce the influence of extreme tails.

### The idea mathematically

If `x` is an income variable and:

- `q1 = 1st percentile`
- `q99 = 99th percentile`

then the capped value is:

```text
x_capped = min(max(x, q1), q99)
```

This is a simple form of **winsorization**.

### Additional cleanup

The notebook also forces selected income values to be non-negative:

```python
df[col] = df[col].apply(lambda x: max(x, 0))
```

That is reasonable here because negative values are likely invalid for these specific household-income fields.

---

## Part 8: Feature Engineering

This is one of the most useful parts of the project because it shows how raw fields can be turned into more meaningful features.

### Engineered feature 1: `INCOME_FROM_INVESTMENTS`

```python
df['INCOME_FROM_INVESTMENTS'] = (
    df['INCOME_OF_ALL_MEMBERS_FROM_DIVIDEND'] +
    df['INCOME_OF_ALL_MEMBERS_FROM_INTEREST'] +
    df['INCOME_OF_ALL_MEMBERS_FROM_FD_PF_INSURANCE']
)
```

This combines related income channels into one more interpretable signal.

### Engineered feature 2: `IS_HIGH_INCOME`

```python
high_income_threshold = df['TOTAL_INCOME'].quantile(0.90)
df['IS_HIGH_INCOME'] = (df['TOTAL_INCOME'] >= high_income_threshold).astype(int)
```

This creates a binary top-income indicator.

### Engineered feature 3: `HOUSEHOLD_SIZE_NUM`

This converts `SIZE_GROUP` from a text bucket into an approximate numeric size.

Important implementation detail:

- ranges like `"3-5 Members"` are converted using the lower bound
- `"> 15 Members"` is converted to `16`
- `"Data Not Available"` becomes missing

This is a practical approximation, not an exact reconstruction of household size.

### Engineered feature 4: `DEPENDENCY_RATIO`

```python
df['DEPENDENCY_RATIO'] = df['HOUSEHOLD_SIZE_NUM'] / (df['TOTAL_INCOME'] + 1)
```

This variable is useful, but I should be precise about what it is.

**Important note to myself:** this is **not** the classical dependency ratio used in demography or credit affordability analysis. It is really:

```text
household size relative to income
```

So the name `DEPENDENCY_RATIO` is convenient, but technically it is an engineered size-to-income ratio.

### Engineered feature 5: `HAS_GOV_SUPPORT`

```python
df['HAS_GOV_SUPPORT'] = (
    df['INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS'] > 0
).astype(int)
```

### Engineered feature 6: `HAS_RENTAL_INCOME`

```python
df['HAS_RENTAL_INCOME'] = (
    df['INCOME_OF_HOUSEHOLD_FROM_RENT'] > 0
).astype(int)
```

### Why feature engineering matters

Feature engineering often improves ML performance because it makes patterns easier for the model to learn.

Instead of forcing the model to discover every useful interaction on its own, I can encode domain logic such as:

- grouped income sources
- binary support indicators
- household-size approximations
- high-income thresholds
- interpretable economic ratios

This is also exactly what I do in credit work when I create utilization ratios, payment burden ratios, delinquency counts, trend variables, and behavior flags.

---

## Part 9: Train-Test Split and Preprocessing

The notebook then creates:

- `X` = all predictors
- `y` = `REGION_TYPE`

and performs an **80/20 train-test split** with stratification.

### Why stratification matters

If the target is imbalanced, a random split can accidentally distort class balance.

Stratification helps preserve roughly the same Rural/Urban mix in both training and test sets.

That is important because I want the test set to remain representative.

### Resulting split

- **Training rows:** `95,108`
- **Test rows:** `23,777`

### Numerical scaling

The notebook scales numerical features using `StandardScaler`.

For a variable `x`, scaling computes:

```text
z = (x - mean) / standard deviation
```

This gives the feature mean 0 and variance 1 on the training set.

### Why scaling matters

Scaling is especially useful for:

- logistic regression
- neural networks
- PCA

It is less critical for trees, but once I build a single shared preprocessing workflow, it can still be convenient.

### Categorical encoding

The notebook then uses one-hot encoding on categorical columns.

If a variable like `OCCUPATION_GROUP` has categories such as:

- Farmers
- Entrepreneurs
- Wage Labourers

one-hot encoding turns it into binary columns like:

```text
OCCUPATION_GROUP_Farmers
OCCUPATION_GROUP_Entrepreneurs
OCCUPATION_GROUP_Wage Labourers
```

### Final feature space

After encoding, the model matrix becomes:

```text
109 features
```

---

## Part 10: Class Imbalance

The notebook later shows the training target distribution:

- `Urban (1): 64,095`
- `Rural (0): 31,013`

So the positive class is actually the majority class in the binary encoding used for some of the later models.

### Why this matters

If I only optimize overall accuracy, the model may learn to favor the majority class.

That is why I have to inspect:

- precision
- recall
- F1-score
- confusion matrix

and not just accuracy.

### The real question

In this project, a model with slightly lower accuracy may still be better if it captures **Rural households** more reliably.

That is the same logic used in credit-risk modeling when I care about detecting bad accounts rather than just maximizing raw accuracy.

---

## Part 11: Logistic Regression — The Baseline Linear Classifier

Logistic regression is one of the most important models for me to understand deeply, because it sits at the foundation of many scorecard-style systems.

### The model

For inputs `x`, logistic regression computes:

```text
z = β0 + β1x1 + β2x2 + ... + βpxp
```

and then transforms it into a probability:

```text
P(Y = 1 | x) = 1 / (1 + e^(-z))
```

This is the **sigmoid function**.

### Intuition

- a positive coefficient increases the log-odds of the target class
- a negative coefficient decreases the log-odds
- the model is linear in the feature space
- after the sigmoid, the output becomes a probability between 0 and 1

### Notebook result

The notebook reports:

- **Training accuracy:** `81.84%`
- **Test accuracy:** `81.81%`

This is a good sign because train and test are very close, so the model is stable and not badly overfitting.

### Rural vs Urban performance

The notebook also shows that logistic regression performs much better on the majority Urban class than on Rural households.

Rural recall is only about **56%**, which means many Rural households are being missed.

### What I learn from this

Logistic regression is a strong and clean baseline, but because the underlying boundary is probably non-linear, it cannot capture all the structure in the data.

That is why it is useful, but not final.

---

## Part 12: Decision Tree — Learning Non-Linear Rules

A decision tree repeatedly splits the data into smaller regions.

At each node, it asks a rule like:

```text
Is TOTAL_INCOME < threshold?
Is OCCUPATION_GROUP = Farmers?
Is DEPENDENCY_RATIO > threshold?
```

### Core idea

The model chooses splits that improve class purity.

For classification trees, impurity is often measured using **Gini impurity**:

```text
Gini = 1 - Σ p_k^2
```

where `p_k` is the proportion of class `k` in the node.

A good split reduces impurity.

### Untuned decision tree result

The notebook reports:

- **Training accuracy:** `98.85%`
- **Test accuracy:** `76.27%`

This is a classic overfitting pattern.

### Why it overfits

A deep or unconstrained tree can memorize the training data:

- many very specific splits
- near-perfect fit in-sample
- poor generalization out-of-sample

### Tuned decision tree result

After tuning:

- **Best parameters:** `max_depth = 10`, `min_samples_leaf = 4`, `min_samples_split = 20`
- **Best CV accuracy:** `82.68%`
- **Test accuracy:** `82.56%`

This is much better because the tree is now constrained and generalizes more effectively.

### What I learn from this

This project gives me a very clean lesson:

- **unconstrained trees overfit**
- **regularized trees can become strong practical models**

That is an interview-ready point.

---

## Part 13: Artificial Neural Network — Flexible Non-Linear Function Learning

The notebook also builds ANN models using TensorFlow/Keras.

### Architecture idea

A neural network computes repeated transformations:

```text
input → hidden layer → hidden layer → output
```

Each neuron computes:

```text
a = activation(w·x + b)
```

The final output layer for binary classification uses a sigmoid activation.

### Why ANN can help

Neural networks can capture:

- non-linear effects
- interactions
- complex boundaries

### Important notebook limitation

The notebook trains ANN models, including an improved architecture with:

- multiple dense layers
- dropout
- batch normalization
- early stopping

But in the uploaded notebook version, the final ANN performance summary is not clearly preserved in the stored output cells.

So I can explain the ANN setup and why it was attempted, but I should **not pretend I have a clean final ANN metric if the notebook output does not clearly show it**.

That is important for accuracy.

### What I learn from the ANN section

This project still helps me understand:

- why scaling matters for neural nets
- why dropout is used to reduce overfitting
- why batch normalization can stabilize training
- why early stopping helps prevent unnecessary epochs
- why not every tabular problem is automatically best solved by deep learning

---

## Part 14: Random Forest — Many Trees, Better Generalization

A random forest builds many decision trees and averages their predictions.

### Why it works

Instead of relying on one unstable tree, random forest introduces:

- bootstrap sampling of rows
- random subsets of features
- averaging across many trees

This reduces variance and improves robustness.

### Baseline random forest result

The notebook reports:

- **Test accuracy:** `81.97%`

Classification report highlights:

- Rural recall about `53%`
- Urban recall about `96%`

So the model is strong overall but still somewhat biased toward Urban.

### Tuned random forest result

After randomized search, the notebook reports:

- **Test accuracy:** `83.56%`

This is one of the strongest clearly documented results in the notebook.

### What this tells me

Random forest handles:

- non-linearity
- interactions
- mixed feature patterns

better than logistic regression, and it usually generalizes better than a single decision tree.

This is why random forest becomes a very strong benchmark in tabular classification projects.

---

## Part 15: Feature Importance and Top-20 Feature Subset

The notebook uses random-forest feature importances and selects a **top 20 feature subset** for later XGBoost experiments.

Some of the top selected features include:

- `OCCUPATION_GROUP_Small/Marginal Farmers`
- `INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION`
- `OCCUPATION_GROUP_Organised Farmers`
- `DEPENDENCY_RATIO`
- `TOTAL_INCOME`
- `INCOME_OF_HOUSEHOLD_FROM_ALL_SOURCES`
- `INCOME_OF_ALL_MEMBERS_FROM_ALL_SOURCES`
- `INCOME_OF_ALL_MEMBERS_FROM_WAGES`
- `INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS`
- `HOUSEHOLD_SIZE_NUM`
- education-related dummies

### Why this matters

This aligns with intuition:

- occupation mix matters
- total and source-specific income matter
- household burden and size matter
- education matters
- government-support and self-production features matter

So the model is learning economically meaningful structure rather than pure noise.

---

## Part 16: XGBoost — Gradient Boosted Trees

XGBoost is one of the most important tabular ML models to understand.

### The core idea

Instead of building many independent trees like random forest, boosting builds trees **sequentially**.

Each new tree tries to improve what the previous trees got wrong.

At a high level:

```text
Prediction_t = Prediction_(t-1) + new_tree
```

The model minimizes an objective of the form:

```text
Loss = training loss + regularization
```

So XGBoost is powerful because it combines:

- flexible tree-based structure
- additive boosting
- regularization
- efficient optimization

### Baseline XGBoost on top features

Using the top-20 feature subset, the notebook reports:

- **Test accuracy:** `82.56%`

### Tuned XGBoost on top features

After randomized search:

- **Best parameters:** roughly `n_estimators=300`, `max_depth=8`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=1`
- **Test accuracy:** `82.63%`

This is solid, but in this notebook it does not beat the tuned random forest yet.

### What I learn from this

XGBoost is powerful, but performance still depends on:

- feature set
- class handling
- hyperparameters
- whether I use the full encoded matrix or only selected features

---

## Part 17: SMOTE — Balancing the Classes by Synthetic Oversampling

The notebook also tests **SMOTE**.

### What SMOTE does

SMOTE creates synthetic minority-class examples by interpolating between nearby minority points.

This helps reduce imbalance in the training set.

### Why it can help

If the model keeps ignoring the minority class, SMOTE can improve minority recall.

### Notebook result

After SMOTE + XGBoost:

- **Test accuracy:** `80.78%`

This is lower than the earlier tuned models.

But Rural recall improves from roughly the high-50s to around **67%**.

### What I learn from this

SMOTE improved minority sensitivity, but at a cost:

- lower overall accuracy
- lower Urban performance

This is a very important lesson.

In classification, there is often a tradeoff between:

- raw accuracy
- balance across classes
- minority recall
- false positives vs false negatives

That is exactly the same kind of tradeoff I would discuss in a risk-model interview.

---

## Part 18: Class Weighting — A Useful Idea, but I Need to Read the Code Carefully

The notebook also tests an XGBoost variant described as weighted.

This section teaches me an important accuracy habit: **read the code, not just the label**.

### Earlier weighted run on top features

One earlier branch computes:

```text
scale_pos_weight = negatives / positives = 31013 / 64095 ≈ 0.4839
```

Because the positive class (`Urban = 1`) is actually the majority class here, a value below 1 downweights the positive class rather than upweighting it.

That is mathematically valid, but it is different from the more common situation where `scale_pos_weight > 1` is used to help a minority positive class.

### Result of that earlier weighted run

The notebook reports about:

- **Accuracy:** `80.68%`

So that earlier weighted attempt does not beat the stronger tuned models.

### Crucial notebook inconsistency in the final section

Near the end of the notebook:

```python
scale_pos_weight = 1  # Neutral weight to test if the model runs
```

and then the final XGBoost model is trained using that value.

That means the final best-scoring model is **not really using class weighting in an effective sense**. It is essentially a neutral setting.

### Why this matters

So if I describe the final best model honestly, I should say:

- it is a **final XGBoost model on the full encoded feature set**
- it achieves the best test accuracy in the notebook
- but it should **not** be described as a truly class-weighted XGBoost model, because the final code resets `scale_pos_weight` to `1`

This kind of code-reading precision is very important in interviews.

---

## Part 19: PCA — Dimensionality Reduction

The notebook also applies **PCA** after scaling.

### What PCA does

PCA creates new orthogonal components that capture the maximum variance in the data.

Instead of using the original correlated features directly, it transforms them into a smaller number of linear combinations.

### Why PCA can help

PCA can:

- reduce dimensionality
- reduce multicollinearity
- sometimes improve generalization
- speed up training

### Notebook result

The notebook reduces:

- **Original features:** `109`
- **Retained PCA features:** `80`

Then XGBoost with PCA reaches:

- **Test accuracy:** `82.80%`

This is slightly better than the earlier top-feature XGBoost branch, but still not better than the best final full-feature XGBoost run.

### Tradeoff

PCA may help performance, but interpretability becomes weaker because principal components are not directly meaningful business variables.

That is why PCA is useful, but not always ideal when explainability matters.

---

## Part 20: Voting Ensemble

The notebook then builds a soft-voting ensemble using:

- Logistic Regression
- Random Forest
- XGBoost

### How soft voting works

Each model outputs probabilities, and the ensemble averages them.

At a high level:

```text
Final probability = average of model probabilities
```

Then the final class is chosen from the averaged probability.

### Notebook result

- **Ensemble test accuracy:** `83.00%`

This is respectable, but it still does not beat the best later XGBoost run.

### What I learn from this

Ensembling is not guaranteed to win.

It works best when the base models:

- are individually strong
- make somewhat different kinds of errors

If they are too similar, the gain may be limited.

---

## Part 21: The Final Best-Scoring Model in the Notebook

The final strongest reported score in the notebook is:

- **XGBoost test accuracy:** `84.05%`

This is achieved in the final modeling section using the **full cleaned and encoded feature set**, not the earlier top-20 subset.

### How I should describe it carefully

This is the best-scoring model in the notebook, but I should describe it precisely:

- it is a final **full-feature XGBoost**
- it uses the full encoded feature matrix
- it is labeled as weighted in the notebook output
- but the code sets `scale_pos_weight = 1`, so it is effectively a neutral-weight run

That is the most accurate interpretation.

### Why the score improved

The improvement likely comes from using:

- the full feature space
- strong XGBoost hyperparameters
- a flexible boosted-tree structure

rather than from real class-weight adjustment.

---

## Part 22: Top Features in the Final Model

The notebook’s final interpretation highlights features such as:

1. `DEPENDENCY_RATIO`
2. `TOTAL_INCOME`
3. `INCOME_OF_HOUSEHOLD_FROM_ALL_SOURCES`
4. `INCOME_OF_HOUSEHOLD_FROM_GOVERNMENT_TRANSFERS`
5. `INCOME_OF_HOUSEHOLD_FROM_SELF_PRODUCTION`
6. `INCOME_OF_ALL_MEMBERS_FROM_ALL_SOURCES`
7. `MONTH_SLOT_Apr 2021`
8. `INCOME_OF_ALL_MEMBERS_FROM_WAGES`
9. `HOUSEHOLD_SIZE_NUM`
10. `EDUCATION_GROUP_Households Of All Literates`

### Economic interpretation

These features make sense:

- **income level** matters
- **income composition** matters
- **government support** matters
- **self-production** matters
- **household burden and size** matter
- **education** matters
- **occupation categories** matter

That is exactly what I would expect in a Rural vs Urban classification task.

---

## Part 23: What This Project Teaches Me About Metrics

A very important lesson from this notebook is that **accuracy is not enough**.

### Confusion matrix thinking

A confusion matrix helps me see:

- how many Rural households were predicted as Urban
- how many Urban households were predicted as Rural

That is often much more informative than one single accuracy number.

### Precision and recall

For a given class:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

### Why this matters here

Several models achieve similar overall accuracy, but they behave differently on Rural households.

That means the best model depends on the business objective:

- do I want the highest overall accuracy?
- do I want stronger Rural recall?
- do I want a more balanced classifier?
- do I care more about false positives or false negatives?

This is exactly how I should speak in a real interview.

---

## Part 24: Beginner-Friendly Code Skeleton of the Workflow

```python
# 1. Load data
df = pd.read_csv(file_path)

# 2. Drop irrelevant fields
df = df.drop(columns=cols_to_drop + cols_to_drop_for_sure)

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Replace missing sentinels
df.replace(-99, np.nan, inplace=True)

# 5. Drop rows where all key income fields are missing
df.dropna(subset=income_cols, how='all', inplace=True)

# 6. Cap outliers and clean negatives
for col in income_cols:
    lower_cap = df[col].quantile(0.01)
    upper_cap = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
    df[col] = df[col].clip(lower=0)

# 7. Feature engineering
df['INCOME_FROM_INVESTMENTS'] = ...
df['IS_HIGH_INCOME'] = ...
df['HOUSEHOLD_SIZE_NUM'] = ...
df['DEPENDENCY_RATIO'] = ...
df['HAS_GOV_SUPPORT'] = ...
df['HAS_RENTAL_INCOME'] = ...

# 8. Split into X and y
X = df.drop(columns=['REGION_TYPE'])
y = df['REGION_TYPE']

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 10. Scale numeric variables
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 11. One-hot encode categoricals
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_ohe = ohe.fit_transform(X_train[cat_cols])
X_test_ohe = ohe.transform(X_test[cat_cols])

# 12. Build final matrices
X_train_final = ...
X_test_final = ...

# 13. Train models
log_reg.fit(X_train_final, y_train)
rf.fit(X_train_final, y_train)
xgb.fit(X_train_final, y_train_binary)

# 14. Evaluate
print(accuracy_score(...))
print(classification_report(...))
print(confusion_matrix(...))
```

---

## Part 25: How This Connects to My Other Notes

Even though this is a socio-economic classification project, it connects strongly to my broader system.

### Connection to PD modeling

In a PD model, I also:

- define a target
- clean data
- examine missingness
- transform variables
- split train/test
- evaluate performance
- compare model behavior
- think carefully about business interpretation

The main difference is that for credit-risk scorecards I usually prefer stronger explainability and regulatory defensibility.

### Connection to monitoring

This project also helps me understand what I would monitor later:

- class balance drift
- feature drift
- missing value drift
- changes in income distributions
- deterioration in minority-class recall
- changes in confusion matrix structure

That links directly to my monitoring note.

### Connection to other ML projects

This note also connects well to:

- churn prediction
- employee attrition
- fraud flags
- customer segmentation
- application risk segmentation

because the workflow is the same even when the domain changes.

---

## Part 26: What I Would Say in an Interview

If I had to summarize this project clearly, I would say:

> I built a binary classification pipeline to classify households as Rural or Urban using socio-economic and income-based variables. I started with raw survey-style household data, removed irrelevant administrative columns, handled duplicate records, converted sentinel missing values like `-99` into `NaN`, capped extreme income outliers, engineered economically meaningful features such as investment income, household-size approximation, and size-to-income ratio, then scaled numeric variables and one-hot encoded categorical variables. I compared logistic regression, decision trees, random forests, XGBoost, SMOTE-based balancing, PCA, and ensemble voting. The best notebook score came from a final full-feature XGBoost model at about 84.05% test accuracy, while tuned random forest was one of the strongest clean benchmark models at about 83.56%. A key learning from the project was that overall accuracy alone was not enough because the Rural class was harder to capture, so recall and confusion-matrix analysis were important.

That is a compact but technically strong explanation.

---

## Part 27: Important Caveats I Should Know

This is one of the most useful sections for me because it keeps me honest.

### Caveat 1: Final “weighted” model is not truly weighted in the final code

The notebook’s final best-scoring XGBoost run resets:

```python
scale_pos_weight = 1
```

So I should not oversell it as a genuine class-weighted model.

### Caveat 2: `DEPENDENCY_RATIO` is not a classical dependency ratio

It is really:

```text
household size / (total income + 1)
```

So I should describe it accurately.

### Caveat 3: Household size conversion is approximate

Turning grouped size ranges into a numeric value is a practical approximation.

### Caveat 4: PCA improves compression, but weakens interpretability

If explainability matters, original variables are easier to discuss.

### Caveat 5: Accuracy alone can hide imbalance problems

A model can look good overall while still underperforming on Rural households.

---

## Part 28: What I Would Improve if I Rebuilt This Project Again

If I rebuild this project in a more production-grade way, I would improve it like this.

### 1. Use a single sklearn pipeline

That would make preprocessing and modeling cleaner and less error-prone.

### 2. Add cross-validated model comparison with multiple metrics

Instead of relying mainly on one test split, I would compare models using:

- accuracy
- balanced accuracy
- precision
- recall
- F1
- ROC-AUC

### 3. Tune decision thresholds

For some business objectives, threshold tuning may matter more than switching models.

### 4. Use SHAP for XGBoost explainability

That would make feature interpretation much stronger.

### 5. Consider calibration

If I need reliable probabilities rather than just class labels, calibration would matter.

### 6. Review whether survey weights should be incorporated

The raw data includes weighting-related fields. In a more rigorous survey-analytics setup, I would think carefully about whether and how those weights should enter estimation or evaluation.

---

## Part 29: Final Takeaways I Want to Retain

### 1. This is a complete end-to-end classification project

It teaches me the whole supervised-learning workflow on messy tabular data.

### 2. Cleaning matters as much as modeling

Dropping duplicates, fixing sentinel missing values, and handling outliers materially change results.

### 3. Feature engineering still matters

Hand-built variables like grouped income and size-to-income features can be very useful.

### 4. Trees and boosting capture non-linearity well

That is why they outperform a purely linear baseline here.

### 5. Metrics must match the objective

Accuracy is useful, but recall, precision, F1, and the confusion matrix matter just as much.

### 6. Reading the code carefully matters

The final notebook label says weighted XGBoost, but the code uses neutral weighting in the last run. I should always trust the code over a label.

### 7. This project transfers directly to credit-risk thinking

Even though the target is Rural vs Urban, the workflow, discipline, and evaluation logic transfer strongly to credit, risk analytics, and quantitative modeling.

---

## Quick Revision Sheet

### Problem type
- Binary classification

### Target
- `REGION_TYPE` = Rural vs Urban

### Key preprocessing
- drop irrelevant administrative columns
- remove duplicates
- convert `-99` to missing
- drop rows with all key income fields missing
- cap outliers
- enforce non-negative income values
- scale numeric variables
- one-hot encode categorical variables

### Engineered features
- `INCOME_FROM_INVESTMENTS`
- `IS_HIGH_INCOME`
- `HOUSEHOLD_SIZE_NUM`
- `DEPENDENCY_RATIO`
- `HAS_GOV_SUPPORT`
- `HAS_RENTAL_INCOME`

### Important model lessons
- logistic regression = stable baseline
- untuned decision tree = overfits badly
- tuned decision tree = much better
- tuned random forest = strong benchmark
- tuned/top-feature XGBoost = solid
- SMOTE improves minority recall but lowers accuracy
- PCA gives a modest improvement in compressed space
- ensemble is decent but not best
- final best notebook score comes from full-feature XGBoost

### Best-scoring notebook result
- final XGBoost on full encoded features
- test accuracy ≈ `84.05%`
- but final code uses `scale_pos_weight = 1`, so it is not truly a weighted final run

---

## Closing Note

This project is one of my best beginner-to-intermediate classification notes because it forces me to understand the full lifecycle of a real tabular ML problem:

- business framing
- data understanding
- cleaning
- feature engineering
- preprocessing
- model comparison
- metric interpretation
- caveat handling
- honest final model selection

That is exactly the kind of thinking I want to carry into all the rest of my notes.
