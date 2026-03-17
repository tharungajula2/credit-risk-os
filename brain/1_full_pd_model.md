---
title: "PD Model: End-to-End Credit Scorecard — Lending Club"
date: 2026-03-19
tags:
  - PD
  - probability-of-default
  - logistic-regression
  - scorecard
  - WoE
  - IV
  - KS-statistic
  - AUROC
  - PDO
  - missing-data
  - feature-engineering
  - model-validation
  - SR-11-7
  - FCRA
  - ECOA
  - credit-risk
  - lending-club
  - IRB

links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[2_monitoring_model]]"
---

---

# PD Model: End-to-End Credit Scorecard — Lending Club

> This note is the full technical record of how I built a Probability of Default (PD) model and converted it into a production-grade credit scorecard using the Lending Club dataset. Every section covers the concept from first principles, the math behind it, the Python code I wrote, and why the approach matters in a regulated banking environment.
>
> This is the anchor note for the entire Quant OS system. Everything else — WoE, validation metrics, PSI monitoring, deployment — connects back to the decisions made here.

---

## The Project at a Glance

**Dataset:** Lending Club loan data (~390,000 loans, 2007–2014)

**Objective:** Build a binary classification model that scores each borrower's probability of default, then scale that model into an interpretable credit scorecard with scores between 300 and 850.

**Why Lending Club:** It is one of the cleanest publicly available retail credit datasets that mirrors what a real bank would have — loan application attributes, bureau-derived variables, and a performance outcome column (`loan_status`). The data dictionary is attached and referenced throughout this note.

**The full pipeline I built:**

```
Raw Lending Club CSV
        │
        ▼
Target Variable Definition (good_bad flag)
        │
        ▼
Train / Test Split (80/20)
        │
        ▼
Raw Preprocessing (string parsing, datetime math, missing values)
        │
        ▼
WoE Binning + IV Filtering (fine classing → coarse classing)
        │
        ▼
Dummy Variable Creation + Reference Category Removal
        │
        ▼
Logistic Regression with Custom Fisher Information Matrix (P-Values)
        │
        ▼
Validation Suite (Confusion Matrix, AUROC, Gini, KS Statistic)
        │
        ▼
Scorecard Scaling (Min-Max → then PDO framework)
        │
        ▼
Cut-off Strategy Table for Business Use
```

Each stage is a section in this note.

---

## Part 1: Target Variable Definition — What Is "Default"?

### The Concept

Before any model training can happen, I need to define what I am actually predicting. The raw dataset has a column called `loan_status`, which contains values like `'Fully Paid'`, `'Current'`, `'Charged Off'`, `'Default'`, `'Late (31-120 days)'`, `'In Grace Period'`, `'Late (16-30 days)'`, and `'Does not meet the credit policy'`.

A machine learning model cannot take these strings as a target. I need to collapse them into a single binary column — `good_bad` — where **1 = Good** (the borrower is performing) and **0 = Bad** (the borrower has defaulted or is severely delinquent).

### The Business Decision Behind the Label

This is not an arbitrary split. It is a formal risk definition. In retail credit, the near-universal industry standard for declaring a borrower "bad" is reaching **90 Days Past Due (DPD)** at least once within a defined observation window (typically 12 to 24 months after origination).

`'Charged Off'` means the bank has already written off the loan as unrecoverable — these are clearly Bad.

`'Default'` is explicitly Bad.

`'Late (31-120 days)'` is the judgment call. At 31+ days, the borrower has already missed a full monthly payment. At 90+ days, the regulatory bad definition is triggered. I included this category as Bad, which is a slightly aggressive definition but is consistent with industry practice.

`'Late (16-30 days)'`, `'In Grace Period'`, and current loans — these are treated as Good.

### The Indeterminate Zone

There is a concept that most beginner tutorials skip entirely: **indeterminate accounts**. Borrowers who are sitting in the 30–60 DPD range are genuinely ambiguous. Some of them will cure (they pay the overdue amount and return to good standing). Others will deteriorate further into default.

If I force the model to learn from these ambiguous cases — treating a 45-DPD borrower as "Bad" — I pollute the training signal. These borrowers look statistically very similar to Good borrowers in terms of their application characteristics, but I am asking the model to classify them as Bad. This confuses the model and reduces its ability to separate clearly safe borrowers from clearly risky ones.

**The production practice:** In a real bank model build, indeterminate accounts (typically defined as 30–60 DPD) are **excluded entirely** from the training data. The model learns only from the cleanest signal: clearly performing loans on one side, clearly defaulted loans on the other.

In my notebook, `'Late (16-30 days)'` sits in the indeterminate zone and I assigned it as Good. This is a simplification. In production, I would exclude it.

### The Code

```python
import numpy as np
import pandas as pd

# Define the binary target variable
# 0 = Bad (defaulted / severely late)
# 1 = Good (performing)

loan_data['good_bad'] = np.where(
    loan_data['loan_status'].isin([
        'Charged Off',
        'Default',
        'Late (31-120 days)'
    ]),
    0,   # Bad
    1    # Good
)

print(loan_data['good_bad'].value_counts(normalize=True))
# Typical output: Good ~80-85%, Bad ~15-20%
# This class imbalance matters — explained in the Validation section
```

The `np.where` call reads: "If the loan_status is in the bad categories, assign 0. Otherwise assign 1."

---

## Part 2: Train / Test Split — The Right Way vs. The Easy Way

### The Random Split (What I Did)

```python
from sklearn.model_selection import train_test_split

# Separate inputs from the target
X = loan_data.drop('good_bad', axis=1)
y = loan_data['good_bad']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

An 80/20 random split is correct for an offline project. The `random_state=42` makes the split reproducible — running the same code always produces the same train/test division.

`test_size=0.2` means 20% of the data is held out. The model never sees this during training. After the model is fully built, I apply it to this held-out set and measure how it performs. If the model only performs well on training data but poorly on the test set, it has memorized the training data rather than learned generalizable patterns — that is called **overfitting**.

### Why a Random Split Fails in Production

The Lending Club dataset spans loans originated from 2007 to 2014. A random split pulls records from all years into both the training and test sets.

This means:
- A loan from 2008 (during the financial crisis) might end up in the test set
- A loan from 2013 (during the recovery) might end up in the training set
- The model gets to "learn" macro patterns from both pre-crisis and post-crisis data simultaneously

In real banking, a model is built on historical data and then deployed to score future applicants. The model never has the luxury of seeing future macroeconomic conditions during training. A random split disguises this problem.

### The Production Standard: Out-of-Time (OOT) Validation

Under **SR 11-7** (the US Federal Reserve's Model Risk Management guidelines), models deployed in live banking environments are required to be validated on an **Out-of-Time (OOT)** sample.

The correct split for this dataset would be:

```
Training set:    Loans originated 2007 – 2012  (builds the model)
Validation set:  Loans originated 2013         (tunes hyperparameters)
OOT Test set:    Loans originated 2014         (final unbiased performance test)
```

This way, the model proves it can generalise across a completely different macroeconomic period. If a model built on 2007–2012 data (which includes the financial crisis) cannot accurately rank-order risk on 2014 loans (during recovery), it fails validation.

**Why this matters for what I built:** My random split gives inflated performance metrics. The true OOT performance would likely be lower. This does not mean the model is wrong — it means the evaluation is optimistic, and a validator would flag it immediately.

---

## Part 3: Raw Preprocessing — Cleaning Before Modelling

### What the Raw Data Looks Like

Several columns in the Lending Club dataset are not model-ready:

- `emp_length` (Employment Length) is stored as a string: `"< 1 year"`, `"1 year"`, `"10+ years"`. The model needs a number.
- `term` (Loan Term) is stored as `" 36 months"` or `" 60 months"` — a string with a space.
- `earliest_cr_line` (Date of Earliest Credit Line) is stored as `"Apr-2000"` — a date string, not a number.
- `issue_d` (Loan Origination Date) is also a date string.

These need to be converted into numeric features before any analysis.

### The Code: String Parsing and Date Engineering

```python
# --- Employment Length: strip the text, extract the number ---
loan_data['emp_length'].replace('10+ years', '10', inplace=True)
loan_data['emp_length'].replace('< 1 year', '0', inplace=True)

# Remove ' years' and ' year' from all remaining values
loan_data['emp_length'] = loan_data['emp_length'].str.replace(' years', '')
loan_data['emp_length'] = loan_data['emp_length'].str.replace(' year', '')
loan_data['emp_length'] = pd.to_numeric(loan_data['emp_length'])

# --- Loan Term: strip ' months', convert to integer ---
loan_data['term'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

# --- Date Features: convert to datetime, then calculate months elapsed ---
# Reference date: last date in the dataset (December 2017 for this Kaggle extract)
# In production this would be parameterised to system_current_date
reference_date = pd.to_datetime('2017-12-01')

loan_data['earliest_cr_line'] = pd.to_datetime(
    loan_data['earliest_cr_line'], format='%b-%Y'
)
loan_data['mths_since_earliest_cr_line'] = round(
    pd.to_numeric(
        (reference_date - loan_data['earliest_cr_line']) / np.timedelta64(1, 'M')
    )
)

loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'], format='%b-%Y')
loan_data['mths_since_issue_d'] = round(
    pd.to_numeric(
        (reference_date - loan_data['issue_d']) / np.timedelta64(1, 'M')
    )
)
```

The result: instead of `"Apr-2000"` in `earliest_cr_line`, the model now sees `212` — the number of months between April 2000 and December 2017. This captures credit history length numerically.

**Production note on the reference date:** I hardcoded `'2017-12-01'` because that is when this dataset was extracted. In a live Loan Origination System (LOS), this date must be parameterised to `system_current_date`. If it is hardcoded, every applicant scored after 2017 will have an incorrectly calculated credit history length.

---

## Part 4: Missing Data — The Three Mechanisms and Why They Matter

### Why Missing Data Is Not Just a Nuisance

Before touching a single null value, the most important question to ask is: **why is this value missing?** The answer completely determines the correct treatment.

There are three distinct mechanisms of missingness, and they have very different implications.

### MCAR — Missing Completely at Random

The missing values have no relationship to any other variable in the dataset. A server crashed and dropped 0.5% of records at random. Income, employment length, credit history — none of it influenced which records were dropped.

In this case, simply imputing with the **mean** (for continuous variables) or **mode** (for categorical variables) is statistically acceptable, because the missing records are a random sample of the full population. The imputation will not introduce systematic bias.

This is the rarest case in credit data.

### MAR — Missing at Random

The missingness is related to *other observable variables*, but not to the missing value itself. For example: borrowers in certain states may more often skip the `emp_title` field because the application interface in that state made it optional. The missingness depends on geography (an observable feature), not on what the actual job title would have been.

Here, **conditional imputation** works well. We can predict the missing value from the other variables — for example, using KNN imputation or regression imputation based on known features.

### MNAR — Missing Not at Random

The missingness is caused by the underlying value itself. This is the dominant case in credit risk data.

**Examples:**
- `annual_inc` is blank because the borrower chose not to disclose it. Borrowers who hide their income typically do so because it is low or unstable. The missing value is itself a risk signal.
- `mths_since_last_delinq` (months since the borrower's last delinquency) is blank because the borrower has *never been delinquent*. In this case, missing means "no history of delinquency" — which is actually a positive signal.

**The danger of mean/zero imputation in the MNAR case:**

If I fill a missing `annual_inc` with the portfolio average income ($65,000), I am artificially making a potentially high-risk, low-income borrower look like an average-risk borrower. The model then scores them as average risk. This is not a data quality fix — it is risk fabrication.

If I fill `mths_since_last_delinq` with zero, I am telling the model "this person became delinquent zero months ago" — i.e., they are currently delinquent. That is the exact opposite of the truth.

### The Industry Best Practice: Missing as a Separate Category

For MNAR data in credit, the correct approach is to treat **"Missing"** as a separate, meaningful bin during the Weight of Evidence transformation.

By creating a dedicated "Missing" bucket, I let the data determine what risk level is associated with having no response. If borrowers with missing income have a systematically higher default rate than the average, the model will assign that "Missing" bin a high WoE (high risk score). If borrowers with no delinquency history (missing `mths_since_last_delinq`) have a lower default rate, the "Missing" bin will receive a low WoE (low risk).

The signal is preserved. Nothing is assumed.

### My Specific Imputation Decisions

For the Lending Club dataset, the following imputation decisions were made before WoE binning:

```python
# total_rev_hi_lim: Total revolving credit limit
# Missing likely because the bureau lookup returned no revolving trades
# Fill with funded_amnt as a proxy (the loan itself is the credit exposure)
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)

# annual_inc: Self-reported income
# NOTE: This is a simplification. In production this is MNAR.
# Better practice: create a 'missing' indicator, then impute separately
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)

# mths_since_earliest_cr_line: Months since first credit line opened
# Missing = no credit history found by the bureau
# Fill with 0 initially; the WoE binning will create a Missing category
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)

# mths_since_last_delinq: The ideal MNAR treatment
# I created an explicit binary 'missing' flag first
loan_data['mths_since_last_delinq_missing'] = np.where(
    loan_data['mths_since_last_delinq'].isnull(), 1, 0
)
loan_data['mths_since_last_delinq'].fillna(0, inplace=True)
```

The `mths_since_last_delinq` treatment is the one I am most satisfied with. By creating the binary `_missing` indicator before imputing, I preserve the information that "this data was not available" as a separate feature. The WoE transformation will then find the risk associated with having no recorded delinquency.

---

## Part 5: Weight of Evidence (WoE) and Information Value (IV)

### Why Not Feed Raw Numbers Into the Model?

The raw features in the dataset — `annual_inc = 64321.0`, `dti = 18.32`, `inq_last_6mths = 2` — are all on completely different scales. A logistic regression treats a 1-unit change in each variable identically unless the data is scaled. But more importantly, risk is not linear.

The difference in credit risk between earning $20,000 and $30,000 per year is substantial. The difference between earning $150,000 and $160,000 is nearly zero. If I feed raw income into the model, it assumes the risk change is the same per $10,000 across the entire income range. That assumption is false.

**WoE transformation solves this** by binning the variable and replacing each bin with a number that directly represents its relative riskiness in the portfolio. The model then works in a space where risk relationships are already linearised.

### The WoE Formula

For each bin $i$ of a variable:

$$
\text{WoE}_i = \ln\left(\frac{\text{Distribution of Events}_i}{\text{Distribution of Non-Events}_i}\right)
$$

Where:

$$
\text{Distribution of Events}_i = \frac{\text{Number of Bads in bin } i}{\text{Total number of Bads in dataset}}
$$

$$
\text{Distribution of Non-Events}_i = \frac{\text{Number of Goods in bin } i}{\text{Total number of Goods in dataset}}
$$

**Interpreting WoE values:**
- $\text{WoE} > 0$: This bin contains a higher-than-average concentration of defaulters. High risk.
- $\text{WoE} < 0$: This bin contains a lower-than-average concentration of defaulters. Low risk.
- $\text{WoE} = 0$: The default rate in this bin exactly equals the portfolio average.

### The Information Value Formula

WoE tells me the risk of each individual bin. IV aggregates across all bins to give a single number measuring the variable's overall predictive power:

$$
\text{IV} = \sum_{i=1}^{n} \left(\text{Dist. Events}_i - \text{Dist. Non-Events}_i\right) \times \text{WoE}_i
$$

| IV Range | Predictive Power | Decision |
|---|---|---|
| < 0.02 | Useless | Drop the variable |
| 0.02 – 0.10 | Weak | Use with caution |
| 0.10 – 0.30 | Medium | Strong candidate |
| 0.30 – 0.50 | Strong | Keep |
| > 0.50 | Suspicious | **Investigate for target leakage** |

### The Target Leakage Warning

An IV above 0.50 is not a sign of a great variable — it is almost always a sign of a data problem. In credit risk modeling, "target leakage" means the feature accidentally contains information from the future.

For example: `total_rec_late_fee` (late fees received to date) would have an extremely high IV for predicting default. But a borrower is only charged late fees *after* they have been late — which means this variable is a consequence of default, not a predictor of it. At the moment of loan origination (when we need to score a new applicant), this value will always be zero. Including it makes the model useless in production.

Variables to scrutinise for leakage in the Lending Club data include anything from the "payment history" group: `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`.

### My WoE Code (Discrete Variables)

```python
def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    """
    Calculates WoE and IV for a categorical (discrete) variable.
    
    df: DataFrame containing the variable
    discrete_variable_name: column name of the feature
    good_bad_variable_df: the binary target column (1=Good, 0=Bad)
    """
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat(
        [
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
            df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()
        ],
        axis=1
    )
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()   # Distribution of Goods
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()      # Distribution of Bads

    # Clip to avoid log(0)
    df['WoE'] = np.log(
        df['prop_n_good'].clip(lower=0.0001) / df['prop_n_bad'].clip(lower=0.0001)
    )
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df
```

### Fine Classing and Coarse Classing

For continuous variables like `annual_inc` or `dti`, the process has two stages:

**Fine Classing:** Cut the variable into many small, equal-frequency buckets (I used 50–100 bins). This gives a first look at how WoE behaves across the variable's range.

```python
# Fine classing: slice annual_inc into 50 equal-frequency bins
loan_data['annual_inc_factor'] = pd.qcut(
    loan_data['annual_inc'],
    q=50,
    duplicates='drop'
)
```

**Coarse Classing:** Look at the WoE chart for each bin. Adjacent bins with similar WoE values are merged together. The goal is to create a smaller number of final bins where the WoE trend is **monotonic** — consistently moving in one direction as the variable value increases.

For `annual_inc`: as income increases, the WoE should consistently decrease (higher income = lower risk). If there are any inversions (a higher income bin suddenly showing higher risk than a lower income bin), those bins are merged until the monotonicity holds.

### Monotonicity: Why It Is a Regulatory Requirement

Under **SR 11-7**, a model must be **conceptually sound**. A zig-zagging WoE profile for income — where $60k earners appear riskier than $40k earners — is not economically defensible. It is the model fitting noise in the training sample.

Any variable that cannot be forced into a monotonic WoE pattern after coarse classing is rejected from the scorecard entirely. The cost of keeping a non-monotonic variable is a model that behaves erratically for individual borrowers and cannot be explained to regulators.

### Discrete Variables: The Home Ownership Example

For `home_ownership`, the values are `'RENT'`, `'OWN'`, `'MORTGAGE'`, `'OTHER'`, `'NONE'`, `'ANY'`. After computing WoE for each:

- `'MORTGAGE'` and `'OWN'` might have similar, low-risk WoE values
- `'RENT'`, `'OTHER'`, `'NONE'`, `'ANY'` might cluster together as higher-risk

I merged `'RENT'`, `'OTHER'`, `'NONE'`, and `'ANY'` into a single category because their WoE values were indistinguishable. Keeping them separate would create fragile dummy variables with insufficient sample sizes.

---

## Part 6: Feature Validation — Multicollinearity, Fair Lending, and the Dummy Variable Trap

### Creating Dummy Variables

After WoE binning, each continuous variable becomes a set of interval categories (e.g., `annual_inc:20K-30K`, `annual_inc:30K-40K`, etc.) and each categorical variable becomes grouped categories. I convert these into binary dummy variables for the logistic regression.

```python
# Example: after coarse classing, loan_data_dummies might contain:
# home_ownership:RENT_OTHER_NONE_ANY   |   home_ownership:OWN   |   home_ownership:MORTGAGE
# Each row has exactly one 1 and the rest 0s

loan_data_dummies = pd.get_dummies(loan_data_woe_binned)
```

### The Dummy Variable Trap and How to Avoid It

For every categorical variable, one dummy must be dropped. This is called the **reference category**.

If `home_ownership` has 3 categories (RENT, OWN, MORTGAGE), I create only 2 dummies and drop one. If I keep all three, the sum of those three columns equals exactly 1 for every row — this creates **perfect multicollinearity** with the model's intercept term.

**Why perfect multicollinearity destroys the model:**

In logistic regression, coefficients are estimated by solving the system:

$$
\hat{\boldsymbol{\beta}} = \text{argmax}_{\boldsymbol{\beta}} \; \ell(\boldsymbol{\beta})
$$

where $\ell$ is the log-likelihood function. The solution requires inverting the matrix $(\mathbf{X}^T \mathbf{X})$. If the design matrix $\mathbf{X}$ has a column that is a perfect linear combination of others, this matrix becomes **singular** (its determinant equals zero) and **cannot be inverted**. There is no unique solution for $\boldsymbol{\beta}$.

The model either crashes or outputs numerically unstable coefficients with enormous standard errors.

```python
# Reference categories: one bin from each variable to be dropped
ref_categories = [
    'grade:G',                          # Worst credit grade — riskiest
    'home_ownership:RENT_OTHER_NONE_ANY',
    'addr_state:ND_NE_IA_NV_FL_HI_AL',
    'verification_status:Verified',
    'purpose:small_business',
    'initial_list_status:f',
    'term:60',                          # Longer term = higher risk
    'emp_length:0',
    # ... one per variable
]

# Drop reference categories from training inputs
X_train_final = X_train_dummies.drop(ref_categories, axis=1)
X_test_final = X_test_dummies.drop(ref_categories, axis=1)
```

**Choosing the reference category strategically:**

I chose the riskiest bin (e.g., `'grade:G'`) as the reference for each variable. This means all other $\beta$ coefficients represent *improvement over the riskiest option* — they will all be positive, making the scorecard additive and easy to explain. The score increases as the borrower's profile improves relative to the worst case.

An alternative is to choose the highest-volume bin as reference, which gives smaller standard errors for all other coefficients (because variance scales with $1/n$).

### Multicollinearity: The VIF Test

Even after dropping reference categories, non-dummy variables can be correlated with each other. `funded_amnt` (loan amount) and `installment` (monthly payment) are almost perfectly correlated — a larger loan always means a larger monthly payment.

**Variance Inflation Factor (VIF):**

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

Where $R_j^2$ is the R-squared from regressing feature $j$ on all other features. If VIF exceeds 5 (or 10 under stricter bank policies), I need to act.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(df_features):
    vif = pd.DataFrame()
    vif['feature'] = df_features.columns
    vif['VIF'] = [
        variance_inflation_factor(df_features.values, i)
        for i in range(df_features.shape[1])
    ]
    return vif.sort_values('VIF', ascending=False)

# If funded_amnt VIF = 18.4 and installment VIF = 17.9:
# Compare their IV scores. Keep the higher IV, drop the other.
```

When two variables are highly collinear, I keep the one with the higher IV (stronger predictive power) and drop the other.

### Fair Lending: The `addr_state` Problem

I used `addr_state` (US State) as a predictor in the model. It has genuine predictive power — borrowers in some states default at higher rates than others due to foreclosure laws, economic conditions, and demographics.

However, under the **Equal Credit Opportunity Act (ECOA)** and the principle of **Disparate Impact**, using geography as a credit factor risks creating a proxy for race or national origin. This is sometimes called "redlining" — historically, certain ZIP codes and states were systematically associated with minority populations, and using them as credit factors perpetuates discrimination.

In a real US bank model submission, the Fair Lending team would almost certainly require the removal of `addr_state` regardless of its IV, and would demand a Disparate Impact Analysis before any geographically correlated variable is used. This is a non-negotiable compliance requirement.

---

## Part 7: The Logistic Regression Engine and P-Values

### Why Logistic Regression and Not a Neural Network?

The US **Fair Credit Reporting Act (FCRA)** requires that every applicant who is denied credit must receive written notification of the specific reasons for the denial — called **Adverse Action reason codes**. 

A neural network cannot produce this. The relationship between inputs and outputs passes through hundreds of non-linear transformations, and there is no clean way to say "your score was reduced by 12 points due to your recent credit inquiry history."

Logistic regression with WoE-transformed inputs can do this precisely. Because every feature is on the same log-odds scale and the model is linear in that space, each feature's contribution to the final score is directly calculable. The top 3–4 features with the largest negative point contributions become the Adverse Action reasons.

### The Model Mathematics

The logistic regression models the probability of being Good (1):

$$
P(\text{Good} | \mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_k x_k)}}
$$

Re-arranging to the log-odds form:

$$
\ln\left(\frac{P(\text{Good})}{1 - P(\text{Good})}\right) = \beta_0 + \sum_{j=1}^{k} \beta_j x_j
$$

The left side is the **logit** or log-odds of being Good. Since all $x_j$ are WoE values (which are themselves log-odds ratios), the model is entirely working in log-odds space. The relationship is perfectly linear.

### Maximum Likelihood Estimation

The $\beta$ coefficients are found by maximising the likelihood of observing the actual outcomes:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]
$$

Where $y_i \in \{0, 1\}$ is the actual Good/Bad label and $p_i$ is the model's predicted probability. There is no closed-form solution — optimisation uses iterative methods like **Newton-Raphson** or gradient descent.

### The Problem: sklearn Does Not Give P-Values

Scikit-learn's `LogisticRegression` only returns the fitted coefficients $\hat{\boldsymbol{\beta}}$. It does not return standard errors or p-values.

P-values are critical in credit scoring. A feature with a high IV but a p-value of 0.40 is statistically indistinguishable from zero. If I include that feature in a deployed scorecard and a declined customer challenges the Adverse Action reason code in court, the bank is exposed — it cannot defend a credit decision based on a variable that is not statistically significant.

**My solution:** I built a custom Python class that extends sklearn's `LogisticRegression` by computing p-values manually using the **Fisher Information Matrix**.

### The Fisher Information Matrix: The Full Math

After MLE produces $\hat{\boldsymbol{\beta}}$, the statistical uncertainty around each coefficient is given by the **Cramér-Rao lower bound**, which says that the variance of any unbiased estimator is at least as large as the inverse of the Fisher Information:

$$
\text{Var}(\hat{\boldsymbol{\beta}}) \geq \mathcal{I}(\boldsymbol{\beta})^{-1}
$$

For logistic regression, the Fisher Information Matrix is the **negative expected Hessian** of the log-likelihood. The Hessian measures the curvature of the log-likelihood surface — how sharply it peaks at the MLE solution:

$$
\mathcal{I}(\boldsymbol{\beta}) = -\mathbb{E}\left[\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}\right] = \mathbf{X}^T \mathbf{W} \mathbf{X}
$$

Where $\mathbf{W}$ is a diagonal matrix with $W_{ii} = p_i(1 - p_i)$ — the variance of each observation's Bernoulli outcome under the model's predicted probability.

The standard error of each coefficient is the square root of the corresponding diagonal of $\mathcal{I}^{-1}$:

$$
\text{SE}(\hat{\beta}_j) = \sqrt{\left[\mathcal{I}(\hat{\boldsymbol{\beta}})^{-1}\right]_{jj}}
$$

The **Z-statistic** (Wald statistic) is:

$$
Z_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}
$$

Under the null hypothesis $H_0: \beta_j = 0$, this follows a standard normal distribution. The two-tailed p-value is:

$$
p_j = 2 \times \left(1 - \Phi(|Z_j|)\right)
$$

Where $\Phi$ is the standard normal CDF.

### The Custom Class Code

```python
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy import stats

class LogisticRegression_with_p_values:
    """
    Extends sklearn LogisticRegression to compute Wald p-values
    via the Fisher Information Matrix.
    
    This is equivalent to what statsmodels.Logit() computes,
    but preserves sklearn's API for pipeline compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        
        # Step 1: Get predicted probabilities for each observation
        p_hat = self.model.predict_proba(X)[:, 1]  # P(Good)
        
        # Step 2: Build W, the diagonal weight matrix
        # W_ii = p_i * (1 - p_i)  [Bernoulli variance]
        W = np.diag(p_hat * (1 - p_hat))
        
        # Step 3: Compute the Fisher Information Matrix
        # F = X^T * W * X
        # Prepend intercept column (column of 1s)
        X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
        F = X_with_intercept.T @ W @ X_with_intercept
        
        # Step 4: Invert the Fisher Information Matrix to get variance-covariance
        F_inv = np.linalg.inv(F)
        
        # Step 5: Standard errors = sqrt of diagonal of variance-covariance matrix
        # Combine intercept coef with feature coefs
        all_coefs = np.hstack([self.model.intercept_, self.model.coef_[0]])
        self.standard_errors = np.sqrt(np.diag(F_inv))
        
        # Step 6: Z-statistics (Wald statistic)
        self.z_scores = all_coefs / self.standard_errors
        
        # Step 7: Two-tailed p-values from standard normal distribution
        self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.z_scores)))
        
        # Attach coefficients and feature names for later use
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)


# --- Fitting the model ---
lr = LogisticRegression_with_p_values(max_iter=1000)
lr.fit(X_train_final, y_train)

# --- Inspect results ---
summary = pd.DataFrame({
    'Feature': ['Intercept'] + list(X_train_final.columns),
    'Coefficient': np.hstack([lr.intercept_, lr.coef_[0]]),
    'Std Error': lr.standard_errors,
    'Z-Score': lr.z_scores,
    'P-Value': lr.p_values
})

print(summary.to_string())
```

### Pruning: Removing Statistically Insignificant Variables

After running the custom class, any feature with $p > 0.05$ is removed. In a production scorecard, a variable with a high p-value means its coefficient is statistically indistinguishable from zero. The model is better off without it.

The pruning loop:

```python
# Remove features with p-value > 0.05 iteratively
# (Sometimes removing one feature changes other p-values — re-fit after each removal)

features_to_keep = [
    f for f, p in zip(X_train_final.columns, lr.p_values[1:])
    if p <= 0.05
]

X_train_pruned = X_train_final[features_to_keep]
X_test_pruned = X_test_final[features_to_keep]

# Re-fit final model on pruned feature set
lr_final = LogisticRegression_with_p_values(max_iter=1000)
lr_final.fit(X_train_pruned, y_train)
```

**Why iterative removal?** When a highly collinear feature is removed, the remaining features' standard errors decrease and their p-values improve. A feature that initially failed the significance test may become significant after the collinear partner is removed.

---

## Part 8: The Validation Suite — AUROC, Gini, and KS

### Why Accuracy Is the Wrong Metric

The Lending Club dataset has approximately 80–85% Good borrowers and 15–20% Bad borrowers. This is called **class imbalance**.

If I build a completely useless model that approves every single applicant (predicts "Good" for all), it achieves 80–85% accuracy. That is a high number, and it is completely misleading.

In credit risk, the cost of errors is asymmetric. A **False Negative** (approving someone who defaults) means the bank loses the full outstanding principal — often £5,000–£50,000+. A **False Positive** (declining someone who would have performed) means the bank loses only the net interest income on that loan — typically much smaller.

The cost of a False Negative is roughly 10–20× the cost of a False Positive. Accuracy, which treats both errors equally, is therefore an inappropriate optimisation target for credit risk.

### The Confusion Matrix

At any given score threshold $\tau$, borrowers are split into predicted Good (score above $\tau$) and predicted Bad (score below $\tau$). The outcomes:

|  | **Actual Good** | **Actual Bad** |
|---|---|---|
| **Predicted Good** | True Negative (TN) | False Negative (FN) — **Dangerous** |
| **Predicted Bad** | False Positive (FP) | True Positive (TP) |

$$
\text{Recall} = \frac{TP}{TP + FN} \quad \text{(What fraction of all actual defaults did we catch?)}
$$

$$
\text{Precision} = \frac{TP}{TP + FP} \quad \text{(Of those we flagged as bad, how many actually were?)}
$$

Recall is the metric I care most about in credit — I want to catch as many future defaulters as possible.

```python
from sklearn.metrics import confusion_matrix, classification_report

# Apply a threshold for classification
threshold = 0.9  # Score probabilities above 0.9 predicted as Good

y_pred = np.where(
    lr_final.predict_proba(X_test_pruned)[:, 1] > threshold,
    1,  # Good
    0   # Bad
)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### AUROC — Area Under the ROC Curve

The ROC (Receiver Operating Characteristic) curve plots **True Positive Rate (Recall)** against **False Positive Rate** at every possible threshold. It shows the trade-off between catching defaulters and incorrectly rejecting good borrowers.

$$
\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})
$$

**The probabilistic interpretation:** AUROC is the probability that, if I randomly pick one Good borrower and one Bad borrower, my model gives the Good borrower a higher score. An AUROC of 0.80 means the model correctly rank-orders 80% of Good/Bad pairs.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_pred_proba = lr_final.predict_proba(X_test_pruned)[:, 1]
auroc = roc_auc_score(y_test, y_pred_proba)
print(f'AUROC: {auroc:.4f}')

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auroc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve — Lending Club PD Model')
plt.legend()
plt.show()
```

**Industry benchmarks for retail credit:**

| AUROC | Interpretation |
|---|---|
| 0.50 | Random guessing — model has no value |
| 0.60 – 0.70 | Weak discrimination |
| 0.70 – 0.80 | Acceptable; typical for first-generation scorecards |
| 0.80 – 0.90 | Good; typical for mature scorecards |
| > 0.90 | Excellent — audit for data leakage |

### Gini Coefficient

The Gini Coefficient is the dominant summary metric in European and UK credit risk. It is mathematically tied to AUROC:

$$
\text{Gini} = 2 \times \text{AUROC} - 1
$$

It ranges from 0 (random) to 1 (perfect discrimination). A Gini of 0.60 means the model adds 60% of the maximum possible discrimination over random.

```python
gini = 2 * auroc - 1
print(f'Gini Coefficient: {gini:.4f}')
```

### KS Statistic — The Most Important Credit Risk Metric

The **KS (Kolmogorov-Smirnov) Statistic** is the single most-quoted discrimination metric in retail credit model documentation. It measures the maximum separation between the cumulative distribution functions of Good and Bad borrowers' scores.

$$
\text{KS} = \max_{\tau} \left| F_{\text{Good}}(\tau) - F_{\text{Bad}}(\tau) \right|
$$

Where $F_{\text{Good}}(\tau)$ is the proportion of Good borrowers scoring below threshold $\tau$, and $F_{\text{Bad}}(\tau)$ is the proportion of Bad borrowers scoring below the same threshold.

**What KS physically means:**

Sort all borrowers from lowest score to highest. At every score level, ask: "What percentage of all Goods have I passed, and what percentage of all Bads have I passed?" The KS is the maximum vertical gap between these two running totals.

A high KS means there exists a score threshold where the model very cleanly separates Goods from Bads. That threshold is also the natural starting point for the business's cut-off decision.

```python
def compute_ks(y_true, y_scores):
    """
    Computes the KS Statistic manually.
    y_true: actual labels (1=Good, 0=Bad)
    y_scores: predicted probability of being Good
    """
    # Sort by score descending (highest score first)
    df = pd.DataFrame({'score': y_scores, 'label': y_true})
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    total_good = df['label'].sum()
    total_bad = (df['label'] == 0).sum()
    
    df['cumulative_good'] = (df['label'] == 1).cumsum() / total_good
    df['cumulative_bad'] = (df['label'] == 0).cumsum() / total_bad
    df['KS'] = abs(df['cumulative_good'] - df['cumulative_bad'])
    
    ks_stat = df['KS'].max()
    ks_threshold_idx = df['KS'].idxmax()
    ks_score = df.loc[ks_threshold_idx, 'score']
    
    return ks_stat, ks_score

ks_stat, ks_score = compute_ks(y_test.values, y_pred_proba)
print(f'KS Statistic: {ks_stat:.4f}')
print(f'KS Score Threshold: {ks_score:.4f}')
```

**KS industry benchmarks:**

| KS | Interpretation |
|---|---|
| < 20% | Poor — model lacks separation |
| 20% – 30% | Fair |
| 30% – 40% | Good — industry target for retail |
| 40% – 50% | Very Good |
| > 50% | Excellent — verify for leakage |

**The KS score threshold** — the score at which KS is maximised — is often the starting point for the business when choosing a loan approval cut-off.

**SR 11-7 requirement:** All these metrics must be computed on the **out-of-sample test set**, never on training data. If training KS = 45% but test KS = 22%, the model is overfit and fails validation. The gap between training and test performance is itself a model quality indicator.

---

## Part 9: Scorecard Scaling — From Log-Odds to 300–850

### The Business Requirement

The Chief Risk Officer does not make decisions based on "predicted probability of 0.812" or "log-odds of 1.45". The entire consumer banking infrastructure — from credit bureau interfaces to customer-facing decline letters — is built around **integer credit scores** in a defined range.

I need to translate the logistic regression's mathematical output into a score between 300 and 850.

### My Method: Min-Max Scaling

In the notebook, I used a direct min-max transformation:

```python
# Find the theoretical worst-case score (every feature takes its riskiest bin value)
# and the theoretical best-case score (every feature takes its safest bin value)

# Gather all scorecard coefficients
df_scorecard = summary_pruned.copy()

# For each variable, the worst bin = most negative coefficient (riskiest)
# The best bin = most positive coefficient (safest)
# Sum all worst bins to get the minimum possible raw score
# Sum all best bins to get the maximum possible raw score

min_raw_score = df_scorecard['Coefficient'].min_sum  # theoretical worst case
max_raw_score = df_scorecard['Coefficient'].max_sum  # theoretical best case

min_score = 300
max_score = 850

# Scale each coefficient to the 300-850 range
df_scorecard['Score'] = (
    df_scorecard['Coefficient'] *
    (max_score - min_score) / (max_raw_score - min_raw_score)
)

# Round to integers (scorecard points must be whole numbers)
df_scorecard['Score - Final'] = df_scorecard['Score'].round()
```

The rounding step sometimes causes the maximum possible score to be 849 or 851 due to accumulated rounding. Adjusting a single reference category's points by 1 to restore the 850 maximum is a standard calibration step — provided it is documented and proven not to affect the model's rank-ordering.

### The Correct Industry Method: PDO Scaling

Min-max scaling anchors the range to 300–850 but does not attach a probabilistic meaning to any specific score. A score of 600 under min-max means "you are roughly in the middle of the historical range" — but it does not tell the business what the default odds are at 600.

**PDO (Points to Double the Odds)** is the industry standard. It anchors the score to actual risk:

The score is defined as a linear function of the log-odds:

$$
\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds}_{\text{Good:Bad}})
$$

Three parameters define the scale:

- **Base Score ($S_0$):** The score assigned at a reference Good:Bad odds ratio. Typically 600.
- **Base Odds ($\Theta_0$):** The Good:Bad odds at which the base score is awarded. Typically 50:1 (50 Good borrowers for every 1 Bad).
- **PDO:** How many score points are needed to double the Good:Bad odds. Typically 20 points.

**Deriving Factor and Offset:**

At base odds $\Theta_0$:

$$
S_0 = \text{Offset} + \text{Factor} \times \ln(\Theta_0)
$$

At double the base odds $2\Theta_0$ (score increases by PDO):

$$
S_0 + \text{PDO} = \text{Offset} + \text{Factor} \times \ln(2\Theta_0)
$$

Subtracting:

$$
\text{PDO} = \text{Factor} \times \ln(2)
$$

$$
\boxed{\text{Factor} = \frac{\text{PDO}}{\ln(2)}}
$$

$$
\boxed{\text{Offset} = S_0 - \text{Factor} \times \ln(\Theta_0)}
$$

**Worked example with typical bank parameters:**

$$
\text{Factor} = \frac{20}{\ln(2)} = \frac{20}{0.6931} \approx 28.85
$$

$$
\text{Offset} = 600 - 28.85 \times \ln(50) = 600 - 28.85 \times 3.912 \approx 487.1
$$

A borrower with log-odds $= \ln(50) = 3.912$ receives a score of exactly 600. A borrower with log-odds $= \ln(100) = 4.605$ receives a score of exactly 620 (50 odds doubled to 100 → score goes up by 20 points).

**The total score decomposition:**

$$
\text{Total Score} = \text{Offset} + \text{Factor} \times \beta_0 + \sum_{j=1}^{k} \text{Factor} \times \beta_j \times \text{WoE}_{ij}
$$

Each term $\text{Factor} \times \beta_j \times \text{WoE}_{ij}$ is the **points contribution of variable $j$ at bin $i$**. This decomposition is what makes Adverse Action reason codes possible — the top negative contributors are the reasons for a low score.

### Applying the Score to New Applicants

```python
def score_applicant(applicant_woe_vector, intercept, coefs, factor, offset):
    """
    Converts a WoE-transformed applicant vector into a credit score.
    
    applicant_woe_vector: array of WoE values for each feature
    intercept: beta_0 from logistic regression
    coefs: beta_1 ... beta_k coefficients
    factor, offset: PDO scaling parameters
    """
    log_odds = intercept + np.dot(coefs, applicant_woe_vector)
    score = offset + factor * log_odds
    return round(score)
```

---

## Part 10: Cut-off Strategy — Translating the Score Into a Decision

### The Quant Team Does Not Set the Cut-off

This is one of the most important governance principles in credit risk. The model produces scores. The business produces decisions. These are different functions and they must not be conflated.

Under **SR 11-7**, the model development team is responsible for building the mathematical engine and measuring its performance. The **Credit Policy team** (Chief Risk Officer, Head of Credit Policy, Risk Committee) is responsible for deciding where to draw the approval/decline line.

If the quant team sets the cut-off, they are making a credit policy decision without proper oversight. This is a governance violation.

My role is to build the **Strategy Table** — the tool that allows the business to see exactly what each cut-off choice means in terms of approval rates, expected default rates, and revenue.

### Building the Strategy Table

```python
# Get all unique probability thresholds from the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Map each probability threshold to its corresponding scorecard score
def prob_to_score(prob, factor, offset, intercept_offset=0):
    """Converts predicted probability to scorecard score."""
    # P(Good) = prob → log-odds of Good = log(prob / (1-prob))
    log_odds = np.log(prob / (1 - prob))
    return factor * log_odds + offset

# Build the strategy table
strategy_rows = []
for prob_threshold in np.linspace(0.01, 0.99, 1000):
    score_cutoff = prob_to_score(prob_threshold, factor=28.85, offset=487.1)
    n_approved = (y_pred_proba >= prob_threshold).sum()
    approval_rate = n_approved / len(y_pred_proba)
    
    # Among approved, what fraction actually defaulted?
    approved_mask = y_pred_proba >= prob_threshold
    if approved_mask.sum() > 0:
        bad_rate_approved = (y_test[approved_mask] == 0).mean()
    else:
        bad_rate_approved = 0
    
    strategy_rows.append({
        'Score Cutoff': round(score_cutoff),
        'Prob Threshold': round(prob_threshold, 3),
        'N Approved': n_approved,
        'Approval Rate': round(approval_rate * 100, 1),
        'Bad Rate (Approved)': round(bad_rate_approved * 100, 2)
    })

df_strategy = pd.DataFrame(strategy_rows)
print(df_strategy.to_string(index=False))
```

**Example output (illustrative):**

| Score Cutoff | Approval Rate | Bad Rate (Approved) |
|---|---|---|
| 500 | 95.2% | 18.4% — too many defaults |
| 580 | 78.3% | 11.2% |
| 620 | 65.1% | 7.8% — reasonable for sub-prime |
| 660 | 50.4% | 4.3% — industry norm for prime |
| 720 | 31.7% | 1.9% — near-prime/super-prime |
| 780 | 15.2% | 0.8% — very conservative |

The business reads this table as a dial. If Marketing needs a 65% approval rate to hit revenue targets, they look at the row showing 65.1% — score cutoff 620, expected bad rate 7.8%. They then decide if the bank's risk appetite can absorb a 7.8% bad rate.

### The Champion-Challenger Deployment Protocol

When the business agrees on a cut-off and the model passes all validation gates, it does not go straight to production. It enters **shadow mode** first.

The current model in production (the **Champion**) continues making all actual approve/decline decisions. My new model (the **Challenger**) runs silently in the background, scoring the same applicants without affecting any decisions.

After 3–6 months of parallel running, the performance of both models on the same cohort is compared. If the Challenger demonstrates better discrimination (higher KS, lower bad rate at the same approval rate) without causing unstable score distribution shifts, the Risk Committee approves the swap. The Challenger becomes the new Champion.

This protocol protects the bank from deploying an untested model that performs well on historical data but fails on live applicants.

---

## Connections to the Rest of the Quant OS Brain

This note is the technical core of the entire Quant OS system. Every other note connects back to a decision made here:

- [[Tharun-Kumar-Gajula]] — The master profile anchoring this portfolio.
- [[2_monitoring_model]] — After this model goes live, the monitoring engine takes over to validate PSI, CSI, and score distribution shifts.


---

## Key Concepts Summary

| Concept | What It Is | Where It Appears in This Project |
|---|---|---|
| **Target Variable (90 DPD)** | Binary label: 0=Bad, 1=Good | `loan_status` → `good_bad` |
| **OOT Validation** | Train on early years, test on later years | Should replace 80/20 random split |
| **MNAR Missing Data** | Missing *because of* the underlying value | `annual_inc`, `mths_since_last_delinq` |
| **WoE Transformation** | Replaces raw values with log-odds risk scores | All continuous and categorical features |
| **IV (Information Value)** | Measures each feature's overall predictive power | Feature selection gate before modelling |
| **Monotonicity** | WoE must move consistently in one direction | Required for conceptual soundness under SR 11-7 |
| **Perfect Multicollinearity** | Design matrix becomes singular | Caused by forgetting to drop reference category |
| **VIF** | Detects collinearity between features | Run before finalising feature set |
| **Fisher Information Matrix** | Source of standard errors and p-values | Custom class extends sklearn |
| **P-Value Pruning** | Remove features with p > 0.05 | Post-fitting step before scaling |
| **AUROC** | Global rank-ordering probability | Target: 0.70–0.80 for retail credit |
| **KS Statistic** | Maximum separation of Good/Bad CDFs | Primary metric; target: 30–40% |
| **PDO Scaling** | Anchors score to probabilistic risk odds | Production standard; replaces min-max |
| **Adverse Action Codes** | Legal requirement: explain each denial | Enabled by WoE + logistic decomposition |
| **Fair Lending / ECOA** | Cannot proxy for protected characteristics | Reason to scrutinise `addr_state` |
| **Champion-Challenger** | Shadow deployment before full cutover | Production deployment protocol |
| **SR 11-7** | US Fed model risk management guidelines | Governs all validation and governance decisions |

---

*This note is Version 1.0. As the model is refined and extended to LGD and EAD, additional sections will be appended. Next: [[LGD-EAD-Model-Lending-Club.md]].*
