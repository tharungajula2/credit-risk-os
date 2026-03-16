---
title: "Quant OS Credit Risk Curriculum"
cluster: "System Atlas"
graph_exclude: true
---

# Quant OS: End-to-End Retail Credit Risk Modeling & Machine Learning
### A Zero-to-Hero Practitioner's Curriculum

> **Quant OS Flagship Curriculum** | Version 1.0 | Cluster: Credit Risk  
> *A definitive, practitioner-grade knowledge graph for quantitative finance professionals.*

---

## Table of Contents

- [[Phase-1-Macro-Picture]] — The Macro Picture & End-to-End Workflow
- [[Phase-2-Data-Architecture]] — Data Architecture & Preprocessing
- [[Phase-3-Algorithmic-Engine]] — The Algorithmic Engine & Scorecard Development
- [[Phase-4-Model-Validation]] — Model Validation & Performance Metrics
- [[Phase-5-Production-Monitoring]] — Production Monitoring & Stability
- [[Phase-6-Deployment-Lifecycle]] — The Deployment Lifecycle & Client Engineering
- [[Zettelkasten-Architecture]] — Zettelkasten Architecture

---

# Phase 1: The Macro Picture & End-to-End Workflow

## 1.1 What Is Credit Risk Modeling?

**Credit risk** is the probability that a borrower will fail to meet their contractual obligations — i.e., they will default on a loan, miss payments, or become severely delinquent. In the retail banking context, this encompasses personal loans, credit cards, mortgages, auto finance, and buy-now-pay-later (BNPL) products.

A **credit risk model** is a mathematical and statistical system that quantifies this probability for every applicant or existing borrower. The output is typically one of:

- **A Probability of Default (PD)** — a continuous score between 0 and 1.
- **A Credit Score** — a scaled, human-readable integer (e.g., 300–850 on the FICO scale).
- **A Risk Segment** — a categorical band (e.g., Low / Medium / High Risk).

### Why Does This Matter?

The business stakes are enormous. A bank deploying a 1% improvement in default prediction accuracy across a £10 billion retail loan portfolio can recover tens of millions in avoided losses. Conversely, a poorly calibrated model silently destroys capital over months before anyone detects the degradation.

Credit risk models serve three primary commercial functions:

1. **Origination** — Should we approve this application, and at what interest rate?
2. **Account Management** — Should we increase or decrease this customer's credit limit?
3. **Collections Prioritization** — Among delinquent customers, who is most recoverable?

---

## 1.2 The End-to-End Project Lifecycle

A credit risk modeling project is not just a modeling exercise. It is a full engineering and governance pipeline. The diagram below maps the canonical lifecycle:

```
[Raw Data Sources]
        │
        ▼
[Data Ingestion & ETL]
        │
        ▼
[Exploratory Data Analysis (EDA)]
        │
        ▼
[Data Preprocessing & Feature Engineering]
    (Missing value imputation, outlier capping, WoE transformation)
        │
        ▼
[Model Development]
    (Logistic Regression / Tree-Based / Ensemble)
        │
        ▼
[Model Validation & Performance Benchmarking]
    (AUROC, KS Statistic, Gini, PSI)
        │
        ▼
[Technical & Business Review]
        │
        ▼
[Deployment to Scoring Engine / API]
        │
        ▼
[Live Production Monitoring]
    (PSI, CSI, Outcome Tracking)
        │
        ▼
[Model Refresh / Redevelopment Trigger]
```

Each arrow represents a decision gate. Quantitative analysts must master not just the modeling nodes, but the data and governance nodes on either side.

### The Three Pillars of a Credit Risk Project

| Pillar | Core Question | Key Deliverable |
|---|---|---|
| **Data** | Do we have signal? | Cleaned dataset, IV report |
| **Model** | Can we quantify risk? | Scorecard / PD model |
| **Governance** | Can we defend it? | Model Risk documentation |

---

# Phase 2: Data Architecture & Preprocessing

## 2.1 How Retail Credit Data Actually Looks

A typical retail credit portfolio dataset is a **wide, panel-structured table** where each row represents a unique loan application or account snapshot. Understanding the column taxonomy is the first step.

### Standard Feature Categories

| Category | Example Variables | Notes |
|---|---|---|
| **Application Attributes** | Age, Income, Employment Status, Loan Amount Requested | Collected at origination |
| **Bureau Attributes** | Number of trade lines, worst delinquency (months), total outstanding debt, bureau score | Sourced from Equifax / Experian / TransUnion |
| **Behavioral Attributes** | Months on book, average utilization last 6 months, payment trend | Only available for existing customers |
| **Macroeconomic** | GDP growth rate, unemployment rate at origination | Used in stress testing |
| **Target Variable** | `BAD` flag (1 = default, 0 = good) | Binary; definition varies by lender |

### Defining the Target Variable

**This is the single most consequential decision in a credit risk project.** A "bad" borrower is typically defined as anyone who has been **90+ days past due (DPD)** at least once within an **observation window** (typically 12–24 months post-origination). This definition must be locked down before any modeling begins.

```
Performance Window: [Origination Date] → [12 or 24 months later]
Bad Definition: ≥ 90 DPD at least once in that window
Good Definition: Never exceeded 30 DPD in that window
Indeterminate: Between 30 and 90 DPD (often excluded from training data)
```

The exclusion of indeterminate accounts is a critical industry practice. Including them pollutes the signal at the decision boundary.

---

## 2.2 Handling Dirty Data: Advanced Missing Data Strategies

Real credit data is never clean. Missing values arise from system migrations, optional application fields, and bureau lookup failures. The method of imputation must match the **mechanism** of missingness.

### The Three Mechanisms of Missingness

1. **Missing Completely At Random (MCAR):** The missingness has no relationship to any variable. Simple imputation is valid.
2. **Missing At Random (MAR):** Missingness is related to *observed* variables but not the missing value itself. Conditional imputation is appropriate.
3. **Missing Not At Random (MNAR):** Missingness is related to the unobserved value itself (e.g., applicants who refuse to disclose income tend to have lower income). This is the most dangerous case.

### Imputation Strategies for Credit Data

| Strategy | When to Use | Risk |
|---|---|---|
| **Mean/Median Imputation** | MCAR only; low-signal variables | Distorts variance; masks MNAR |
| **Mode Imputation** | Categorical features, MCAR | Can over-inflate a category |
| **Regression Imputation** | MAR; when predictors exist | Understates standard error |
| **KNN Imputation** | MAR; moderate missing rate (<30%) | Computationally expensive at scale |
| **Multiple Imputation (MICE)** | MAR; research-grade rigor needed | High complexity |
| **Missing as a Category** | MNAR; bureau attribute not found | **Best practice in credit** — preserves the signal that "missing" itself carries |

> **Practitioner Note:** In credit risk, a missing bureau attribute (e.g., "No credit history found") is often a powerful predictor in its own right. Creating a binary indicator `variable_is_missing = 1` and then imputing the original with a neutral value is the industry-standard approach for MNAR data.

### Outlier Capping (Winsorization)

Extreme outliers in financial data (e.g., an income of $50,000,000 in a consumer dataset) can catastrophically distort model coefficients. The standard remedy is **Winsorization**:

$$
x_{\text{capped}} = \max\left(P_1, \min\left(x, P_{99}\right)\right)
$$

Where $P_1$ and $P_{99}$ are the 1st and 99th percentiles of the training distribution. Values below $P_1$ are floored; values above $P_{99}$ are capped. This must be computed **only on the training set** and then applied identically to validation and production data.

```python
# Python implementation of Winsorization
import numpy as np

def winsorize(series, lower_pct=0.01, upper_pct=0.99):
    """
    Caps a series at the lower and upper percentile thresholds.
    CRITICAL: Fit only on training data; apply to all subsequent splits.
    """
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower=lower, upper=upper), lower, upper
```

---

## 2.3 Statistical Assumptions: Homoscedasticity & Multicollinearity

### Homoscedasticity

**Homoscedasticity** means that the variance of the model's residuals (errors) is **constant** across all levels of the independent variables. Its opposite, **heteroscedasticity**, means the error variance is non-constant.

In the context of **Logistic Regression** for credit scoring (see [[Phase-3-Algorithmic-Engine]]), the model does not produce continuous residuals in the OLS sense, so classical homoscedasticity tests (Breusch-Pagan, White's test) are not directly applicable. However, the concept matters because:

- **Heteroscedastic inputs** (e.g., income that fans out dramatically at higher levels) can create unstable coefficient estimates.
- **WoE transformation** (see [[Phase-3-Algorithmic-Engine#Weight-of-Evidence]]) implicitly addresses this by binning continuous variables, removing the raw scale.

For any sub-model using linear regression (e.g., predicting Loss Given Default), heteroscedasticity must be formally tested and corrected via **robust standard errors** (Huber-White sandwich estimator) or **Weighted Least Squares (WLS)**.

### Multicollinearity

**Multicollinearity** occurs when two or more predictor variables are highly linearly correlated. In credit risk, this is ubiquitous — bureau score correlates with delinquency count, utilization correlates with outstanding balance, and income correlates with loan-to-value ratio.

**The Danger:** Multicollinearity inflates the **variance of coefficient estimates**. Individual $\beta$ coefficients become unstable — small changes in the data produce large swings in estimated coefficients, destroying the scorecard's stability and interpretability.

**Detection: Variance Inflation Factor (VIF)**

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

Where $R_j^2$ is the R-squared from regressing variable $j$ on all other predictors.

| VIF Value | Interpretation |
|---|---|
| 1.0 | No collinearity |
| 1–5 | Moderate; generally acceptable |
| 5–10 | High; investigate |
| > 10 | **Severe; requires remediation** |

**Remediation Strategies:**

1. **Remove one of the correlated pair** — prefer the variable with higher [[Phase-4-Model-Validation#Information-Value|Information Value]].
2. **Principal Component Analysis (PCA)** — transforms correlated features into orthogonal components. Loses interpretability.
3. **Regularization** (Ridge/Lasso) — see [[Phase-3-Algorithmic-Engine#Regularization]]. Ridge regression is specifically designed to handle multicollinearity.

```python
# VIF Calculation
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def compute_vif(df_features: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_features.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df_features.values, i)
        for i in range(df_features.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False)
```

---

# Phase 3: The Algorithmic Engine & Scorecard Development

## 3.1 Feature Engineering: Weight of Evidence (WoE) & Information Value (IV)

Weight of Evidence and Information Value are the **cornerstone** of classical credit scorecard development. They were developed in the 1950s-60s within the consumer credit industry and remain indispensable today.

### Why WoE Exists: The Problem It Solves

Raw continuous variables (e.g., Age = 34.7 years) violate several practical requirements of credit modeling:

1. **Non-linearity:** Risk does not change linearly with age. A 20-year-old and a 22-year-old may have similar risk, but the jump from 22 to 23 might be large.
2. **Outlier sensitivity:** A single extreme value skews the model.
3. **Missing value handling:** Raw variables cannot natively encode "not available."
4. **Regulatory interpretability:** Regulators demand bin-level explanations of why a score was assigned.

**WoE transformation solves all four problems simultaneously.**

### The WoE Formula

For each bin $i$ of a variable:

$$
\text{WoE}_i = \ln\left(\frac{\%\text{Events}_i}{\%\text{Non-Events}_i}\right) = \ln\left(\frac{P(\text{Bad}|X \in \text{bin}_i)}{P(\text{Good}|X \in \text{bin}_i)}\right)
$$

Where:
- **Events** = "Bad" borrowers (defaulters)
- **Non-Events** = "Good" borrowers (performing)
- $\%\text{Events}_i = \frac{\text{Number of Bads in bin } i}{\text{Total number of Bads}}$
- $\%\text{Non-Events}_i = \frac{\text{Number of Goods in bin } i}{\text{Total number of Goods}}$

**Interpretation:**
- $\text{WoE} > 0$: The bin has a higher concentration of Bads than the overall portfolio (elevated risk).
- $\text{WoE} < 0$: The bin has a lower concentration of Bads (reduced risk).
- $\text{WoE} = 0$: The bin's bad rate equals the portfolio average.

### The Information Value Formula

Information Value aggregates WoE across all bins to produce a single **feature importance** metric:

$$
\text{IV} = \sum_{i=1}^{n} \left(\%\text{Events}_i - \%\text{Non-Events}_i\right) \times \text{WoE}_i
$$

**The Standard IV Thresholds (Industry Convention):**

| IV Range | Predictive Power | Action |
|---|---|---|
| < 0.02 | Useless | Discard |
| 0.02 – 0.10 | Weak | Use with caution |
| 0.10 – 0.30 | Medium | Strong candidate |
| 0.30 – 0.50 | Strong | Very predictive |
| > 0.50 | Suspicious | **Check for data leakage** |

> **Critical Warning:** An IV above 0.50 is almost always a sign of **target leakage** — the feature is accidentally encoding future information (e.g., "days past due at end of performance window" inadvertently included as a predictor). This is one of the most dangerous and common errors in credit model development.

### WoE Binning Strategy

Binning must be done carefully. Two approaches exist:

- **Equal-width binning:** Fixed interval width. Simple but ignores distribution shape.
- **Fine classing then coarse classing:** Start with 20 equal-frequency bins (fine), then merge adjacent bins with similar WoE values into fewer, monotonically-ordered final bins (coarse). **This is the industry standard.**

Monotonicity in WoE bins is a regulatory and interpretability requirement. A scorecard where "higher income → higher WoE (more bad)" is indefensible to a regulator.

```python
# Conceptual WoE/IV computation
import numpy as np
import pandas as pd

def compute_woe_iv(df: pd.DataFrame, feature: str, target: str,
                   n_bins: int = 10) -> pd.DataFrame:
    """
    Computes WoE and IV for a single feature.
    df: training DataFrame
    feature: column name of the predictor
    target: binary target (1=Bad, 0=Good)
    """
    total_events = df[target].sum()
    total_non_events = (df[target] == 0).sum()

    df_binned = df.copy()
    df_binned['bin'] = pd.qcut(df_binned[feature], q=n_bins,
                                duplicates='drop')

    grouped = df_binned.groupby('bin')[target].agg(
        events=lambda x: x.sum(),
        non_events=lambda x: (x == 0).sum()
    ).reset_index()

    grouped['pct_events'] = grouped['events'] / total_events
    grouped['pct_non_events'] = grouped['non_events'] / total_non_events

    # Avoid log(0)
    grouped['pct_events'] = grouped['pct_events'].replace(0, 0.0001)
    grouped['pct_non_events'] = grouped['pct_non_events'].replace(0, 0.0001)

    grouped['WoE'] = np.log(grouped['pct_events'] / grouped['pct_non_events'])
    grouped['IV_component'] = (
        (grouped['pct_events'] - grouped['pct_non_events']) * grouped['WoE']
    )

    iv = grouped['IV_component'].sum()
    grouped['IV_total'] = iv
    return grouped
```

---

## 3.2 Logistic Regression: The Mechanics

Logistic Regression is the canonical algorithm for credit scoring. Its primacy is not due to superior predictive power (ensemble methods outperform it), but because of its **interpretability, regulatory acceptance, and scorecard scaling properties**.

### From Linear to Logistic: The Link Function

We want to model $P(\text{Bad} | \mathbf{X})$. A linear model $\hat{y} = \mathbf{X}\boldsymbol{\beta}$ produces values outside $[0,1]$. The logistic transformation maps any real number to the probability interval:

$$
P(\text{Bad} | \mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_k x_k)}}
$$

This is the **sigmoid function**. Taking the inverse (the logit link):

$$
\ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 x_1 + \ldots + \beta_k x_k
$$

The left-hand side, $\ln\left(\frac{P}{1-P}\right)$, is the **log-odds** (or logit). It is a linear function of the features. **This linearity in the log-odds space is what makes scorecard construction possible.**

### Maximum Likelihood Estimation (MLE)

Logistic Regression coefficients $\boldsymbol{\beta}$ are estimated by **Maximum Likelihood Estimation**. Given $n$ independent observations, the **Likelihood function** is:

$$
\mathcal{L}(\boldsymbol{\beta}) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}
$$

Where $p_i = P(\text{Bad}_i | \mathbf{x}_i)$ and $y_i \in \{0, 1\}$. Taking the log (for numerical stability):

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]
$$

MLE finds the $\boldsymbol{\beta}$ that **maximizes** this log-likelihood. There is no closed-form solution; optimization proceeds via **Newton-Raphson** or gradient-based methods.

### Interpreting $\beta$ Coefficients

For a unit increase in $x_j$ (holding all else constant):
- The **log-odds** of being "Bad" change by $\beta_j$.
- The **odds** of being "Bad" are multiplied by $e^{\beta_j}$.

If $\beta_j = 0.45$ for "Number of delinquencies":

$$
e^{0.45} \approx 1.57
$$

Each additional delinquency is associated with a **57% increase in the odds** of default. This interpretability is why Logistic Regression dominates regulatory submissions.

---

## 3.3 Scorecard Scaling: From Log-Odds to Points

A raw logistic regression outputs a probability. Credit bureaus and lenders communicate risk through **integer scores** (e.g., 300–850). The transformation from log-odds to scorecard points is mathematically precise.

### The Points-to-Double-Odds (PDO) System

The scorecard is defined by three parameters:

1. **Base Score ($S_0$):** The score assigned at a reference odds-to-bad ratio.
2. **Base Odds ($\Theta_0$):** The reference odds (e.g., 50 Goods : 1 Bad = 50).
3. **PDO (Points to Double the Odds):** The number of scorecard points required to double the Good:Bad odds (i.e., halve the risk). Typically 20 points.

**Deriving the Scaling Constants:**

The score is a linear function of the log-odds:

$$
\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})
$$

Using the two boundary conditions:

At the base odds $\Theta_0$:

$$
S_0 = \text{Offset} + \text{Factor} \times \ln(\Theta_0)
$$

At double the base odds $2\Theta_0$ (score increases by PDO):

$$
S_0 + \text{PDO} = \text{Offset} + \text{Factor} \times \ln(2\Theta_0)
$$

Subtracting the first equation from the second:

$$
\text{PDO} = \text{Factor} \times \ln(2)
$$

$$
\boxed{\text{Factor} = \frac{\text{PDO}}{\ln(2)}}
$$

$$
\boxed{\text{Offset} = S_0 - \text{Factor} \times \ln(\Theta_0)}
$$

### Converting Log-Odds to Individual Scorecard Points

The total log-odds from logistic regression is:

$$
\ln(\hat{\Theta}) = \beta_0 + \sum_{j=1}^{k} \beta_j \text{WoE}_{ij}
$$

The total score is decomposed into a **base score** plus **characteristic scores**:

$$
\text{Total Score} = \text{Offset} + \text{Factor} \times \beta_0 + \sum_{j=1}^{k} \text{Factor} \times \beta_j \times \text{WoE}_{ij}
$$

Each term $\text{Factor} \times \beta_j \times \text{WoE}_{ij}$ is the **points contribution of variable $j$, bin $i$**. This decomposition allows a credit analyst to explain exactly which factors raised or lowered a specific applicant's score.

**Worked Example:**

| Parameter | Value |
|---|---|
| Base Score ($S_0$) | 600 |
| Base Odds ($\Theta_0$) | 50 (50:1 Good:Bad) |
| PDO | 20 |

$$
\text{Factor} = \frac{20}{\ln(2)} = \frac{20}{0.6931} \approx 28.85
$$

$$
\text{Offset} = 600 - 28.85 \times \ln(50) = 600 - 28.85 \times 3.912 = 600 - 112.9 \approx 487.1
$$

A borrower with $\ln(\hat{\Theta}) = 3.912$ (i.e., odds of 50:1) receives a score of exactly 600.

---

## 3.4 Regularization: Preventing Overfitting

In high-dimensional credit datasets (potentially thousands of bureau attributes), models are prone to **overfitting** — learning noise patterns in the training data that do not generalize to new applicants.

Regularization adds a **penalty term** to the MLE objective, shrinking coefficients toward zero and preventing the model from becoming too complex.

### Lasso Regression (L1 Regularization)

$$
\ell_{\text{Lasso}}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \lambda \sum_{j=1}^{k} |\beta_j|
$$

- **Effect:** Drives some coefficients to **exactly zero**, performing automatic feature selection.
- **Use Case:** When you have many features and want a sparse model — ideal for large bureau data dumps.
- **Limitation:** Among a group of correlated features, Lasso arbitrarily selects one and zeroes the rest. See [[Phase-2-Data-Architecture#Multicollinearity]].

### Ridge Regression (L2 Regularization)

$$
\ell_{\text{Ridge}}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \lambda \sum_{j=1}^{k} \beta_j^2
$$

- **Effect:** Shrinks all coefficients uniformly toward zero but **never to exactly zero**. All features are retained.
- **Use Case:** When you have many correlated features (e.g., multiple utilization metrics). Ridge distributes the predictive load across the correlated group.
- **Solves:** Multicollinearity (see [[Phase-2-Data-Architecture#Multicollinearity]]).

### Elastic Net

$$
\ell_{\text{ElasticNet}}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \lambda_1 \sum|\beta_j| - \lambda_2 \sum\beta_j^2
$$

A convex combination of L1 and L2 penalties. Best of both worlds: sparsity *and* collinearity handling. The mixing ratio $\alpha = \frac{\lambda_1}{\lambda_1 + \lambda_2}$ is a hyperparameter.

| Method | Feature Selection | Handles Collinearity | Sparsity |
|---|---|---|---|
| Ridge (L2) | No | ✅ Yes | No |
| Lasso (L1) | ✅ Yes | Partial | ✅ Yes |
| Elastic Net | ✅ Yes | ✅ Yes | ✅ Yes |

---

## 3.5 Tree-Based Models

### 3.5.1 Decision Trees: The Foundation

A Decision Tree recursively partitions the feature space by selecting the **split** that best separates "Good" and "Bad" borrowers at each node.

**Splitting Criteria:**

**Gini Impurity:** Measures the probability of misclassifying a randomly drawn sample.

$$
\text{Gini}(t) = 1 - \sum_{c \in \{0,1\}} p(c|t)^2 = 1 - p_{\text{Good}}^2 - p_{\text{Bad}}^2
$$

A pure node (all Good or all Bad) has Gini = 0. The algorithm selects the split minimizing the **weighted Gini** across child nodes.

**Entropy / Information Gain:**

$$
H(t) = -\sum_{c} p(c|t) \log_2 p(c|t)
$$

$$
\text{Information Gain} = H(\text{parent}) - \sum_{\text{children}} \frac{n_{\text{child}}}{n_{\text{parent}}} H(\text{child})
$$

Decision Trees are prone to **overfitting** (growing until every leaf has one sample). Pruning via `max_depth`, `min_samples_leaf`, and `max_leaf_nodes` is essential.

### 3.5.2 Ensemble Methods

Ensembles combine many weak learners (shallow trees) to produce a strong learner.

#### Random Forest (Bagging Ensemble)

Random Forest trains $B$ decision trees, each on a **bootstrap sample** of the training data, and with only a **random subset of features** ($\sqrt{k}$ for classification) available at each split.

$$
\hat{P}_{\text{RF}} = \frac{1}{B} \sum_{b=1}^{B} \hat{P}_b(\mathbf{x})
$$

- **Reduces variance** (averaging over many trees smooths out noisy individual predictions).
- **Does not reduce bias** (each tree is still trained on the same data distribution).
- Features selected via **Mean Decrease in Impurity (MDI)** or **Permutation Importance**.
- Excellent **out-of-bag (OOB) error** estimate as a built-in validation tool.

#### XGBoost (Gradient Boosting)

XGBoost builds trees **sequentially**, where each new tree corrects the residual errors of the ensemble built so far. The objective function is:

$$
\mathcal{L}^{(t)} = \sum_{i=1}^n \ell(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)
$$

Where $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$ is the regularization term ($T$ = number of leaves, $w_j$ = leaf weights).

XGBoost's innovations:
- **Second-order Taylor expansion** of the loss function for faster, more stable optimization.
- **Column subsampling** (like Random Forest) to prevent overfitting.
- **Sparsity-aware** split finding — handles missing values natively.

#### LightGBM (Gradient Boosting, Optimized)

LightGBM extends the boosting framework with two critical algorithmic innovations:

1. **Gradient-based One-Side Sampling (GOSS):** Retains all samples with large gradients (high error) and subsamples instances with small gradients. This dramatically reduces training data while preserving split quality.
2. **Exclusive Feature Bundling (EFB):** Bundles mutually exclusive sparse features together, reducing the effective feature dimensionality.

**Practical Comparison:**

| Property | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| **Training Speed** | Parallelizable | Moderate | ✅ Very Fast |
| **Memory Usage** | High | Moderate | ✅ Low |
| **Overfitting Resistance** | High (bagging) | Moderate (tuning needed) | Moderate (tuning needed) |
| **Performance on Tabular** | Very Good | ✅ Excellent | ✅ Excellent |
| **Interpretability** | SHAP values | SHAP values | SHAP values |
| **Missing Value Handling** | Requires imputation | ✅ Native | ✅ Native |
| **Regulatory Acceptance** | Low (needs SHAP) | Low (needs SHAP) | Low (needs SHAP) |

---

## 3.6 Hyperparameter Tuning

Model performance is highly sensitive to hyperparameter choices. Three search strategies exist.

### Grid Search

Exhaustively evaluates every combination in a predefined parameter grid.

- **Advantage:** Guaranteed to find the best combination within the grid.
- **Disadvantage:** Exponentially expensive. A grid of 5 parameters × 10 values each = $10^5 = 100,000$ fits.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500],
    'subsample': [0.7, 0.85, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
```

### Random Search

Randomly samples hyperparameter combinations from the search space.

- **Key insight (Bergstra & Bengio, 2012):** With the same compute budget, Random Search finds better configurations than Grid Search because in most models, **only a few hyperparameters are critical**. Random Search is more likely to explore the critical dimensions.
- **Practical rule:** Use 60–100 random trials for most credit risk models.

### Bayesian Optimization

Uses a probabilistic surrogate model (typically a **Gaussian Process** or **Tree-structured Parzen Estimator / TPE**) to model the relationship between hyperparameters and model performance. It selects the *next* hyperparameter configuration to try by maximizing an **acquisition function** that balances exploration (uncertain regions) and exploitation (known good regions).

$$
\hat{x}_{t+1} = \arg\max_{x} \text{EI}(x | \mathcal{D}_t)
$$

Where $\text{EI}$ is **Expected Improvement** over the current best result.

- **Advantage:** Significantly more sample-efficient than Grid or Random Search.
- **Libraries:** `Optuna`, `Hyperopt`, `BayesianOptimization`.

| Method | Sample Efficiency | Implementation Complexity | Best For |
|---|---|---|---|
| Grid Search | ❌ Low | Low | Small grids, final fine-tuning |
| Random Search | Moderate | Low | Good baseline; most practical cases |
| Bayesian Optimization | ✅ High | Moderate | Large search spaces; expensive models |

---

# Phase 4: Model Validation & Performance Metrics

## 4.1 The Validation Philosophy

Model validation is **not** a post-hoc formality. It is the scientific proof that the model generalizes to new, unseen borrowers. A model validated only on its training data is worthless — it may have memorized the training set entirely.

**The Standard Validation Protocol:**

```
Full Labeled Dataset
        │
        ├──── Training Set (70%) ─────→ Model is fitted here
        │
        ├──── Validation Set (15%) ───→ Hyperparameter tuning & early stopping
        │
        └──── Test Set (15%) ─────────→ Final unbiased performance estimate
                                         (Touch only ONCE, at the very end)
```

For time-series data (credit data is longitudinal), **Out-of-Time (OOT) validation** is mandatory. The test set must consist of applications from a *later time period* than the training set, mimicking real deployment conditions.

---

## 4.2 Confusion Matrix, Precision & Recall

At a given score threshold $\tau$, every borrower is classified as Predicted Good (score > $\tau$) or Predicted Bad (score ≤ $\tau$).

| | **Actual Good** | **Actual Bad** |
|---|---|---|
| **Predicted Good** | True Negative (TN) | False Negative (FN) |
| **Predicted Bad** | False Positive (FP) | True Positive (TP) |

**Key Metrics:**

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP} \quad \text{(Of those flagged as Bad, how many actually were?)}
$$

$$
\text{Recall (Sensitivity)} = \frac{TP}{TP + FN} \quad \text{(Of all actual Bads, how many did we catch?)}
$$

$$
\text{Specificity} = \frac{TN}{TN + FP} \quad \text{(Of all actual Goods, how many did we correctly approve?)}
$$

### Why Recall Is Often Prioritized in Credit

Credit risk is an **asymmetric cost problem**. The cost of a **False Negative** (approving a borrower who then defaults) is typically 5–20× larger than the cost of a **False Positive** (declining a borrower who would have performed). A defaulted £10,000 loan costs the bank the full principal plus collection expenses. A declined good customer costs only the foregone net interest income.

Therefore, lenders typically:
1. Set a **Recall target** first (e.g., "catch at least 70% of all future defaulters").
2. Then maximize **Precision** given that Recall constraint.

The **F-beta score** formalizes this trade-off:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \times \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

With $\beta > 1$ placing greater weight on Recall. In credit, $\beta = 2$ is common.

---

## 4.3 ROC Curve & AUROC

The **Receiver Operating Characteristic (ROC) Curve** plots the **True Positive Rate (Recall)** against the **False Positive Rate** ($1 - \text{Specificity}$) at every possible threshold $\tau$.

$$
\text{TPR}(\tau) = P(\text{score} \leq \tau | \text{Bad}) \quad \text{False Positive Rate}(\tau) = P(\text{score} \leq \tau | \text{Good})
$$

The **Area Under the ROC Curve (AUROC or AUC)** summarizes the model's discrimination across all thresholds:

$$
\text{AUROC} = \int_0^1 \text{TPR}(FPR) \, d(FPR)
$$

**Probabilistic Interpretation:** AUROC equals the probability that the model assigns a higher risk score to a randomly chosen Bad borrower than to a randomly chosen Good borrower.

$$
\text{AUROC} = P(\hat{P}_{\text{Bad}} > \hat{P}_{\text{Good}})
$$

**Industry Benchmarks for Retail Credit:**

| AUROC | Discrimination Power |
|---|---|
| 0.50 | No better than random |
| 0.60 – 0.70 | Weak |
| 0.70 – 0.80 | Acceptable |
| 0.80 – 0.90 | ✅ Good (typical for mature scorecards) |
| > 0.90 | Excellent — verify for data leakage |

---

## 4.4 Gini Coefficient

The **Gini Coefficient** in credit scoring is directly related to AUROC:

$$
\boxed{\text{Gini} = 2 \times \text{AUROC} - 1}
$$

It ranges from 0 (random) to 1 (perfect). The Gini is the ratio of the area between the ROC curve and the diagonal to the maximum possible such area. It is the most commonly quoted discriminatory metric in credit risk model documentation.

**Example:** An AUROC of 0.82 corresponds to a Gini of $2(0.82) - 1 = 0.64$ (64%).

---

## 4.5 KS Statistic (Kolmogorov-Smirnov)

The **KS Statistic** measures the maximum separation between the cumulative distribution functions (CDFs) of scores for Good and Bad borrowers.

$$
\text{KS} = \max_{\tau} \left| F_{\text{Bad}}(\tau) - F_{\text{Good}}(\tau) \right|
$$

Where $F_{\text{Bad}}(\tau) = P(\text{score} \leq \tau | \text{Bad})$ and $F_{\text{Good}}(\tau) = P(\text{score} \leq \tau | \text{Good})$.

**What KS Measures in Credit Context:**

At every possible score threshold, KS asks: "What percentage of Bads have I captured, minus what percentage of Goods have I captured?" The maximum of this difference is the KS statistic. A high KS means the score distributions of Good and Bad borrowers are widely separated — the model creates a clear "fault line" between the risk populations.

**Visualizing KS:**

```
CDF
1.0 |                              .....Good CDF (reaches 1.0 faster)
    |                         .....
    |                    .....
    |               .....  ↑
    |          .....       |  KS = max vertical gap
    |     .....            ↓
    | .....
    |_________________________ Bad CDF
0.0 |_____________________________
    Low Score                  High Score
```

**Industry Benchmarks:**

| KS Value | Interpretation |
|---|---|
| < 20% | Poor |
| 20% – 30% | Fair |
| 30% – 40% | Good |
| 40% – 50% | Very Good |
| > 50% | Excellent |

**Relationship to Logistic Regression:** The KS score corresponds to the optimal threshold from the logistic model (see [[Phase-3-Algorithmic-Engine#Logistic-Regression]]). The threshold at which KS is maximized is often used as the **operational cutoff** for approval/decline decisions.

```python
from sklearn.metrics import roc_curve
import numpy as np

def compute_ks(y_true, y_scores):
    """
    Computes KS Statistic for a binary classifier.
    y_true: array of actual labels (1=Bad, 0=Good)
    y_scores: array of predicted probabilities of being Bad
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    ks_stat = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks_stat, ks_threshold
```

---

# Phase 5: Production Monitoring & Stability

## 5.1 Why Models Degrade: The Concept of Drift

A model is a snapshot of the world at the time its training data was collected. The world does not stand still. **Population drift** occurs when the distribution of applicants — their incomes, employment patterns, credit behaviors — shifts over time due to:

- **Economic cycles:** Recessions dramatically increase default rates and change applicant profiles.
- **Policy changes:** New lending policy may attract a different demographic.
- **Product changes:** A new loan product changes the borrower mix.
- **Regulatory changes:** New data protection laws may alter what bureau data is available.

A model built on 2021 data deployed in 2024 will face applicants whose characteristics may bear little resemblance to the training population. If the model is not monitored, this degradation is **invisible until charge-offs spike**.

---

## 5.2 Population Stability Index (PSI)

The **PSI** measures the overall shift in the distribution of **model scores** (or any key variable) between the development population and the current live population.

### The PSI Formula

$$
\text{PSI} = \sum_{i=1}^{n} \left( \%\text{Actual}_i - \%\text{Expected}_i \right) \times \ln\left( \frac{\%\text{Actual}_i}{\%\text{Expected}_i} \right)
$$

Where:
- **Expected distribution** = score distribution on the development (training) dataset.
- **Actual distribution** = score distribution on the current live population.
- $n$ = number of bins (typically 10 deciles).

### PSI: Business Rule Thresholds

| PSI Value | Signal | Action |
|---|---|---|
| < 0.10 | ✅ Stable | No action required |
| 0.10 – 0.25 | ⚠️ Moderate Shift | Investigate; monitor more frequently |
| > 0.25 | 🚨 **Significant Shift** | **Trigger model review; consider redevelopment** |

### Computing PSI in Practice

```python
import numpy as np

def compute_psi(expected_scores: np.ndarray, actual_scores: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Computes Population Stability Index.
    expected_scores: model scores from development/baseline population
    actual_scores: model scores from current live population
    """
    # Define bin edges on the expected (development) distribution
    breakpoints = np.percentile(expected_scores,
                                np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_dist = np.histogram(expected_scores, bins=breakpoints)[0]
    actual_dist = np.histogram(actual_scores, bins=breakpoints)[0]

    # Convert to proportions; clip to avoid log(0)
    expected_pct = np.clip(expected_dist / len(expected_scores), 1e-6, None)
    actual_pct = np.clip(actual_dist / len(actual_scores), 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi
```

**Critical Implementation Note:** The bin breakpoints must be defined on the **development dataset** and then applied identically to all monitoring periods. Recomputing breakpoints on the live data would defeat the purpose.

---

## 5.3 Characteristic Stability Index (CSI)

While PSI monitors the **aggregate score distribution**, CSI monitors the distribution of **individual input features (characteristics)**. CSI isolates *which features* are drifting, enabling surgical diagnosis of model degradation.

$$
\text{CSI}_{j} = \sum_{i=1}^{n} \left( \%\text{Actual}_{j,i} - \%\text{Expected}_{j,i} \right) \times \ln\left( \frac{\%\text{Actual}_{j,i}}{\%\text{Expected}_{j,i}} \right)
$$

Where $j$ denotes the specific feature (characteristic) being monitored, and $i$ indexes the bins.

**CSI uses the same formula and the same thresholds as PSI**, but applied at the individual feature level:

- **CSI < 0.10:** Feature is stable.
- **0.10 ≤ CSI < 0.25:** Feature is shifting; investigate data pipeline.
- **CSI ≥ 0.25:** Feature has shifted significantly; assess impact on model.

### PSI vs. CSI: The Diagnostic Hierarchy

Think of PSI and CSI as a two-stage diagnostic:

```
Step 1: Monitor PSI (the overall score)
        │
        ├── PSI < 0.10 → ✅ No action
        │
        └── PSI > 0.10 → ⚠️ Investigate
                │
                Step 2: Compute CSI for ALL model features
                        │
                        ├── Feature X: CSI > 0.25 → 🚨 Data quality issue / real shift
                        │
                        └── All features CSI < 0.10 but PSI > 0.10 →
                            Score shifting despite stable inputs →
                            Check for label drift / outcome shift
```

### Monitoring Dashboard Structure

A production credit model should generate a monthly monitoring report containing:

1. **Score Distribution:** PSI vs. development benchmark.
2. **Default Rate by Decile:** Actual bad rate vs. expected bad rate by score band.
3. **Characteristic Report (CSI):** Top 10 features by CSI value.
4. **Performance Metrics Trend:** Monthly AUROC and KS (requires 12+ months of outcome data).
5. **Override Rate Report:** Percentage of model decisions manually overridden by analysts.

---

# Phase 6: The Deployment Lifecycle & Client Engineering

## 6.1 The Three Stages of Model Approval

A credit risk model is not deployed the moment it achieves a good AUROC. It must pass through a formal **Model Risk Management (MRM)** process before any production use. The three stages are universal across regulated financial institutions.

### Stage 1: Technical Validation (Quantitative Review)

**Who conducts it:** An independent Model Validation team (separate from the development team — this independence is a regulatory requirement under SR 11-7 and SS1/23).

**What is validated:**

- **Mathematical correctness:** Are the statistical techniques applied correctly? Is MLE implemented properly? Are the WoE bins monotonic?
- **Code review:** Is the implementation reproducible? Are there data leakage risks in the feature engineering pipeline?
- **Performance benchmarking:** Does the model meet minimum AUROC and KS thresholds on the OOT test set?
- **Stability testing:** Does performance degrade when small perturbations are applied to inputs (sensitivity analysis)?
- **Challenger model comparison:** How does the proposed model compare against the existing production model (the "champion")?

**Output:** A **Technical Validation Report** with a pass/fail/conditional recommendation.

### Stage 2: Qualitative & Business Review

**Who conducts it:** Business stakeholders — credit policy, risk strategy, compliance, and sometimes the regulator.

**What is reviewed:**

- **Business logic:** Do the model's directional relationships make intuitive sense? (Higher income → lower risk; more delinquencies → higher risk). A statistically valid model that violates business logic will fail this stage.
- **Regulatory compliance:**
  - Is any protected characteristic (race, gender, religion, national origin) — or a proxy for it — used as a feature? This triggers **Fair Lending / ECOA** review.
  - Can the model's decisions be explained at the individual level? (GDPR / CCPA "right to explanation")
- **Adverse Action Coding:** Every declined applicant is legally entitled to a reason. The model must produce **top reason codes** (e.g., "Too many recent delinquencies; High utilization ratio") — only possible with WoE-based scorecards or SHAP-augmented tree models.

**Output:** A **Model Approval Memo** signed by the Chief Risk Officer (or equivalent).

### Stage 3: Client Handover & Non-Technical Communication

**Who is involved:** The quant team, the client's IT/data engineering team, and senior business stakeholders.

**The Central Challenge:** Translating complex mathematics into business language without losing accuracy.

**Framework for Explaining Black-Box Math to Non-Technical Stakeholders:**

| Concept | Technical Reality | Client-Facing Explanation |
|---|---|---|
| Logistic Regression $\beta$ | Log-odds coefficient | "Each additional missed payment lowers your score by approximately X points" |
| AUROC = 0.82 | Area under ROC curve | "The model correctly ranks 82% of riskier borrowers above safer ones" |
| PSI = 0.18 | Distribution shift metric | "The profile of new applicants has shifted moderately from our training data — we recommend reviewing the model within 90 days" |
| WoE = +0.6 for a bin | Log-ratio of bad/good rates | "Applicants in this income range are 1.8× more likely to default than the average" |
| SHAP value = -12 pts | Shapley attribution | "This applicant's score was reduced by 12 points primarily due to their recent credit inquiry history" |

**The Champion-Challenger Framework:**

The deployment is never a hard cutover. New models are deployed in **shadow mode** first:

```
All live applications
        │
        ├──→ Champion Model (current production) → Drives actual decisions
        │
        └──→ Challenger Model (new model) → Scores silently; no business action taken

After 3–6 months of parallel running:
→ Compare challenger vs. champion performance on matched cohorts
→ If challenger statistically outperforms → promote to champion
→ Retire old champion; establish new challenger
```

---

## 6.2 Mapping Algorithms to Client Business Problems

The choice of algorithm is not a technical decision alone — it must map precisely to the **business problem the client is actually trying to solve**.

| Business Problem | Recommended Algorithm | Why |
|---|---|---|
| **"Should we approve this application?"** (Binary decision, need reason codes) | **Logistic Regression Scorecard** | Regulatory acceptance; reason codes; score decomposition; champion model for most lenders |
| **"Which segment of our portfolio is highest risk?"** (Ranking/prioritization) | **XGBoost or LightGBM** with calibration | Superior discrimination; SHAP for post-hoc explainability |
| **"Is this transaction fraudulent?"** (Rare event, imbalanced classes) | **Isolation Forest, One-Class SVM, or Autoencoder** (Anomaly Detection) | Fraud is an anomaly problem, not a standard classification problem; labeled fraud data is scarce |
| **"Who among delinquent customers should collections call first?"** (Ranking within subpopulation) | **Survival Analysis (Cox PH) or XGBoost ranking** | Time-to-event matters; behavioral data drives this |
| **"Find me customers similar to my best customers"** (Customer targeting) | **K-Nearest Neighbors (KNN) or Clustering (K-Means)** | Similarity search in feature space; no label required |
| **"Predict the next 12 months' default rate under stress"** (Macroeconomic scenario) | **Panel regression / econometric model with macro overlays** | Time-series properties; macro factor sensitivity required |

---

# Zettelkasten Architecture

## Splitting This Curriculum into 8–10 Interconnected Files

The following is the recommended structure for the `brain/` directory of the Quant OS knowledge graph. Each file is a self-contained conceptual note with formal YAML frontmatter and `[[wikilink]]` cross-references connecting it to the broader graph.

---

### File 1: `01-credit-risk-foundations.md`

```yaml
---
title: "Credit Risk Foundations"
tags:
  - credit-risk
  - fundamentals
  - PD
  - retail-banking
  - lifecycle
cluster_phase: Phase 1
links:
  - "[[02-data-architecture-preprocessing]]"
  - "[[06-deployment-lifecycle]]"
---
```

**Contents:** Definition of credit risk, PD/LGD/EAD, the end-to-end project lifecycle diagram, target variable definition (90 DPD), performance window, indeterminate exclusion.

---

### File 2: `02-data-architecture-preprocessing.md`

```yaml
---
title: "Data Architecture & Preprocessing for Credit"
tags:
  - data-engineering
  - missing-data
  - imputation
  - outliers
  - homoscedasticity
  - multicollinearity
  - VIF
cluster_phase: Phase 2
links:
  - "[[01-credit-risk-foundations]]"
  - "[[03-woe-iv-feature-engineering]]"
  - "[[04-logistic-regression-scorecard]]"
---
```

**Contents:** Feature taxonomy table, missingness mechanisms (MCAR/MAR/MNAR), imputation strategies, Winsorization formula, VIF formula and thresholds, multicollinearity remediation.

---

### File 3: `03-woe-iv-feature-engineering.md`

```yaml
---
title: "Weight of Evidence & Information Value"
tags:
  - WoE
  - IV
  - feature-engineering
  - feature-selection
  - binning
  - scorecard
cluster_phase: Phase 3
links:
  - "[[02-data-architecture-preprocessing]]"
  - "[[04-logistic-regression-scorecard]]"
  - "[[07-model-validation-metrics]]"
---
```

**Contents:** WoE formula derivation, IV formula, IV interpretation thresholds, data leakage warning for IV > 0.5, fine/coarse classing procedure, monotonicity requirement, Python implementation.

---

### File 4: `04-logistic-regression-scorecard.md`

```yaml
---
title: "Logistic Regression & Scorecard Scaling"
tags:
  - logistic-regression
  - MLE
  - log-odds
  - scorecard
  - PDO
  - beta-coefficients
  - regularization
cluster_phase: Phase 3
links:
  - "[[03-woe-iv-feature-engineering]]"
  - "[[05-tree-based-models]]"
  - "[[07-model-validation-metrics]]"
---
```

**Contents:** Sigmoid/logit derivation, MLE likelihood function, log-likelihood, Newton-Raphson, $\beta$ interpretation, PDO/base score/offset/factor derivation, full worked example, Lasso/Ridge/ElasticNet formulas and comparison table.

---

### File 5: `05-tree-based-models.md`

```yaml
---
title: "Tree-Based Models: Decision Trees to LightGBM"
tags:
  - decision-tree
  - random-forest
  - XGBoost
  - LightGBM
  - gradient-boosting
  - hyperparameter-tuning
  - SHAP
cluster_phase: Phase 3
links:
  - "[[04-logistic-regression-scorecard]]"
  - "[[07-model-validation-metrics]]"
  - "[[09-deployment-client-engineering]]"
---
```

**Contents:** Gini impurity, entropy, information gain, bagging vs. boosting, Random Forest averaging formula, XGBoost objective function, LightGBM GOSS/EFB, comparison table, Grid/Random/Bayesian optimization, Optuna example.

---

### File 6: `06-model-validation-metrics.md`

```yaml
---
title: "Model Validation & Performance Metrics"
tags:
  - validation
  - AUROC
  - KS-statistic
  - Gini
  - precision
  - recall
  - confusion-matrix
  - OOT
cluster_phase: Phase 4
links:
  - "[[04-logistic-regression-scorecard]]"
  - "[[05-tree-based-models]]"
  - "[[07-production-monitoring-PSI]]"
---
```

**Contents:** Train/validation/test split, OOT validation, confusion matrix, precision/recall/F-beta with asymmetric cost justification, ROC formula, AUROC probabilistic interpretation, AUROC benchmarks, Gini-AUROC relationship, KS formula, KS visualization, KS benchmarks, Python implementations.

---

### File 7: `07-production-monitoring-PSI.md`

```yaml
---
title: "Production Monitoring: PSI & CSI"
tags:
  - PSI
  - CSI
  - model-drift
  - population-stability
  - monitoring
  - model-risk
cluster_phase: Phase 5
links:
  - "[[06-model-validation-metrics]]"
  - "[[08-deployment-lifecycle]]"
  - "[[01-credit-risk-foundations]]"
---
```

**Contents:** Concept of drift, PSI formula, PSI thresholds (0.10/0.25), PSI Python implementation with bin-fixing caveat, CSI formula, PSI vs. CSI diagnostic hierarchy, monitoring dashboard components.

---

### File 8: `08-deployment-lifecycle.md`

```yaml
---
title: "Deployment Lifecycle & Model Risk Management"
tags:
  - deployment
  - model-risk-management
  - MRM
  - SR-11-7
  - champion-challenger
  - adverse-action
  - client-handover
cluster_phase: Phase 6
links:
  - "[[07-production-monitoring-PSI]]"
  - "[[04-logistic-regression-scorecard]]"
  - "[[09-algorithm-selection-guide]]"
---
```

**Contents:** Three-stage approval process (Technical → Business → Handover), SR 11-7 / SS1/23 regulatory references, Fair Lending / ECOA, adverse action reason codes, shadow deployment / champion-challenger framework, non-technical communication translation table.

---

### File 9: `09-algorithm-selection-guide.md`

```yaml
---
title: "Algorithm Selection: Matching Models to Business Problems"
tags:
  - algorithm-selection
  - logistic-regression
  - XGBoost
  - anomaly-detection
  - KNN
  - clustering
  - survival-analysis
  - fraud-detection
cluster_phase: Phase 6
links:
  - "[[04-logistic-regression-scorecard]]"
  - "[[05-tree-based-models]]"
  - "[[08-deployment-lifecycle]]"
---
```

**Contents:** Full algorithm-to-use-case mapping table, fraud (Isolation Forest/Autoencoder), collections (survival analysis), customer targeting (KNN/K-Means), stress testing (econometric panel), origination (Logistic Scorecard), portfolio ranking (XGBoost).

---

### File 10: `00-quant-os-credit-risk-index.md` *(Master Index)*

```yaml
---
title: "Quant OS: Credit Risk Curriculum Index"
tags:
  - index
  - MOC
  - credit-risk
  - quant-os
cluster_phase: Index
links:
  - "[[01-credit-risk-foundations]]"
  - "[[02-data-architecture-preprocessing]]"
  - "[[03-woe-iv-feature-engineering]]"
  - "[[04-logistic-regression-scorecard]]"
  - "[[05-tree-based-models]]"
  - "[[06-model-validation-metrics]]"
  - "[[07-production-monitoring-PSI]]"
  - "[[08-deployment-lifecycle]]"
  - "[[09-algorithm-selection-guide]]"
---
```

**Contents:** This file is the **Map of Content (MOC)** — the navigation hub. It contains the Phase structure, a clickable index of all 9 files, the end-to-end workflow diagram, and a visual graph showing the interconnections between notes.

---

## Zettelkasten Graph Architecture

```
[00-index] ──────────────────────────────────────────────────
    │                                                         │
    ▼                                                         ▼
[01-foundations] → [02-data-arch] → [03-woe-iv] → [04-logistic-scorecard]
                                                         │
                                                    [05-tree-models]
                                                         │
                                                    [06-validation]
                                                         │
                                                    [07-PSI-monitoring]
                                                         │
                                                    [08-deployment]
                                                         │
                                                    [09-algorithm-selection]
```

Cross-cluster connections (non-linear links that make this a true graph, not a chain):

- `[[03-woe-iv]]` ↔ `[[06-validation]]` (IV as feature selection; KS as validation metric share conceptual DNA)
- `[[04-logistic-scorecard]]` ↔ `[[07-PSI-monitoring]]` (scorecard score distribution → PSI monitoring)
- `[[02-data-arch]]` ↔ `[[07-PSI-monitoring]]` (CSI monitors the same features preprocessed in Phase 2)
- `[[05-tree-models]]` ↔ `[[09-algorithm-selection]]` (tree models map to specific business problems)

---

*End of Quant OS Flagship Curriculum — Credit Risk Edition*

*© Quant OS Open-Source Knowledge Graph. Distributed under the Quant OS Open Publication License.*
