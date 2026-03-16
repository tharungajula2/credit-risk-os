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

In the context of **Logistic Regression** for credit scoring (see [[03-woe-iv-feature-engineering]]), the model does not produce continuous residuals in the OLS sense, so classical homoscedasticity tests (Breusch-Pagan, White's test) are not directly applicable. However, the concept matters because:

- **Heteroscedastic inputs** (e.g., income that fans out dramatically at higher levels) can create unstable coefficient estimates.
- **WoE transformation** (see [[03-woe-iv-feature-engineering#Weight-of-Evidence]]) implicitly addresses this by binning continuous variables, removing the raw scale.

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

1. **Remove one of the correlated pair** — prefer the variable with higher [[06-model-validation-metrics#Information-Value|Information Value]].
2. **Principal Component Analysis (PCA)** — transforms correlated features into orthogonal components. Loses interpretability.
3. **Regularization** (Ridge/Lasso) — see [[04-logistic-regression-scorecard#Regularization]]. Ridge regression is specifically designed to handle multicollinearity.

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
