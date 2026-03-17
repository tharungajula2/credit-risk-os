---
title: "LGD and EAD Modeling — Recovery, Exposure, and Loss Severity"
date: 2026-03-19
tags:
  - LGD
  - EAD
  - recovery-rate
  - CCF
  - hurdle-model
  - logistic-regression
  - linear-regression
  - beta-regression
  - random-forest
  - xgboost
  - SHAP
  - downturn-LGD
  - credit-risk
  - lending-club
  - model-validation
  - SR-11-7
cluster: Phase 4 — Loss & Exposure Modeling
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_full_pd_model]]"
  - "[[2_monitoring_model]]"
---

---

# LGD and EAD Modeling — Recovery, Exposure, and Loss Severity

> This note is my full technical record of how I think about **Loss Given Default (LGD)** and **Exposure at Default (EAD)** using the Lending Club project as the anchor example. I use the project to explain the concepts from first principles, the minimum math I should know for interviews, the Python logic behind the models, the practical limitations of the dataset, and how these ideas extend to real banking environments.
>
> This note sits after the PD scorecard build in [[1_full_pd_model]] and before full loss aggregation. PD tells me **whether** a borrower is likely to default. LGD and EAD tell me **how much I lose if default happens**.

---

## The Project at a Glance

**Dataset:** Lending Club retail loan data

**Scope of this note:** Model the two remaining credit-risk parameters after PD:

- **LGD** — the fraction of exposure I lose after default
- **EAD** — the exposure outstanding at the time of default

**Why this matters:** A bank does not lose money merely because a borrower has a high PD. It loses money when all three pieces combine:

```text
Expected Loss (EL) = PD × LGD × EAD
```

That is the full economic story:

- **PD** answers: how likely is default?
- **LGD** answers: if default happens, what fraction is not recovered?
- **EAD** answers: how much money is actually outstanding when default happens?

**What I modeled in this project:**

```text
Defaulted loans only
        │
        ▼
Create recovery_rate = recoveries / funded_amnt
        │
        ├── Stage 1: Predict whether recovery > 0
        │
        ├── Stage 2: Predict recovery rate given recovery > 0
        │
        ▼
Combine both stages to estimate expected recovery
        │
        ▼
Convert to LGD = 1 - expected recovery
        │
        ▼
Create CCF / EAD target from principal still outstanding at default
        │
        ▼
Model exposure severity and discuss better alternatives
```

**Important dataset caveat:** Lending Club is a strong learning dataset, but it is mostly an **unsecured installment-loan** environment. That means the recovery dynamics here are not the same as mortgage, auto, or secured wholesale books. I can learn the mechanics very well from this project, but I should not pretend the absolute recovery levels transfer directly to collateralized bank portfolios.

---

## Part 1: Why PD Alone Is Not Enough

### The Concept

A borrower with a high default probability is not automatically the worst economic risk.

Two borrowers can have the same PD and still create very different losses:

- one may default on a small outstanding balance and recover part of it
- another may default on a much larger balance and recover almost nothing

That is why a complete credit-risk framework separates:

1. **default frequency** → PD
2. **loss severity** → LGD
3. **exposure size at default** → EAD

### A Simple Numerical Example

Suppose two borrowers each have a PD of 10%.

- Borrower A: `LGD = 20%`, `EAD = $5,000`
- Borrower B: `LGD = 80%`, `EAD = $20,000`

Then:

```text
EL_A = 0.10 × 0.20 × 5,000 = $100
EL_B = 0.10 × 0.80 × 20,000 = $1,600
```

Same PD. Very different loss.

That is the core reason this note matters.

### Practical Banking Interpretation

In origination, the PD model is often the visible front-end model because it drives scorecards and approval cut-offs. But from a portfolio, reserving, capital, collections, and stress-testing perspective, LGD and EAD are equally important. They convert risk ranking into **money**.

---

## Part 2: Loss Given Default (LGD)

### The Definition

**LGD** is the fraction of the exposure that remains lost after all recoveries are collected.

The cleanest way to define it is through the recovery rate:

```text
Recovery Rate = Recoveries / Exposure
LGD = 1 - Recovery Rate
```

In this project, the working denominator was the original funded amount:

```text
recovery_rate = recoveries / funded_amnt
LGD = 1 - recovery_rate
```

### Why LGD Is Modeled Only on Defaulted Loans

LGD is **conditional on default**. If a loan never defaulted, LGD is not observed. That means the LGD training sample should come only from loans that have already defaulted or charged off.

This is a very important conceptual distinction:

- PD is modeled on the full origination population
- LGD is modeled on the defaulted sub-population
- EAD is also modeled conditional on default

If I mix performing loans into LGD training, I am no longer modeling loss severity after default. I am mixing frequency and severity together.

### The Practical Challenge

Recovery behavior is messy:

- many defaulted loans recover **nothing at all**
- some recover a small amount
- a small minority recover a meaningful amount

So the target distribution is not a clean bell curve. It is usually:

- a large spike at **zero**
- then a continuous spread between **0 and 1** for positive recoveries

That is exactly why a single plain linear regression is usually not the best first model.

### The Code Logic

```python
loan_data_defaults = loan_data[loan_data['good_bad'] == 0].copy()

loan_data_defaults['recovery_rate'] = (
    loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
)

# Safety cap for data errors or small numerical issues
loan_data_defaults['recovery_rate'] = loan_data_defaults['recovery_rate'].clip(0, 1)

loan_data_defaults['LGD'] = 1 - loan_data_defaults['recovery_rate']
```

### What I Should Understand Intuitively

A high LGD means recovery is weak. In unsecured retail credit, that is common because there may be little or no collateral to liquidate. In secured lending, recovery depends much more on collateral value, legal process, workout cost, time to resolution, and the economic cycle.

---

## Part 3: The Two-Stage LGD Model (Hurdle Logic)

### Why a Two-Stage Model Is Needed

A one-stage regression struggles because a very large number of loans have `recovery_rate = 0` exactly.

That means the target is not just continuous. It is **semi-continuous**:

- first there is a binary event: **any recovery or no recovery?**
- only after that comes the continuous question: **if recovery happens, how large is it?**

So I split the problem into two models.

### Stage 1 — Probability of Any Recovery

First I define a binary target:

```python
loan_data_defaults['recovery_rate_0_1'] = np.where(
    loan_data_defaults['recovery_rate'] > 0,
    1,
    0
)
```

This stage is a **classification** problem.

The standard model is logistic regression:

```text
log(p / (1 - p)) = β0 + β1x1 + β2x2 + ... + βkxk
```

where:

- `p` = probability that recovery is positive
- `x1 ... xk` = borrower or loan features
- `β` coefficients = direction and strength of each feature's effect

The logistic function maps the linear score into a valid probability:

```text
p = 1 / (1 + e^(-z))
```

where `z = β0 + β1x1 + ... + βkxk`.

### Stage 2 — Recovery Size Given Positive Recovery

Then I restrict the sample to loans where recovery is actually positive and fit a continuous model:

```python
loan_data_defaults_positive = loan_data_defaults[
    loan_data_defaults['recovery_rate'] > 0
].copy()
```

A simple first pass is linear regression:

```text
ŷ = β0 + β1x1 + β2x2 + ... + βkxk
```

Here `ŷ` is the predicted recovery rate for loans that already cleared the first hurdle.

### Combining Both Stages

The final expected recovery is the product of:

```text
Expected Recovery
= P(recovery > 0 | x) × E(recovery_rate | recovery > 0, x)
```

If I call the stage-1 output `p_i` and the stage-2 output `r_i`, then:

```text
expected_recovery_i = p_i × r_i
expected_LGD_i      = 1 - expected_recovery_i
```

### The Code Pattern

```python
# Stage 1
lgd_stage_1_pred = lgd_stage_1_model.predict_proba(X_test)[:, 1]

# Stage 2
lgd_stage_2_pred = lgd_stage_2_model.predict(X_test)

# Combine
expected_recovery = lgd_stage_1_pred * lgd_stage_2_pred
expected_recovery = np.clip(expected_recovery, 0, 1)

expected_lgd = 1 - expected_recovery
```

### Why This Structure Makes Sense

This is a cleaner mental model than forcing one regression to solve both problems simultaneously.

- Stage 1 learns the **incidence of recovery**
- Stage 2 learns the **size of recovery**

That separation is conceptually strong and easy to defend.

---

## Part 4: Exposure at Default (EAD)

### The Definition

**EAD** is the exposure outstanding when default happens.

In very simple terms, it answers:

> How much money is still on the line at the moment the borrower defaults?

### Why EAD Is Easy for Some Products and Hard for Others

EAD depends heavily on product type.

#### Closed-End Installment Loans

For a term loan or personal loan, the exposure path is relatively structured. The outstanding balance is driven by the amortization schedule, prepayments, missed payments, and charge-off timing.

That is the case closest to the Lending Club project.

#### Revolving Products

For a credit card or revolving line, EAD is harder because the borrower can draw more money **before** default. That is where the **Credit Conversion Factor (CCF)** becomes important.

The generic revolving-credit relationship is:

```text
EAD = Drawn Amount + (CCF × Undrawn Limit)
```

If the borrower is under stress, the unused line may get drawn down before default. That behavior is one of the hardest parts of EAD modeling.

### What Was Done in This Project

In the project, the target was built as a CCF-style ratio using principal not yet repaid:

```python
loan_data_defaults['CCF'] = (
    (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp'])
    / loan_data_defaults['funded_amnt']
).clip(0, 1)
```

This is best understood as:

- the fraction of original funded amount still economically exposed at default
- a practical EAD proxy for an installment-loan setting

If I want the dollar EAD from that ratio, I can multiply back:

```text
EAD = CCF × funded_amnt
```

### What I Should Say Carefully

For the Lending Club project, I should be precise:

- this is **not** the full revolving-credit CCF problem
- it is a simpler installment-loan exposure problem
- the concept still teaches me the core EAD logic very well

That distinction matters in interviews because it shows I know the difference between:

- **remaining balance at default** on a term loan
- **future drawdown risk before default** on a revolving facility

---

## Part 5: Why Plain Linear Regression Struggles Here

### The First Problem: The Target Is Bounded

Recovery rates, LGD, and CCFs are percentages. They live in a fixed interval:

```text
0 ≤ target ≤ 1
```

Ordinary Least Squares (OLS) does not know that. It assumes the response can extend across the whole real line.

So OLS can predict nonsense values such as:

- `-0.12`
- `1.18`

Those are impossible for recovery rate, LGD, or CCF.

### The Second Problem: Zero Inflation

The LGD setup contains many exact zeros in recovery. That produces a distribution that is poorly matched to OLS assumptions.

### The Third Problem: Error Variance Is Not Stable

In bounded targets, the prediction error often behaves differently near the edges than in the middle. That means the variance of errors is usually not constant. OLS standard errors become less trustworthy.

### The Operational Fix Used in Practice

A common quick fix is manual clipping:

```python
y_hat = model.predict(X_test)
y_hat = np.where(y_hat < 0, 0, y_hat)
y_hat = np.where(y_hat > 1, 1, y_hat)
```

This is acceptable for an educational prototype, but it is not the cleanest modeling choice because the model is still producing impossible values internally.

### Better Alternatives

#### 1. Fractional Logit

Useful when the target is a proportion between 0 and 1. The link function respects the boundary better than OLS.

#### 2. Beta Regression

Designed for variables in `(0, 1)`. This is elegant for recovery rates or CCFs, but exact zeros and ones usually need separate handling or transformation. That is one reason the two-stage LGD setup remains attractive.

#### 3. Tree-Based Models

Decision trees, random forests, gradient boosting, XGBoost, and LightGBM can handle nonlinear relationships much better than a single straight line. They are often strong challengers for LGD and EAD.

---

## Part 6: Regularization and Machine-Learning Upgrades

### Why Regularization Comes First

Before jumping to more complex algorithms, I should know how to stabilize a linear model.

#### Ridge (L2)

Ridge shrinks coefficients toward zero but usually does not force them exactly to zero.

Use case:

- many correlated predictors
- I want stability more than feature elimination

#### Lasso (L1)

Lasso can shrink some coefficients all the way to zero.

Use case:

- I want automatic variable selection
- I need a sparser model

#### Elastic Net

Elastic Net blends L1 and L2.

Use case:

- I want both stability and some feature selection
- correlated variables exist, but I do not want Lasso behaving too aggressively

### Tree-Based Alternatives

#### Decision Tree

Easy to understand, but a single tree is unstable and can overfit quickly.

#### Random Forest

Builds many trees on resampled data and averages them. Usually more stable than one tree and often a strong baseline for nonlinear severity modeling.

#### XGBoost / LightGBM

Boosting models build trees sequentially so later trees learn from earlier errors. These methods are often extremely strong on tabular credit data.

### What I Gain and What I Lose

| Model Family | Main Strength | Main Weakness |
|---|---|---|
| Linear / Regularized Linear | Transparent, easy to explain | Misses nonlinear effects, boundary problem remains |
| Random Forest | Strong nonlinear fit, robust baseline | Harder to explain, less clean for extrapolation |
| XGBoost / LightGBM | Often highest predictive power | Governance burden is higher |

### Explainability for Stronger Models

If I use a tree-based challenger, I still need to explain feature contribution.

The most important explainability tool to know here is **SHAP**:

- it decomposes a prediction into feature-level contributions
- it helps me understand why the model predicted higher LGD or EAD for a particular loan
- it is useful for internal challenge, sensitivity review, and documentation

I should still remember that explainability tooling does not remove governance obligations. It only makes the model more inspectable.

---

## Part 7: Validation of LGD and EAD Models

### LGD Stage 1 Validation (Binary Recovery / No Recovery)

Because Stage 1 is a classification model, I validate it like a classifier:

- confusion matrix
- precision / recall if needed
- AUROC
- KS, if I want a rank-ordering view

### LGD Stage 2 Validation (Continuous Recovery Rate)

Because Stage 2 predicts a continuous target, I validate it like a regression model:

#### Mean Absolute Error (MAE)

```text
MAE = average of |actual - predicted|
```

Interpretation: on average, how far off am I in absolute percentage terms?

#### Root Mean Squared Error (RMSE)

```text
RMSE = sqrt(average of (actual - predicted)^2)
```

Interpretation: similar to MAE, but large errors are penalized more heavily.

#### R²

Useful as a summary of explained variation, but not sufficient by itself. A low R² is common in recovery modeling because recoveries are noisy and operationally driven.

### EAD / CCF Validation

For EAD or CCF regression, I usually focus on:

- MAE
- RMSE
- segment-level bias analysis
- out-of-time stability

### The Validation Questions I Should Always Ask

1. Are errors larger for specific subgroups such as loan purpose, grade, or term?
2. Does the model systematically underpredict losses in stressed periods?
3. Does performance degrade materially on newer vintages?
4. Are the predictions directionally sensible when key inputs worsen?

### What Good Validation Looks Like in Practice

A technically good LGD or EAD model is not just one with low error. It should also show:

- stable behavior across time
- sensible segmentation performance
- no obvious bias toward underestimating severe losses
- defensible preprocessing and feature logic

---

## Part 8: Downturn LGD, Stress, and Practical Banking Use

### The Core Idea

Average recoveries in good times can be misleading. When the economy weakens:

- default volumes tend to rise
- collateral values can fall
- workout timelines can lengthen
- realized recoveries may deteriorate

That is why **downturn sensitivity** matters in LGD.

### The Minimum Interview Understanding I Need

I do not need to overstate the regulation. What I do need to understand clearly is:

- downturn LGD is a core prudential concept in Basel/IRB-style thinking
- recovery assumptions should not be blindly anchored to benign historical averages when stress conditions are relevant
- stress testing, capital, and accounting can use related but not identical calibrations

### Why This Matters in This Project

The Lending Club notebook is a useful learning build, but it is not a full downturn LGD framework by itself. A production-grade extension would require:

- macro segmentation or macro overlays
- stress-period benchmarking
- conservative treatment where data is thin
- clear separation between development-period average loss and stressed loss assumptions

---

## Part 9: What I Should Be Able to Explain Clearly

These are the points I should be able to explain without hesitation.

### 1. The difference between PD, LGD, and EAD

- PD = chance of default
- LGD = fraction lost if default occurs
- EAD = amount exposed when default occurs

### 2. Why the LGD model uses only defaulted loans

Because LGD is conditional on default. Non-defaulted loans do not carry an observed loss severity outcome.

### 3. Why the two-stage hurdle setup is sensible

Because many loans have zero recovery, and I need to separate:

- probability of any recovery
- size of recovery given recovery exists

### 4. Why EAD is easier in installment loans than revolving products

Installment exposure is mostly balance mechanics. Revolving exposure includes pre-default drawdown behavior.

### 5. Why OLS is not fully satisfactory for bounded targets

Because recovery rate, LGD, and CCF must stay between 0 and 1, while OLS can predict values outside that range.

### 6. Why tree-based models become attractive

Because recovery and exposure behavior are nonlinear and interaction-heavy. Trees often capture these patterns better than a single linear equation.

### 7. Why explainability still matters even for internal models

Because strong predictive power is not enough. I still need conceptual soundness, validation evidence, documentation, and challenge.

---

## Part 10: A Clean Python Skeleton

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1) Restrict to defaulted loans for LGD / EAD work
# --------------------------------------------------
defaults = loan_data[loan_data['good_bad'] == 0].copy()

# --------------------------------------------------
# 2) Targets
# --------------------------------------------------
defaults['recovery_rate'] = (
    defaults['recoveries'] / defaults['funded_amnt']
).clip(0, 1)

defaults['recovery_rate_0_1'] = (defaults['recovery_rate'] > 0).astype(int)

defaults['CCF'] = (
    (defaults['funded_amnt'] - defaults['total_rec_prncp'])
    / defaults['funded_amnt']
).clip(0, 1)

# --------------------------------------------------
# 3) Feature matrix
# --------------------------------------------------
feature_cols = [
    'grade', 'term_int', 'annual_inc', 'dti', 'emp_length_int'
]

X = defaults[feature_cols]
y_stage_1 = defaults['recovery_rate_0_1']

# --------------------------------------------------
# 4) LGD Stage 1: any recovery?
# --------------------------------------------------
lgd_stage_1 = LogisticRegression(max_iter=1000)
lgd_stage_1.fit(X, y_stage_1)

p_positive_recovery = lgd_stage_1.predict_proba(X)[:, 1]
print('Stage 1 AUROC:', roc_auc_score(y_stage_1, p_positive_recovery))

# --------------------------------------------------
# 5) LGD Stage 2: recovery size given positive recovery
# --------------------------------------------------
pos = defaults['recovery_rate'] > 0
X_pos = defaults.loc[pos, feature_cols]
y_pos = defaults.loc[pos, 'recovery_rate']

lgd_stage_2 = LinearRegression()
lgd_stage_2.fit(X_pos, y_pos)

pred_positive_size = lgd_stage_2.predict(X)
expected_recovery = np.clip(p_positive_recovery * pred_positive_size, 0, 1)
expected_lgd = 1 - expected_recovery

# --------------------------------------------------
# 6) EAD / CCF model (simple baseline)
# --------------------------------------------------
ead_model = LinearRegression()
ead_model.fit(X, defaults['CCF'])

pred_ccf = np.clip(ead_model.predict(X), 0, 1)

print('EAD MAE :', mean_absolute_error(defaults['CCF'], pred_ccf))
print('EAD RMSE:', mean_squared_error(defaults['CCF'], pred_ccf) ** 0.5)
```

This skeleton is intentionally simple. Its job is to keep the flow clear:

- define the right conditional sample
- create the right severity targets
- separate the recovery-incidence problem from the recovery-size problem
- validate classification and regression pieces differently

---

## Common Mistakes I Want to Avoid

1. **Using all loans for LGD training** instead of only defaulted loans.
2. **Confusing installment-loan EAD with revolving-line EAD**.
3. **Treating the Lending Club recovery levels as universally transferable** to secured portfolios.
4. **Reporting only one error metric** and calling validation complete.
5. **Assuming linear regression is conceptually correct** just because it runs.
6. **Using average-period LGD without thinking about stressed conditions**.
7. **Talking about machine learning accuracy without discussing governance and explanation**.

---

## Connections to the Rest of the Notes

- [[1_full_pd_model]] — This note completes the parameter set that begins with PD. The PD model estimates default likelihood; this note estimates loss severity and exposure conditional on default.
- [[2_monitoring_model]] — Once LGD and EAD are in production, their distributions and errors also need monitoring. The same governance logic used for PD monitoring extends to recovery and exposure models.
- [[Tharun-Kumar-Gajula]] — This note supports my broader quantitative credit-risk system by connecting banking practice, model development, validation, and explainability.

---

## Key Concepts Summary

| Concept | What It Is | Where It Appears in This Note |
|---|---|---|
| **Expected Loss (EL)** | Portfolio loss expectation from combining frequency and severity | `PD × LGD × EAD` |
| **Recovery Rate** | Fraction of exposure recovered after default | `recoveries / funded_amnt` |
| **LGD** | Fraction of exposure not recovered | `1 - recovery_rate` |
| **EAD** | Exposure outstanding at default | Dollar exposure or ratio-scaled proxy |
| **CCF** | Conversion factor from available or original exposure to default exposure | Revolving formula and installment proxy |
| **Two-Stage / Hurdle Model** | Separate model for positive recovery event and recovery size | Stage 1 logistic + Stage 2 regression |
| **Zero Inflation** | Large mass of exact zero recoveries | Reason one-stage OLS struggles |
| **Bounded Target Problem** | LGD / recovery / CCF must remain between 0 and 1 | Main weakness of plain OLS |
| **Fractional Logit / Beta Regression** | Better-behaved alternatives for proportion targets | Conceptual upgrade path |
| **Regularization** | Stabilizes linear models under multicollinearity | Ridge, Lasso, Elastic Net |
| **Tree-Based Models** | Nonlinear algorithms for complex severity patterns | Random Forest, XGBoost, LightGBM |
| **SHAP** | Feature-level explainability for complex models | Internal challenge and documentation |
| **Downturn LGD** | Recovery assumptions should reflect stress where relevant | Capital / stress sensitivity concept |
| **Out-of-Time Validation** | Test performance on later periods, not just random splits | Important for severity models too |

---

*This note is Version 1.0. The next natural extension is to combine the PD model from [[1_full_pd_model]] with the severity logic here into a full expected-loss and CECL-style forecasting note.*
