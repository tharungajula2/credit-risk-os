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
- **Limitation:** Among a group of correlated features, Lasso arbitrarily selects one and zeroes the rest. See [[02-data-architecture-preprocessing#Multicollinearity]].

### Ridge Regression (L2 Regularization)

$$
\ell_{\text{Ridge}}(\boldsymbol{\beta}) = \ell(\boldsymbol{\beta}) - \lambda \sum_{j=1}^{k} \beta_j^2
$$

- **Effect:** Shrinks all coefficients uniformly toward zero but **never to exactly zero**. All features are retained.
- **Use Case:** When you have many correlated features (e.g., multiple utilization metrics). Ridge distributes the predictive load across the correlated group.
- **Solves:** Multicollinearity (see [[02-data-architecture-preprocessing#Multicollinearity]]).

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
