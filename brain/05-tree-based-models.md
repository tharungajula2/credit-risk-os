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
