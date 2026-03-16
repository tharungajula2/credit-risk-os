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

**Relationship to Logistic Regression:** The KS score corresponds to the optimal threshold from the logistic model (see [[04-logistic-regression-scorecard#Logistic-Regression]]). The threshold at which KS is maximized is often used as the **operational cutoff** for approval/decline decisions.

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
