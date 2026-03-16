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
