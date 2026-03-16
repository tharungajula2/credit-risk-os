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
