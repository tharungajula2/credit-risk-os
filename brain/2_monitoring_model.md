---
title: "Production Pipeline Integrity & Model Monitoring — PSI, CSI, Drift"
date: 2026-03-19
tags:
  - PSI
  - CSI
  - model-monitoring
  - population-drift
  - concept-drift
  - pipeline-integrity
  - data-lineage
  - BCBS-239
  - SR-11-7
  - recalibration
  - post-model-overlay
  - credit-risk
  - lending-club
  - sklearn-pipeline
  - serialization

links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_full_pd_model]]"
---

---

# Production Pipeline Integrity & Model Monitoring — PSI, CSI, Drift

> This note covers everything that happens *after* the PD model from [[1_full_pd_model]] is deployed. Building the model is the first half of the job. Keeping it honest over time is the second half — and in regulated banking, it is arguably harder and more important.
>
> The reference for this note is the monitoring notebook where I applied the trained Lending Club PD model to a 2015 dataset (`loan_data_2015.csv`) and built a full stability monitoring engine. Every section covers what was done in that notebook, why it matters, and what the correct production-grade approach looks like.

---

## The Central Problem: A Model Is a Frozen Snapshot of the Past

When the Lending Club PD model was trained, it learned the relationship between borrower characteristics and default *as they existed in the 2007–2014 dataset*. The moment that model is deployed, two things begin to happen simultaneously:

1. **The population applying for loans may change.** New geographies, different economic conditions, changed marketing campaigns — the types of people walking in the door are no longer identical to the training population.

2. **The fundamental relationships may change.** After a major recession, the relationship between income and default risk might shift. What used to be a "safe" income level may no longer protect against default in a high-inflation environment.

The first problem is called **population drift**. The second is **concept drift**. They require completely different responses, and most practitioners conflate them.

This note builds the tools to detect both, diagnose which type is occurring, and decide whether to recalibrate or rebuild.

---

## Part 1: Pipeline Integrity — The Preprocessing Re-Run

### What Was Done in the Monitoring Notebook

The monitoring notebook imports `loan_data_2015.csv` — new loans from 2015 — and needs to score them using the frozen 2007–2014 PD model.

Before scoring, the 2015 data must pass through the exact same preprocessing pipeline the training data went through:

- String parsing for `emp_length` and `term`
- Date arithmetic to compute `mths_since_earliest_cr_line` and `mths_since_issue_d`
- Missing value handling for `annual_inc`, `total_rev_hi_lim`, `mths_since_last_delinq`, and others
- WoE bin assignment using the *exact thresholds* from the training-time coarse classing
- Dummy variable creation and reference category removal

In the offline notebook, the approach was to copy-paste the entire preprocessing code block from the development notebook into the monitoring notebook, with one manual change: the reference date was updated from `2017-12-01` to `2018-12-01`.

This gets the right answer on a static dataset. In production, it is an operational risk.

### Why Copy-Pasting Code Is a Governance Problem

Under **BCBS 239** (*Principles for Effective Risk Data Aggregation and Risk Reporting*, Basel Committee on Banking Supervision), banks are required to ensure that risk data is accurate, complete, and produced through reliable, controlled processes.

A copy-pasted notebook cell is none of these things:

- A developer might update the imputation logic in the development notebook and forget to update the monitoring notebook. The two pipelines silently diverge.
- The reference date change was done manually. A human making this change a year later might forget, producing systematically wrong credit history lengths for every applicant.
- There is no audit trail proving that the 2015 preprocessing was identical to the 2007 preprocessing.

### The Production Standard: Serialized Sklearn Pipelines

The correct approach is to encapsulate the entire preprocessing sequence into a **sklearn `Pipeline` object**, fit it once on the training data, and serialise (save) it alongside the model. All future data — 2015 data, 2020 data, live applicants — is transformed by calling the same frozen pipeline object.

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# Step 1: Build a custom Transformer that encapsulates all
#         preprocessing logic for the Lending Club dataset.
#         This class is fitted ONCE on training data and then
#         frozen. It stores all the parameters needed to
#         transform any future dataset identically.
# ---------------------------------------------------------------

class LendingClubPreprocessor(BaseEstimator, TransformerMixin):
    """
    Encapsulates all Lending Club preprocessing steps.
    
    On fit():  learns all parameters from training data
               (means for imputation, WoE bin thresholds, etc.)
    On transform(): applies those frozen parameters to any new data.
    """
    
    def fit(self, X, y=None):
        # Freeze the training mean for annual_inc imputation
        # (though 'Missing as Category' is preferred — see Note 1)
        self.annual_inc_train_mean_ = X['annual_inc'].mean()
        
        # Freeze the training mean for total_rev_hi_lim
        self.rev_hi_lim_fallback_ = X['funded_amnt'].mean()
        
        # Freeze the WoE bin boundaries from coarse classing
        # These were determined during training — see Note 1 Part 5
        self.income_bins_ = [0, 20000, 30000, 40000, 60000, 80000,
                             100000, 120000, np.inf]
        self.dti_bins_ = [0, 10, 15, 20, 25, 30, 35, np.inf]
        self.inq_bins_ = [0, 1, 2, 3, 4, np.inf]
        # ... (all other coarse-classed bin boundaries)
        
        return self  # fit() always returns self
    
    def transform(self, X):
        X = X.copy()
        
        # --- String cleaning ---
        X['emp_length'] = X['emp_length'].replace('10+ years', '10')
        X['emp_length'] = X['emp_length'].replace('< 1 year', '0')
        X['emp_length'] = pd.to_numeric(
            X['emp_length'].str.replace(' years', '').str.replace(' year', ''),
            errors='coerce'
        )
        X['term'] = pd.to_numeric(
            X['term'].str.replace(' months', ''), errors='coerce'
        )
        
        # --- Dynamic date arithmetic ---
        # Reference date is DYNAMIC — uses system time, not hardcoded
        reference_date = pd.Timestamp.now().floor('D')
        
        X['earliest_cr_line'] = pd.to_datetime(
            X['earliest_cr_line'], format='%b-%Y', errors='coerce'
        )
        X['mths_since_earliest_cr_line'] = round(
            pd.to_numeric(
                (reference_date - X['earliest_cr_line']) / np.timedelta64(1, 'M')
            )
        )
        
        X['issue_d'] = pd.to_datetime(X['issue_d'], format='%b-%Y', errors='coerce')
        X['mths_since_issue_d'] = round(
            pd.to_numeric(
                (reference_date - X['issue_d']) / np.timedelta64(1, 'M')
            )
        )
        
        # --- Missing value handling ---
        # annual_inc: impute with TRAINING mean (not live mean — see below)
        X['annual_inc'] = X['annual_inc'].fillna(self.annual_inc_train_mean_)
        
        # total_rev_hi_lim: use funded_amnt as proxy
        X['total_rev_hi_lim'] = X['total_rev_hi_lim'].fillna(X['funded_amnt'])
        
        # mths_since_last_delinq: preserve the MNAR signal
        X['mths_since_last_delinq_missing'] = X['mths_since_last_delinq'].isnull().astype(int)
        X['mths_since_last_delinq'] = X['mths_since_last_delinq'].fillna(0)
        
        # --- WoE bin assignment using FROZEN training boundaries ---
        X['annual_inc_bin'] = pd.cut(
            X['annual_inc'],
            bins=self.income_bins_,
            labels=False,
            include_lowest=True
        )
        # ... (apply all other frozen bin boundaries)
        
        return X


# ---------------------------------------------------------------
# Step 2: Build the full pipeline — preprocessor + model
# ---------------------------------------------------------------

full_pipeline = Pipeline(steps=[
    ('preprocessor', LendingClubPreprocessor()),
    ('model', LogisticRegression(max_iter=1000))
])

# Fit ONCE on training data
full_pipeline.fit(X_train_raw, y_train)

# ---------------------------------------------------------------
# Step 3: Serialise (freeze) the entire pipeline
#         Everything — bin boundaries, training mean, coefficients — 
#         is saved in a single binary file.
# ---------------------------------------------------------------

with open('lending_club_pd_pipeline_v1.pkl', 'wb') as f:
    pickle.dump(full_pipeline, f)

print("Pipeline serialised. Version 1.0 locked.")


# ---------------------------------------------------------------
# Step 4: Scoring new data in 2015, 2020, or any future year
#         The preprocessing is automatically IDENTICAL to training.
# ---------------------------------------------------------------

with open('lending_club_pd_pipeline_v1.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# loan_data_2015 is the raw, unprocessed 2015 CSV
# The pipeline handles ALL cleaning, binning, and transformation
scores_2015 = loaded_pipeline.predict_proba(loan_data_2015_raw)[:, 1]
```

The key point is that `loaded_pipeline` contains the frozen `LendingClubPreprocessor` with all parameters (`annual_inc_train_mean_`, `income_bins_`, etc.) computed from the 2007–2014 training data. There is no human in the loop making manual changes.

### The Static Training Mean Rule

In the offline notebook, missing `annual_inc` values in the 2015 data were filled using the *mean of the 2015 data itself*. This is a subtle but significant error.

If the economic conditions in 2015 caused average incomes to be lower than in 2007–2014, using the 2015 mean as the imputation value means the borrowers in 2015 get imputed with a different value than identical borrowers in 2007 would have received.

The model was trained with one mean. It is being scored with a different one. The model's calibration breaks silently.

**The rule:** Any statistical parameter used during preprocessing — imputation means, standard deviations for scaling, bin boundaries — must be computed **exclusively on the training data** and stored frozen. All future data must be transformed using those stored values, regardless of what the live data looks like.

```python
# WRONG: Computing imputation mean from live data
loan_data_2015['annual_inc'].fillna(loan_data_2015['annual_inc'].mean())

# CORRECT: Using the frozen training mean stored in the pipeline
loan_data_2015['annual_inc'].fillna(self.annual_inc_train_mean_)
```

The serialised pipeline handles this automatically because `self.annual_inc_train_mean_` is saved at `fit()` time and never changes.

### The Dynamic Reference Date

The original development notebook hardcoded `'2017-12-01'` as the reference date for computing credit history length. In the monitoring notebook, this was manually changed to `'2018-12-01'`.

This manual change is an operational risk. If someone runs the monitoring notebook in 2026 without updating the date, every applicant's `mths_since_earliest_cr_line` will be calculated as if it is 2018. An applicant with a credit history dating to 1990 would appear to have 336 months of history when they actually have 432 months. They would be dropped into the wrong WoE bin.

The correct production approach is `pd.Timestamp.now()` — the system clock provides the current date at the exact moment of scoring. The pipeline shown above implements this.

---

## Part 2: Population Stability Index (PSI)

### The Question the Score Average Cannot Answer

When the 2015 data was scored, the average credit score dropped from approximately 650 (training era) to around 610 (2015 cohort). The natural question is: has the model degraded, or did the economy simply produce riskier borrowers in 2015?

Looking at the average score alone cannot answer this. A drop in average score could mean:

1. The *same types of borrowers* as before, but the model is miscalibrated and scoring them too low (model problem).
2. *Different types of borrowers* — lower income, more delinquencies, higher DTI — are now applying, and the model is correctly scoring them lower (population problem).
3. Both are happening simultaneously.

PSI answers the question mathematically by measuring whether the *distribution* of scores has shifted, not just its central tendency.

### The PSI Formula

PSI compares the distribution of scores (or any variable) between a reference (training) population and a current (monitoring) population by dividing the range into bins and measuring how the proportions in each bin have changed:

$$
\text{PSI} = \sum_{i=1}^{n} \left( \%\text{Actual}_i - \%\text{Expected}_i \right) \times \ln\left( \frac{\%\text{Actual}_i}{\%\text{Expected}_i} \right)
$$

Where:
- **Expected** distribution = proportion of *training* population in bin $i$
- **Actual** distribution = proportion of *current (2015)* population in bin $i$
- $n$ = number of bins (typically 10 deciles)

The formula structure is the same as the **Kullback-Leibler divergence** from information theory — it measures the information lost when the Expected distribution is used to approximate the Actual distribution. PSI is actually a symmetrised version:

$$
\text{PSI} = \text{KL}(\text{Actual} \| \text{Expected}) + \text{KL}(\text{Expected} \| \text{Actual})
$$

This symmetry means PSI penalises shifts in both directions equally — both an influx of low-risk borrowers and an influx of high-risk borrowers trigger a high PSI.

### Why the Bin Boundaries Must Be Fixed

This is one of the most common implementation mistakes in production monitoring.

The PSI bins must be defined using the **training data distribution** — typically decile breakpoints — and those boundaries must be stored and reused for all future monitoring periods. The 2015 data is then dropped into those same fixed bins.

If the bin boundaries are recomputed from the 2015 data, the 2015 data will always fall into 10 roughly equal bins by construction. The PSI will be near zero regardless of how much the actual population has shifted. The alarm has been disabled.

```
WRONG: Compute decile breakpoints from 2015 data → PSI ≈ 0 always
CORRECT: Compute decile breakpoints from training data → Apply to 2015 data → PSI reflects true shift
```

This is the "Fixed Boundaries Mandate" — and it applies identically to PSI (score-level monitoring) and CSI (feature-level monitoring).

### PSI Implementation

```python
import numpy as np
import pandas as pd

def compute_psi(expected_scores: np.ndarray,
                actual_scores: np.ndarray,
                n_bins: int = 10,
                return_detail: bool = False) -> float:
    """
    Computes Population Stability Index.
    
    expected_scores: model scores from TRAINING (reference) population
    actual_scores:   model scores from CURRENT (monitoring) population
    n_bins:          number of bins — use 10 (deciles) as industry standard
    return_detail:   if True, returns bin-level breakdown for diagnostics
    
    CRITICAL: Bin breakpoints are ALWAYS computed from expected_scores only.
    """
    # Step 1: Define bin edges from the TRAINING distribution
    breakpoints = np.nanpercentile(
        expected_scores,
        np.linspace(0, 100, n_bins + 1)
    )
    # Extend edges to capture all future values
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # Step 2: Count how many observations fall in each bin
    expected_counts = np.histogram(expected_scores, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual_scores,   bins=breakpoints)[0]
    
    # Step 3: Convert counts to proportions
    # Clip at small epsilon to prevent log(0) which is undefined
    expected_pct = np.clip(expected_counts / len(expected_scores), 1e-6, None)
    actual_pct   = np.clip(actual_counts   / len(actual_scores),   1e-6, None)
    
    # Step 4: Compute PSI contributions per bin
    psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = psi_components.sum()
    
    if return_detail:
        detail_df = pd.DataFrame({
            'Bin': range(1, n_bins + 1),
            'Lower': breakpoints[:-1],
            'Upper': breakpoints[1:],
            'Expected %': (expected_pct * 100).round(2),
            'Actual %':   (actual_pct   * 100).round(2),
            'PSI Component': psi_components.round(4)
        })
        return psi_total, detail_df
    
    return psi_total


# --- Applying to the Lending Club monitoring scenario ---

# Training-era scores (from the test set of the 2007-2014 model)
psi_score, psi_detail = compute_psi(
    expected_scores=scores_train,
    actual_scores=scores_2015,
    n_bins=10,
    return_detail=True
)

print(f"\nOverall PSI: {psi_score:.4f}")
print(f"\nBin-level breakdown:")
print(psi_detail.to_string(index=False))
```

### PSI Thresholds and Business Rules

| PSI Value | Zone | Signal | Required Action |
|---|---|---|---|
| < 0.10 | 🟢 Green | Population is stable | No action required |
| 0.10 – 0.25 | 🟡 Amber | Moderate shift detected | Investigate data pipeline; notify Risk Committee; increase monitoring frequency |
| > 0.25 | 🔴 Red | Significant shift | Trigger immediate model review; consider halting automated decisioning; engage Model Validation team |

These thresholds are the **universal industry standard** under US Model Risk Management frameworks. They are not derived from statistical theory — they are empirically established governance rules based on decades of scorecard monitoring experience. They apply equally to PSI (score level) and CSI (feature level).

---

## Part 3: Characteristic Stability Index (CSI)

### PSI Tells You the Fire Alarm Went Off. CSI Tells You Which Room Is Burning.

PSI monitors the aggregate score distribution. It is a portfolio-level indicator. When PSI trips into amber or red, it tells you *something has changed*, but not *what*.

CSI applies the exact same mathematical formula as PSI, but at the level of individual input features (characteristics). It answers the diagnostic question: which specific variable is driving the score distribution shift?

$$
\text{CSI}_{j} = \sum_{i=1}^{n} \left( \%\text{Actual}_{j,i} - \%\text{Expected}_{j,i} \right) \times \ln\left( \frac{\%\text{Actual}_{j,i}}{\%\text{Expected}_{j,i}} \right)
$$

Where $j$ is the specific feature being monitored and $i$ is its bin index. The formula is identical to PSI — only the input changes from scores to individual feature distributions.

### What Was Actually Built in the Monitoring Notebook

In the monitoring notebook, the code did more than compute a single PSI number on the score. It:

1. Created dummy variables for every bin of every model feature — `grade:A`, `grade:B`, `grade:G`, `emp_length:0`, `emp_length:1`, `annual_inc:20K-30K`, etc.
2. Computed the proportion of training borrowers in each dummy variable bin → `PSI_calc_train`
3. Computed the proportion of 2015 borrowers in each dummy variable bin → `PSI_calc_2015`
4. Applied `(Actual% - Expected%) × ln(Actual% / Expected%)` for every bin
5. **Grouped contributions by original feature name and summed** → this is exactly CSI

The result was a table showing the stability index for each of the model's input variables. This is a full CSI engine.

### CSI Implementation

```python
def compute_csi_from_dummies(
        df_train_dummies: pd.DataFrame,
        df_monitor_dummies: pd.DataFrame,
        feature_names: list) -> pd.DataFrame:
    """
    Computes CSI for each model feature using the WoE dummy variables.
    
    df_train_dummies:   dummy-encoded training data (one column per bin)
    df_monitor_dummies: dummy-encoded monitoring data (same columns)
    feature_names:      list of original feature names (e.g. 'annual_inc')
    
    Returns a DataFrame with one row per feature, sorted by CSI descending.
    """
    results = []
    
    for feature in feature_names:
        # Find all dummy columns belonging to this feature
        # Convention: columns are named 'feature_name:bin_label'
        feature_cols = [c for c in df_train_dummies.columns
                        if c.startswith(feature + ':')]
        
        feature_csi = 0.0
        
        for col in feature_cols:
            # Proportion of training population in this bin
            expected_pct = df_train_dummies[col].mean()
            # Proportion of monitoring population in this bin
            actual_pct   = df_monitor_dummies[col].mean()
            
            # Clip to avoid log(0)
            expected_pct = max(expected_pct, 1e-6)
            actual_pct   = max(actual_pct,   1e-6)
            
            component = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            feature_csi += component
        
        results.append({
            'Feature': feature,
            'CSI': round(feature_csi, 4),
            'N Bins Monitored': len(feature_cols)
        })
    
    csi_df = pd.DataFrame(results).sort_values('CSI', ascending=False)
    csi_df['Signal'] = pd.cut(
        csi_df['CSI'],
        bins=[-np.inf, 0.10, 0.25, np.inf],
        labels=['🟢 Stable', '🟡 Moderate Shift', '🔴 Significant Shift']
    )
    
    return csi_df.reset_index(drop=True)


# --- Running CSI on the Lending Club monitoring data ---

model_features = [
    'grade', 'home_ownership', 'addr_state', 'verification_status',
    'purpose', 'initial_list_status', 'term', 'emp_length',
    'annual_inc', 'dti', 'mths_since_issue_d', 'int_rate',
    'mths_since_earliest_cr_line', 'inq_last_6mths',
    'revol_util', 'total_rev_hi_lim', 'open_acc',
    'delinq_2yrs', 'pub_rec', 'mths_since_last_delinq'
]

csi_report = compute_csi_from_dummies(
    df_train_dummies=X_train_dummies,
    df_monitor_dummies=X_2015_dummies,
    feature_names=model_features
)

print(csi_report.to_string(index=False))
```

### The CSI Diagnostic Hierarchy in Practice

When PSI signals a problem, the CSI report drives the investigation. The pattern of CSI values points to specific root causes:

**Pattern 1: CSI is high for `annual_inc`, `dti`, and `emp_length`**

These are borrower demographic and financial variables. A shift here means the *population applying for loans* has genuinely changed — different income levels, different employment patterns. This is **population drift**. The model's math is probably fine; the applicant pool has changed.

**Pattern 2: CSI is high for `mths_since_earliest_cr_line` or `mths_since_last_delinq`**

These are bureau-derived time variables. A CSI spike here, especially if it is very large or approaches infinity (because one bin becomes empty and the log ratio blows up), almost always means a **data pipeline failure** — a broken API connection to the credit bureau, a schema change in the bureau data feed, or a system migration that changed how timestamps are formatted. The economy did not change; the data did.

**Pattern 3: CSI is high for `addr_state`**

As happened in the Lending Club monitoring notebook — the bank expanded into California, a state that was barely represented in the training data. The `addr_state` geographic distribution shifted dramatically. This is a business decision causing population drift, not a model failure.

**Pattern 4: All feature CSIs are low (<0.10) but PSI is high (>0.25)**

This is the most sophisticated case. The individual features appear stable but the model's output distribution has shifted significantly. This typically indicates **concept drift** — the *relationship* between the features and default probability has changed, even though the features themselves look similar. See Part 4 below.

### CSI for Key Lending Club Features

Illustrative expected CSI values for the Lending Club monitoring exercise (training: 2007–2014 → monitoring: 2015):

| Feature | Likely CSI Direction | Typical Root Cause |
|---|---|---|
| `grade` | Moderate — economic cycle affects grade distribution | Tightened underwriting criteria |
| `annual_inc` | Low-moderate — income levels relatively stable year-over-year | Normal economic variation |
| `addr_state` | **High if California expansion** | Deliberate business policy change |
| `inq_last_6mths` | Low-moderate | Credit-seeking behaviour changes slowly |
| `mths_since_issue_d` | Expected to shift — 2015 data has shorter tenure | Dataset construction artefact, not real drift |
| `revol_util` | Moderate — utilisation tracks economic conditions | Macro environment change |
| `mths_since_last_delinq` | Potentially high | Bureau API issues or genuine behavioural shift |

---

## Part 4: Population Drift vs. Concept Drift

This is the most important distinction in model monitoring, and it is consistently undertaught.

### Population Drift

**Definition:** The statistical distribution of the *input features* has changed, but the underlying relationship between features and default probability has remained the same.

**Example:** Before 2015, average applicant income was $55,000. After Lending Club expanded marketing to recent graduates, average income dropped to $38,000. The model has never seen many applicants at that income level, so its score distribution shifts. But the fundamental truth — lower income increases default risk — has not changed. The model is still correct; it is just seeing a different population.

**Detectable by:** High CSI on demographic/financial input features. PSI is elevated. Model's rank-ordering (KS, AUROC) on the new cohort may still be acceptable once outcome data arrives.

**Response:** Recalibration or post-model overlay (see Part 5). Full redevelopment is not required.

### Concept Drift

**Definition:** The underlying relationship between input features and the target variable has changed. The model has not "drifted" in an input sense — it is seeing similar borrowers — but those borrowers now behave differently than they used to.

**Example:** Before 2020, a borrower with 3 credit inquiries in 6 months (`inq_last_6mths = 3`) had a default rate of 15%. After a period of high inflation and rising interest rates, that same borrower profile now defaults at 28% — because they are rate-shopping out of financial desperation rather than out of normal credit management behaviour. The WoE for `inq_last_6mths = 3` was computed in the training data and is now stale.

**Detectable by:** Feature CSIs are low (borrower profiles look similar), but the **realised default rate by score band** diverges from the expected rate. The model's score of 650 used to correspond to a 3% bad rate. It now corresponds to a 7% bad rate. The rank-ordering may still work (KS/AUROC stable) but the calibration has broken.

**Response:** Full model redevelopment is ultimately required. Short-term: post-model overlay or intercept recalibration.

### Why the Distinction Matters

| | Population Drift | Concept Drift |
|---|---|---|
| **CSI** | High for input features | Low for input features |
| **PSI** | Elevated | May be moderate or elevated |
| **KS / AUROC on new data** | Usually still acceptable | May degrade significantly |
| **Realised bad rate vs. expected** | Deviation concentrated in specific segments | Systematic deviation across all segments |
| **Short-term fix** | Recalibrate / overlay | Overlay; begin redevelopment |
| **Long-term response** | Monitor; redevelop if persistent | **Mandatory redevelopment** |

---

## Part 5: Recalibrate vs. Retrain — The Decision Framework

When PSI trips into red, the immediate instinct might be to retrain the model. This is almost always the wrong first response. Model redevelopment in a regulated bank is an expensive, months-long governance process involving independent validation, business review, and Risk Committee approval. The correct response is proportionate.

### The Cost Hierarchy

From cheapest and fastest to most expensive and time-consuming:

```
1. Do nothing — log the flag, continue monitoring more frequently
   Cost: Zero. Risk: Continued use of a degraded model.

2. Post-model overlay (Score Penalty / Boost)
   Cost: Low. Risk: A crude fix; does not address root cause.

3. Intercept Recalibration
   Cost: Low-moderate. Risk: Fixes calibration but not rank-ordering.

4. Partial redevelopment (update specific features or bins)
   Cost: Moderate. Risk: Creates a hybrid model; harder to validate.

5. Full model redevelopment from scratch
   Cost: High (months of quant time + full MRM process).
   Risk: Model gap period; deployment risk of new model.
```

The goal is always to solve the problem at the lowest level of the hierarchy that is sufficient.

### Post-Model Overlays

A **post-model overlay** is a rule applied *after* the model scores an applicant, adjusting the raw model score by a fixed or conditional offset.

**Example:** The bank expands into California (which was barely in training data). The CSI for `addr_state` is 0.41 (red zone) for California applicants. Rather than rebuilding the model, the credit policy team applies a temporary penalty:

```python
# Post-model overlay: California applicants receive a -15 point score penalty
# until sufficient California performance data is accumulated (typically 12 months)

def apply_overlay(raw_score: float, addr_state: str) -> float:
    overlay_rules = {
        'CA': -15,   # California expansion — insufficient training data
        'NY': 0,     # No adjustment
        # ... other state adjustments
    }
    adjustment = overlay_rules.get(addr_state, 0)
    return raw_score + adjustment

# This is fully documented in the model's change log and must be
# approved by the Risk Committee before implementation
```

Overlays must be:
- Formally documented in the **Model Change Log**
- Approved by the Risk Committee or delegated authority
- Time-limited — they require a review date after which the overlay is either retired, adjusted, or replaced by a full model update
- Communicated to Model Validation for their ongoing monitoring records

### Intercept Recalibration

If the model's rank-ordering is still valid (KS and AUROC on the new cohort are acceptable) but the *absolute* default rate predictions are off, the problem is calibration rather than discrimination. The model correctly ranks borrowers from lowest to highest risk, but the probability estimates are too optimistic or too pessimistic.

This can be corrected by adjusting the logistic regression intercept $\beta_0$ without changing any of the feature coefficients $\beta_1, \ldots, \beta_k$.

The recalibrated intercept is found by solving:

$$
\beta_0^{\text{new}} = \ln\left(\frac{\bar{p}_{\text{actual}}}{1 - \bar{p}_{\text{actual}}}\right) - \sum_{j=1}^k \beta_j \bar{x}_j
$$

Where $\bar{p}_{\text{actual}}$ is the observed bad rate in the new population and $\bar{x}_j$ is the mean WoE value of feature $j$ in the new population.

```python
def recalibrate_intercept(model, X_new, y_new_actual_bad_rate):
    """
    Adjusts the logistic regression intercept to match the observed bad rate
    in a new population, without touching the feature coefficients.
    
    model: fitted LogisticRegression object
    X_new: WoE-transformed features of the new population
    y_new_actual_bad_rate: observed bad rate in the new population (0 to 1)
    """
    # Current predicted mean score (log-odds)
    mean_feature_contribution = np.dot(model.coef_[0], X_new.mean(axis=0))
    
    # Target log-odds for the observed bad rate
    # Note: model predicts P(Good), so bad rate = 1 - P(Good)
    # We want the intercept that produces the right overall calibration
    target_log_odds_good = np.log(
        (1 - y_new_actual_bad_rate) / y_new_actual_bad_rate
    )
    
    new_intercept = target_log_odds_good - mean_feature_contribution
    
    print(f"Original intercept: {model.intercept_[0]:.4f}")
    print(f"Recalibrated intercept: {new_intercept:.4f}")
    print(f"Adjustment: {new_intercept - model.intercept_[0]:.4f}")
    
    # Apply the new intercept
    model.intercept_[0] = new_intercept
    return model
```

Intercept recalibration must be treated as a **Minor Model Change** under SR 11-7. It requires documentation and Risk Committee notification, but does not require the full independent validation process that a new model would need.

### When Full Redevelopment Is Unavoidable

Three conditions independently trigger mandatory model redevelopment:

1. **Sustained PSI > 0.25 for 3+ consecutive monitoring periods** — the population has fundamentally changed and no overlay or recalibration is sufficient.
2. **KS or AUROC on new cohorts has degraded by more than 10–15 percentage points** from the initial approved model — the model has lost significant discriminatory power.
3. **Realised bad rates by score band systematically exceed model predictions** — concept drift is confirmed and calibration is structurally broken.

---

## Part 6: The Full Production Monitoring Cycle

Bringing everything together, here is what a complete monthly monitoring run looks like for the Lending Club PD model in production:

```python
# Monthly monitoring pipeline
# Runs on the first business day of each month

def monthly_monitoring_run(
        pipeline_path: str,
        training_scores: np.ndarray,
        training_dummies: pd.DataFrame,
        current_month_data: pd.DataFrame,
        current_month_outcomes: pd.Series,  # available after 12-month lag
        model_features: list,
        report_date: str) -> dict:
    
    # Load frozen pipeline
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    # Score current month
    current_scores = pipeline.predict_proba(current_month_data)[:, 1]
    
    # --- PSI ---
    psi_total, psi_detail = compute_psi(
        expected_scores=training_scores,
        actual_scores=current_scores,
        return_detail=True
    )
    
    # --- CSI ---
    current_dummies = create_woe_dummies(current_month_data, pipeline)
    csi_report = compute_csi_from_dummies(
        training_dummies, current_dummies, model_features
    )
    
    # --- Discrimination metrics (requires 12+ month outcome lag) ---
    if current_month_outcomes is not None:
        auroc = roc_auc_score(current_month_outcomes, current_scores)
        gini  = 2 * auroc - 1
        ks, _ = compute_ks(current_month_outcomes, current_scores)
    else:
        auroc = gini = ks = None
    
    # --- Bad rate by score band ---
    score_band_performance = compute_bad_rate_by_decile(
        current_scores, current_month_outcomes
    )
    
    # --- Assemble report ---
    report = {
        'report_date': report_date,
        'PSI': round(psi_total, 4),
        'PSI_signal': 'Red' if psi_total > 0.25 else ('Amber' if psi_total > 0.10 else 'Green'),
        'top_3_CSI_features': csi_report.head(3)[['Feature', 'CSI', 'Signal']].to_dict(),
        'AUROC': auroc,
        'Gini': gini,
        'KS': ks,
        'score_band_performance': score_band_performance
    }
    
    # Auto-trigger alert if red zone
    if psi_total > 0.25:
        trigger_model_review_alert(report)
    
    return report
```

### Monthly Monitoring Dashboard Components

A production monitoring report should contain five sections:

**1. Score Distribution (PSI)**
The current month's score histogram overlaid on the training baseline. PSI value prominently displayed with traffic-light colour coding.

**2. Characteristic Report (CSI)**
All model features ranked by CSI value. Top 5 flagged features highlighted with directional shift (e.g., `annual_inc` average shifted from $52K to $41K).

**3. Discrimination Trend**
Monthly AUROC and KS plotted over the model's lifetime. Requires 12+ months of outcome data (the performance window from [[1_full_pd_model]] Part 1).

**4. Bad Rate by Score Band**
Realised default rate in each score decile vs. the rate predicted at model development. Divergence between these lines is the clearest early warning of concept drift.

**5. Override and Exception Report**
What percentage of model decisions were manually overridden by credit officers. A rising override rate is a business signal that the model's decisions are no longer trusted internally — itself a trigger for model review.

---

## Connections to the Rest of the Quant OS Brain

- [[Tharun-Kumar-Gajula]] — The master profile anchoring this portfolio.
- [[1_full_pd_model]] — This entire note is about what happens to that model after deployment. The serialised pipeline preserves the WoE bins and preprocessing logic from the PD score development natively.

---

## Key Concepts Summary

| Concept | What It Is | Where It Appears in This Project |
|---|---|---|
| **Serialised Pipeline** | Frozen sklearn Pipeline saved as a binary file | `lending_club_pd_pipeline_v1.pkl` |
| **BCBS 239** | Basel principle requiring controlled, auditable risk data processes | Justifies why copy-paste notebooks are unacceptable |
| **Static Training Mean** | Imputation parameters must come from training data, not live data | `annual_inc` imputation in the monitoring notebook |
| **Dynamic Reference Date** | `pd.Timestamp.now()` replaces hardcoded `2017-12-01` | `mths_since_earliest_cr_line` calculation |
| **PSI** | Measures overall score distribution shift | Portfolio-level monitoring; 0.10/0.25 thresholds |
| **Fixed Bin Boundaries** | PSI/CSI bins defined on training data only, never recomputed on live data | Core implementation requirement |
| **CSI** | PSI applied at individual feature level | Diagnoses which features are driving score shift |
| **Population Drift** | The *who* has changed; model relationship intact | `addr_state` CSI spike from California expansion |
| **Concept Drift** | The *relationship* between features and default has changed | All feature CSIs stable but bad rate by score band diverging |
| **Post-Model Overlay** | Score adjustment rule applied after model output | California `-15 points` temporary penalty |
| **Intercept Recalibration** | Adjust $\beta_0$ to match new population's bad rate | Minor model change; lighter governance than full rebuild |
| **Champion-Challenger** | New model runs in shadow mode before full deployment | See 08-deployment-lifecycle |
| **SR 11-7** | US Fed model risk management guidelines | Governs monitoring, recalibration, and redevelopment decisions |

---

*This note is Version 1.0. The next monitoring extension is LGD and EAD outcome tracking — monitoring whether recovery rates and exposure estimates are stable post-deployment. That note will be: [[3_lgd_ead_model_rewritten]].*
