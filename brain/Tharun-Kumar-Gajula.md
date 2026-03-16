---
title: "Tharun Kumar Gajula"
date: 2026-03-19
tags:
  - profile
  - MOC
  - credit-risk
  - model-validation
  - quantitative-finance
  - data-science
  - banking
  - zettelkasten
cluster: "System Atlas"
links:
  - "[[00-quant-os-credit-risk-index]]"
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

---

# Tharun Kumar Gajula

I work at the intersection of credit risk, quantitative analytics, model development, and model validation.

My background is neither purely academic nor purely technical. It is grounded in real banking and lending workflows — understanding how risk is measured, monitored, and governed — with a strong quantitative and model-focused layer built on top.

This vault is how I organize that work. It is my working system for understanding how risk models are built, challenged, monitored, and used inside real institutions.

---

## What This Vault Is For

The purpose of this system is to connect five things properly:

- credit risk fundamentals and regulation
- model development and scorecard construction
- model validation and performance measurement
- production monitoring and stability
- practical deployment and governance

The new brain I have built — the **Quant OS Credit Risk Curriculum** — is the core technical engine of this vault. Every note below connects back to it.

The entry point is [[00-quant-os-credit-risk-index]].

---

## How the New Brain Maps to My Work

The Quant OS curriculum is structured across six phases. Here is how each phase connects to my actual background and direction.

---

### Phase 1 — Foundations → [[01-credit-risk-foundations]]

This is the conceptual base. PD definition, the 90 DPD target variable, the end-to-end project lifecycle from data ingestion to production monitoring.

This maps directly to the credit-risk fundamentals I built at Jana Small Finance Bank, where I worked close to portfolio monitoring and PD/EAD/LGD-style risk tracking. It also connects to my existing notes:

- [[Probability-of-Default.md]]
- [[Loss-Given-Default.md]]
- [[Exposure-at-Default.md]]
- [[Expected-vs-Unexpected-Loss.md]]
- [[Basel-IRB-Framework.md]]
- [[IFRS-9-and-ECL.md]]

The lifecycle diagram in [[01-credit-risk-foundations]] is the spine that holds the entire system together. Every other phase is a node on that spine.

---

### Phase 2 — Data Architecture → [[02-data-architecture-preprocessing]]

Feature taxonomy (application, bureau, behavioral, macroeconomic attributes), missingness mechanisms (MCAR / MAR / MNAR), Winsorization, VIF, and multicollinearity detection.

This connects to the implementation and data-pipeline side of my work at Lentra AI, where I was close to business requirements, UAT, API validation, and how data flows through lending systems before it ever reaches a model.

The practical note here: MNAR missing data in credit is not a data quality problem. It is a signal. The section on "missing as a category" in [[02-data-architecture-preprocessing]] is one of the most important decisions in retail credit modeling.

---

### Phase 3 — Algorithmic Engine → [[03-woe-iv-feature-engineering]], [[04-logistic-regression-scorecard]], [[05-tree-based-models]]

This is the densest technical cluster in the brain. Three interconnected files:

**[[03-woe-iv-feature-engineering]]** — WoE derivation, IV thresholds (0.02 / 0.10 / 0.30 / 0.50), the data leakage warning at IV > 0.50, fine/coarse classing, monotonicity. This is the foundation of everything in [[04-logistic-regression-scorecard]].

**[[04-logistic-regression-scorecard]]** — Full MLE derivation, log-odds, $\beta$ interpretation, the complete PDO scorecard scaling formula (Factor, Offset, Base Score, Base Odds), regularization (L1/L2/ElasticNet). This connects directly to [[IRB-Credit-Scorecard.md]] and is the most regulatory-facing technical note in the system.

**[[05-tree-based-models]]** — Decision trees through XGBoost and LightGBM. Gini impurity, entropy, bagging vs. boosting, GOSS/EFB, hyperparameter tuning (Grid / Random / Bayesian Optimization). This connects to [[Advanced-ML-in-Risk.md]].

This phase is where the IRB Credit Scorecard project lives technically:

- [[IRB-Credit-Scorecard.md]] → uses [[03-woe-iv-feature-engineering]] + [[04-logistic-regression-scorecard]]

---

### Phase 4 — Validation → [[06-model-validation-metrics]]

Confusion matrix, Precision/Recall with asymmetric cost justification, AUROC, Gini coefficient, and the KS Statistic — the dominant discrimination metric in retail credit.

The KS formula and interpretation is the most-used validation metric in production credit environments. AUROC = 0.82 → Gini = 64%. These are not abstract statistics. They are the language of model sign-off.

This connects to:

- [[Model-Performance-Metrics.md]]
- [[SR-11-7-Model-Governance.md]] (validation independence requirement)
- [[Logistic-Regression-Scorecards.md]]

The OOT (Out-of-Time) validation protocol in [[06-model-validation-metrics]] is specifically relevant to model validation roles. A model validated only on training data is not validated.

---

### Phase 5 — Monitoring → [[07-production-monitoring-PSI]]

PSI formula and the 0.10 / 0.25 threshold business rules. CSI at the feature level. The two-stage diagnostic hierarchy: PSI triggers investigation → CSI isolates the drifting feature.

This is the production-facing half of model governance. A model does not stop being a risk the day it goes live. It becomes a risk that must be monitored every month.

This connects to:

- [[Population-Stability-Index-PSI.md]]
- [[Macro-Stress-Testing.md]] (population shifts under macro stress)
- [[SR-11-7-Model-Governance.md]] (ongoing monitoring as a governance requirement)

The Python PSI implementation in [[07-production-monitoring-PSI]] is reusable across any credit model in production. The bin-fixing caveat (breakpoints must be computed on the development population, never the live population) is a critical implementation detail.

---

### Phase 6 — Deployment & Governance → [[08-deployment-lifecycle]], [[09-algorithm-selection-guide]]

**[[08-deployment-lifecycle]]** — The three-stage model approval process: Technical Validation → Business/Regulatory Review → Client Handover. SR 11-7 and SS1/23 references. Adverse action reason codes. The champion-challenger shadow deployment framework.

**[[09-algorithm-selection-guide]]** — Mapping algorithms to business problems: Logistic Regression Scorecard for origination, XGBoost for portfolio ranking, Isolation Forest for fraud, KNN/K-Means for customer targeting, Survival Analysis for collections.

This phase is the governance layer. It is what separates a model that works from a model that is approved, deployed, and defensible to a regulator.

This connects to:

- [[SR-11-7-Model-Governance.md]]
- [[AML-and-Financial-Crime-Models.md]] (anomaly detection use case)
- [[Low-Default-Portfolios-LDP.md]] (specialist validation constraints)

---

## The Full Brain: Node Map

```
[00-tharun-kumar-gajula-profile]  ← You are here
            │
            ▼
[00-quant-os-credit-risk-index]   ← Master curriculum index
            │
    ┌───────┼───────────────────────────────────┐
    ▼       ▼                                   ▼
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

**Cross-cluster links (non-linear):**

- [[03-woe-iv-feature-engineering]] ↔ [[06-model-validation-metrics]] — IV is feature selection; KS is the validation metric. Both are ratios of Bad/Good distributions.
- [[04-logistic-regression-scorecard]] ↔ [[07-production-monitoring-PSI]] — The scorecard score distribution is exactly what PSI monitors in production.
- [[02-data-architecture-preprocessing]] ↔ [[07-production-monitoring-PSI]] — CSI monitors the same features preprocessed in Phase 2.
- [[08-deployment-lifecycle]] ↔ [[SR-11-7-Model-Governance.md]] — The three-stage approval process is SR 11-7 in practice.

---

## What Roles This System Is Built For

The system is structured for roles where these areas overlap:

- credit risk analytics and modeling
- quantitative risk and model risk
- model validation (independent review)
- ML model monitoring and governance
- regulated model development and deployment

That is why the note structure does not stop at credit concepts. Validation, monitoring, deployment, and governance are each first-class citizens in the graph — not appendices.

---

## The Three Things to Know

If someone reads this note, they should understand three things quickly:

1. My base is credit risk and banking — built at Jana Small Finance Bank through portfolio monitoring, risk analytics, and PD/EAD/LGD thinking.
2. My direction is toward quantitative risk and model validation — the Quant OS brain is the technical system that supports that direction.
3. This vault connects theory, regulation, modeling, and practical implementation — not as separate silos, but as a single working graph.

The new brain is the engine. This note is the map.

Start at [[00-quant-os-credit-risk-index]].