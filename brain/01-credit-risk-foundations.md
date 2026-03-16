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
