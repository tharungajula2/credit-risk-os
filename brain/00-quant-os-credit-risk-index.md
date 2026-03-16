---
title: "Quant OS: Credit Risk Curriculum Index"
tags:
  - index
  - MOC
  - credit-risk
  - quant-os
cluster: "System Atlas"
links:
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

# Quant OS: End-to-End Retail Credit Risk Modeling & Machine Learning
### A Zero-to-Hero Practitioner's Curriculum

> **Quant OS Flagship Curriculum** | Version 1.0 | Cluster: Credit Risk  
> *A definitive, practitioner-grade knowledge graph for quantitative finance professionals.*

---

## Table of Contents

- [[01-credit-risk-foundations]] — The Macro Picture & End-to-End Workflow
- [[02-data-architecture-preprocessing]] — Data Architecture & Preprocessing
- [[03-woe-iv-feature-engineering]] — The Algorithmic Engine & Scorecard Development
- [[06-model-validation-metrics]] — Model Validation & Performance Metrics
- [[07-production-monitoring-PSI]] — Production Monitoring & Stability
- [[08-deployment-lifecycle]] — The Deployment Lifecycle & Client Engineering
- [[Zettelkasten-Architecture]] — Zettelkasten Architecture

## Zettelkasten Graph Architecture

```
[00-index] ──────────────────────────────────────────────────
    │                                                         │
    ▼                                                         ▼
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

Cross-cluster connections (non-linear links that make this a true graph, not a chain):

- `[[03-woe-iv]]` ↔ `[[06-validation]]` (IV as feature selection; KS as validation metric share conceptual DNA)
- `[[04-logistic-scorecard]]` ↔ `[[07-PSI-monitoring]]` (scorecard score distribution → PSI monitoring)
- `[[02-data-arch]]` ↔ `[[07-PSI-monitoring]]` (CSI monitors the same features preprocessed in Phase 2)
- `[[05-tree-models]]` ↔ `[[09-algorithm-selection]]` (tree models map to specific business problems)

---
