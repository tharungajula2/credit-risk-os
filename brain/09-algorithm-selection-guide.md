---
title: "Algorithm Selection: Matching Models to Business Problems"
tags:
  - algorithm-selection
  - logistic-regression
  - XGBoost
  - anomaly-detection
  - KNN
  - clustering
  - survival-analysis
  - fraud-detection
cluster_phase: Phase 6
links:
  - "[[04-logistic-regression-scorecard]]"
  - "[[05-tree-based-models]]"
  - "[[08-deployment-lifecycle]]"
---

## 6.2 Mapping Algorithms to Client Business Problems

The choice of algorithm is not a technical decision alone — it must map precisely to the **business problem the client is actually trying to solve**.

| Business Problem | Recommended Algorithm | Why |
|---|---|---|
| **"Should we approve this application?"** (Binary decision, need reason codes) | **Logistic Regression Scorecard** | Regulatory acceptance; reason codes; score decomposition; champion model for most lenders |
| **"Which segment of our portfolio is highest risk?"** (Ranking/prioritization) | **XGBoost or LightGBM** with calibration | Superior discrimination; SHAP for post-hoc explainability |
| **"Is this transaction fraudulent?"** (Rare event, imbalanced classes) | **Isolation Forest, One-Class SVM, or Autoencoder** (Anomaly Detection) | Fraud is an anomaly problem, not a standard classification problem; labeled fraud data is scarce |
| **"Who among delinquent customers should collections call first?"** (Ranking within subpopulation) | **Survival Analysis (Cox PH) or XGBoost ranking** | Time-to-event matters; behavioral data drives this |
| **"Find me customers similar to my best customers"** (Customer targeting) | **K-Nearest Neighbors (KNN) or Clustering (K-Means)** | Similarity search in feature space; no label required |
| **"Predict the next 12 months' default rate under stress"** (Macroeconomic scenario) | **Panel regression / econometric model with macro overlays** | Time-series properties; macro factor sensitivity required |
