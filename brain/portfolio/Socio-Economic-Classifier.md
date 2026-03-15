---
title: "Socio-Economic Household Classification"
date: "2026-03-15"
summary: "End-to-end pipeline handling noisy national survey datasets to classify household status (84% accuracy)."
tags: ["Python", "XGBoost", "Ensemble", "Data Science"]
---

# India Survey Data: Socio-Economic Classifier

Real-world survey data is notoriously noisy, imbalanced, and filled with high-cardinality categorical variables. This project demonstrates end-to-end data pipeline engineering to derive policy-making insights.

## Project Architecture
- **Objective:** Classify households (Urban vs. Rural) based on a massive, complex national socio-economic survey dataset.
- **Data Pipeline:** Handled severe class imbalances using SMOTE (Synthetic Minority Over-sampling Technique) and reduced dimensionality using PCA (Principal Component Analysis).
- **Core Methodology:** Deployed a sophisticated Voting Classifier (Ensemble) leveraging XGBoost and other Scikit-learn models.
- **Outcomes:** Achieved **84% accuracy** on unseen data, deriving critical feature importance insights useful for governmental or NGO policymaking.

*(Extensive EDA visualizations, PCA variance plots, and the SMOTE pipeline code will be added here).*