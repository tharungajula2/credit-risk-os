---
title: "IRB Credit Scorecard & PD Modeling"
date: "2026-03-15"
summary: "Developed & validated PD/EAD/LGD models with strict KS/AUC validation and PSI stability monitoring."
tags: ["Python", "XGBoost", "Scikit-learn", "SQL", "Credit Risk"]
---

# AI-Powered Credit Risk Modeling (PD Scorecard)

In the highly regulated domain of credit risk, black-box models are often insufficient. This project focuses on building a highly interpretable, regulatory-compliant application scorecard to predict the Probability of Default (PD) for lending decisions.

## Project Architecture
- **Objective:** Optimize credit risk assessment and automate lending decisions based on applicant data.
- **Core Methodology:** Engineered a robust Logistic Regression pipeline, transforming raw features using Weight of Evidence (WOE) to handle non-linearities and missing data smoothly.
- **Validation:** Ensured model robustness using industry-standard metrics, including strict KS (Kolmogorov-Smirnov) and AUC (Area Under the Curve) validation. 
- **Monitoring:** Implemented Population Stability Index (PSI) tracking to detect data drift over time.

*(Jupyter Notebook code blocks for data preprocessing, WOE transformation, and model fitting will be added here).*