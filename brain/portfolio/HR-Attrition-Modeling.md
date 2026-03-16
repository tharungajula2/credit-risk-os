---
title: "Employee Retention & Predictive Attrition Modeling"
date: "2026-03-19"
summary: "Evaluated Logistic Regression and advanced Tree-Based ensembles, optimizing a Random Forest to achieve 96.5% accuracy in predicting HR flight risks."
tags: ["Python", "Scikit-learn", "Random Forest", "GridSearchCV", "Predictive ML"]
cluster: "Quantitative Portfolio"
---

# HR Attrition & Performance Analytics

Applying quantitative risk techniques to human resources. This project treats employee attrition as a predictable risk event, moving beyond gut-feeling HR management to deploy targeted, data-driven retention strategies based on predictive modeling.

## Project Architecture
- **Objective:** Identify the core drivers of employee turnover and predict individual attrition risks (whether an employee will 'leave' or 'stay').
- **Core Methodology:** Built a comparative machine learning pipeline starting with a Logistic Regression baseline, before scaling up to hyperparameter-tuned Decision Trees and Random Forest ensembles.
- **Outcomes:** The optimized Random Forest model achieved a remarkable **96.5% test accuracy** with a highly balanced Precision/Recall profile, allowing HR to reliably identify flight-risk talent without generating excessive false alarms.

*(Model comparison tables, hyperparameter tuning grids, and feature importance charts will be added here).*

---

## 1. Data Architecture & Preprocessing

The foundation of the pipeline utilized a dataset of 14,999 employee records. Strict data hygiene protocols were applied prior to modeling.

- **Deduplication:** Identified and removed 3,008 duplicate records to prevent data leakage between training and validation splits.
- **Variable Standardization:** Renamed system columns for programmatic consistency (e.g., `time_spend_company` $\rightarrow$ `tenure`, `Department` $\rightarrow$ `department`).
- **Outlier Detection (IQR Method):** Applied the Interquartile Range (IQR) method specifically to the `tenure` variable. 
  - Calculated the 25th and 75th percentiles.
  - Established a lower limit of 1.5 and an upper limit of 5.5 years.
  - Successfully identified and removed 824 outlier rows to ensure the model learned generalizable trends rather than edge cases.
- **Categorical Encoding:** Applied One-Hot Encoding (`pd.get_dummies`) to nominal variables (`department`, `salary`), enabling distance and tree-based algorithms to process them mathematically.

---

## 2. Feature Engineering & EDA

Before feeding data into the predictive engines, engineered features were developed to capture latent behavioral patterns.

- **The 'Overworked' Trigger:** Extracted minimum (96) and maximum (310) `average_monthly_hours`. Engineered a new binary feature, `overworked`, strictly defined as employees logging **> 175 hours/month**.
- **Correlation Profiling:** A heatmap analysis revealed that `satisfaction_level` had a strong negative correlation with attrition, while `tenure` and `number_project` showed complex, non-linear relationships with employee exit rates.
- **Departmental Risk:** Visualized churn across departments, identifying Sales, Technical, and Support as the highest absolute volume drivers of turnover.

---

## 3. The Baseline: Logistic Regression

To establish a benchmark, a Logistic Regression model (`max_iter=500`) was trained.

- **Performance Profile:** - Overall Accuracy: 82%
  - **Recall (Class 1 - Left): 0.26**
  - **Precision (Class 1 - Left): 0.49**
- **Business Conclusion:** While the accuracy seemed acceptable, the model was functionally useless for HR purposes. A recall of 0.26 meant it missed 74% of the employees who actually left. It proved that the relationship between HR metrics and attrition is highly non-linear.

---

## 4. Tree-Based Ensembles & Hyperparameter Optimization

To capture the non-linear dynamics missed by the baseline, the architecture transitioned to tree-based models. Both models utilized `GridSearchCV` with 4-fold Cross Validation (`cv=4`), optimizing strictly for the `roc_auc` metric.

### Model 1: Decision Tree Classifier
- **Best Parameters Extracted:** `max_depth`: 6, `min_samples_leaf`: 1, `min_samples_split`: 4.
- **CV AUC Score:** 0.953
- **Test Performance:** Reached 96.0% accuracy, significantly improving the ability to map complex decision boundaries (like the interplay between low salary and high hours).

### Model 2: The Champion - Random Forest Classifier
To prevent the high variance and overfitting inherent to single decision trees, a Random Forest ensemble was deployed.

- **Hyperparameter Grid Search:** Evaluated combinations of `max_depth`, `max_features`, `max_samples`, `min_samples_leaf`, and `n_estimators` (300 vs 500).
- **Optimal Architecture:** - `n_estimators`: 300
  - `max_depth`: None (Nodes expanded until all leaves were pure)
  - `max_features`: 1.0
  - `max_samples`: 0.7 (Bootstrapping 70% of data per tree)
  - `min_samples_leaf`: 3
  - `min_samples_split`: 2
- **Cross-Validation AUC:** 0.965

---

## 5. Final Evaluation & Business Translation

The optimized Random Forest model was evaluated on the unseen test set, yielding exceptional results that directly translate to business ROI:

| Metric | Score | Business Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **0.965** | The model correctly classifies 96.5% of all employee outcomes. |
| **AUC** | **0.935** | Excellent discriminatory power between flight-risk and retained talent. |
| **Precision** | **0.898** | When the model flags an employee as a flight risk, it is correct nearly 90% of the time. (High efficiency for HR budgets). |
| **Recall** | **0.889** | The model successfully captures 88.9% of all employees who actually leave. (Minimizes catastrophic talent loss). |
| **F1-Score** | **0.894** | A near-perfect harmonic balance, ensuring HR neither wastes retention bonuses on false alarms nor misses critical departures. |

**Next Steps for Deployment:**
Integrate the Random Forest prediction pipeline into the HR dashboard. Any employee crossing a specific probability threshold triggers an automated alert to their department head, alongside their specific risk drivers (e.g., "Overworked flag triggered").