---
title: "Socio-Economic Household Classification"
date: "2026-03-15"
summary: "End-to-end pipeline handling noisy national survey datasets to classify household status, peaking at 83.5% accuracy via Ensembles."
tags: ["Python", "XGBoost", "Ensemble", "PCA", "Data Science"]
---

# India Survey Data: Socio-Economic Classifier

Real-world national survey data is notoriously noisy, highly imbalanced, and filled with high-cardinality categorical variables. This project demonstrates end-to-end data pipeline engineering, transforming raw survey inputs into a robust predictive engine to classify households (Urban vs. Rural) and derive macroeconomic policy insights.

## Project Architecture
- **Objective:** Classify household regional status based on a massive socio-economic dataset to uncover the fundamental drivers of the Urban-Rural economic divide.
- **Data Pipeline:** Processed 176,661 records, performing aggressive outlier capping, dimensionality reduction via Principal Component Analysis (PCA), and handling severe class imbalances using SMOTE and algorithm-level class weights.
- **Core Methodology:** Built a comparative pipeline scaling from Logistic Regression and Artificial Neural Networks (ANN) up to a sophisticated Soft-Voting Ensemble Classifier leveraging Random Forest and XGBoost.
- **Outcomes:** The optimized Random Forest and Soft Voting Ensemble achieved **~83.5% accuracy** on unseen data, while feature importance extraction revealed the critical impact of occupational segmentation and dependency ratios on household classification.

*(Extensive EDA visualizations, PCA variance plots, and the SMOTE pipeline code will be added here).*

---

## 1. Data Architecture & Feature Engineering

The initial dataset contained 176,661 rows and 38 columns, heavily polluted with administrative metadata and extreme economic outliers. Strict preprocessing protocols were enforced before modeling.

- **Dimensionality Pruning:** Eliminated 15 redundant columns, including survey population weights (`HH_WEIGHT_MS`), single-value columns (`MONTH`), and non-predictive identifiers.
- **Data Hygiene:** Standardized missing values (converting `-99` flags to `NaN`) and dropped rows where all numerical income values were missing, resulting in a clean baseline of 118,885 records.
- **Outlier Mitigation:** Applied strict 1st and 99th percentile capping across all 15 continuous income variables to prevent extreme wealth outliers from dominating the gradient descent and tree-splitting algorithms.
- **Custom Feature Engineering:** Extracted 6 latent economic indicators to provide deeper predictive context:
  1. `INCOME_FROM_INVESTMENTS`: Aggregated dividends, interest, and FD/PF insurance.
  2. `IS_HIGH_INCOME`: Binary flag for the top 10% of total earners.
  3. `HOUSEHOLD_SIZE_NUM`: Converted categorical string bands into a continuous numeric scale (1 to 16).
  4. `DEPENDENCY_RATIO`: Calculated as `HOUSEHOLD_SIZE_NUM / TOTAL_INCOME` to measure financial stress.
  5. `HAS_GOV_SUPPORT` & `HAS_RENTAL_INCOME`: Binary flags isolating passive income streams.

---

## 2. Dimensionality Reduction & Preprocessing

To prepare the dataset for machine learning, categorical variables underwent **One-Hot Encoding (OHE)**, expanding the feature space to 109 dimensions.

### Principal Component Analysis (PCA)
To combat the curse of dimensionality and remove multicollinearity (e.g., `TOTAL_INCOME` having a 0.95 correlation with `WAGES`), PCA was applied to the scaled features.
- **Variance Retention:** The architecture was configured to retain **95% of the variance**.
- **Result:** The feature space was successfully compressed from 109 raw features down to **80 Principal Components**, optimizing training times for complex tree algorithms without sacrificing predictive signal.

---

## 3. Combating Structural Imbalance (SMOTE vs. Class Weights)

The dataset exhibited a severe geographic imbalance: Urban (64,095) vs. Rural (31,013). Left untreated, baseline models heavily favored the Urban majority. Two distinct balancing strategies were tested:

1. **SMOTE (Synthetic Minority Over-sampling Technique):** - Synthetically generated new Rural records to achieve a perfect 1:1 balance in the training set.
   - *Result:* Pushed Rural Recall from a poor 58% up to 67%. However, overall test accuracy dropped to 80.7% due to the introduction of synthetic noise.
2. **Algorithmic Class Weighting:**
   - Calculated a `scale_pos_weight` ratio of **0.484** to heavily penalize the XGBoost loss function for misclassifying Rural households.
   - *Result:* Achieved a 68% Rural Recall while maintaining a slightly tighter overall model fit. 
- **Business Trade-off:** While balancing the data drops top-line "Accuracy," it is a mandatory step to ensure the model actually learns the underlying characteristics of Rural households rather than just guessing "Urban" by default.

---

## 4. The Predictive Ensemble (Model Evolution)

The modeling phase evolved iteratively, culminating in a sophisticated voting architecture.

### 1. The Baselines
- **Logistic Regression (Tuned):** Set a strong, interpretable baseline at **81.7%** test accuracy.
- **Decision Tree:** Highlighted the danger of unconstrained trees, heavily overfitting the training data (98.8% Train vs. 76.2% Test) before tuning.
- **Artificial Neural Network (ANN):** A 3-layer deep learning model utilizing `BatchNormalization` and `Dropout(0.3)` to prevent overfitting, converging via the Adam optimizer with Early Stopping.

### 2. The Champions (Tree Ensembles)
- **Random Forest (Tuned):** Optimized via `RandomizedSearchCV`. Using 300 estimators and bootstrapping, it effectively mapped the complex, non-linear economic boundaries, peaking at **83.5% Accuracy**.
- **XGBoost (Tuned + PCA):** Achieved **82.8% Accuracy**, proving that gradient boosting could maintain exceptional predictive power even on a mathematically compressed (PCA) feature space.

### 3. The Ultimate Architecture: Soft Voting Classifier
To squeeze out maximum generalization, a `VotingClassifier` was deployed, combining the distinct mathematical approaches of Logistic Regression, Random Forest, and XGBoost.
- By utilizing **Soft Voting**, the ensemble averaged the raw probability outputs (`predict_proba()`) of all three models rather than their hard class predictions, yielding a highly stable, production-ready accuracy of **83.0%** with exceptionally balanced Precision/Recall matrices.

---

## 5. Macro-Economic Interpretability (Feature Importance)

Machine Learning models are only as valuable as the business logic they reveal. Extracting the SHAP/Feature Importances from the XGBoost and Random Forest architectures exposed the true fault lines of the Urban-Rural divide:

1. **Occupational Stratification is Supreme:** The absolute heaviest predictors of a household's region were specific occupational flags, primarily `Agricultural Labourers`, `Small/Marginal Farmers`, and `Organised Farmers` (driving Rural classification), contrasted against `Entrepreneurs` and `White-collar Professionals` (driving Urban).
2. **The Dependency Ratio:** The custom engineered `DEPENDENCY_RATIO` proved highly predictive, signaling that the mathematical relationship between household size and total income fundamentally shifts between urban and rural environments.
3. **Wage Dominance:** `INCOME_OF_ALL_MEMBERS_FROM_WAGES` heavily outranked passive income sources, proving that direct labor compensation remains the primary delineator of geographic socio-economic status in the dataset.