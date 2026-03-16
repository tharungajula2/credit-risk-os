---
title: "Bank Churn Prediction using Neural Networks"
date: "2026-03-19"
summary: "Built an Artificial Neural Network classifying bank customer churn, optimizing thresholds for 75% recall."
tags: ["Python", "TensorFlow", "Keras", "Deep Learning"]
cluster: "Quantitative Portfolio"
---

# Bank Customer Churn Prediction

Customer acquisition is significantly more expensive than retention. This project leverages Deep Learning to identify at-risk bank customers before they leave the institution.

## Project Architecture
- **Objective:** Classify banking customers into 'likely to churn' vs 'retained' to enable proactive intervention by relationship managers.
- **Core Methodology:** Engineered a deep Artificial Neural Network (ANN). To prevent overfitting on tabular data, the architecture heavily utilized `Dropout` layers and `Batch Normalization`.
- **Optimization Strategy:** In churn prediction, false negatives (missing a churning customer) are costlier than false positives. Therefore, the model threshold was specifically tuned for **Recall**, successfully capturing 75% of actual churning customers.

*(TensorFlow/Keras model architecture, training history plots, and confusion matrix will be added here).*

---

## 1. Exploratory Data Analysis (EDA) & Business Insights

Before building the neural network, thorough EDA was conducted on the 10,000-record dataset to extract baseline behavioral insights:

- **Geographic Risk:** Customers from Germany exhibited a significantly higher churn rate compared to those in France and Spain, indicating a need for localized retention strategies.
- **Age Dynamics:** Younger customers (20-40) have lower churn rates, while customers aged 45+ show a distinctly higher probability of exiting.
- **The "Product Sweet Spot":** Customers holding exactly 2 products showed the lowest churn. Customers with only 1 product, or 3-4 products, were highly likely to churn.
- **The Balance Paradox:** Churned customers actually tended to have a higher median account balance, proving that financial capacity does not guarantee customer satisfaction or loyalty.

---

## 2. Data Engineering & Preprocessing

Deep learning models require strictly numerical, scaled, and noise-free inputs to allow gradient descent to converge properly.

- **Dimensionality Reduction:** Dropped identifiers with zero predictive power (`RowNumber`, `CustomerId`, `Surname`).
- **Outlier Treatment:** Applied the Interquartile Range (IQR) method to remove extreme outliers in `CreditScore` (15 removed) and `Age` (359 removed).
- **Categorical Encoding:** - `Gender` was Label Encoded (Male: 1, Female: 0).
  - `Geography` was One-Hot Encoded, systematically dropping the first column to prevent multicollinearity.
- **Feature Scaling:** Applied `StandardScaler` to all continuous numerical columns (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`) to stabilize the neural network's weight updates.

---

## 3. ANN Architecture & Iterative Optimization

The modeling phase was approached iteratively, establishing a baseline and systematically tuning the architecture to resolve performance bottlenecks.

### Iteration 1: Baseline SGD Model
- **Structure:** 64-node Input Layer $\rightarrow$ 32-node Hidden Layer $\rightarrow$ 1-node Output Layer (Sigmoid).
- **Result:** ~86% Accuracy, but a poor Recall of **0.49**. The model was missing 51% of actual churners due to the severe class imbalance (7963 non-churners vs. 2037 churners).

### Iteration 2 & 3: Adam Optimizer & Dropout Regularization
- **Enhancement:** Switched to the `Adam` optimizer for faster convergence and added `Dropout(0.3)` layers to penalize the network for relying too heavily on specific neuron pathways, mitigating minor overfitting seen in later epochs.
- **Result:** Improved stability and slightly better precision, but Recall remained stagnant around 0.49.

### Iteration 4: Hyperparameter Tuning (GridSearchCV)
- **Methodology:** Wrapped the Keras model using `scikeras` and performed a grid search across `batch_size`, `epochs`, `learning_rate`, and node counts.
- **Optimal Params:** 128 neurons (Layer 1), 32 neurons (Layer 2), Learning Rate of 0.001, Batch Size 64.
- **Result:** AUC improved slightly to 0.8547, and Recall crept up to **0.54**, but the fundamental class imbalance was still holding the model back.

---

## 4. The Breakthrough: SMOTE & Recall Optimization

Because the business cost of a False Negative (missing a churner) is vastly higher than a False Positive (sending a retention email to a loyal customer), the model's primary evaluation criterion was shifted entirely to **Recall**.

To fix the structural data deficit, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training set, balancing the classes to exactly 6,141 churners and 6,141 non-churners.

### Final SMOTE Model Performance:
- **Recall for Churn (Class 1):** Skyrocketed to **0.75** (up from 0.49). The model now successfully identifies 3 out of every 4 customers planning to leave.
- **Overall Accuracy:** Dropped intentionally to 80%.
- **ROC-AUC Score:** Remained highly discriminative at 0.8515.
- **Business Trade-off:** Precision dropped to 0.51, meaning false positives increased. However, in the context of retail banking, the ROI of catching 75% of actual flight-risk accounts completely justifies the minimal cost of over-communicating with stable accounts.

---

## 5. Interview Talking Points: Quantitative HR & CRM Strategy

- **Metric Alignment:** Successfully argued against using raw "Accuracy" as a success metric in highly imbalanced datasets. Demonstrated the business value of sacrificing overall accuracy (dropping from 86% to 80%) to achieve a massive 26-point jump in actionable Recall.
- **Algorithmic Iteration:** Showcased a structured engineering pipeline: starting with a simple SGD baseline, moving to Adam, combating overfitting with Dropout, tuning via GridSearch, and ultimately solving the core data problem via SMOTE.
- **Actionable Triggers:** The model transforms from a theoretical exercise into a live CRM tool; any customer crossing the predictive threshold can instantly trigger an automated retention workflow (e.g., a targeted rate-reduction offer for high-balance German customers).