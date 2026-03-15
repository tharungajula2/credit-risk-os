---
title: "Bank Churn Prediction using Neural Networks"
date: "2026-03-15"
summary: "Built an Artificial Neural Network classifying bank customer churn, optimizing thresholds for 75% recall."
tags: ["Python", "TensorFlow", "Keras", "Deep Learning"]
---

# Bank Customer Churn Prediction

Customer acquisition is significantly more expensive than retention. This project leverages Deep Learning to identify at-risk bank customers before they leave the institution.

## Project Architecture
- **Objective:** Classify banking customers into 'likely to churn' vs 'retained' to enable proactive intervention by relationship managers.
- **Core Methodology:** Engineered a deep Artificial Neural Network (ANN). To prevent overfitting on tabular data, the architecture heavily utilized `Dropout` layers and `Batch Normalization`.
- **Optimization Strategy:** In churn prediction, false negatives (missing a churning customer) are costlier than false positives. Therefore, the model threshold was specifically tuned for **Recall**, successfully capturing 75% of actual churning customers.

*(TensorFlow/Keras model architecture, training history plots, and confusion matrix will be added here).*