---
title: "IRB Credit Scorecard & PD Modeling"
date: "2026-03-19"
summary: "Developed & validated PD/EAD/LGD models with strict KS/AUC validation and PSI stability monitoring."
tags: ["Python", "XGBoost", "Scikit-learn", "SQL", "Credit Risk"]
cluster: "Quantitative Portfolio"
---

# AI-Powered Credit Risk Modeling (PD Scorecard)

In the highly regulated domain of credit risk, black-box models are often insufficient. [cite_start]This project focuses on building a highly interpretable, regulatory-compliant application scorecard to predict the Probability of Default (PD) for lending decisions[cite: 8, 30].

## Project Architecture
- [cite_start]**Objective:** Optimize credit risk assessment and automate lending decisions based on applicant data[cite: 30].
- [cite_start]**Core Methodology:** Engineered a robust Logistic Regression pipeline, transforming raw features using Weight of Evidence (WOE) to handle non-linearities and missing data smoothly[cite: 31, 58].
- [cite_start]**Validation:** Ensured model robustness using industry-standard metrics, including strict KS (Kolmogorov-Smirnov) and AUC (Area Under the Curve) validation[cite: 145, 149]. 
- [cite_start]**Monitoring:** Implemented Population Stability Index (PSI) tracking to detect data drift over time[cite: 8, 261].

*(Jupyter Notebook code blocks for data preprocessing, WOE transformation, and model fitting will be added here).*

---

## 1. Data Engineering & Setup

[cite_start]The foundation of the framework relies on a granular, loan-level dataset (similar to Lending Club data) containing borrower information (grade, income, DTI), loan characteristics, and credit history[cite: 6, 13, 17].

- **Good/Bad Definition:** The dependent variable is strictly defined. 
  - [cite_start]**Bad (0):** Charged Off, Default, Late 31-120 days[cite: 25, 52].
  - [cite_start]**Good (1):** Fully Paid, Current, In Grace Period, Late 16-30 days[cite: 25, 51].
- [cite_start]**Preprocessing & Imputation:** Missing credit history values were zero-filled, while annual income utilized mean imputation[cite: 19].
- **Feature Engineering:**
  - [cite_start]Categorical variables underwent one-hot encoding (dummy variables)[cite: 21].
  - [cite_start]Continuous variables (Income, DTI, Interest Rate) were binned into categories to capture non-linear relationships[cite: 22, 355].
  - [cite_start]Created time-based features (e.g., `mths_since_issue_d`)[cite: 23].
- [cite_start]**Multicollinearity Prevention:** Reference categories for dummy variables were systematically excluded from the models[cite: 27, 360].

---

## 2. Probability of Default (PD) Modeling

[cite_start]The PD model estimates the likelihood of a borrower defaulting, serving as the core engine for credit decisioning and pricing[cite: 30]. [cite_start]The model utilizes Logistic Regression with custom p-value calculations (Fisher Information Matrix)[cite: 31].

### The Log-Odds Equation

$$\ln \left( \frac{\text{Non-defaults}}{\text{Defaults}} \right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_m X_m$$

### Feature Selection via WoE and IV

[cite_start]Interpretability is critical for regulatory compliance[cite: 49]. [cite_start]Instead of raw inputs, features were transformed using **Weight of Evidence (WoE)** to evaluate how well each bin explains the dependent variable[cite: 58].

$$\text{WoE}_i = \ln \left( \frac{\% \text{good}_i}{\% \text{bad}_i} \right)$$

[cite_start]**Information Value (IV)** was then used to systematically select variables with strong predictive power[cite: 63, 64]:

$$\text{IV} = \sum_{i=1}^{k} \left[ (\% \text{good}_i - \% \text{bad}_i) \times \ln \left( \frac{\% \text{good}_i}{\% \text{bad}_i} \right) \right]$$

| Information Value ($IV$) | Predictive Power |
| :--- | :--- |
| $< 0.02$ | No predictive power |
| $0.02 - 0.1$ | Weak predictive power |
| $0.1 - 0.3$ | Medium predictive power |
| $0.3 - 0.5$ | Strong predictive power |
| $> 0.5$ | Suspiciously high, too good to be true |

[cite_start]*Result:* The final model retained 85 statistically significant features (p-value < 0.05)[cite: 69, 70, 71].

---

## 3. The Credit Scorecard Derivation

[cite_start]To make the logistical output actionable for business users, the model coefficients were scaled into a standard FICO-like scorecard ranging from **300 to 850**[cite: 194, 386].

$$\text{Score} = \left( \frac{\text{coef} - \text{min\_sum\_coef}}{\text{max\_sum\_coef} - \text{min\_sum\_coef}} \right) \times (\text{max\_score} - \text{min\_score}) + \text{min\_score}$$

[cite_start]An individual's final credit score is the weighted sum of their feature scores, which can mathematically map back to their exact Probability of Default via logistic transformation[cite: 195, 196, 197].

---

## 4. LGD & EAD Modeling

[cite_start]To complete the ECL equation, models for Loss Given Default (LGD) and Exposure at Default (EAD) were developed using the same feature set[cite: 285, 298].

### Loss Given Default (LGD)
[cite_start]LGD represents the percentage of exposure lost upon default ($1 - \text{Recovery Rate}$)[cite: 286, 377]. [cite_start]A **two-stage approach** was implemented to handle zero-inflated recovery data[cite: 287].
1. [cite_start]**Stage 1 (Logistic):** Predicts if recovery will be $> 0$ or $= 0$[cite: 288].
2. [cite_start]**Stage 2 (Linear):** Predicts the exact recovery rate for loans where recovery $> 0$[cite: 289].
3. [cite_start]**Final LGD:** $\text{Predicted Recovery} = P(\text{recovery}>0) \times \text{Predicted\_Recovery\_Amount}$ (Capped between 0 and 1)[cite: 292, 297].

### Exposure at Default (EAD)
[cite_start]EAD represents the outstanding amount at the time of default[cite: 307].
- [cite_start]**Calculation:** Modeled via a single-stage linear regression predicting the Credit Conversion Factor (CCF)[cite: 300, 301].
- [cite_start]**Formula:** $\text{EAD} = \text{CCF} \times \text{funded\_amount}$ (CCF capped between 0 and 1)[cite: 300, 306].

---

## 5. Expected Credit Loss (ECL) & Portfolio Aggregation

[cite_start]Aligning with Basel and IFRS 9 regulatory frameworks, Expected Credit Loss is calculated at both the individual loan and portfolio levels to drive capital reserve planning[cite: 308, 327, 347].

$$\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$$

**Business Insights Extracted:**
- [cite_start]**Total Expected Loss:** The sum of EL across all portfolio loans[cite: 319].
- [cite_start]**Expected Loss Rate:** $\text{Total EL} / \text{Total Funded Amount}$[cite: 321].
- [cite_start]**Risk-Based Pricing:** Adjusting interest rates proportionally to a segment's calculated PD/LGD/EAD[cite: 325, 364].

---

## 6. Model Validation & Performance Metrics

[cite_start]The PD model was rigorously evaluated out-of-sample (on the 20% test split) using the following metrics[cite: 14, 144, 353]:

- **AUROC (Area Under ROC Curve):** Measures discriminatory power. [cite_start]A higher AUROC indicates better ranking of good vs. bad loans[cite: 145, 173].
- **Gini Coefficient:** Derived as $(2 \times \text{AUROC}) - 1$. [cite_start]Quantifies inequality in the distribution[cite: 146, 385].
- **KS Statistic (Kolmogorov-Smirnov):** Measures the maximum vertical distance between the cumulative distribution of Good and Bad borrowers. [cite_start]It dictates how well the model separates the two populations[cite: 149, 182, 383].
- [cite_start]**Confusion Matrix & Cutoff Analysis:** Evaluated True Positive/False Positive rates at various threshold probabilities to optimize approval/rejection rates[cite: 150, 166, 168].

| AUROC Range | Discriminatory Power |
| :--- | :--- |
| $50\% - 60\%$ | Bad |
| $60\% - 70\%$ | Poor |
| $70\% - 80\%$ | Fair |
| $80\% - 90\%$ | Good |
| $90\% - 100\%$ | Excellent |

---

## 7. Model Monitoring & Population Stability Index (PSI)

[cite_start]To ensure the model remains robust in production, **Population Stability Index (PSI)** was implemented to monitor data drift by comparing new loan distributions against the original training baseline[cite: 259, 260, 261].

$$\text{PSI} = \sum_{j=1}^{k} \left[ (\% \text{actual}_j - \% \text{expected}_j) \times \ln \left( \frac{\% \text{actual}_j}{\% \text{expected}_j} \right) \right]$$

**PSI Thresholds & Action Plan:**

| PSI Value | Interpretation & Action |
| :--- | :--- |
| $< 0.1$ | Little to no difference; [cite_start]Model is stable[cite: 266, 277]. |
| $0.1 - 0.25$ | Moderate drift; [cite_start]Monitor closely (No immediate action)[cite: 266, 278]. |
| $> 0.25$ | Significant drift; [cite_start]Trigger model review and recalibration[cite: 266, 279]. |

---

## 8. Interview Talking Points: Business Impact

- [cite_start]**End-to-End Pipeline:** Built a complete, production-ready credit risk pipeline from raw data engineering to ECL calculation and drift monitoring[cite: 331, 332].
- [cite_start]**Regulatory Alignment:** The framework perfectly aligns with Basel Capital Accords and IFRS 9 provisioning requirements by separating PD, LGD, and EAD[cite: 346, 347].
- [cite_start]**Real-World Utility:** The output is not just a probability; it is an actionable 300-850 application scorecard utilized for automated decisioning and risk-adjusted pricing[cite: 361, 363, 364].