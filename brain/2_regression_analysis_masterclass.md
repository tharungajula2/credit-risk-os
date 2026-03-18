
---
title: "Regression Analysis Masterclass — From Linear Regression to Logistic Regression and Statistical Testing"
date: 2026-03-19
tags:
  - regression
  - linear-regression
  - multiple-linear-regression
  - logistic-regression
  - OLS
  - MLE
  - residuals
  - assumptions
  - R-squared
  - adjusted-r-squared
  - regularization
  - lasso
  - ridge
  - elastic-net
  - ANOVA
  - chi-squared
  - VIF
  - multicollinearity
  - hypothesis-testing
  - data-science
  - statistics
  - beginner-friendly
cluster: "01 — Foundations"
progress: 0
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_lending_club_credit_risk_masterclass]]"
  - "[[5_bank_churn_neural_networks_masterclass]]"
  - "[[6_employee_retention_tree_models_masterclass]]"
  - "[[7_socio_economic_household_classification_masterclass]]"
  - "[[8_twitter_sentiment_nlp_masterclass]]"
---

---

# Regression Analysis Masterclass — From Linear Regression to Logistic Regression and Statistical Testing

> This note is my beginner-friendly master note for regression analysis and the statistical toolkit around it. I am using it as a foundation note inside my brain system so that I can understand how data science moves from raw relationships to prediction, interpretation, hypothesis testing, and business communication.
>
> My goal here is not only to memorize definitions. I want to understand what each method is trying to do, what math sits behind it, what assumptions I need to check, what Python tools are commonly used, and how these ideas connect back to the projects I already know.

---

## The Note at a Glance

This note gives me one connected path through the regression ecosystem:

```text
Business question
→ choose outcome variable
→ choose predictor variables
→ inspect relationships
→ build a regression model
→ check assumptions
→ evaluate fit
→ avoid overfitting
→ interpret coefficients carefully
→ communicate findings without claiming causation falsely
```

It also helps me understand why the regression family is so important:

- **simple linear regression** teaches me how one variable relates to another
- **multiple linear regression** teaches me how many predictors work together
- **logistic regression** teaches me how classification models probabilities
- **ANOVA and chi-squared tests** teach me how to compare groups and categories
- **regularization** teaches me how to stop models from becoming too noisy

So this is not just one topic. It is one of the core languages of data science.

---

## Why This Note Matters for My Other Notes

Regression analysis is not isolated theory. It shows up directly across my project notes:

- in [[1_lending_club_credit_risk_masterclass]], logistic regression is the core PD model
- in [[6_employee_retention_tree_models_masterclass]], logistic regression is one of the baseline classifiers I compare against tree models
- in [[5_bank_churn_neural_networks_masterclass]], evaluation concepts like precision, recall, thresholding, and ROC-AUC connect back to logistic classification logic
- in [[7_socio_economic_household_classification_masterclass]], multicollinearity, feature selection, and regularization ideas help me think more clearly about large tabular datasets
- in [[8_twitter_sentiment_nlp_masterclass]], the binary and multiclass classification mindset is an extension of the same prediction logic, even though the features come from text

This note gives me the statistics foundation underneath those projects.

---

# Part 1 — What Regression Is Actually Trying to Do

## 1.1 The core goal

Regression tries to estimate the relationship between:

- a **dependent variable**  
  the outcome I care about

and

- one or more **independent variables**  
  the predictors I think help explain that outcome

Examples:

- predicting **house price** from square footage
- predicting **salary** from years of experience and education
- predicting **default probability** from borrower characteristics
- predicting whether a customer will **churn** based on account behaviour

The basic idea is always the same:

> I want to use information in one set of variables to explain or predict another variable.

---

## 1.2 Regression is part of a larger workflow

A useful way to frame regression is through the common data workflow:

### Plan
- define the business problem
- identify the outcome variable
- decide what predictors might matter

### Analyze
- explore the data
- visualize relationships
- check variable types
- inspect missing values and outliers

### Construct
- build the model
- estimate coefficients
- check assumptions
- compare model variants

### Execute
- communicate results
- explain uncertainty
- translate coefficients into business meaning
- avoid overstating what the model can prove

That matters because regression is not just a formula. It is one step inside a full analytical process.

---

# Part 2 — Simple Linear Regression

## 2.1 What simple linear regression means

**Simple Linear Regression (SLR)** models the relationship between:

- one continuous dependent variable
- one continuous independent variable

The model is:

\[
y = \beta_0 + \beta_1 x + \varepsilon
\]

Where:

- \(y\) = outcome variable
- \(x\) = predictor
- \(\beta_0\) = intercept
- \(\beta_1\) = slope
- \(\varepsilon\) = error term

### What the coefficients mean

- **Intercept \(\beta_0\)**  
  the predicted value of \(y\) when \(x = 0\)

- **Slope \(\beta_1\)**  
  the expected change in \(y\) for a one-unit increase in \(x\)

If \(\beta_1\) is positive, the relationship moves upward.  
If \(\beta_1\) is negative, the relationship moves downward.

---

## 2.2 Correlation is not causation

This is one of the most important things to remember.

Two variables can move together without one causing the other.

Examples:

- ice cream sales and drowning incidents may both rise in summer
- credit utilization and default may move together, but that does not automatically prove utilization alone causes default
- marketing spend and sales may move together, but other drivers may also be involved

Regression can show **association**.  
It does not automatically prove **causation**.

That is why I should be careful in interviews and business communication.

---

## 2.3 The math engine: Ordinary Least Squares (OLS)

Simple linear regression usually estimates coefficients using **Ordinary Least Squares (OLS)**.

### The key idea

For every observation, the model makes a prediction:

\[
\hat{y}_i = \beta_0 + \beta_1 x_i
\]

The difference between the actual value and the predicted value is the **residual**:

\[
e_i = y_i - \hat{y}_i
\]

OLS chooses the line that minimizes the **sum of squared residuals**:

\[
SSR = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

Why square them?

- positive and negative errors do not cancel
- larger errors get penalized more heavily
- the math becomes differentiable and easy to optimize

### Intuition

OLS is finding the line that leaves the smallest overall squared prediction error.

---

## 2.4 Residuals and error

A **residual** is the observed error in my sample:

\[
\text{Residual} = \text{Actual} - \text{Predicted}
\]

Residuals matter because they help me check whether the model is behaving properly.

If the model is appropriate, residuals should look like random noise, not like a hidden pattern.

---

## 2.5 The main assumptions of simple linear regression

### 1. Linearity
The relationship between \(x\) and \(y\) should be approximately linear.

How I check it:
- scatter plot
- fitted line plot

If the points follow a curve instead of a line, simple linear regression may not be appropriate.

### 2. Normality of residuals
The residuals should be approximately normally distributed.

How I check it:
- histogram of residuals
- Q-Q plot

This matters mainly for inference, confidence intervals, and hypothesis testing.

### 3. Independent observations
One observation should not depend on another.

Examples where this can fail:
- repeated measurements on the same customer
- time-series dependence
- grouped data without adjustment

### 4. Homoscedasticity
The residual variance should be roughly constant across the fitted range.

In plain English:
the spread of errors should stay fairly even, not widen into a cone shape.

How I check it:
- residuals vs fitted plot

If the spread increases with the prediction level, I may have **heteroscedasticity**.

---

# Part 3 — Multiple Linear Regression

## 3.1 Why simple linear regression is usually not enough

Real business outcomes rarely depend on only one variable.

For example, salary may depend on:
- education
- years of experience
- industry
- role
- geography

That leads to **Multiple Linear Regression (MLR)**:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \varepsilon
\]

Now each coefficient measures the effect of one predictor **while holding the others constant**.

That phrase matters a lot.

---

## 3.2 How to interpret coefficients in multiple regression

Suppose:

\[
\text{Salary} = \beta_0 + \beta_1(\text{Experience}) + \beta_2(\text{Education Years}) + \varepsilon
\]

Then:

- \(\beta_1\) tells me how salary changes with experience, **after accounting for education**
- \(\beta_2\) tells me how salary changes with education, **after accounting for experience**

This is why multiple regression is more realistic than a one-variable model.

---

## 3.3 Handling categorical variables

Regression equations need numeric inputs.  
But many business variables are categorical:

- city
- state
- product type
- grade
- home ownership

To include them, I convert them into **dummy variables** or **one-hot encoded columns**.

### Example

If `City` has:
- Hyderabad
- Bengaluru
- Pune

I create:

- `City_Bengaluru`
- `City_Pune`

and leave one category out as the **reference category**.

Why leave one out?

Because if I include all categories, I create perfect multicollinearity.

This idea connects directly to my Lending Club note, where I drop one risk bucket as the reference category before fitting logistic regression.

---

## 3.4 Interaction terms

Sometimes the effect of one variable depends on another.

Example:
the effect of experience on salary may differ by industry.

Then I can create an interaction:

\[
x_1 \times x_2
\]

This lets the model say:

> the impact of one predictor changes when another predictor changes.

That is a very important modeling idea.

---

## 3.5 The fifth assumption: no severe multicollinearity

In multiple regression, predictors should not be too highly correlated with each other.

Why this is a problem:

- coefficient estimates become unstable
- standard errors inflate
- interpretation becomes unreliable

### The usual diagnostic: VIF

**Variance Inflation Factor (VIF)** measures how much the variance of a coefficient is inflated because of collinearity.

A common interpretation:

- VIF near 1 → little collinearity
- VIF above 5 → worth investigating
- VIF above 10 → serious problem in many practical settings

This concept matters a lot in structured tabular modeling.

---

# Part 4 — Model Fit, Overfitting, and Selection

## 4.1 R-squared

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

Where:

- \(SS_{res}\) = residual sum of squares
- \(SS_{tot}\) = total sum of squares

### What it means

\(R^2\) tells me the proportion of variation in the dependent variable explained by the model.

Examples:

- \(R^2 = 0.70\) means the model explains 70% of the variation in the outcome
- \(R^2 = 0.20\) means it explains much less

### Important caution

A high \(R^2\) does not automatically mean:
- the model is causal
- the model is unbiased
- the model will generalize well

---

## 4.2 Adjusted R-squared

Standard \(R^2\) never decreases when I add more predictors.

That is dangerous because useless variables can make the model look better.

**Adjusted \(R^2\)** corrects for this by penalizing unnecessary complexity.

So if I add a useless variable:
- ordinary \(R^2\) may go up
- adjusted \(R^2\) may stay flat or go down

This makes adjusted \(R^2\) better for comparing models with different numbers of predictors.

---

## 4.3 The overfitting problem

A model can fit the training data very well but still perform poorly on new data.

That is **overfitting**.

### Underfitting vs overfitting

- **Underfitting**  
  model is too simple and misses real structure

- **Overfitting**  
  model is too complex and memorizes noise

The goal is not to maximize training fit.  
The goal is to generalize well.

---

## 4.4 Bias-variance tradeoff

This is one of the most important big-picture ideas in modeling.

- **High bias** → model is too rigid and misses structure
- **High variance** → model is too sensitive to noise

A good model balances both.

This tradeoff appears everywhere:
- regression
- tree models
- neural networks
- ensemble methods

---

## 4.5 Holdout validation

To test whether a model generalizes, I usually split the data into:

- **training set**
- **test set**

The model learns on training data and gets evaluated on unseen test data.

This matters because a model should be judged on data it has not already seen.

That same idea appears again in my classification projects.

---

## 4.6 Variable selection strategies

### Forward selection
Start with no variables, then add the best one at each step.

### Backward elimination
Start with all variables, then remove the weakest one at each step.

### Stepwise approaches
Move forward and backward based on model criteria.

These can be useful for learning, but in real work I should also use domain knowledge, stability logic, and validation performance instead of blindly trusting automated selection.

---

## 4.7 Regularization

Regularization adds a penalty term to prevent coefficients from becoming too large and unstable.

### Ridge Regression (L2)
Penalizes squared coefficient size.

- shrinks coefficients toward zero
- keeps all variables in the model
- useful when many predictors are correlated

### Lasso Regression (L1)
Penalizes absolute coefficient size.

- can shrink some coefficients exactly to zero
- acts like automatic feature selection

### Elastic Net
Combines L1 and L2 penalties.

- useful when I want a compromise between lasso and ridge

These methods are extremely important because they link statistics and machine learning.

---

# Part 5 — Hypothesis Testing for Categorical and Group Questions

Regression is not the whole toolkit.  
Sometimes the question is not “predict a number,” but:

- do these categories differ?
- are these variables related?
- are these group means significantly different?

That is where the following tools matter.

---

## 5.1 Chi-squared test

### Goodness of fit
Checks whether the observed categorical distribution matches an expected theoretical distribution.

### Test of independence
Checks whether two categorical variables are associated or independent.

Example:
- is churn independent of geography?
- is default independent of loan grade?

This is useful when both variables are categorical.

---

## 5.2 ANOVA

**ANOVA = Analysis of Variance**

It tests whether the means of multiple groups differ significantly.

### One-way ANOVA
One categorical grouping variable, one continuous outcome.

Example:
- do average salaries differ across 3 departments?

### Two-way ANOVA
Two grouping variables.

Example:
- do average outcomes differ by department and gender together?

ANOVA tells me whether at least one group differs.  
It does not tell me which one differs.

---

## 5.3 Tukey's HSD

After ANOVA finds a significant difference, I use **Tukey's Honest Significant Difference** test to identify which specific group pairs differ.

So the sequence is:

```text
ANOVA says: "At least one mean is different"
Tukey says: "These exact pairs are different"
```

---

## 5.4 ANCOVA, MANOVA, and MANCOVA

### ANCOVA
ANOVA plus continuous covariates.

This helps compare groups while controlling for another continuous variable.

### MANOVA
Multiple dependent variables analyzed together.

### MANCOVA
Multiple dependent variables plus covariates.

These are more advanced, but it is good for me to know where they fit in the family.

---

# Part 6 — Logistic Regression: When the Outcome Is Yes/No

This is the most important bridge section for my other notes.

Linear regression predicts continuous numbers.

But many real problems are binary:

- default vs non-default
- churn vs no churn
- fraud vs non-fraud
- approved vs declined

That is where **logistic regression** comes in.

---

## 6.1 Why linear regression is not suitable for binary targets

Suppose I try to predict a binary variable using a straight line.

Problems:
- predictions can fall below 0
- predictions can exceed 1
- errors are not normally distributed in the same way
- the relationship is not really linear in probability space

So I need a model that keeps outputs between 0 and 1.

---

## 6.2 The logistic function

Logistic regression maps any real-valued score into a probability using the sigmoid function:

\[
p = \frac{1}{1 + e^{-z}}
\]

Where:

\[
z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
\]

This creates an S-shaped curve.

That is why logistic regression can model probabilities.

---

## 6.3 Log-odds and the logit link

Instead of modeling probability directly as a straight line, logistic regression models the **log-odds**:

\[
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
\]

This is called the **logit link**.

### Why this matters

- the left side can take any real value
- the right side is linear in the predictors
- converting back gives me a valid probability

This section is crucial for my Lending Club PD scorecard note.

---

## 6.4 The estimation engine: Maximum Likelihood Estimation (MLE)

Linear regression uses OLS.

Logistic regression usually uses **Maximum Likelihood Estimation (MLE)**.

### The idea

MLE chooses the coefficients that make the observed outcomes most likely under the model.

So instead of minimizing squared error, I maximize the probability of the observed labels.

That is the right estimation framework for binary outcomes.

---

## 6.5 Confusion matrix

Once I turn probabilities into class labels using a threshold, I can summarize the results in a **confusion matrix**.

For a positive class:

- **TP** = true positive
- **TN** = true negative
- **FP** = false positive
- **FN** = false negative

This matrix is the base for classification metrics.

---

## 6.6 Precision, recall, accuracy, and F1 score

### Precision
Out of the cases I predicted as positive, how many were truly positive?

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

### Recall
Out of the actual positive cases, how many did I catch?

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

### Accuracy
Out of all cases, how many did I classify correctly?

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

### F1 score
The harmonic mean of precision and recall:

\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

This matters when I want a balance between precision and recall.

### Why these matter differently by problem

- in **credit risk**, missing bad borrowers can be costly, so recall for the bad class can matter a lot
- in **churn**, I may care about recall to catch more likely exits
- in **fraud**, I may care about precision to reduce wasteful investigations
- in imbalanced datasets, accuracy alone can be misleading

This is why metric choice must match the business decision.

---

## 6.7 ROC curve and AUC

The **ROC curve** plots:

- **True Positive Rate**  
  same as recall

against

- **False Positive Rate**

across many thresholds.

The **AUC** is the area under that curve.

### Intuition

AUC tells me how well the model separates positives from negatives across all thresholds.

- 0.50 = random guessing
- closer to 1.00 = stronger discrimination

This is one of the most common threshold-independent evaluation metrics.

It directly connects to my credit-risk and churn notes.

---

# Part 7 — Python Tools I Need to Know

The common Python ecosystem for regression and statistical analysis includes:

- **pandas**  
  data loading, cleaning, reshaping

- **numpy**  
  numerical arrays and linear-algebra style calculations

- **matplotlib / seaborn**  
  scatter plots, residual plots, distributions, diagnostics

- **statsmodels**  
  OLS summaries, p-values, ANOVA tables, inference-heavy output

- **scikit-learn**  
  train/test split, preprocessing, regularized regression, logistic regression, model evaluation

A very common real workflow is:

- use **pandas** for preparation
- use **seaborn / matplotlib** for EDA and assumption checks
- use **statsmodels** when I want full statistical summaries
- use **scikit-learn** when I want pipeline-based prediction workflows

---

# Part 8 — How I Should Explain Regression in Interviews

A strong interview answer should show that I understand both the math and the business interpretation.

## What I should say clearly

### For linear regression
- it estimates the relationship between a continuous outcome and predictors
- OLS chooses coefficients by minimizing squared residuals
- I need to check assumptions such as linearity, independence, normality of residuals, and homoscedasticity
- coefficient interpretation depends on the presence of other variables and reference categories

### For multiple regression
- each coefficient is interpreted holding the others constant
- multicollinearity can distort interpretation
- adjusted \(R^2\) is more useful than raw \(R^2\) for comparing models of different sizes

### For logistic regression
- it is used for binary outcomes
- it models log-odds and converts them into probabilities through the logistic function
- it is estimated by maximum likelihood, not OLS
- evaluation depends on the problem, so I should discuss precision, recall, F1, and ROC-AUC in business context

### For communication
- I should describe relationships carefully
- I should not claim causality without the right research design
- I should connect metrics and thresholds to the real business cost of mistakes

---

# Part 9 — The Full Regression Family in One Connected View

```text
Continuous target
        │
        ├── One predictor
        │      └── Simple Linear Regression
        │
        └── Many predictors
               └── Multiple Linear Regression
                         │
                         ├── check assumptions
                         ├── handle categorical variables
                         ├── inspect multicollinearity
                         ├── evaluate with R² / Adjusted R²
                         └── regularize if needed

Categorical / group comparison
        │
        ├── Chi-Squared
        ├── ANOVA
        ├── Tukey HSD
        ├── ANCOVA
        └── MANOVA / MANCOVA

Binary target
        │
        └── Logistic Regression
                 │
                 ├── logit link
                 ├── MLE
                 ├── confusion matrix
                 ├── precision / recall / F1
                 └── ROC-AUC
```

---

## Connections to the Rest of My Notes

- [[1_lending_club_credit_risk_masterclass]] — the logistic-regression section in this note is the statistical foundation for the PD scorecard
- [[5_bank_churn_neural_networks_masterclass]] — the classification metrics here help me understand churn evaluation properly
- [[6_employee_retention_tree_models_masterclass]] — this note helps me compare linear probability thinking with tree-based nonlinearity
- [[7_socio_economic_household_classification_masterclass]] — regularization, variable selection, and multicollinearity help me think more clearly about large tabular data
- [[8_twitter_sentiment_nlp_masterclass]] — the logistic-classification mindset extends naturally to text classification after feature extraction

---

## Key Concepts Summary

| Concept | What it means | Clean takeaway |
|---|---|---|
| **Dependent Variable** | Outcome being predicted | The target I want to explain |
| **Independent Variable** | Predictor used to explain the outcome | Input feature |
| **Residual** | Actual minus predicted value | Error left after fitting |
| **OLS** | Minimizes squared residuals | Core engine of linear regression |
| **Linearity** | Relationship is approximately straight-line | First assumption to inspect |
| **Homoscedasticity** | Error variance stays roughly constant | Residual spread should stay even |
| **Multicollinearity** | Predictors are too correlated with each other | Makes coefficient interpretation unstable |
| **VIF** | Measures collinearity severity | Higher means more inflation |
| **R-squared** | Share of outcome variation explained | Fit summary, not proof of quality |
| **Adjusted R-squared** | R-squared penalized for useless variables | Better for comparing models |
| **Overfitting** | Model memorizes noise | Strong training fit, weak generalization |
| **Bias-Variance Tradeoff** | Simplicity vs flexibility balance | Central idea in all modeling |
| **Ridge** | L2 regularization | Shrinks coefficients, keeps all variables |
| **Lasso** | L1 regularization | Can shrink some coefficients to zero |
| **Elastic Net** | Mix of L1 and L2 | Hybrid regularization |
| **Chi-Squared** | Test for categorical distributions / relationships | Used for categories, not continuous means |
| **ANOVA** | Test of group mean differences | Detects whether at least one group differs |
| **Logistic Regression** | Binary-outcome regression model | Outputs probabilities |
| **Logit** | Log-odds transformation | Makes binary probabilities modelable |
| **MLE** | Fits logistic-regression coefficients | Maximizes likelihood, not squared-error fit |
| **Precision** | Correct positives among predicted positives | Useful when false positives are costly |
| **Recall** | Found positives among actual positives | Useful when false negatives are costly |
| **F1 Score** | Balance between precision and recall | Useful in imbalanced settings |
| **ROC-AUC** | Threshold-independent discrimination metric | Higher means better ranking ability |

---

*This note is my regression-analysis foundation note. If I understand this note clearly, then many of the ideas in my credit-risk and machine-learning notes become easier to explain from first principles.*
