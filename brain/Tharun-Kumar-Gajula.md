---
title: "Tharun Kumar Gajula"
date: 2026-03-17
tags:
  - profile
  - MOC
  - credit-risk
  - model-development
  - model-validation
  - portfolio-monitoring
  - quantitative-analytics
  - data-science
  - banking
  - machine-learning
cluster: "System Atlas"
links:
  - "[[1_full_pd_model]]"
  - "[[2_monitoring_model]]"
  - "[[3_lgd_ead_model_rewritten]]"
  - "[[4_ecl_cecl_stress_testing_rewritten]]"
  - "[[5_bank_churn_neural_networks_masterclass]]"
  - "[[6_employee_retention_tree_models_masterclass]]"
  - "[[7_socio_economic_household_classification_masterclass]]"
  - "[[8_twitter_sentiment_nlp_masterclass]]"
---

---

# Tharun Kumar Gajula

> This note is the master map of my note system. It explains who I am, what I am building, how the eight core notes connect, and how my background in banking, risk analytics, business analysis, and machine learning comes together as one coherent direction.
>
> I use this vault as both a learning system and a professional system. It helps me understand concepts from first principles, connect them to project work, and keep the big picture clear: how data becomes a model, how a model becomes a business decision, and how that decision must be monitored, validated, and governed.

---

## My Direction in One View

I work at the intersection of **credit risk**, **quantitative analytics**, **model development**, **model validation**, and **practical banking workflows**.

My background is not limited to one narrow lane. It sits across three layers that matter a lot in real institutions:

1. **Banking and risk context** — understanding lending portfolios, default risk, monitoring, provisioning logic, and the way risk decisions are actually used.
2. **Analytical and modeling depth** — using Python, SQL, statistics, machine learning, and model-evaluation methods to convert raw data into measurable risk insight.
3. **Business and implementation thinking** — documenting workflows, translating requirements, validating systems, and making sure solutions work inside real operational environments.

That combination comes from the way my experience has evolved:

- At **Jana Small Finance Bank**, I worked close to credit-risk analytics, portfolio monitoring, reporting, and model-oriented thinking across lending portfolios.
- At **Lentra AI**, I worked on lending-system workflows, BRDs/FRDs, UAT, API validation, and the translation layer between business requirements and technical implementation.
- Through my academic path in **banking and finance** at NIBM and later **deep learning** at IISc, I built a stronger quantitative layer on top of that practical foundation.
- Through these notes and projects, I am turning that combined base into a more complete system for **credit risk modeling, validation, governance, and broader quantitative problem solving**.

This vault is the place where all of those threads are tied together.

---

## What I Bring Technically

The note system is not separate from my skill set. It reflects the tools and methods I repeatedly use and want to keep sharpening:

- **Python** for data cleaning, modeling, validation, automation, and analytical storytelling
- **SQL** for extraction, portfolio analysis, reporting logic, and data validation
- **Statistical and ML methods** such as logistic regression, scorecards, random forests, XGBoost, neural networks, and NLP pipelines
- **Risk metrics and validation tools** such as KS, AUC, Gini, PSI, cut-off analysis, and stability interpretation
- **Documentation and workflow thinking** through BRDs, FRDs, testing, process mapping, and handoff between business and technical teams

That combination is important to me because I do not want to be strong in only one layer of the work. I want to be able to move from raw data and process understanding all the way to model design, monitoring, and business interpretation.

---

## What This Vault Is Really For

This is not just a collection of isolated notes.

I built it to connect six things properly:

- **credit-risk fundamentals**
- **data science foundations**
- **math and coding behind models**
- **model development and evaluation**
- **monitoring, validation, and governance**
- **practical business use inside regulated environments**

A very important personal goal sits underneath all of this: I want to understand every major concept through projects, not as disconnected textbook fragments.

That means even when a concept is broader than a specific project, I still want to anchor it back to a project and ask:

- what the concept means
- why it matters
- what math sits behind it
- what the code is doing
- how it would be used in practice
- how I would explain it in an interview

That is why these notes are written the way they are. They are intentionally detailed, beginner-friendly, technical, and practical at the same time.

---

## The Architecture of My Brain Notes

The eight core notes form two broad clusters.

### Cluster 1 — Credit Risk, Loss Modeling, Monitoring, and Governance

These notes are the core of my professional direction in risk analytics and model validation:

- [[1_full_pd_model]]
- [[2_monitoring_model]]
- [[3_lgd_ead_model_rewritten]]
- [[4_ecl_cecl_stress_testing_rewritten]]

### Cluster 2 — Applied Machine Learning Foundations Through Projects

These notes expand my data-science depth across tabular classification, tree models, neural networks, and NLP:

- [[5_bank_churn_neural_networks_masterclass]]
- [[6_employee_retention_tree_models_masterclass]]
- [[7_socio_economic_household_classification_masterclass]]
- [[8_twitter_sentiment_nlp_masterclass]]

Together they form one connected system.

The first cluster gives me the **risk-specific spine**.
The second cluster gives me the **broader machine-learning range**.

---

## The Full Brain as a Concept Map

```text
[Tharun-Kumar-Gajula]
        │
        ├── Credit Risk Core
        │      ├── [1_full_pd_model]
        │      ├── [2_monitoring_model]
        │      ├── [3_lgd_ead_model_rewritten]
        │      └── [4_ecl_cecl_stress_testing_rewritten]
        │
        └── Applied Machine Learning Foundation
               ├── [5_bank_churn_neural_networks_masterclass]
               ├── [6_employee_retention_tree_models_masterclass]
               ├── [7_socio_economic_household_classification_masterclass]
               └── [8_twitter_sentiment_nlp_masterclass]
```

The logic of this structure is simple:

- the **credit-risk notes** show how a lending model is built, interpreted, monitored, and extended into loss estimation
- the **project notes** help me internalize the data-science concepts that appear across many domains, not only credit risk
- the profile note sits above both and explains the big picture

---

## How the Credit-Risk Core Fits Together

### [[1_full_pd_model]] — The Core Modeling Engine

This is the technical anchor of the whole system.

It covers the end-to-end build of a **Probability of Default (PD)** model using Lending Club data, including:

- target definition
- good/bad labeling
- missing-data treatment
- feature engineering
- Weight of Evidence (WoE)
- Information Value (IV)
- logistic regression
- validation metrics such as KS, AUROC, and Gini
- scorecard scaling and cut-off logic

This note matters because it ties together many of the most important ideas in risk modeling:

- how a business problem becomes a statistical target
- why preprocessing decisions change model behavior
- how linear models become probability models through the logistic function
- why scorecards remain important in regulated lending environments
- how interpretability and control matter as much as prediction

This note is where I connect statistics, business logic, and model building most directly.

---

### [[2_monitoring_model]] — The Governance and Production Layer

A model is not finished when it is trained.

This note captures the second half of the model lifecycle:

- frozen preprocessing pipelines
- model deployment integrity
- Population Stability Index (PSI)
- Characteristic Stability Index (CSI)
- population drift versus concept drift
- recalibration versus redevelopment
- post-deployment control thinking

This note matters because many people understand model building, but fewer understand what makes a model defensible after deployment.

This is the bridge between analytics and model governance.

It also maps very naturally to the kind of thinking needed in real institutions:

- is the pipeline consistent?
- has the incoming population changed?
- is the model still ranking risk properly?
- is the issue only calibration, or has the underlying relationship changed?
- does the model still deserve to be used?

---

### [[3_lgd_ead_model_rewritten]] — Loss Severity and Exposure Thinking

Once PD is understood, the next question is not only *whether* default happens, but *how much money is lost if it happens*.

This note extends the risk framework into:

- **LGD** — Loss Given Default
- **EAD** — Exposure at Default
- recovery modeling
- two-stage LGD logic
- exposure severity logic
- practical limitations of installment-loan data

This note matters because it completes the move from **default probability** to **loss economics**.

It also helps me think more clearly about the fact that two borrowers with the same PD do not necessarily create the same loss.

That is a very important conceptual shift in credit risk.

---

### [[4_ecl_cecl_stress_testing_rewritten]] — Portfolio Loss, Provisioning, and Stress Thinking

This note brings the three risk parameters together:

- **PD**
- **LGD**
- **EAD**

and connects them into:

- **Expected Loss (EL)**
- **Unexpected Loss (UL)**
- portfolio aggregation
- CECL-style lifetime loss thinking
- macro-scenario extensions
- stress-testing logic

This is where individual-loan modeling becomes **portfolio-level financial impact**.

This note matters because it brings model outputs closer to the language of provisions, reserves, capital pressure, and scenario analysis.

In other words, this is where statistical modeling becomes balance-sheet relevant.

---

## How the Applied Machine-Learning Notes Fit In

These project notes are not side projects in a random sense. I use them to strengthen the general data-science foundation that supports the risk work.

### [[5_bank_churn_neural_networks_masterclass]] — Neural Networks, Classification, and Imbalance

This note teaches me:

- binary classification
- feature preparation
- scaling and encoding
- neural-network basics
- loss functions and optimization
- class imbalance
- threshold tuning
- precision versus recall trade-offs

Even though the project is framed as churn, the transferable lessons are broader:

- early-warning systems
- retention-risk logic
- probability scoring
- decision thresholds
- recall-oriented intervention models

It gives me a practical base for understanding when and why a neural network is useful, and when a simpler model may still be preferable.

---

### [[6_employee_retention_tree_models_masterclass]] — Logistic Regression, Trees, and Random Forests

This note helps me understand:

- linear versus non-linear modeling
- logistic regression as a probability model
- decision trees and recursive splitting
- random forests and ensemble averaging
- feature importance
- feature engineering
- model comparison across metrics

This is a very useful bridge note because it places **logistic regression** and **tree-based models** side by side.

That comparison is extremely valuable in interviews because many real questions are really about model trade-offs:

- interpretability versus flexibility
- bias versus variance
- simple baseline versus complex model
- probability calibration versus raw predictive strength

---

### [[7_socio_economic_household_classification_masterclass]] — Large-Scale Tabular Modeling and XGBoost

This note strengthens my understanding of:

- messy real-world tabular data
- missing values and sentinel values
- categorical encoding
- outlier handling
- dimensionality reduction with PCA
- class imbalance
- XGBoost and boosted ensembles
- model comparison on larger feature spaces

This project matters because it pushes me toward more realistic data complexity.

It is the note where I get more comfortable with the idea that not every dataset is clean, tidy, or well-structured, and that strong modeling often depends as much on data handling as on the model itself.

---

### [[8_twitter_sentiment_nlp_masterclass]] — NLP, Feature Extraction, and Unstructured Data

This note expands my range beyond tabular modeling into text analytics.

It covers:

- text cleaning
- tokenization
- lemmatization
- CountVectorizer
- TF-IDF
- multi-class classification
- feature extraction from raw text
- model evaluation in imbalanced text settings

This note matters because real institutions do not only work with structured tables.

There are also:

- complaint narratives
- call-center notes
- customer feedback
- collections remarks
- adverse media signals
- policy and regulatory text

So this note helps me think about how unstructured data can enter an analytics workflow.

---

## The Big Picture: What Concepts This System Covers

One of the most important reasons I built this note system is to avoid learning concepts in isolation.

I want the big picture to stay visible at all times.

### 1. Data Understanding and Data Quality

Across the notes, I repeatedly work with:

- target definition
- missing values
- duplicate handling
- outliers
- encoding
- scaling
- leakage risks
- feature engineering

This is the first layer of almost every serious analytical problem.

### 2. Statistics and Probability

The notes also connect back to core quantitative ideas:

- probability of an event
- conditional loss given an event
- expectation
- variance and uncertainty
- train-test separation
- sampling logic
- signal versus noise
- calibration versus discrimination

This layer matters because good modeling is not just coding. It is applied statistical reasoning.

### 3. Supervised Learning

The projects collectively cover multiple model families:

- logistic regression
- scorecards
- decision trees
- random forests
- XGBoost
- neural networks
- NLP classification pipelines

That breadth is useful because it helps me compare models rather than memorizing one favorite model.

### 4. Model Evaluation

A major theme across the vault is that prediction quality is multi-dimensional.

I work with metrics such as:

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- KS statistic
- Gini
- confusion matrix logic
- cut-off selection
- business-threshold interpretation

This matters because a model cannot be judged by one metric in a vacuum.

### 5. Monitoring, Validation, and Governance

A strong part of my direction is not only model building, but also model challenge and control.

The notes therefore include:

- stability monitoring
- PSI and CSI
- drift diagnosis
- recalibration logic
- pipeline integrity
- score interpretation
- documentation thinking
- governance framing

This is one of the clearest ways this vault reflects my model-validation and risk-governance direction rather than only a generic data-science direction.

### 6. Credit-Risk and Banking Context

The risk-specific notes keep the banking context explicit:

- PD, LGD, EAD
- expected loss
- lifetime loss thinking
- scorecards
- portfolio monitoring
- provisioning logic
- scenario analysis
- regulated deployment thinking

This is the layer that keeps the entire system anchored to my domain rather than floating as purely generic machine learning.

---

## How This Maps to My Background

### Banking and Risk Analytics

My work in credit-risk analytics gave me direct exposure to the way risk is measured and monitored across lending portfolios.

That background is why the first four notes feel natural and central in this system. They match the way I think about real portfolio questions:

- what is default risk?
- how should it be measured?
- how stable is the portfolio?
- how is loss quantified?
- how do analytics support decisions and controls?

### Business Analysis and Lending-System Workflows

My experience in business analysis strengthened a different but equally important layer: the implementation side.

That part of my background helps me think carefully about:

- requirements clarity
- workflow design
- testability
- documentation
- handoff between business and technical teams
- whether an analytical solution can actually survive inside a working process

That is one reason the notes repeatedly discuss not only math and code, but also practical deployment and governance.

### Quantitative and Technical Development

My deeper work in Python, machine learning, and project-based learning is what expands this system beyond traditional business or reporting analytics.

It allows me to move from:

- describing risk
- monitoring risk
- predicting risk
- explaining model behavior
- comparing algorithms
- thinking about validation and remediation

This is the layer that connects the broader machine-learning projects back to the risk notes.

---

## What This Note System Says About Me Professionally

If I had to summarize what this system shows about me, it is this:

### 1. I am building from a banking base, not from abstraction alone.

The work is anchored in lending, risk, portfolio behavior, and real operational thinking.

### 2. I care about the full model lifecycle.

I do not want to stop at training a model. I want to understand preprocessing, validation, monitoring, governance, and business use.

### 3. I learn concepts through implementation.

I prefer to understand math and code through projects rather than through disconnected definitions.

### 4. I am comfortable working across both structured and unstructured problems.

The vault now includes scorecards, tree models, neural networks, ensemble methods, and NLP pipelines.

### 5. My direction is toward stronger quantitative risk and model-validation capability.

The core note system is increasingly shaped around defensible models, practical controls, and institution-grade analytical thinking.

---

## How I Use This Vault

This vault serves three purposes for me.

### As a learning system

It helps me revisit concepts from first principles until I can explain them clearly.

### As an interview system

It gives me a way to move from a project to the bigger conceptual question behind it.

For example:

- a churn project becomes a discussion about classification, thresholding, and intervention strategy
- a PD scorecard becomes a discussion about default definition, WoE, logistic regression, validation, and interpretability
- a monitoring note becomes a discussion about drift, governance, recalibration, and model controls

### As a long-term professional system

It helps me organize my direction clearly: credit risk, quantitative analytics, data science, validation, and practical business implementation all sit inside one map rather than in separate boxes.

---

## Recommended Reading Flow Inside the Vault

When I want the most coherent path through the system, the cleanest sequence is:

1. [[1_full_pd_model]]
2. [[2_monitoring_model]]
3. [[3_lgd_ead_model_rewritten]]
4. [[4_ecl_cecl_stress_testing_rewritten]]
5. [[6_employee_retention_tree_models_masterclass]]
6. [[5_bank_churn_neural_networks_masterclass]]
7. [[7_socio_economic_household_classification_masterclass]]
8. [[8_twitter_sentiment_nlp_masterclass]]

That order keeps the credit-risk spine first, then broadens into the wider machine-learning foundation.

---

## Final Anchor

This note is the map.

The rest of the vault is the working machinery.

Together, they represent how I think about my professional direction:

- grounded in banking and credit risk
- strengthened by quantitative and machine-learning depth
- aware of implementation reality
- focused on model quality, monitoring, and governance
- built as a connected system rather than a loose collection of projects

That is the real purpose of this brain.
