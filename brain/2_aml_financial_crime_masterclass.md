---
title: "AML and Financial Crime Analytics Masterclass"
date: 2026-03-17
tags:
  - aml
  - bsa-aml
  - financial-crime
  - transaction-monitoring
  - sanctions-screening
  - kyc
  - cdd
  - edd
  - sar
  - ofac
  - anomaly-detection
  - xgboost
  - nlp
  - sr-11-7
cluster: "05 — Sandbox Environment"
links:
  - "[[Tharun-Kumar-Gajula_links_updated|Tharun Kumar Gajula]]"
  - "[[1_lending_club_credit_risk_masterclass|Lending Club Credit Risk Masterclass]]"
  - "[[5_bank_churn_neural_networks_masterclass_links_updated|Bank Churn Prediction with Neural Networks]]"
  - "[[8_twitter_sentiment_nlp_masterclass_links_updated|Twitter Sentiment Analysis with NLP]]"
---

# AML and Financial Crime Analytics Masterclass

*This note is my beginner-friendly master note for Anti-Money Laundering (AML) and Financial Crime Analytics. I am treating it as the bridge between my credit risk foundation, my machine learning projects, and the kind of financial-crime work I may need to discuss in interviews. I am not pretending I already built a full AML production system. I am using this note to understand the domain properly, learn the technical vocabulary, and connect my existing Python, SQL, modeling, NLP, and monitoring skills to AML problems in a clean and honest way.*

---

## 1. What AML is actually trying to solve

AML is not mainly asking **"Will this customer repay?"** That is a credit-risk question.

AML is asking a different set of questions:

- Is this customer who they claim to be?
- Does the activity make sense for the customer profile?
- Is the money source and movement consistent with legitimate business or personal behavior?
- Is the account being used to hide, layer, move, or disguise illicit funds?
- Is the bank missing suspicious behavior because the rules are too weak or too noisy?

So the core shift is:

- **Credit risk** focuses on **solvency and loss**.
- **AML / Financial crime** focuses on **integrity, behavior, movement, and suspicion**.

A bank's BSA/AML framework usually includes customer due diligence, suspicious activity monitoring and reporting, and OFAC/sanctions compliance. That is the high-level operational environment in which AML analytics sits.

---

## 2. The big picture workflow inside an AML program

This is the simple end-to-end picture I should keep in my head.

### Stage 1: Onboarding and customer understanding
At account opening, the bank collects identity information, business information, expected activity, geography, and ownership details. This is the KYC / CDD layer.

### Stage 2: Sanctions and watchlist screening
The bank checks customer names, counterparties, entities, and sometimes payments against sanctions lists and watchlists. Exact matching is not enough because names can be noisy, transliterated, abbreviated, or misspelled.

### Stage 3: Transaction monitoring
Once the account is active, the bank watches deposits, withdrawals, wires, transfers, cash behavior, counterparties, and movement patterns. This can be done using rules, ML models, anomaly detection, and network analysis.

### Stage 4: Alert review and case investigation
Potential issues become alerts. Analysts review the alert, gather more context, and decide whether it is explainable or suspicious.

### Stage 5: Escalation and SAR filing
If the activity remains suspicious after investigation, the institution may file a **SAR (Suspicious Activity Report)** with **FinCEN**.

### Stage 6: Model and rule monitoring
The bank monitors the performance of its AML rules and models: false positives, missed suspicious activity, drift, threshold quality, investigation workload, and control effectiveness.

This is the lifecycle where analytics fits.

---

## 3. Core AML vocabulary I need to know cold

### AML
Anti-Money Laundering. The broad control framework used to detect and prevent the movement of illicit funds through the financial system.

### BSA / AML
In the US context, AML work usually sits inside the broader **Bank Secrecy Act / Anti-Money Laundering** framework.

### KYC
Know Your Customer. The process of identifying the customer and establishing who they are.

### CDD
Customer Due Diligence. Understanding the customer relationship, beneficial ownership where relevant, expected activity, and risk profile.

### EDD
Enhanced Due Diligence. Additional scrutiny for higher-risk customers, products, or geographies.

### Sanctions screening
Checking customers, counterparties, and transactions against sanctions lists and related watchlists.

### OFAC
Office of Foreign Assets Control. In practice, OFAC screening is one of the most visible sanctions-control requirements for US financial institutions.

### Transaction monitoring
Ongoing surveillance of transactions and account behavior to identify unusual or suspicious activity.

### Alert
A rule, model, or screening hit that requires review.

### False positive
An alert that looks suspicious to the system but is ultimately cleared by analysts.

### True positive
A useful alert that genuinely surfaces suspicious behavior.

### SAR
Suspicious Activity Report. A formal report filed with FinCEN when a financial institution identifies reportable suspicious activity.

### Typology
A recognizable suspicious pattern, such as structuring, rapid pass-through behavior, account takeover, mule activity, or sanctions evasion behavior.

### Structuring / Smurfing
Breaking a larger transaction into many smaller transactions to avoid reporting or detection thresholds.

### ATL / BTL testing
Industry shorthand often used in AML validation. **Above-the-line** testing reviews alerts that fired at or above a threshold. **Below-the-line** testing reviews a sample just below the threshold to see whether the system may be missing suspicious activity. Internal definitions vary by institution, but this is the practical idea.

---

## 4. How AML is different from credit risk

| Dimension | Credit Risk | AML / Financial Crime |
|---|---|---|
| Main question | Will the customer default? | Is the activity suspicious or illicit? |
| Typical target | default / non-default | SAR / no SAR, escalation / no escalation, suspicious / non-suspicious |
| Core signal | repayment behavior and borrower risk | transaction behavior, identity, counterparties, networks, typologies |
| Time pattern | slower deterioration is common | abrupt spikes and unusual velocity are common |
| Labels | usually cleaner than AML | often noisy, delayed, and incomplete |
| Cost of error | bad lending decisions and credit loss | missed crime, regulatory exposure, wasted investigation effort |
| Monitoring focus | score stability, delinquency, calibration | alert volume, false positives, drift, missed suspicious activity |

The most important lesson is that AML labels are much messier than credit-risk labels.

In credit risk, default is not perfect, but it is still more concrete. In AML, a SAR is not the same as a criminal conviction. It is a suspicious-case outcome inside the institution's process. That makes AML modeling harder.

---

## 5. The main AML analytics problems

I should think of AML analytics as several technical problems, not one single model.

### 5.1 Customer risk scoring at onboarding
This is closer to classic scorecard thinking.

Goal:
- assign a higher-risk rating to customers who deserve stronger due diligence or tighter monitoring

Possible features:
- customer type
- occupation or business type
- geography
- expected transaction volume
- cash intensity
- ownership complexity
- PEP / sanctions proximity flags

Typical outputs:
- low / medium / high AML risk
- decision to route to standard CDD or EDD

This is the closest AML problem to a traditional scorecard mindset.

---

### 5.2 Transaction monitoring and alert generation
This is the core AML monitoring problem.

Goal:
- identify unusual or suspicious transactions or behavior sequences

Possible signals:
- sudden cash spikes
- high-velocity movement through the account
- round-dollar transactions
- rapid in-and-out transfers
- unusual geographies
- new counterparties with high volume
- activity inconsistent with known customer profile

There are two broad approaches:

#### Rules-based monitoring
Examples:
- flag cash deposits above a threshold
- flag many cash deposits just below a threshold
- flag rapid movement of funds after inbound wires

Strength:
- easy to explain

Weakness:
- very high false positives
- easy for criminals to adapt around static rules

#### Model-based monitoring
Examples:
- supervised alert triage
- anomaly detection
- graph/network scoring

Strength:
- can prioritize cases more intelligently

Weakness:
- needs good design, good governance, and careful validation

---

### 5.3 Alert triage using supervised machine learning
This is one of the cleanest places where my current skill set transfers.

#### Problem
A bank's rules can create too many alerts. Human investigators cannot deeply review everything.

#### Idea
Use a supervised model to rank or classify alerts based on historical outcomes.

#### Possible target
- `1 = escalated / SAR / confirmed suspicious`
- `0 = cleared / false positive`

#### Candidate models
- logistic regression
- random forest
- XGBoost / gradient boosting

#### Why this maps well from my background
This is still classification.

The logic is very similar to my other projects:
- build features
- train a classifier
- optimize thresholds
- balance recall against workload
- monitor drift after deployment

#### Beginner intuition
A triage model does not replace investigators.
It helps **rank** alerts so the best human attention goes to the most important cases first.

---

### 5.4 Anomaly detection for unknown suspicious behavior
Not all bad behavior will look like past suspicious cases.

That is why unsupervised or semi-supervised methods matter.

#### Typical techniques
- Isolation Forest
- Local Outlier Factor
- clustering
- autoencoders
- sequence anomaly methods

#### What the model is doing
It is not predicting a formal label directly.
It is measuring how unusual the behavior is relative to a learned baseline.

#### Why this matters in AML
Criminal behavior adapts. A system based only on historical labeled cases can miss new typologies.

#### Example intuition
If a low-activity retail account suddenly behaves like a high-throughput business account, the behavior may be mathematically abnormal even if no static rule was explicitly broken.

---

### 5.5 Sanctions screening and name matching
This is another major AML / financial-crime analytics area.

#### Problem
Simple exact matching does not work well for names because of:
- spelling variation
- abbreviations
- initials
- transliteration differences
- aliases
- ordering differences
- entity name noise

#### Technical solution space
- normalization and text cleaning
- token-based comparison
- phonetic methods
- edit-distance methods
- fuzzy matching
- embedding-based semantic similarity

#### My skill bridge
This connects strongly to:
- preprocessing discipline
- string cleaning
- NLP thinking
- threshold setting
- precision / recall trade-offs

This is where my text-classification and NLP notes become relevant conceptually, even though sentiment analysis itself is a different business problem.

---

### 5.6 Adverse media and unstructured text analytics
AML is not only about structured transaction tables.

Banks also look at public information, case narratives, news, investigation text, and screening notes.

Possible NLP tasks:
- adverse media classification
- entity extraction
- name disambiguation
- document triage
- narrative summarization
- relationship extraction

This is where my NLP project becomes a useful mental bridge.

---

### 5.7 Graph and network analytics
Financial crime often happens through connected entities, not isolated transactions.

Useful graph ideas:
- shared addresses
- shared devices
- shared beneficiaries
- circular fund movement
- hub-and-spoke patterns
- unusually dense transaction communities

Useful graph features:
- degree
- centrality
- edge frequency
- community structure
- multi-hop exposure

I do not need to become a graph-theory expert immediately, but I should know that modern AML analytics often becomes a **network problem**, not just a row-by-row classification problem.

---

## 6. The data used in AML analytics

AML data is more heterogeneous than many beginner ML datasets.

### Customer data
- name
- date of birth / incorporation
- address
- nationality / domicile
- occupation or business type
- beneficial ownership details
- expected account activity

### Account data
- account age
- account type
- channel access
- linked accounts
- product type

### Transaction data
- amount
- currency
- timestamp
- transaction type
- originator and beneficiary information
- counterparty geography
- payment rail
- cash vs non-cash

### Screening data
- sanctions/watchlist hits
- similarity scores
- list source
- alias information

### Case-management data
- alert status
- investigator decision
- escalation reason
- SAR / no SAR
- closure notes

### External / contextual data
- country risk
- industry risk
- branch information
- device or channel metadata
- public adverse media

This is why feature engineering in AML is often more complex than in simple tabular ML projects.

---

## 7. What good AML features look like

A raw transaction row is usually not enough.

Good AML features are often **behavioral aggregates** over time windows.

### Customer-profile features
- declared expected monthly volume
- customer risk rating
- business type risk flag
- geographic risk bucket

### Behavioral aggregate features
- average daily transaction count
- 7-day inbound amount
- 30-day outbound amount
- ratio of cash to total activity
- count of new counterparties in last 30 days
- nighttime transaction frequency
- round-dollar transaction ratio

### Deviation features
- current activity / expected activity
- current amount / historical median
- current counterparties / historical counterparties

### Network features
- number of connected accounts
- number of shared beneficiaries
- circular-flow indicator
- pass-through ratio

### Text and screening features
- name similarity score
- adverse media score
- number of alias matches
- narrative keywords or embeddings

This is a very important interview point:

> AML models often win or fail more because of feature design and workflow integration than because of one fancy algorithm.

---

## 8. The machine learning and math I should know for AML interviews

I do not need every formula on earth. I need the right conceptual math.

### 8.1 Logistic regression for alert triage
If I build a supervised AML classifier, logistic regression is the easiest place to start.

It models:

\[
P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k)}}
\]

Interpretation:
- the model converts a linear score into a probability
- each coefficient changes the log-odds of the outcome
- in AML, the target could be suspicious / not suspicious or SAR / no SAR

Why it matters:
- simple
- transparent
- explainable
- often a strong baseline

---

### 8.2 Gradient boosting / XGBoost for alert prioritization
XGBoost is useful when relationships are nonlinear and interactions matter.

Why it helps in AML:
- transaction behavior is rarely linear
- interaction effects matter
- thresholding matters more than raw accuracy

Typical questions I should be ready for:
- Why use boosting instead of only rules?
- How do you avoid overfitting?
- How do you tune threshold by workload capacity?
- How do you explain a boosted model to investigators or validators?

---

### 8.3 Precision, recall, and the false-positive problem
AML is one of the clearest domains where accuracy can be misleading.

Suppose only 1 out of 1,000 alerts is truly suspicious.
A model that predicts everything as non-suspicious may look highly accurate but is useless.

So the more important metrics are often:
- **precision**: when the model says suspicious, how often is it correct?
- **recall**: of the truly suspicious cases, how many did it catch?
- **PR-AUC**: useful when classes are highly imbalanced

A high false-positive rate creates major operational pain because investigators waste time on poor alerts.
A low recall can be even worse because suspicious activity may be missed.

AML is therefore a **threshold optimization and workflow design** problem, not only a modeling problem.

---

### 8.4 Class imbalance
True suspicious cases are rare.

That means I should know:
- class weighting
- resampling carefully
- precision-recall trade-offs
- cost-sensitive evaluation
- why business capacity matters when setting thresholds

---

### 8.5 Anomaly scores
In anomaly detection, the model may not output a probability of crime.
It may output an **anomaly score**.

Interpretation:
- higher score means more unusual behavior
- the threshold determines how many cases become alerts
- the threshold should be chosen with risk appetite and investigation capacity in mind

---

### 8.6 String similarity
For sanctions and name matching, I should understand basic similarity ideas.

#### Edit distance intuition
A string similarity method measures how many changes are needed to turn one name into another.

Example:
- `MOHAMED ALI`
- `MUHAMMAD ALI`

These are not identical strings, but they may still refer to the same person or related variants.

Key point:
- a screening system often produces a similarity score
- then the bank chooses a threshold
- too low a threshold misses real matches
- too high a threshold floods analysts with false positives

Again, AML is full of threshold-management problems.

---

## 9. Typologies I should understand as patterns, not just vocabulary

I do not need to memorize endless lists. I need to understand the logic behind the patterns.

### Structuring
Many smaller transactions designed to avoid a larger visible reportable or suspicious amount.

### Funnel account behavior
Funds collected from many locations or parties and rapidly consolidated or moved.

### Rapid pass-through / layering
Funds enter and quickly leave with little economic reason to stay.

### Mule activity
An account appears to be receiving and forwarding funds on behalf of someone else.

### Dormant-account reactivation
An inactive account suddenly becomes active with unusual volume or counterparty patterns.

### Geographical inconsistency
Activity shows countries, corridors, or counterparties that do not fit the customer profile.

### Sanctions evasion signals
Name variants, indirect counterparties, unusual routing patterns, or payments involving restricted parties or geographies.

The general AML principle is:

> suspicious behavior is usually about **inconsistency** - inconsistency with the customer profile, with historical behavior, with economic purpose, or with expected money movement.

---

## 10. Validation and monitoring in AML

This is where my credit-risk monitoring background transfers well.

### 10.1 Why monitoring matters
An AML model that worked last year may degrade because:
- customer behavior changed
- payment channels changed
- criminals adapted
- business mix changed
- thresholds no longer reflect reality
- labels changed due to investigation practices

### 10.2 PSI / CSI style logic
The same monitoring logic I used in credit risk can be translated into AML.

#### PSI-like questions
- Did the score distribution move?
- Are more alerts landing in high-score buckets?
- Did the transaction-risk population shift?

#### CSI-like questions
- Which underlying variables shifted?
- Did international wire behavior change?
- Did cash intensity change?
- Did country-risk mix change?

### 10.3 Performance monitoring questions
- What is the current alert volume?
- What percentage becomes true escalation or SAR?
- Did false positives increase?
- Did investigator turnaround time worsen?
- Did model score rank-ordering weaken?

### 10.4 Threshold reviews
A model may still rank well, but operational thresholds may become outdated.

So AML monitoring is not only:
- **Is the model mathematically stable?**

It is also:
- **Is the alerting system still operationally useful?**

---

## 11. How model risk management applies to AML

AML models still live inside a controlled model-risk environment.

That means I should think in the same broad SR 11-7 style structure:
- clear purpose
- sound data and assumptions
- documented methodology
- implementation controls
- independent validation
- ongoing monitoring
- change management

Key AML twist:
The domain is noisy, investigator-dependent, and operationally constrained. So validators often care a lot about:
- label quality
- governance around thresholds
- explainability
- below-the-line testing
- segmentation logic
- stability over time
- whether the model actually improves investigation quality

---

## 12. How my existing brain notes connect to AML

This is one of the most important sections for me because it tells me I am not starting from zero technically.

### Lending Club Credit Risk Masterclass -> AML transfer
What transfers directly:
- classification thinking
- feature engineering discipline
- threshold choice
- model validation
- monitoring logic
- PSI / CSI intuition
- governance mindset

What changes:
- labels become noisier
- suspicious behavior is more adaptive than default behavior
- operations and investigations matter more

### Bank Churn note -> AML transfer
What transfers directly:
- recall-oriented thinking
- threshold tuning
- class imbalance awareness
- intervention prioritization

AML parallel:
- prioritize suspicious alerts instead of likely churners

### Twitter Sentiment NLP note -> AML transfer
What transfers directly:
- text cleaning
- vectorization intuition
- NLP pipeline thinking
- classification of unstructured text

AML parallel:
- sanctions screening support
- adverse media triage
- narrative review support

### Employee Retention / Tree Models -> AML transfer
What transfers directly:
- feature interactions
- tree-based classification
- importance ranking
- explaining segments and nonlinear behavior

AML parallel:
- complex transactional interaction patterns

### General conclusion
I may be new to AML domain knowledge, but I am **not** new to:
- tabular ML
- monitoring
- classification
- preprocessing
- NLP foundations
- model explanation
- workflow thinking

That is the honest bridge I should communicate.

---

## 13. How I should talk about AML in interviews without faking experience

I should never pretend I already ran a production AML monitoring engine if I did not.

The better answer is:

> My direct project background is stronger in credit risk and applied ML, but the underlying modeling toolkit transfers well to AML. I already understand classification, thresholding, drift monitoring, feature engineering, NLP basics, and model-governance thinking. What I am building now is the domain layer: KYC, sanctions, transaction monitoring, alert triage, and AML-specific validation practices.

That answer is honest and strong.

---

## 14. A simple AML technical architecture I should be able to explain

This is a good generic end-to-end design.

### Layer 1: Data ingestion
- customer master data
- account data
- transactions
- external watchlists
- case outcomes
- text or narrative data

### Layer 2: Data quality and preprocessing
- deduplication
- entity resolution
- missing-value handling
- normalization of names, addresses, and countries
- transaction windowing and aggregation

### Layer 3: Detection engines
- rules engine
- supervised alert-triage model
- anomaly-detection engine
- sanctions / screening engine
- network-risk engine

### Layer 4: Alert orchestration
- score aggregation
- thresholds
- priority queues
- investigator assignment

### Layer 5: Case management
- analyst review
- evidence collection
- escalation decisions
- SAR filing where required

### Layer 6: Monitoring and governance
- performance dashboards
- drift monitoring
- validation testing
- threshold review
- documentation and change logs

This is a strong mental model even before I know every vendor platform.

---

## 15. What I should study next if I want to become interview-ready fast

### First priority: domain basics
I should become comfortable with:
- KYC
- CDD / EDD
- sanctions screening
- transaction monitoring
- SAR lifecycle
- common suspicious typologies

### Second priority: AML metrics and validation
I should be able to explain:
- false positives
- recall
- precision
- PR-AUC
- threshold tuning
- below-the-line testing
- model monitoring

### Third priority: technical workflows
I should learn how AML analytics is actually operationalized:
- rules + models together
- feature windows
- alert queues
- investigator workflow
- case outcomes as labels

### Fourth priority: graph and network thinking
Even a basic grasp here will make me sound much stronger.

---

## 16. Interview questions I should be able to answer after this note

### Conceptual
- What is the difference between credit risk modeling and AML modeling?
- Why is AML harder from a labeling perspective?
- Why are false positives such a major problem in transaction monitoring?
- What is the difference between KYC, CDD, and EDD?
- What is a SAR?
- What is sanctions screening?

### Technical
- How would I build a model to prioritize AML alerts?
- Why might anomaly detection matter in AML?
- How would I monitor an AML model after deployment?
- Which metrics matter more than accuracy in AML and why?
- How would I handle highly imbalanced suspicious-activity labels?
- How would I validate a sanctions-screening name-match threshold?

### Honest bridge questions
- How does my credit-risk background transfer into AML?
- What parts transfer directly, and what parts are new?
- How would I explain model risk management in AML?

---

## 17. The most important beginner summary

If I had to compress this whole note into one page in my head, it would be this:

1. AML is about suspicious money movement, customer integrity, and financial-crime risk - not default risk.
2. The big operational engines are KYC/CDD, sanctions screening, transaction monitoring, case investigation, and SAR filing.
3. AML analytics includes rules, supervised classification, anomaly detection, NLP, and sometimes graph analytics.
4. False positives are a major operational problem, so threshold tuning and investigator workflow matter a lot.
5. Precision, recall, PR-AUC, class imbalance, and drift monitoring are more useful than raw accuracy.
6. My credit-risk and ML background transfers well into alert triage, model monitoring, threshold optimization, and feature engineering.
7. I should speak honestly: I am not claiming finished AML production experience, but I can clearly explain how my current toolkit applies to AML problems.

---

## 18. My final takeaway

AML is not a totally different universe from what I already know.

The business objective changes, the labels become noisier, the data becomes more behavioral and networked, and the operational workflow becomes more investigation-heavy. But the technical spine still feels familiar:

- define the problem clearly
- create meaningful features
- choose the right model class
- evaluate with the right metrics
- set thresholds carefully
- monitor drift and effectiveness
- keep the whole system explainable and governed

That is exactly why this note belongs inside my broader brain system. It connects my credit-risk foundation, my ML projects, my NLP exposure, and my interview preparation into one coherent AML starting point.
