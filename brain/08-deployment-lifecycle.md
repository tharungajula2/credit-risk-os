---
title: "Deployment Lifecycle & Model Risk Management"
tags:
  - deployment
  - model-risk-management
  - MRM
  - SR-11-7
  - champion-challenger
  - adverse-action
  - client-handover
cluster_phase: Phase 6
links:
  - "[[07-production-monitoring-PSI]]"
  - "[[04-logistic-regression-scorecard]]"
  - "[[09-algorithm-selection-guide]]"
---

# Phase 6: The Deployment Lifecycle & Client Engineering

## 6.1 The Three Stages of Model Approval

A credit risk model is not deployed the moment it achieves a good AUROC. It must pass through a formal **Model Risk Management (MRM)** process before any production use. The three stages are universal across regulated financial institutions.

### Stage 1: Technical Validation (Quantitative Review)

**Who conducts it:** An independent Model Validation team (separate from the development team — this independence is a regulatory requirement under SR 11-7 and SS1/23).

**What is validated:**

- **Mathematical correctness:** Are the statistical techniques applied correctly? Is MLE implemented properly? Are the WoE bins monotonic?
- **Code review:** Is the implementation reproducible? Are there data leakage risks in the feature engineering pipeline?
- **Performance benchmarking:** Does the model meet minimum AUROC and KS thresholds on the OOT test set?
- **Stability testing:** Does performance degrade when small perturbations are applied to inputs (sensitivity analysis)?
- **Challenger model comparison:** How does the proposed model compare against the existing production model (the "champion")?

**Output:** A **Technical Validation Report** with a pass/fail/conditional recommendation.

### Stage 2: Qualitative & Business Review

**Who conducts it:** Business stakeholders — credit policy, risk strategy, compliance, and sometimes the regulator.

**What is reviewed:**

- **Business logic:** Do the model's directional relationships make intuitive sense? (Higher income → lower risk; more delinquencies → higher risk). A statistically valid model that violates business logic will fail this stage.
- **Regulatory compliance:**
  - Is any protected characteristic (race, gender, religion, national origin) — or a proxy for it — used as a feature? This triggers **Fair Lending / ECOA** review.
  - Can the model's decisions be explained at the individual level? (GDPR / CCPA "right to explanation")
- **Adverse Action Coding:** Every declined applicant is legally entitled to a reason. The model must produce **top reason codes** (e.g., "Too many recent delinquencies; High utilization ratio") — only possible with WoE-based scorecards or SHAP-augmented tree models.

**Output:** A **Model Approval Memo** signed by the Chief Risk Officer (or equivalent).

### Stage 3: Client Handover & Non-Technical Communication

**Who is involved:** The quant team, the client's IT/data engineering team, and senior business stakeholders.

**The Central Challenge:** Translating complex mathematics into business language without losing accuracy.

**Framework for Explaining Black-Box Math to Non-Technical Stakeholders:**

| Concept | Technical Reality | Client-Facing Explanation |
|---|---|---|
| Logistic Regression $\beta$ | Log-odds coefficient | "Each additional missed payment lowers your score by approximately X points" |
| AUROC = 0.82 | Area under ROC curve | "The model correctly ranks 82% of riskier borrowers above safer ones" |
| PSI = 0.18 | Distribution shift metric | "The profile of new applicants has shifted moderately from our training data — we recommend reviewing the model within 90 days" |
| WoE = +0.6 for a bin | Log-ratio of bad/good rates | "Applicants in this income range are 1.8× more likely to default than the average" |
| SHAP value = -12 pts | Shapley attribution | "This applicant's score was reduced by 12 points primarily due to their recent credit inquiry history" |

**The Champion-Challenger Framework:**

The deployment is never a hard cutover. New models are deployed in **shadow mode** first:

```
All live applications
        │
        ├──→ Champion Model (current production) → Drives actual decisions
        │
        └──→ Challenger Model (new model) → Scores silently; no business action taken

After 3–6 months of parallel running:
→ Compare challenger vs. champion performance on matched cohorts
→ If challenger statistically outperforms → promote to champion
→ Retire old champion; establish new challenger
```
