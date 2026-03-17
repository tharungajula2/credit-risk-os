---
title: "Expected Loss, CECL, and Stress Testing — Portfolio Credit Loss Integration"
date: 2026-03-19
tags:
  - ECL
  - EL
  - expected-loss
  - unexpected-loss
  - CECL
  - IFRS-9
  - CCAR
  - DFAST
  - stress-testing
  - macroeconomics
  - PIT
  - lifetime-loss
  - survival-analysis
  - portfolio-risk
  - concentration-risk
  - credit-risk
  - lending-club
  - model-validation
cluster: "03 — Loss, Exposure, and Recovery Modeling"
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_full_pd_model]]"
  - "[[2_monitoring_model]]"
  - "[[3_lgd_ead_model_rewritten]]"
---

---

# Expected Loss, CECL, and Stress Testing — Portfolio Credit Loss Integration

> This note is my full technical record of how I bring **PD**, **LGD**, and **EAD** together into one portfolio-level loss framework. I use the Lending Club project as the anchor example, but I also extend it into the practical areas I need for interviews and real banking work: **Expected Loss**, **Unexpected Loss**, **portfolio aggregation**, **US CECL**, and **stress testing**.
>
> This note sits after [[1_full_pd_model]] and [[3_lgd_ead_model_rewritten]]. The PD note explains default likelihood. The LGD/EAD note explains severity and exposure conditional on default. This note combines those pieces into the loss number that matters at portfolio level.

---

## The Project at a Glance

**Core objective:** Combine the outputs of the PD, LGD, and EAD models into one economically meaningful measure of loss.

**The baseline project logic is simple:**

```text
For each loan
        │
        ├── Predict PD
        ├── Predict LGD
        └── Predict EAD
        │
        ▼
Compute EL = PD × LGD × EAD
        │
        ▼
Sum across all loans
        │
        ▼
Get total portfolio expected loss
```

**Why this matters:**

A model is only truly useful when it turns prediction into money.

- **PD** tells me how likely default is.
- **LGD** tells me what fraction is lost if default happens.
- **EAD** tells me how much exposure is outstanding when default happens.
- **EL** turns those three pieces into an expected dollar loss.

**What this note covers:**

1. the math of **Expected Loss (EL)**
2. the difference between **EL and Unexpected Loss (UL)**
3. why portfolio EL is additive and UL is not
4. how this baseline project connects to **US CECL**
5. how the same framework is extended into **stress testing**

**Important scope note:**

The Lending Club project is a very good baseline learning build for one-period expected loss, but it is not by itself a full production CECL engine or a full supervisory stress-testing system. I use it here as the anchor and then explain how the architecture would evolve in a real bank.

---

## Part 1: The Core Equation — What Expected Loss Actually Means

### The Concept

For one loan, expected loss is the average loss I should expect over many similar cases.

The standard expression is:

```text
EL = PD × LGD × EAD
```

where:

- `PD` = Probability of Default
- `LGD` = Loss Given Default
- `EAD` = Exposure at Default

### A More Formal View

Let:

- `D` be a default indicator, where `D = 1` if default occurs and `D = 0` otherwise
- `L` be realized loss

Then a simple one-period loss representation is:

```text
L = D × LGD × EAD
```

Taking expectation:

```text
E[L] = E[D × LGD × EAD]
```

If `PD = E[D]` and LGD/EAD are already interpreted as conditional expectations on default, the operational formula becomes:

```text
EL = PD × LGD × EAD
```

That is the formula banks use constantly because it is interpretable, modular, and directly usable for pricing, provisioning, stress testing, and capital conversations.

### A Beginner Numerical Example

Suppose one loan has:

- `PD = 8% = 0.08`
- `LGD = 60% = 0.60`
- `EAD = $10,000`

Then:

```text
EL = 0.08 × 0.60 × 10,000 = $480
```

This does **not** mean the bank will definitely lose $480 on that specific loan.

It means that, on average, loans with this same risk profile generate an expected loss of $480 per account.

### The Most Important Technical Caution: Horizon Consistency

This is one of the most important things to get right in interviews.

All three components must refer to a **consistent horizon**.

Examples:

- a **12-month PD** should not be multiplied with a **lifetime LGD/EAD framework** unless I explicitly align the horizon
- a **lifetime PD curve** belongs with a lifetime exposure path and a lifetime severity view
- if I change the forecast horizon, the EL interpretation changes too

So the formula is simple, but the horizon discipline behind it is not optional.

---

## Part 2: What I Actually Did in the Project

### The Practical Build

In the project notebook, I used the model outputs as columns and computed expected loss loan by loan.

A simple pattern looks like this:

```python
loan_data['EL'] = (
    loan_data['PD'] *
    loan_data['LGD'] *
    loan_data['EAD']
)
```

Then I aggregated to portfolio level:

```python
portfolio_EL = loan_data['EL'].sum()
portfolio_EL_rate = portfolio_EL / loan_data['funded_amnt'].sum()
```

### What This Means Economically

This turns individual model outputs into:

- an **expected dollar loss per loan**
- a **total expected loss for the portfolio**
- a **portfolio loss rate**, using a chosen denominator such as funded amount or exposure

### Why This Step Matters So Much

Before this step, the project contains separate model components.

After this step, the project becomes a complete credit-risk framework.

That is the transition from:

- model development

to

- risk quantification in money terms

### What This Baseline Still Does Not Capture

A one-shot baseline EL build does not yet capture everything a bank needs. For example:

- time-varying macroeconomic forecasts
- lifetime default timing
- dynamic amortization and prepayment
- stressed loss under recession scenarios
- portfolio correlation effects for capital modeling

That is why the next sections matter.

---

## Part 3: Expected Loss vs. Unexpected Loss

### The Core Distinction

This is one of the most important interview distinctions.

#### Expected Loss (EL)

Expected Loss is the **average** credit loss I expect as part of normal business.

It is the central tendency of loss.

Banks use this kind of number in conversations around:

- pricing
- reserves / allowances
- profitability
- portfolio planning

#### Unexpected Loss (UL)

Unexpected Loss is the **variability around that average**.

It represents the tail-risk problem:

> What if realized losses are materially worse than the expected average?

That is the loss uncertainty that capital exists to absorb.

### Intuition

If EL is the average rainfall in a city, UL is the storm risk.

A city can plan around average rainfall.
A flood is the reason it builds emergency infrastructure.

### A Simple Statistical Interpretation

In a highly simplified one-loan setting, if LGD and EAD were fixed constants, then realized loss depends on a Bernoulli default event.

A rough variability expression would be tied to:

```text
Var(L) ≈ PD × (1 - PD) × (LGD × EAD)^2
```

So the standard deviation contribution is roughly:

```text
UL-like volatility ≈ sqrt(PD × (1 - PD)) × LGD × EAD
```

This is only a simplified intuition, not a full production capital formula.

What matters is the idea:

- EL is about the **mean**
- UL is about the **dispersion around the mean**

### Why This Matters Practically

A bank can often absorb expected loss through:

- pricing margins
- provisioning
- normal earnings

But unexpected loss is why capital buffers matter.

That is why EL and UL should never be treated as interchangeable.

---

## Part 4: Why Portfolio EL Is Additive but Portfolio UL Is Not

### Expected Loss Is Additive

This is a direct consequence of the linearity of expectation.

If a portfolio contains losses `L1, L2, ..., Ln`, then:

```text
E[L1 + L2 + ... + Ln] = E[L1] + E[L2] + ... + E[Ln]
```

So:

```text
Portfolio EL = Σ EL_i
```

This is why summing expected losses across loans is mathematically valid.

### Unexpected Loss Is Not Additive in the Same Way

Variance does not behave like expectation.

For two loans:

```text
Var(L1 + L2) = Var(L1) + Var(L2) + 2Cov(L1, L2)
```

That covariance term is the entire story.

If losses are highly correlated:

- the portfolio becomes more fragile
- diversification benefit shrinks
- tail loss gets worse

If losses are weakly correlated:

- diversification helps
- portfolio volatility is lower than the simple sum of stand-alone volatilities

### Concentration Risk

This is why banks care so much about concentration by:

- geography
- industry
- product type
- collateral type
- vintage
- borrower segment

If too much of the portfolio is exposed to the same macro driver, the correlation rises and the diversification benefit collapses.

### The Interview-Level Conclusion

The clean answer I want to be able to give is:

- **Expected Loss is additive because expectation is linear.**
- **Unexpected Loss depends on correlation, so it is not additive in the same way.**

That one distinction immediately shows statistical maturity.

---

## Part 5: Portfolio Aggregation and Loss Rate Interpretation

### Dollar EL vs. EL Rate

There are two common ways to report portfolio expected loss.

#### 1. Dollar Expected Loss

```text
Portfolio EL = Σ (PD_i × LGD_i × EAD_i)
```

This answers:

> How many dollars do I expect to lose?

#### 2. Portfolio EL Rate

A simple rate form is:

```text
EL Rate = Portfolio EL / Total Exposure
```

In the project, funded amount is a reasonable simple denominator.

In other settings, the denominator may be:

- total EAD
- outstanding balance
- average receivables
- exposure by segment

### Why Segment-Level Aggregation Matters

A portfolio total is useful, but it is not enough.

I should also aggregate by:

- grade or score band
- loan purpose
- term
- origination vintage
- geography

That helps answer questions such as:

- Which segment is driving most expected loss?
- Is higher EL caused by higher PD, higher LGD, or both?
- Is a low-balance segment actually harmless, or just numerous?

### A Very Practical Decomposition

One of the best ways to understand portfolio loss is to decompose contribution by component.

For example, for each segment I can compare:

- average PD
- average LGD
- average EAD
- total EL contribution

That is often much more useful than only showing one final grand total.

---

## Part 6: CECL — The US Accounting Extension of This Framework

### Why This Section Matters

For a US banking context, the relevant accounting framework is **CECL** rather than IFRS 9.

The key practical point is this:

- this project gives me a baseline EL architecture
- CECL extends that architecture into a **lifetime expected credit loss** framework

### The Core CECL Idea

Under CECL, expected credit losses are recognized from initial recognition of the asset and are measured using lifetime expected credit losses for financial assets carried at amortized cost.

That means the bank is not waiting for a missed payment before thinking about loss. It has to estimate expected credit loss over the asset's life using:

- historical credit experience
- current conditions
- reasonable and supportable forecasts

### Why This Changes the Modeling Problem

A one-period EL formula is still conceptually useful, but CECL usually forces a more explicit time structure.

For an installment loan, I may need to think month by month:

- what is the probability of default in month 1, month 2, ..., month T?
- what is the exposure profile through amortization and prepayment?
- what is the severity profile conditional on default at different points in time?

### A Lifetime Form of Expected Loss

A simple lifetime structure can be written as:

```text
Lifetime ECL = Σ_t (Marginal PD_t × LGD_t × EAD_t)
```

where `t` indexes future time periods.

This is a very useful interview formula because it shows the shift from a single-horizon model to a term-structure view.

### Hazard-Rate Interpretation

If `h_t` is the conditional default hazard in month `t`, and `S_t` is survival probability up to time `t`, then:

```text
S_0 = 1
S_t = S_(t-1) × (1 - h_t)
Marginal PD_t = S_(t-1) × h_t
```

Then lifetime loss can be expressed as:

```text
Lifetime ECL = Σ_t (S_(t-1) × h_t × LGD_t × EAD_t)
```

This is the logic behind why survival analysis or discrete-time hazard models become useful in lifetime credit-loss work.

### Why My Baseline PD Scorecard Is Not Automatically a CECL Engine

My origination PD scorecard is a strong baseline risk-ranking model, but CECL usually needs more than a static binary classifier.

Typical gaps include:

- it may be a **through-the-cycle** style build rather than a true point-in-time forecast
- it may produce one probability rather than a **term structure of marginal PDs**
- it may not explicitly incorporate **reasonable and supportable macro forecasts**
- it may not align naturally with monthly exposure runoff

So the right way to say it is:

> The scorecard is a strong starting point, but CECL generally requires additional architecture on top of it.

### Common CECL Implementation Routes

There is not just one valid method. Common approaches include:

- PD × LGD × EAD term-structure frameworks
- discounted cash flow methods
- roll-rate or transition-matrix approaches
- vintage / loss-rate methods for simpler portfolios

The choice depends on product type, data depth, materiality, and governance expectations.

### CECL vs. IFRS 9 — The High-Level Difference I Need to Know

For interviews, the most important high-level distinction is:

- **CECL:** lifetime expected credit loss is recognized from initial recognition
- **IFRS 9:** uses a staging framework, where loss recognition depends on deterioration stage

I do not need to overcomplicate this note with every accounting nuance. I just need to be clear that for a US bank conversation, CECL is the relevant accounting lens.

---

## Part 7: Stress Testing and the Macro Overlay

### The Core Idea

A baseline expected loss number assumes a baseline environment.

But a bank also needs to know:

> What happens if unemployment rises sharply, growth slows, collateral values weaken, and credit conditions deteriorate?

That is the purpose of stress testing.

### Why Stress Changes All Three Components

#### PD under stress

When economic conditions worsen:

- borrowers lose income
- refinancing becomes harder
- delinquencies increase

So **PD usually rises**.

#### LGD under stress

In many portfolios, stress can also worsen recoveries:

- collateral values may fall
- liquidation takes longer
- workout costs rise

So **LGD can also rise**.

#### EAD under stress

For amortizing installment loans, EAD is more structured.

For revolving products, however, stressed borrowers often draw down available credit before default.

So **EAD can rise as well**, especially in revolving books.

### The Multiplicative Effect

Because the loss formula multiplies these components, stress can have a nonlinear effect.

Example:

- baseline: `PD = 5%`, `LGD = 40%`, `EAD = 10,000`
- stressed: `PD = 10%`, `LGD = 60%`, `EAD = 11,000`

Baseline EL:

```text
0.05 × 0.40 × 10,000 = $200
```

Stressed EL:

```text
0.10 × 0.60 × 11,000 = $660
```

The loss does not merely double. It can jump sharply because all three components can move in the wrong direction together.

### How the Macro Link Is Usually Built

A common production architecture uses **satellite models** or macro overlays.

Conceptually:

```text
PD_t = f(borrower_features, unemployment_t, GDP_t, rates_t, ...)
LGD_t = g(collateral_state, macro_t, workout_conditions_t, ...)
EAD_t = h(utilization, product_type, macro_t, ...)
```

These relationships may be estimated with:

- time-series regressions
- panel models
- transition models
- scenario overlays
- expert judgment overlays where data is thin

### The US Supervisory Link

In the US, supervisory stress testing uses macroeconomic scenarios to evaluate how losses and capital evolve under severe conditions.

For interview purposes, the main point I need to understand is:

- a baseline model is not enough
- the model must be able to react to macro scenarios
- the output feeds into portfolio loss projections and capital discussions

### What This Means for My Project

The Lending Club project is the baseline computational skeleton.

A stress-testing extension would require:

1. a scenario path for macro variables
2. a mechanism linking those variables to PD/LGD/EAD
3. re-estimation of losses across future periods
4. aggregation into stressed portfolio loss and capital impact

---

## Part 8: A Practical Modeling Ladder from Baseline EL to Lifetime and Stress

### Step 1: Baseline One-Period EL

```text
EL_i = PD_i × LGD_i × EAD_i
```

Use case:

- foundational project build
- introductory portfolio loss estimation
- first pass segmentation and ranking

### Step 2: Segment-Level Portfolio Diagnostics

Aggregate EL by key portfolio cuts.

Use case:

- concentration review
- business strategy
- pricing discussion
- portfolio steering

### Step 3: Lifetime ECL Structure

Move from one static PD to a term structure of default over time.

Use case:

- CECL-style allowance estimation
- longer-dated products
- forecasting over contractual life or adjusted life assumptions

### Step 4: Stress Overlay

Shock the macro environment and reproject loss under adverse conditions.

Use case:

- stress testing
- capital planning
- downturn sensitivity analysis

This ladder is a clean way to explain progression in an interview.

---

## Part 9: A Clean Python Skeleton

### 1. Baseline Portfolio EL

```python
import pandas as pd

# Assume these are model outputs already attached to each loan
loan_data['EL'] = (
    loan_data['PD'] *
    loan_data['LGD'] *
    loan_data['EAD']
)

portfolio_el = loan_data['EL'].sum()
portfolio_el_rate = portfolio_el / loan_data['funded_amnt'].sum()

segment_summary = (
    loan_data
    .groupby('grade')[['PD', 'LGD', 'EAD', 'EL']]
    .mean()
    .assign(total_EL=loan_data.groupby('grade')['EL'].sum())
)
```

### 2. Lifetime ECL Skeleton

```python
import numpy as np

# monthly_marginal_pd[t], lgd_path[t], ead_path[t] assumed available
lifetime_ecl = 0.0

for t in range(T):
    lifetime_ecl += (
        monthly_marginal_pd[t] *
        lgd_path[t] *
        ead_path[t]
    )
```

### 3. A Simple Stress Overlay Skeleton

```python
# very simplified illustration only
stress_pd_multiplier = 1.50
stress_lgd_multiplier = 1.20
stress_ead_multiplier = 1.05

loan_data['PD_stress'] = (loan_data['PD'] * stress_pd_multiplier).clip(0, 1)
loan_data['LGD_stress'] = (loan_data['LGD'] * stress_lgd_multiplier).clip(0, 1)
loan_data['EAD_stress'] = loan_data['EAD'] * stress_ead_multiplier

loan_data['EL_stress'] = (
    loan_data['PD_stress'] *
    loan_data['LGD_stress'] *
    loan_data['EAD_stress']
)

portfolio_el_stress = loan_data['EL_stress'].sum()
```

This is deliberately simple. Its purpose is to make the flow intuitive:

- baseline loss
- lifetime extension
- stressed re-estimation

---

## Part 10: What I Should Be Able to Explain Clearly

### 1. Why `EL = PD × LGD × EAD` is so important

Because it converts model outputs into expected money loss.

### 2. Why EL is not realized loss

Because EL is an average expectation, not the actual outcome on one specific loan.

### 3. Why EL is additive across loans

Because expectation is linear.

### 4. Why UL is not additive in the same way

Because variance depends on covariance and correlation.

### 5. Why horizon consistency matters

Because PD, LGD, and EAD must refer to the same forecast horizon to make the multiplication meaningful.

### 6. Why CECL needs more than a static scorecard

Because CECL is a lifetime expected credit loss framework and usually needs time structure plus forecast integration.

### 7. Why stress testing can make losses rise sharply

Because PD, LGD, and sometimes EAD can all deteriorate together.

### 8. Why the Lending Club project is still valuable

Because it gives the full conceptual spine of credit-loss integration even though a real bank would add more architecture for lifetime projection and macro scenarios.

---

## Common Mistakes I Want to Avoid

1. **Multiplying mismatched horizons** such as 12-month PD with lifetime severity assumptions.
2. **Treating EL as guaranteed loss** instead of expected average loss.
3. **Adding UL like EL** and ignoring correlation.
4. **Talking about CECL as if it were just one more static score** rather than a lifetime framework.
5. **Assuming baseline loss is enough for a real bank** without macro overlays or scenario analysis.
6. **Ignoring product differences** between installment loans and revolving lines.
7. **Forgetting that model output must be translated into business and accounting interpretation.**

---

## Connections to the Rest of the Notes

- [[1_full_pd_model]] — The PD note provides the default-frequency component that feeds directly into expected loss.
- [[3_lgd_ead_model_rewritten]] — This note provides the loss-severity and exposure components used here.
- [[2_monitoring_model]] — Once expected loss is used in practice, model monitoring extends beyond score stability into portfolio loss stability, realized-vs-expected tracking, and drift in the underlying components.
- [[Tharun-Kumar-Gajula]] — This note supports the larger system by connecting model development, portfolio interpretation, accounting context, and stress testing.

---

## Key Concepts Summary

| Concept | What It Is | Where It Appears in This Note |
|---|---|---|
| **Expected Loss (EL)** | Average expected credit loss | `PD × LGD × EAD` |
| **Unexpected Loss (UL)** | Variability around expected loss | Capital and tail-risk interpretation |
| **Linearity of Expectation** | Why portfolio EL is additive | `Σ EL_i` |
| **Covariance / Correlation** | Why portfolio UL is not simply additive | Portfolio concentration discussion |
| **EL Rate** | Portfolio expected loss scaled by exposure | `Portfolio EL / Total Exposure` |
| **Horizon Consistency** | PD, LGD, EAD must align to same forecast horizon | Core modeling discipline |
| **Lifetime ECL** | Expected loss projected across the life of the asset | CECL section |
| **Marginal PD** | Default probability for a specific future period | Lifetime ECL formula |
| **Hazard Rate** | Conditional default probability in a future period | Survival-style lifetime modeling |
| **CECL** | US lifetime expected credit loss framework | Accounting extension of this project |
| **Reasonable and Supportable Forecasts** | Forward-looking forecast integration in credit-loss estimation | CECL discussion |
| **Stress Testing** | Re-estimating loss under adverse macro scenarios | Macro overlay section |
| **Satellite Models / Macro Overlay** | Link between portfolio loss parameters and macroeconomic variables | Stress architecture |
| **Concentration Risk** | Portfolio fragility when exposures share the same risk driver | UL and portfolio aggregation |

---

*This note is Version 1.0. The next clean extension would be to connect the portfolio loss framework here to a final note on governance, validation expectations for lifetime credit-loss frameworks, and realized-vs-expected backtesting over time.*
