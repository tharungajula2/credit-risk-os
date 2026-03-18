---
title: "Lending Club Credit Risk Masterclass — PD, LGD, EAD, Expected Loss, and Monitoring"
date: 2026-03-19
tags:
  - credit-risk
  - lending-club
  - PD
  - LGD
  - EAD
  - expected-loss
  - scorecard
  - logistic-regression
  - WoE
  - IV
  - monitoring
  - PSI
  - CSI
  - CECL
  - stress-testing
  - model-validation
  - banking
  - data-science
  - beginner-friendly
cluster: "02 — End-to-End Credit Risk Modeling"
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[2_regression_analysis_masterclass]]"
  - "[[3_machine_learning_masterclass]]"
  - "[[5_bank_churn_neural_networks_masterclass]]"
  - "[[6_employee_retention_tree_models_masterclass]]"
  - "[[7_socio_economic_household_classification_masterclass]]"
  - "[[8_twitter_sentiment_nlp_masterclass]]"
---

---

# Lending Club Credit Risk Masterclass — PD, LGD, EAD, Expected Loss, and Monitoring

> This note is my single end-to-end master note for the entire Lending Club credit-risk project. I use one connected story to understand the whole flow from first principles: how I define default, prepare the data, build a PD scorecard, validate it, monitor it after deployment, extend it into LGD and EAD, combine everything into Expected Loss, and then connect that baseline project to CECL and stress testing.
>
> I rewrote this as one note because the split notes felt too fragmented and too code-heavy. My goal here is different: I want a complete beginner to be able to read this note from top to bottom and actually understand the project as one system.

---

## The Project at a Glance

**Dataset family used across the notebooks:**

- Lending Club origination and performance data for **2007–2014** for model development
- Lending Club **2015** data for the monitoring exercise

**What I am trying to learn from one project:**

1. how to define a binary target for default modeling
2. how to prepare messy lending data for modeling
3. how WoE, IV, dummy variables, and reference categories work
4. how logistic regression becomes a scorecard
5. how validation metrics like AUROC, Gini, and KS should be interpreted
6. how a model is monitored after deployment
7. how PD is only one part of a full credit-loss framework
8. how LGD and EAD convert risk ranking into money
9. how Expected Loss, CECL, and stress testing fit into the bigger banking picture

**The full story in one line:**

```text
Raw loan data
→ define Good / Bad
→ preprocess and bin variables
→ build PD scorecard
→ validate it
→ monitor it on new data
→ estimate LGD and EAD
→ combine everything into Expected Loss
→ understand how a real bank would extend this into CECL and stress testing
```

---

## The One Big Idea Behind the Whole Project

A lot of beginner confusion in credit risk comes from seeing the acronyms separately:

- **PD**
- **LGD**
- **EAD**
- **EL**
- **PSI**
- **CECL**

They look like separate chapters, but they are really just different parts of one workflow.

### The full workflow in plain English

When a bank gives a loan, it cares about four big questions:

1. **Will this borrower default?**  
   That is **PD**.

2. **If default happens, what fraction of the exposure will I fail to recover?**  
   That is **LGD**.

3. **How much money will still be outstanding when default happens?**  
   That is **EAD**.

4. **What loss should I expect in money terms?**  
   That is:

```text
Expected Loss = PD × LGD × EAD
```

Then there is one more question after deployment:

5. **Is the model still behaving properly as the world changes?**  
   That is where **monitoring**, **PSI**, **CSI**, and drift analysis come in.

So the project is not really “a PD model plus some extra notes.”  
It is one connected credit-risk system.

---

## What I Actually Built vs. What I Am Learning Beyond the Notebook

This distinction matters a lot because earlier versions mixed project reality and industry extensions too aggressively.

| Topic | What I actually built in the notebooks | What I add here for interview and banking understanding |
|---|---|---|
| Target definition | `good_bad` from `loan_status` | cleaner real-bank bad definitions and indeterminate zone handling |
| PD model | logistic regression scorecard | OOT validation, better deployment framing, PDO intuition |
| Monitoring | re-run preprocessing on 2015 data and compare distributions | cleaner production pipeline design and drift governance logic |
| LGD / EAD | recovery-rate and CCF proxy models on defaulted loans | better modeling alternatives and product-specific caveats |
| Expected Loss | combine PD, LGD, and EAD | CECL and stress testing as extensions |
| Governance | implied in the workflow | clearer explanation of what a real bank would do differently |

That separation makes the note much easier to trust.

---

# Part 1 — The Dataset and the Target Definition

## 1.1 What the raw data gives me

The raw Lending Club data contains:

- borrower application information
- bureau-like credit variables
- loan terms
- performance outcomes through `loan_status`
- collections and recovery fields for defaulted accounts

This is why it is such a useful learning dataset. It is not a toy dataset with only five columns. It looks much more like real retail lending data.

---

## 1.2 The first thing I must define: what is “Bad”?

A model cannot directly learn from text values like:

- `Fully Paid`
- `Current`
- `Charged Off`
- `Default`
- `Late (31-120 days)`
- `Late (16-30 days)`
- `In Grace Period`
- `Does not meet the credit policy. Status:Charged Off`

So I collapse `loan_status` into one binary target:

```python
loan_data['good_bad'] = np.where(
    loan_data['loan_status'].isin([
        'Charged Off',
        'Default',
        'Does not meet the credit policy. Status:Charged Off',
        'Late (31-120 days)'
    ]),
    0,  # Bad
    1   # Good
)
```

### The convention I will use everywhere in this note

This project uses:

- **`1 = Good`**
- **`0 = Bad`**

That means the notebook is coded in the **good-borrower direction**, not the **default = 1** direction.

This is very important because many interview questions assume the opposite convention.

---

## 1.3 So is default the positive event or the negative event?

### Conceptually
Default is the **adverse** event.  
So in business language:

- **Default / Bad** = negative outcome
- **Non-default / Good** = desired outcome

### In this notebook’s coding
The model target is:

- `good_bad = 1` → good
- `good_bad = 0` → bad

So when I later use:

```python
predict_proba(X)[:, 1]
```

I am getting **probability of being Good**, not PD.

If I want **probability of default**, then:

```text
PD = 1 - P(Good)
```

or, equivalently in this notebook setup:

```python
predict_proba(X)[:, 0]
```

because class `0` is the bad/default class.

### Why this matters
This was one of the biggest confusion points in the earlier notes.  
The clean final interpretation is:

> The binary target is coded as `1 = Good` and `0 = Bad`.  
> The logistic model is fit on that target.  
> When I need PD, I take the probability of class `0`, or equivalently `1 - P(Good)`.

---

## 1.4 Is this the same as a full bank default definition?

Not exactly.

### What I did here
I used a practical project definition based on `loan_status`.

### What a bank would usually do more rigorously
A bank would define “bad” using things like:

- 90+ Days Past Due
- charge-off
- bankruptcy
- policy-specific default trigger
- a fixed performance window after origination

So the Lending Club target is a very useful learning target, but it is still a simplified project proxy.

---

## 1.5 The indeterminate zone

One subtle but important issue is that not every delinquency state is equally clean for model development.

For example:

- `Late (31-120 days)` was treated as **Bad**
- `Late (16-30 days)` was treated as **Good**

That is a simplification.

### Why this is tricky
A borrower in early delinquency is not always clearly good or clearly bad. Some cure. Some roll forward into default.

So in production, a bank may create an **indeterminate zone** and exclude ambiguous accounts from development rather than forcing them into one class.

### What I should say in an interview
A clean answer is:

> In the project, I used a practical binary mapping from `loan_status`. For a bank production build, I would tighten the bad definition around a fixed default rule and probably exclude indeterminate delinquency states from development.

That is accurate and safe.

---

# Part 2 — Preprocessing: Turning Raw Lending Data into Model Inputs

This part often overwhelms beginners because the raw columns do not look ready for modeling.

The real lesson is simple:

> A model does not learn from “meaningful column names.”  
> It learns from numeric patterns.  
> So I must convert business fields into numeric, consistent, model-ready features.

---

## 2.1 String cleaning

Some raw fields come as text:

- `emp_length` looks like `"10+ years"` or `"< 1 year"`
- `term` looks like `" 36 months"` or `" 60 months"`

So I convert them into integers.

### Example

```python
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', '0')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

loan_data['term_int'] = loan_data['term'].str.replace(' months', '')
loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])
```

### Why this matters
The model cannot compare the strings `"10+ years"` and `"< 1 year"` mathematically.  
But it can work with `10` and `0`.

---

## 2.2 Date engineering

The raw data also contains dates like:

- `earliest_cr_line`
- `issue_d`

The model usually does not need the raw text date itself.  
It needs a **numerical elapsed-time feature**.

So I convert dates into things like:

- `mths_since_earliest_cr_line`
- `mths_since_issue_d`

### Intuition
If I know a borrower’s earliest credit line was opened a long time ago, that suggests a longer bureau history.  
A longer history can be useful risk information.

### Code idea

```python
loan_data['earliest_cr_line_date'] = pd.to_datetime(
    loan_data['earliest_cr_line'], format='%b-%y'
)

loan_data['mths_since_earliest_cr_line'] = round(
    pd.to_numeric(
        (pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')
    )
)
```

### Important monitoring lesson
In the monitoring notebook, the reference date changes from `2017-12-01` to `2018-12-01` for the 2015 data scoring exercise.

That immediately teaches a production lesson:

> Time-based features must be recomputed consistently using the correct reference date.  
> If I hardcode dates badly, the feature itself becomes wrong.

---

## 2.3 Missing values

Credit data is full of missing values, and the most important thing is **not** to treat them as an annoying cleanup step. Missingness often contains risk information.

### 2.3.1 The three missingness mechanisms I should know

The standard terms are:

- **MCAR — Missing Completely At Random**  
  Missingness has no systematic relationship to either the observed data or the unobserved value itself.

- **MAR — Missing At Random**  
  Missingness depends on other observed variables, even if it does not depend directly on the missing value after conditioning on those observed variables.

- **MNAR — Missing Not At Random**  
  Missingness is related to the missing value itself or to the underlying risk process.

### A simple intuition

- **MCAR** is the least dangerous statistically, because the missing rows are basically a random sample.
- **MAR** can often be handled reasonably well with model-based or segment-based imputation, because other observed fields help explain the gap.
- **MNAR** is the most important case in credit risk, because the fact that something is missing may itself be a risk signal.

### 2.3.2 How this applies to this project

Some examples from the Lending Club workflow:

- `total_rev_hi_lim` is often filled using `funded_amnt`
- some bureau-related fields are filled with `0`
- fields like `mths_since_last_delinq` can be missing because the borrower never had a delinquency

These are **not all the same kind of missingness**.

#### Example 1 — “Never had the event”
If `mths_since_last_delinq` is missing because the borrower never had a delinquency, then missingness is actually informative. In practice, that is often better treated as:

- a special bin
- a flag plus imputed value
- or a “missing / no-event” category in a scorecard

#### Example 2 — proxy-based fill
If `total_rev_hi_lim` is unavailable but `funded_amnt` is available, using a business proxy may be acceptable for a practical project, but I should still document the assumption clearly.

#### Example 3 — operational data failure
If a field is missing because of a pipeline issue, that is a data-quality problem, not a borrower-risk statement. That should be investigated operationally, not hidden blindly with imputation.

### 2.3.3 Best-practice missing-value workflow

A clean workflow is:

1. **ask why the value is missing**
2. **separate structural missingness from accidental missingness**
3. **check whether missingness itself predicts bad rate**
4. **choose treatment based on business meaning**
5. **freeze the treatment for deployment and monitoring**

That fourth step is where beginners often jump too fast.

### 2.3.4 Practical treatment options

#### Option A — Zero fill
Use only when zero has a defensible meaning.

Good examples:

- count of prior events when “missing” really means “none observed”
- utilization-like quantities where the product definition supports zero

Bad example:

- filling income with zero when the borrower clearly does not have zero income

#### Option B — Mean / median / segment imputation
Useful when I need a simple numeric fill and the variable behaves more like a continuous measurement field.

Better than a global mean is often:

- median by segment
- median by grade / product / vintage
- or imputation inside train only, then applied unchanged to test and monitoring data

#### Option C — Proxy imputation
Use a business-related proxy, as in `total_rev_hi_lim ← funded_amnt`, when there is a strong practical relationship.

But this should always be documented because a proxy changes the economic meaning of the feature.

#### Option D — Missing indicator + imputation
A very strong practical pattern is:

- create a binary flag: `is_missing_feature`
- then impute the numeric value

This lets the model learn both:

- the filled numeric level
- and the fact that the original value was missing

#### Option E — Missing as its own scorecard bucket
For WoE / scorecard models, this is often one of the best approaches.

Why?

Because scorecards already work through bins.  
So instead of pretending the missing values are ordinary numeric values, I can make them a dedicated risk bucket and inspect whether that bucket is safer or riskier than the rest.

### 2.3.5 What I should do differently by model family

#### For scorecards / logistic regression
I generally prefer:

- business-rule fills
- missing indicators
- or separate missing buckets

That keeps the treatment interpretable.

#### For tree-based models
Trees can sometimes handle imputed values more flexibly, but I still should not ignore missingness meaning. The model may split on the indicator or on the filled value, but the data design still matters.

### 2.3.6 What I actually did vs what is best practice

#### What I actually did
The notebooks use practical fills that keep the workflow moving and make the feature engineering possible.

#### Best-practice interpretation
For interview purposes, the strongest answer is:

> Missing values are not a single problem. I first classify them as MCAR, MAR, or MNAR in spirit, then I ask whether the missingness itself carries risk information. In scorecards, I often prefer a missing flag or a separate bucket over blindly using a mean fill.

That is the mindset I want to keep.

### 2.3.7 Outliers and extreme values

Outliers are another place where beginners often overreact.  
The goal is **not** to delete every large number. The goal is to distinguish:

- genuine but rare borrower behaviour
- from clear data errors

### What counts as an outlier here?

Potential outlier-type variables in retail lending include:

- `annual_inc`
- `dti`
- revolving utilization
- inquiry counts
- delinquency counts
- months-since variables with impossible negatives or extreme ages due to date parsing issues

### Best-practice order of thinking

#### Step 1 — check for impossible values
These are not “interesting outliers.” They are errors.

Examples:

- negative income where it should not exist
- negative months-since values caused by date conversion problems
- utilization far outside any plausible range because of denominator issues

These should be corrected, capped after investigation, or excluded depending on the root cause.

#### Step 2 — separate skewness from error
A very high income may be rare but real.  
That is not the same as a broken value.

So I should ask:

- is this economically plausible?
- how many such observations exist?
- do they distort the model disproportionately?

#### Step 3 — choose a treatment that matches the model

### Common outlier treatments

#### A. Winsorization / capping
This is one of the most practical methods in credit-risk preprocessing.

Example:

- cap income at the 99th percentile
- cap `dti` at a reasonable upper bound
- cap months-since variables at a sensible extreme

Why it helps:

- protects the model from a few very large values
- keeps all observations in the sample
- is easy to document

#### B. Binning
For scorecards, binning is often even better than raw-value capping.

Why?

Because once I convert a variable into grouped bins or WoE buckets, very large raw values stop dominating the regression numerically. They are absorbed into a top bucket such as:

- `annual_inc: >140K`
- `inq_last_6mths: >6`

This is one reason scorecards are naturally robust to some outlier problems.

#### C. Log transformation
For highly right-skewed variables like income, a log transform can stabilize scale.

This is more common in continuous-modeling workflows than in classic WoE scorecards, but it is still a useful concept to know.

#### D. Robust scaling
This is more relevant in some ML pipelines than in classic scorecards. It uses median and IQR-based scaling instead of mean and standard deviation, so it is less distorted by extremes.

#### E. Deletion
This should be the last resort, not the first instinct.

Deletion is most defensible when the observation is clearly corrupted or impossible. It is much weaker when the value is just rare but real.

### What is the best answer for *this* project?

For this Lending Club scorecard workflow, the strongest answer is:

> For impossible values, I would correct or exclude after investigation. For genuine but extreme values, I prefer capping and, even more importantly, scorecard-style binning and WoE grouping, because that makes the model much less sensitive to a handful of extreme observations.

That fits this project very well.

---
## 2.4 Dummy variables

A model cannot directly estimate coefficients for categories like:

- `grade = A`
- `home_ownership = RENT`
- `purpose = debt_consolidation`

So I convert categories into dummy variables.

Example:

```text
grade:A
grade:B
grade:C
...
grade:G
```

Each dummy is `1` if the loan belongs to that category and `0` otherwise.

---

## 2.5 Reference categories

When I create dummies for all categories, one category must be dropped as the **reference category**.

Why?

Because if I keep every dummy, I create perfect multicollinearity.  
The model can reconstruct one dummy from the others.

### Interpretation
Suppose I keep:

- `grade:A`
- `grade:B`
- ...
- `grade:F`

and drop:

- `grade:G`

Then all the grade coefficients are interpreted **relative to grade G**.

That means:

- if the coefficient for `grade:A` is positive, grade A is safer than grade G
- if the coefficient for `grade:D` is still positive, grade D is also safer than grade G, but maybe less safe than grade A

### Why the riskiest bin as reference is a good teaching choice
Using the riskiest bin as reference makes the scorecard easier to interpret because better categories tend to get positive coefficient contributions relative to the worst bucket.

---

# Part 3 — Weight of Evidence (WoE) and Information Value (IV)

This is one of the most important concepts in scorecard modeling, and also one of the most commonly confused.

So I want this section to be extremely clear.

---

## 3.1 Why I even need WoE

Logistic regression can work with raw numeric variables and with dummies.  
So why do scorecard practitioners spend so much time on WoE?

Because WoE does three useful things at once:

1. it converts categories or bins into a numeric measure tied directly to risk
2. it stabilizes relationships and often makes them more monotonic
3. it makes variable-level interpretation much easier

In scorecards, WoE is not just a transformation.  
It is part of the modeling philosophy.

---

## 3.2 The exact WoE direction in this project

The notebook computes:

```text
WoE = ln(%Good / %Bad)
```

More explicitly:

```text
WoE_bin = ln(
    proportion of all Goods in that bin
    ------------------------------------
    proportion of all Bads  in that bin
)
```

### This means:

- **positive WoE** → safer bin
- **negative WoE** → riskier bin

This is the clean direction I will use everywhere in this note.

### Why?
Because the notebook target is coded as:

- `1 = Good`
- `0 = Bad`

and the WoE function explicitly uses:

```python
df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
```

So if a bin contains relatively more goods than bads, the WoE becomes positive.

---

## 3.3 A simple worked example

Suppose one income bin contains:

- 20% of all Goods
- 10% of all Bads

Then:

```text
WoE = ln(0.20 / 0.10)
    = ln(2)
    ≈ 0.693
```

Positive WoE.

That means the bin is safer than average because it carries a higher share of Goods than Bads.

Now suppose another bin contains:

- 5% of all Goods
- 15% of all Bads

Then:

```text
WoE = ln(0.05 / 0.15)
    = ln(1/3)
    ≈ -1.099
```

Negative WoE.

That means the bin is riskier than average.

---

## 3.4 How I should interpret WoE

### Positive WoE
This bin is relatively concentrated with Goods.  
It is safer.

### Negative WoE
This bin is relatively concentrated with Bads.  
It is riskier.

### WoE near zero
The bin does not strongly separate good and bad.

That is the clean intuition.

---

## 3.5 Fine classing and coarse classing

In practice, the workflow is usually:

1. create many small bins first  
2. inspect bad rates / WoE patterns  
3. combine them into cleaner business bins

This is why the notebook creates grouped ranges such as:

- `emp_length:0`
- `emp_length:1`
- `emp_length:2-4`
- `emp_length:5-6`
- `emp_length:7-9`
- `emp_length:10`

and similar grouped bins for:

- income
- delinquency counts
- inquiry counts
- revolving utilization
- months since issue
- months since earliest credit line

### Why I should do this
Because raw continuous variables are often noisy.  
The grouped bins produce a more stable risk pattern and an easier scorecard story.

---

## 3.6 Information Value (IV)

After WoE comes **IV**, which tells me how useful a variable is for separating Goods from Bads.

The notebook computes IV as:

```text
IV = Σ ( %Good_bin - %Bad_bin ) × WoE_bin
```

### Intuition
If a variable has bins where the Good and Bad distributions differ meaningfully, then it helps the model discriminate risk.

If the Good and Bad distributions look almost identical across all bins, then the variable adds little value.

### How I should talk about IV
A practical beginner explanation is:

> IV tells me whether a variable is worth keeping because it measures how strongly that variable separates good borrowers from bad borrowers.

### Important caution
IV is useful, but it is not the only decision rule.

A variable may have decent IV and still be problematic because of:

- instability over time
- poor business meaning
- fairness concerns
- leakage risk
- governance concerns

So IV helps selection, but it does not replace judgment.

---

# Part 4 — The PD Model: Logistic Regression as a Credit Scorecard

Now that the inputs are prepared, I move to the PD model itself.

This is the central model in the project, but it becomes much easier once the earlier sections are understood.

---

## 4.1 Why logistic regression?

The target is binary:

- Good
- Bad

So I need a model that outputs a probability between `0` and `1`.

Linear regression is not ideal for this because it can predict outside `[0,1]`.

Logistic regression solves that problem by modeling the **log-odds** and then mapping them through the logistic function into a valid probability.

---

## 4.2 The logistic regression equation

The linear score is:

```text
z = β0 + β1x1 + β2x2 + ... + βkxk
```

Then the logistic function gives:

```text
P(Good | x) = 1 / (1 + e^(-z))
```

Because this notebook is coded in the Good direction, the direct model probability is:

```text
P(Good)
```

Then:

```text
PD = 1 - P(Good)
```

or equivalently:

```text
PD = P(Bad)
```

which in the notebook setup is the probability of class `0`.

---

## 4.3 What a coefficient means

A logistic regression coefficient tells me how a feature changes the **log-odds** of being Good.

### For a continuous variable
If `β1` is positive, then as `x1` rises, the log-odds of being Good rise.

### For a dummy variable
If the coefficient for `grade:A` is positive and `grade:G` is the reference, then grade A is safer than grade G, holding other variables constant.

That is why reference categories matter so much.

---

## 4.4 Why logistic regression is so useful in scorecards

Even though more complex models exist, logistic regression remains a strong choice for scorecards because:

- it is interpretable
- coefficients are easy to document
- it works naturally with WoE-transformed variables
- regulators and validators understand it well
- it maps naturally into score scaling

So even if tree models or gradient boosting may perform better in some contexts, logistic regression is still a very important benchmark and often the preferred scorecard model.

---

## 4.5 P-values and variable pruning

The PD notebook goes beyond a plain sklearn logistic regression by using a custom wrapper to approximate p-values.

That lets me look at coefficient significance and prune less useful variables.

### The idea
I do not want a giant model with every possible variable if many of them contribute little or are unstable.

### But I should remember
In production, variable selection is not only about p-values.  
It is also about:

- business meaning
- monotonicity
- stability
- governance
- fairness
- documentation quality

---

## 4.6 Scorecard scaling

After fitting the logistic model, the notebook converts coefficient contributions into a scorecard range of roughly:

- **300**
- to **850**

### What the notebook actually does
It uses a **min-max style scaling** based on the minimum and maximum possible coefficient sums across reference groups and selected bins.

That is a practical project method and it works.

### What a bank often prefers conceptually
A more standard scorecard framing uses:

- **base odds**
- **base score**
- **Points to Double the Odds (PDO)**

The earlier notes talked about PDO because it is useful interview knowledge, but the actual notebook’s scaling logic is closer to min-max score mapping.

### The clean way I should say it
A safe answer is:

> In the project, I converted logistic regression outputs into a scorecard using a coefficient-based score scaling workflow. For interview discussions, I also understand the cleaner industry framing of base odds, base score, and PDO.

That is honest and strong.

---

## 4.7 From score back to PD

Once the score is created, I can move between:

- coefficient sum
- score
- probability

The notebook explicitly reconstructs probability from the score using the logistic relationship.

That is an important conceptual point:

> The score is not magic.  
> It is just a scaled version of the logistic regression output.

Higher score means higher odds of being Good, which means lower PD.

---

# Part 5 — Validation: How I Judge Whether the PD Model Is Useful

A model is not useful just because it runs.  
It has to discriminate risk well.

The PD notebook uses a standard validation suite:

- confusion matrix
- ROC / AUROC
- Gini
- KS

---

## 5.1 The train/test split used in the project

The notebooks use a standard:

- **80/20 random split**
- with a fixed random seed

This is perfectly fine for a learning project.

But it is important to understand the limitation.

### Why it is optimistic
The Lending Club data spans multiple years and macro conditions.  
A random split mixes earlier and later years into both training and test.

So the test set is not a truly future sample.

### What a bank would do
A bank would prefer **Out-of-Time (OOT)** validation, where I train on earlier vintages and test on a later vintage.

That produces a more realistic performance estimate.

### The clean interview answer
> My project used a random train/test split for learning and implementation. In a real banking model validation framework, I would prefer an out-of-time design to avoid optimistic performance estimates.

---

## 5.2 Confusion matrix

A confusion matrix tells me how many predictions fall into each bucket **after I choose a cut-off**.

Because this project is coded as:

- `1 = Good`
- `0 = Bad`

the clean confusion matrix is:

|  | **Actual Good (1)** | **Actual Bad (0)** |
|---|---|---|
| **Predicted Good (1)** | True Good | False Good |
| **Predicted Bad (0)** | False Bad | True Bad |

If I switch into the more standard “positive class = bad” language for validation, then the same four cells are usually written as:

|  | **Actual Bad** | **Actual Good** |
|---|---|---|
| **Predicted Bad** | True Positive (TP) | False Positive (FP) |
| **Predicted Good** | False Negative (FN) | True Negative (TN) |

I care about this second layout more for interviews, because most people discuss default detection with **Bad as the positive class**.

### 5.2.1 What each cell means in plain English

- **TP**: I flagged the borrower as risky, and the borrower really did go bad
- **FP**: I flagged the borrower as risky, but the borrower would actually have been fine
- **FN**: I passed the borrower as acceptable, but the borrower later went bad
- **TN**: I passed the borrower, and the borrower really stayed good

### 5.2.2 Which error is more dangerous here?

In most credit underwriting settings:

- **False Negative** is the more dangerous error  
  because I approved risk that later defaulted

- **False Positive** is still costly  
  because I declined or penalized a borrower who might have performed well

So the costs are **asymmetric**.

That is why a confusion matrix is not just a counting table. It is a business-cost table.

### 5.2.3 Why the cut-off matters so much

Suppose the model outputs a PD for every borrower.

If I set a very **low cut-off** for calling someone “Bad,” then:

- I catch more defaulters
- but I also reject more good borrowers

If I set a very **high cut-off**, then:

- I approve more good borrowers
- but I also miss more future defaulters

So the confusion matrix changes with the threshold.  
It is not a permanent property of the model.  
It is a property of the **model + threshold together**.

### 5.2.4 The core threshold metrics

Using Bad as the positive class:

```text
Recall (Sensitivity / TPR) = TP / (TP + FN)
```

This answers:

> Of all the borrowers who actually went bad, how many did I catch?

```text
Precision = TP / (TP + FP)
```

This answers:

> Of all the borrowers I flagged as bad, how many really were bad?

```text
Specificity (TNR) = TN / (TN + FP)
```

This answers:

> Of all the borrowers who were actually good, how many did I correctly leave alone?

```text
False Positive Rate = FP / (FP + TN) = 1 - Specificity
```

This answers:

> How often did I wrongly reject a good borrower?

```text
False Negative Rate = FN / (FN + TP) = 1 - Recall
```

This answers:

> How often did I miss an actual defaulter?

### 5.2.5 What should I focus on in this project?

For underwriting-style default modeling, **Recall on the Bad class** is often extremely important because missing bad borrowers is expensive.

But precision also matters, because if I chase recall blindly, I may reject far too many good borrowers.

So the right answer is not:

- “only recall matters”

The better answer is:

> I usually care strongly about recall on bad borrowers because missed defaults are expensive, but I still need enough precision and specificity so the business does not over-reject good applicants.

That is a much stronger interview answer.

---

## 5.3 Precision, recall, F1, and threshold trade-offs

This is the part that often feels abstract until I connect it to the project.

### 5.3.1 Recall

```text
Recall = TP / (TP + FN)
```

High recall means:

- I catch a large share of future bads
- I am less likely to miss risky borrowers

In this project, recall is closely tied to the risk appetite question:

> How aggressively do I want to catch bad borrowers?

### 5.3.2 Precision

```text
Precision = TP / (TP + FP)
```

High precision means:

- when I call someone risky, I am usually right
- my bad flags are not mostly noise

This matters because a model that flags too many good borrowers as bad may become unusable commercially.

### 5.3.3 F1 score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is the harmonic mean of precision and recall.

### Why harmonic mean, not average?
Because F1 punishes imbalance.

If one of precision or recall is very low, F1 also becomes low.

### When F1 is useful
F1 is useful when I want one threshold-based metric that forces me to care about both:

- catching bads
- and not flagging too many goods

### Limitation of F1 in banking
F1 still assumes a kind of symmetric compromise between precision and recall.  
But real credit decisions often have **business-specific costs**, not a generic 50/50 trade-off.

So F1 is useful, but it is not the final business truth.

### 5.3.4 Accuracy

```text
Accuracy = (TP + TN) / Total
```

Accuracy can be very misleading in imbalanced datasets.

If 85% of borrowers are good, then a naive model that predicts everyone as good can already achieve around 85% accuracy while being useless for risk detection.

So in this project, accuracy is **not** the metric I should lead with.

### 5.3.5 A simple way to remember the threshold metrics

- **Recall** = “Did I catch the bads?”
- **Precision** = “Are my bad flags believable?”
- **Specificity** = “Did I leave good borrowers alone?”
- **F1** = “Am I balancing precision and recall reasonably?”
- **Accuracy** = “How many total decisions were correct?”  
  Useful, but often misleading by itself.

---

## 5.4 ROC and AUROC

The ROC curve plots:

- **True Positive Rate (Recall)**
against
- **False Positive Rate**

across all possible thresholds.

### 5.4.1 Why ROC exists

A confusion matrix depends on one chosen threshold.  
But before choosing that threshold, I want to know whether the model has good **rank-ordering power** in general.

That is what ROC/AUROC gives me.

### 5.4.2 A clean intuition

Imagine I rank borrowers from safest to riskiest using model score or PD.

A good model should place most future bads closer to the risky end and most goods closer to the safe end.

AUROC answers:

> Across all threshold choices, how well does the model separate the two classes?

### 5.4.3 The probabilistic interpretation

AUROC is the probability that if I randomly choose:

- one actual good borrower
- one actual bad borrower

the model ranks them in the correct order.

Because this project’s raw notebook probability is in the Good direction, I have to be careful with wording. The safest interpretation is:

> AUROC measures whether the model assigns a more favorable score to goods than bads, or equivalently a lower PD to goods than bads.

### 5.4.4 How to interpret AUROC values

- **0.50** → no discrimination; random ordering
- **0.60–0.70** → weak
- **0.70–0.80** → reasonable / acceptable
- **0.80–0.90** → strong
- **>0.90** → unusually high; I should check for leakage or target contamination

### 5.4.5 Why AUROC is so useful

AUROC is threshold-free.

That means it tells me about the model’s general ranking ability before I decide where to cut for approve / reject / refer decisions.

### 5.4.6 What AUROC does *not* tell me

AUROC does **not** tell me:

- the best business cut-off
- the expected money loss
- whether the calibration is perfect
- whether the model is stable over time

It is a discrimination metric, not the whole evaluation story.

---

## 5.5 Gini

Gini is closely related to AUROC:

```text
Gini = 2 × AUROC - 1
```

So if:

- AUROC = 0.50, then Gini = 0
- AUROC = 0.75, then Gini = 0.50
- AUROC = 0.80, then Gini = 0.60

### 5.5.1 Why Gini is used so much in credit risk

Credit-risk scorecard discussions often use Gini because it is just a rescaled version of AUROC that puts random performance at **0** instead of **0.5**.

That makes it feel more “incremental”:

- **0** means no useful discrimination
- higher positive values mean more separation power

### 5.5.2 The easy way to remember it

- **AUROC** is the probability-style discrimination measure
- **Gini** is just the same information on a different scale

So I should never treat them as two unrelated metrics.

---

## 5.6 KS statistic

The **KS statistic** is especially important in retail credit.

It measures the maximum separation between the cumulative distributions of Goods and Bads across the score range.

### 5.6.1 Intuition without math first

Imagine I sort all borrowers by risk score from riskiest to safest.

Now I walk down that sorted list and keep track of two cumulative curves:

- cumulative share of all Bads seen so far
- cumulative share of all Goods seen so far

At each score point, I ask:

> How far apart are these two cumulative curves?

The **largest gap** is the KS statistic.

### 5.6.2 Why KS is useful in scorecards

KS is very intuitive for lending because it directly tells me:

> At the best separation point, how far apart are the bad and good populations?

That makes it very natural for score-band and cut-off discussions.

### 5.6.3 Practical interpretation

- higher KS means stronger separation
- a low KS means goods and bads are too mixed together
- the score band around the KS peak is often a very informative operating region for underwriting discussion

### 5.6.4 How KS differs from AUROC

- **AUROC** summarizes separation across **all thresholds**
- **KS** focuses on the **single best separation point**

So AUROC is more global, while KS is more “maximum gap at the best point.”

---

## 5.7 How these metrics fit together in one picture

This is the cleanest way I know to connect them:

### Threshold-dependent metrics
These depend on the cut-off I choose:

- confusion matrix
- precision
- recall
- specificity
- F1
- accuracy

These answer:

> “How does the model behave if I actually make a yes/no decision at this threshold?”

### Threshold-independent or mostly rank-based metrics
These look at broader discrimination:

- ROC / AUROC
- Gini
- KS

These answer:

> “Before locking a threshold, how well does the model separate good and bad borrowers overall?”

### Why I need both families
If I only look at AUROC, I still do not know the business impact of my chosen cut-off.  
If I only look at the confusion matrix at one threshold, I may miss the broader ranking quality of the model.

So the correct answer is:

> I use AUROC/Gini/KS to judge discrimination and rank ordering, then I use confusion-matrix-based metrics to choose and defend an operating threshold.

That is exactly how these pieces should connect in my head.

---

## 5.8 Metrics are not everything

A model can have acceptable AUROC or KS and still be problematic if:

- the bad definition is weak
- the split is optimistic
- the variables are unstable
- the preprocessing is inconsistent
- the model drifts after deployment
- the model is poorly calibrated
- the cut-off does not match business risk appetite

That is exactly why the monitoring part matters.

---
# Part 6 — Monitoring: What Happens After the PD Model Is Built

This is where many beginners think the project is “finished,” but it is not.

A model is only useful if it remains reliable after deployment.

The monitoring notebook is valuable because it turns the scorecard into a lifecycle story.

---

## 6.1 What the monitoring notebook actually does

It takes new Lending Club data from **2015** and asks:

1. Can I preprocess it the same way as the development data?
2. What scores does the existing model produce on that new data?
3. Has the score distribution shifted?
4. Which characteristics have shifted?
5. Is this population drift, concept drift, or a pipeline problem?

That is the right kind of question after deployment.

---

## 6.2 The first lesson: preprocessing must be identical

The monitoring notebook copies the same preparation logic used in model development:

- string cleaning
- date transformations
- missing value fills
- bin creation
- dummy variables
- reference category handling

### Why this matters
If I change the preprocessing, then I am no longer feeding the same model the same type of input.

The model coefficients stay frozen, but the meaning of the inputs changes.

That is not real monitoring.  
That is accidentally creating a different model environment.

### The production lesson
In a real bank, I should package preprocessing and modeling together into a controlled pipeline rather than manually copy-pasting notebook code.

---

## 6.3 Why time features are especially tricky in monitoring

The monitoring notebook changes the reference date from development to the later monitoring period.

That is correct in spirit because elapsed-time features should move forward as time moves forward.

But it also shows the operational risk:

> If this update is manual, someone can forget it or do it inconsistently.

So monitoring teaches not only model stability, but also **pipeline integrity**.

---

## 6.4 PSI — Population Stability Index

PSI measures whether the **score distribution** has shifted between a reference sample and a new sample.

The usual formula is:

```text
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

### Beginner intuition
Suppose I divide scores into bins.  
Then I compare:

- what percentage of loans used to fall into each bin
- what percentage of new loans now fall into each bin

If those percentages changed a lot, PSI rises.

### What PSI is telling me
PSI does **not** say the model is wrong.  
It says the **population being scored now looks different from the old one**.

That is an important distinction.

---

## 6.5 Common PSI threshold language

A common practitioner rule-of-thumb is:

- **PSI < 0.10** → stable
- **0.10 to 0.25** → moderate shift
- **> 0.25** → significant shift

I treat these as practical monitoring signals, not as sacred laws of nature.

---

## 6.6 CSI-style feature monitoring

The monitoring notebook goes further than portfolio-level PSI.  
It compares dummy-bin proportions between the old and new data and sums contributions by original feature name.

That is effectively a **feature-level stability view**, i.e. CSI-style analysis.

### Why this matters
PSI tells me:

> Something changed.

Feature-level contributions tell me:

> What changed.

For example, if the largest shifts come from:

- `addr_state`
- `annual_inc`
- `dti`

then the borrower mix has changed in a concrete, diagnosable way.

---

## 6.7 Population drift vs. concept drift

This distinction is one of the most useful interview concepts.

### Population drift
The type of borrowers has changed.

Examples:

- different states
- different income ranges
- different loan purposes
- different product mix

The model relationship may still be valid.  
The input population simply moved.

### Concept drift
The relationship between the features and bad outcome changed.

Example:

- income used to be strongly protective
- now, under different macro conditions, the same income level no longer protects as much

That means the coefficients or the model structure may no longer represent reality.

### Clean difference
- **Population drift** = the **who** changed
- **Concept drift** = the **relationship** changed

---

## 6.8 What I should do when drift appears

Monitoring is not just about computing PSI.  
It is about deciding what action to take.

A simple ladder is:

1. verify the data pipeline first  
2. inspect which variables shifted  
3. compare realized outcomes if they are available  
4. decide whether to:
   - keep monitoring
   - recalibrate
   - rebuild the model

### Why pipeline checks come first
Sometimes a dramatic distribution shift is not economics at all.  
It is a broken data feed, a changed schema, or missing bureau information.

That is why monitoring is partly a data-quality exercise.

---

# Part 7 — PD Is Not the Whole Story: Why I Need LGD and EAD Too

A PD scorecard answers:

> How likely is default?

But a bank does not lose money because of default probability alone.  
It loses money when default probability combines with:

- loss severity
- exposure amount

So to move from **risk ranking** to **loss estimation**, I need:

- **LGD**
- **EAD**

---

## 7.1 One simple example

Suppose two borrowers each have the same PD:

- Borrower A → `PD = 10%`
- Borrower B → `PD = 10%`

But:

- Borrower A → `LGD = 20%`, `EAD = 5,000`
- Borrower B → `LGD = 80%`, `EAD = 20,000`

Then:

```text
EL_A = 0.10 × 0.20 × 5,000 = 100
EL_B = 0.10 × 0.80 × 20,000 = 1,600
```

Same PD. Completely different economic risk.

That is the real reason notes 3 and 4 had to be pulled into this master note. Without them, the project is only a scorecard, not a fuller loss framework.

---

# Part 8 — LGD: Loss Given Default

## 8.1 What LGD means

**LGD** is the fraction of exposure that remains lost after recoveries.

The basic identity is:

```text
Recovery Rate = Recoveries / Exposure
LGD = 1 - Recovery Rate
```

In the notebook, the practical working definition is:

```text
recovery_rate = recoveries / funded_amnt
LGD = 1 - recovery_rate
```

### Beginner intuition
If I recover most of the money, LGD is low.  
If I recover very little, LGD is high.

---

## 8.2 Why LGD is modeled only on defaulted loans

This is a foundational concept.

LGD is **conditional on default**.

If a loan never defaulted, I never observed a post-default recovery process, so I do not have an LGD outcome to model.

That means:

- **PD** uses the full origination population
- **LGD** uses the defaulted sample only
- **EAD** is also modeled conditional on default

This separation of frequency and severity is essential.

---

## 8.3 The exact defaulted sample used here

For the LGD / EAD notebook, the severity sample is narrowed to charged-off style default outcomes, because those observations contain realized collections and recovery behavior.

That is stricter than the PD target setup, and that is actually okay.

### Why?
Because PD asks:

> Will this borrower default?

LGD asks:

> For loans that already ended up in default / charge-off resolution, how much did I recover?

The modeling samples do not need to be identical because the parameter meanings are different.

---

## 8.4 Recovery rate is a difficult target

Recovery rate is not a clean bell-shaped variable.

Usually there are:

- many loans with exactly **zero** recovery
- some with small positive recovery
- fewer with larger recovery

So the target is **semi-continuous** and **bounded between 0 and 1**.

That is why a single plain regression is not ideal.

---

## 8.5 The two-stage LGD logic

This is one of the most important modeling ideas in the whole project.

Instead of forcing one model to explain everything, I split LGD into two questions:

### Stage 1
Did I recover **anything at all**?

That is a binary classification problem:

```python
loan_data_defaults['recovery_rate_0_1'] = np.where(
    loan_data_defaults['recovery_rate'] == 0,
    0,
    1
)
```

### Stage 2
If recovery is positive, **how much** did I recover?

That is a continuous regression problem on the positive-recovery subset.

### Combined expected recovery

```text
Expected Recovery
= P(recovery > 0 | x) × E(recovery_rate | recovery > 0, x)
```

Then:

```text
Expected LGD = 1 - Expected Recovery
```

### Why this is a strong idea
Because it separates two very different questions:

- recovery incidence
- recovery magnitude

That is much more realistic than a one-stage OLS model trying to explain a spike-at-zero target.

---

## 8.6 The models used

### Stage 1
Logistic regression

That makes sense because the target is binary:

- `0` = no recovery
- `1` = positive recovery

### Stage 2
Linear regression

That is a useful baseline, but I must understand its limitation:

- recovery rate is bounded between `0` and `1`
- linear regression can predict outside that range

So the notebook clips the predictions back into `[0,1]`.

### What I should say in an interview
> I used a two-stage LGD approach: logistic regression for whether recovery is positive, and linear regression for recovery magnitude conditional on recovery. I understand that OLS is a baseline because bounded proportion targets are often better served by approaches like fractional logit, beta-style modeling, or tree-based alternatives.

That answer is strong and honest.

---

## 8.7 What makes LGD different from PD conceptually

PD is about **whether** default happens.

LGD is about **what happens after default**.

That means LGD is influenced by things like:

- collections effectiveness
- legal process
- product type
- collateral, if any
- economic conditions
- cure and workout behavior

So LGD is often more operational and portfolio-specific than PD.

---

## 8.8 Very important dataset caveat

Lending Club is mostly an **unsecured installment-loan** style environment.

That means:

- no heavy collateral story
- recovery patterns are different from mortgage / auto / secured books
- the absolute recovery levels should not be transferred blindly to other lending products

But the mechanics are still very valuable for learning.

---

# Part 9 — EAD: Exposure at Default

## 9.1 What EAD means

**EAD** is the amount of exposure outstanding when default happens.

In some products that is straightforward.  
In others it is much harder.

### Installment lending intuition
For an amortizing loan, the balance usually declines over time.  
So by the time default happens, EAD may be lower than origination amount.

### Revolving lending intuition
For a credit card or line of credit, EAD can be more complex because borrowers may draw down available credit before default.

That is why EAD is strongly product-dependent.

---

## 9.2 The EAD proxy used in this project

The notebook creates a **CCF-style** target:

```text
CCF = (funded_amnt - total_rec_prncp) / funded_amnt
```

Then estimated EAD is:

```text
EAD = CCF × funded_amnt
```

### Beginner interpretation
This is basically asking:

> What fraction of the originally funded amount was still effectively outstanding at the time the loan went bad?

This is a reasonable installment-loan proxy.

---

## 9.3 Why I should be careful with the term “CCF”

In banking, **CCF** is often strongly associated with off-balance-sheet or revolving products, where the key question is how much undrawn exposure becomes drawn before default.

In this project, the formula is serving as a practical exposure-ratio target inside an installment-loan setting.

So the clean way to say it is:

> I used a CCF-style exposure proxy to estimate EAD for this installment-loan dataset. I understand that full revolving-credit EAD modeling is a different problem.

That is the right interview-safe framing.

---

## 9.4 The modeling approach used here

The notebook uses a regression approach for the CCF target.

Again, because the target is bounded between `0` and `1`, simple linear regression is only a baseline and may need clipping.

So the same caveat applies as for LGD stage 2:

- useful baseline
- not the final word

---

# Part 10 — Expected Loss: Turning Model Outputs into Money

Now the project becomes economically meaningful.

I have:

- PD from the scorecard
- LGD from the recovery model
- EAD from the exposure model

So I can finally calculate:

```text
EL = PD × LGD × EAD
```

---

## 10.1 The most important equation in the whole note

For one loan:

```text
Expected Loss_i = PD_i × LGD_i × EAD_i
```

For the portfolio:

```text
Portfolio Expected Loss = Σ_i EL_i
```

This is the key bridge from machine learning / statistics into banking economics.

---

## 10.2 How PD is actually pulled into the EL notebook

This is an important consistency point.

The EL notebook loads the saved PD model and computes:

```python
reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[:, 0]
```

That is exactly consistent with the target coding:

- class `0` = Bad
- class `1` = Good

So in the EL notebook, PD is explicitly taken as the **probability of class 0**, i.e. bad/default.

That is one of the cleanest ways to explain the earlier confusion.

---

## 10.3 Loan-level EL vs portfolio-level EL

### Loan-level EL
Each loan gets its own:

- PD
- LGD
- EAD
- EL

### Portfolio-level EL
Then I add all those expected losses together.

This is why EL is so useful operationally:

- pricing
- provisioning intuition
- expected loss budgeting
- portfolio strategy
- concentration discussions

---

## 10.4 Expected loss vs realized loss

Expected loss is not what I lose on every single loan.

It is the average loss I expect over many similar loans.

### Example
A specific loan might:

- never default → realized loss = 0
- default and lose a lot → realized loss is large

But EL is the average expected amount before I know which exact path will happen.

That is why it is called **expected** loss.

---

## 10.5 Expected loss vs unexpected loss

A useful interview distinction is:

- **Expected Loss (EL)** → average anticipated loss
- **Unexpected Loss (UL)** → volatility around that average

EL is the central focus of this project.  
UL matters more when thinking about capital, tail risk, and portfolio volatility.

For this note, I mainly need to know the conceptual difference.

---

## 10.6 Horizon consistency

This is one of the most important advanced cautions.

If I multiply:

- a 12-month PD
- with a downturn LGD
- with a point-in-time EAD measured on a different horizon

then the result may be conceptually inconsistent.

So PD, LGD, and EAD should be aligned to a coherent horizon as much as possible.

That is a very strong interview point.

---

# Part 11 — CECL and Stress Testing: Where the Project Extends in a Real Bank

This part is important, but I want to keep the scope honest.

The Lending Club notebooks give me a **baseline one-period expected-loss framework**.

They do **not** by themselves create a full production CECL engine or a full supervisory stress-testing system.

But the project is still an excellent starting skeleton for those extensions.

---

## 11.1 CECL in simple language

For a US banking conversation, the key idea is:

> CECL requires expected credit loss recognition over the relevant lifetime horizon, not only a simple one-period static estimate.

So the basic project formula:

```text
EL = PD × LGD × EAD
```

becomes a multi-period idea.

A simple lifetime structure is:

```text
Lifetime ECL = Σ_t (Marginal PD_t × LGD_t × EAD_t)
```

where `t` runs through future periods.

### What changes conceptually
Instead of one PD number, I may need a **term structure of default risk** through time.

That is why lifetime loss frameworks often use:

- transition approaches
- vintage approaches
- hazard or survival ideas
- macro overlays

---

## 11.2 Why the project PD scorecard is not automatically a CECL model

My scorecard is a strong origination risk-ranking model.

But CECL often needs more:

- lifetime horizon
- runoff through time
- potentially macro-sensitive forecasts
- consistent term structures for PD, LGD, and EAD

So the correct statement is:

> The scorecard is a strong starting point, but CECL usually needs additional architecture on top of it.

That is the right level of sophistication.

---

## 11.3 Stress testing

Stress testing asks:

> What happens to losses if the economy worsens significantly?

### Why stress matters
Under stress:

- **PD** can rise
- **LGD** can worsen because recoveries weaken
- **EAD** can increase for some products, especially revolving products

Because EL multiplies these components, the total loss can jump sharply.

### Simple stressed example

Baseline:

- `PD = 5%`
- `LGD = 40%`
- `EAD = 10,000`

Baseline EL:

```text
0.05 × 0.40 × 10,000 = 200
```

Stressed:

- `PD = 10%`
- `LGD = 60%`
- `EAD = 11,000`

Stressed EL:

```text
0.10 × 0.60 × 11,000 = 660
```

The loss does not just double.  
It can increase nonlinearly because all three components can move adversely.

---

## 11.4 What I should say about CECL and stress in this project

A good answer is:

> My Lending Club project builds the baseline mechanics: PD, LGD, EAD, and portfolio expected loss. I understand that CECL adds a lifetime horizon and that stress testing adds scenario sensitivity. I would describe my project as the core computational skeleton rather than a full production CECL or supervisory stress engine.

That is confident and accurate.

---

# Part 12 — What This Whole Project Teaches Me About Banking, Not Just Modeling

This is one of the most important takeaways.

At first glance, this looks like a machine-learning project.

But it is really teaching me several layers at once:

## 12.1 Risk definition
Before modeling, I must define what “bad” means.

## 12.2 Data engineering
Raw lending data must be turned into stable model inputs.

## 12.3 Statistical modeling
Binary outcomes, bounded targets, and score scaling all require the right model structure.

## 12.4 Validation thinking
Performance metrics must match the business question.

## 12.5 Lifecycle thinking
A model is not finished when it is built. It must be monitored.

## 12.6 Economic interpretation
PD alone is not enough. Loss is money, so LGD and EAD matter.

## 12.7 Real-bank humility
A learning project can be strong and still not be identical to a live bank production system.

That is exactly the maturity I want this note to reflect.

---

# Part 13 — The Most Important “What I Did” vs “What I Would Improve” Table

This is probably the most interview-useful section in the whole note.

| Area | What I did in the project | What I would improve in a real bank build |
|---|---|---|
| Bad definition | built `good_bad` from `loan_status` | use cleaner default window and stricter default policy |
| Train/test design | random 80/20 split | use out-of-time validation |
| Preprocessing | notebook-based, explicit transformations | controlled production pipeline with frozen parameters |
| Missing values | practical fills and grouped bins | more formal missingness strategy by business meaning |
| WoE / IV | used for scorecard-style variable treatment | additionally test stability, fairness, and governance impact |
| PD model | logistic regression scorecard | compare challengers but keep interpretability in mind |
| Score scaling | coefficient-based score mapping | cleaner PDO / base-score framework if needed |
| Monitoring | compared training-era and 2015 distributions | fully automated recurring monitoring with frozen pipeline |
| LGD | two-stage recovery modeling | consider better bounded-target methods and richer recovery drivers |
| EAD | CCF-style installment proxy | product-specific modeling, especially for revolving lines |
| EL | loan-level and portfolio-level expected loss | horizon-aligned production framework |
| CECL / stress | conceptual extension | full lifetime / scenario architecture with governance |

---

# Part 14 — Common Mistakes I Want to Avoid When Explaining This Project

1. **Saying “default = 1” without checking the notebook coding.**  
   In this project, `1 = Good` and `0 = Bad`.

2. **Explaining WoE in the wrong direction.**  
   Here it is `ln(%Good / %Bad)`, so positive WoE is safer.

3. **Pretending the project used a perfect bank default definition.**  
   It used a practical `loan_status` mapping.

4. **Acting like the random split is equivalent to OOT validation.**  
   It is not.

5. **Talking as if monitoring is separate from the model.**  
   Monitoring is part of the same lifecycle.

6. **Training LGD on all loans instead of defaulted loans only.**  
   LGD is conditional on default.

7. **Confusing installment-loan EAD with revolving-line EAD.**  
   The product structure matters.

8. **Pretending the project is already a full CECL engine.**  
   It is the baseline skeleton, not the full final architecture.

9. **Using too much code in the explanation.**  
   The goal is to explain the logic, not drown the reader.

---

# Part 15 — Interview-Ready Questions I Should Be Able to Answer

## What is the target in the PD model?
A binary `good_bad` target built from `loan_status`, with `1 = Good` and `0 = Bad`.

## If the model predicts class 1 probability, is that PD?
No. In this notebook setup, class `1` is Good, so PD is either `1 - P(Good)` or the probability of class `0`.

## What is WoE?
A bin-level transformation that compares the distribution of Goods and Bads:

```text
WoE = ln(%Good / %Bad)
```

In this project, higher WoE means safer.

## Why use logistic regression?
Because the target is binary, logistic regression gives valid probabilities and remains highly interpretable for scorecards.

## Why do reference categories matter?
Each dummy coefficient is interpreted relative to the omitted reference bucket.

## Why is a random split weaker than OOT?
Because a random split mixes time periods and gives an overly optimistic estimate of how the model would perform on future data.

## What does PSI tell me?
Whether the score distribution shifted between the development population and the newer monitoring population.

## What is the difference between population drift and concept drift?
Population drift means the borrower mix changed.  
Concept drift means the relationship between predictors and default changed.

## Why model LGD only on defaulted loans?
Because LGD is observed only after default.

## Why is LGD modeled in two stages?
Because there is a spike at zero recovery, so it is more natural to separate:
- whether any recovery occurs
- how large recovery is if it occurs

## What is EAD in this project?
A CCF-style installment-loan exposure proxy built from principal repayment information.

## What is Expected Loss?
The product of default probability, loss severity, and exposure:

```text
EL = PD × LGD × EAD
```

## How does this connect to CECL?
The project gives the one-period baseline mechanics. CECL extends the same logic over a lifetime horizon.

## How does this connect to stress testing?
Stress scenarios can worsen PD, LGD, and EAD simultaneously, which raises projected loss.

---

# Part 16 — The Full Notebook Flow in One Connected Diagram

```text
Lending Club raw loan data
        │
        ▼
Create binary target from loan_status
(good_bad: 1 = Good, 0 = Bad)
        │
        ▼
Clean raw fields
(emp_length, term, dates, missing values)
        │
        ▼
Create grouped bins and WoE-style transformations
        │
        ▼
Build logistic regression PD model
        │
        ▼
Validate with confusion matrix, AUROC, Gini, KS
        │
        ▼
Scale to credit score
        │
        ▼
Apply same preparation logic to 2015 data
        │
        ▼
Monitor score and feature stability
(PSI / CSI-style logic, drift diagnosis)
        │
        ▼
Take defaulted-loan sample
        │
        ▼
Model recovery and LGD
(two-stage approach)
        │
        ▼
Model CCF / EAD proxy
        │
        ▼
Compute loan-level and portfolio-level EL
        │
        ▼
Understand how the same structure extends to CECL and stress testing
```

---

# Part 17 — Final Big-Picture Summary

This project is the backbone of my credit-risk understanding.

It taught me that a real lending model is not just:

- “fit logistic regression”
- “print AUC”
- “done”

It is a chain of decisions:

1. define the target correctly  
2. preprocess the data correctly  
3. build an interpretable model  
4. validate it properly  
5. monitor it after deployment  
6. extend it from default likelihood into actual loss estimation  
7. understand where project simplification ends and real-bank architecture begins

That is exactly why I wanted one single note instead of several disconnected ones.

---

## Connections to the Rest of My Notes

- [[Tharun-Kumar-Gajula]] — this master note is the core credit-risk anchor inside my broader system
- [[5_bank_churn_neural_networks_masterclass]] — useful contrast for how binary classification appears outside credit-risk scorecards
- [[6_employee_retention_tree_models_masterclass]] — useful comparison for classification with nonlinear tree models
- [[7_socio_economic_household_classification_masterclass]] — useful comparison for structured-tabular supervised learning
- [[8_twitter_sentiment_nlp_masterclass]] — useful contrast for unstructured text pipelines versus structured risk data

---

## Key Concepts Summary

| Concept | What it means in this project | Clean one-line takeaway |
|---|---|---|
| **Good / Bad target** | `good_bad`, with `1 = Good`, `0 = Bad` | The notebook is coded in the Good direction |
| **PD** | Probability of bad/default | `PD = P(class 0) = 1 - P(Good)` |
| **WoE** | `ln(%Good / %Bad)` by bin | Positive WoE means safer in this project |
| **IV** | Variable-level separation strength | Helps tell me which features are useful |
| **Reference category** | Omitted dummy group | Every coefficient is interpreted relative to it |
| **Logistic regression** | PD model core | Gives interpretable probabilities and scorecards |
| **AUROC** | Rank-ordering quality | Higher means better separation of Good and Bad |
| **Gini** | Rescaled AUROC | Common scorecard summary metric |
| **KS** | Max separation between cumulative Good and Bad curves | Very common retail-credit metric |
| **PSI** | Score distribution drift | Tells me whether the population shifted |
| **CSI-style monitoring** | Feature-level drift contribution | Tells me which variables changed |
| **Population drift** | Borrower mix changed | Inputs moved |
| **Concept drift** | Relationship changed | Model logic may no longer fit reality |
| **LGD** | Fraction lost after default | Severity of loss |
| **EAD** | Exposure outstanding at default | Money at risk when default happens |
| **EL** | `PD × LGD × EAD` | Turns model outputs into expected money loss |
| **CECL** | Lifetime expected loss extension | Same logic, longer horizon |
| **Stress testing** | Scenario-driven worsening of risk | PD, LGD, and EAD can all move adversely |

---

*This is my final single master note for the Lending Club credit-risk project. If I can explain this note clearly, then I can explain the full project clearly.*
