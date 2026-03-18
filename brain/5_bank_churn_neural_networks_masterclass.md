---
title: "Bank Churn Prediction with Neural Networks — Classification, Imbalance, and Retention Modeling"
date: 2026-03-19
tags:
  - bank-churn
  - customer-retention
  - neural-networks
  - classification
  - binary-classification
  - SMOTE
  - recall
  - precision
  - ROC-AUC
  - dropout
  - adam
  - SGD
  - hyperparameter-tuning
  - data-preprocessing
  - feature-engineering
  - model-evaluation
  - data-science
  - banking
cluster: "03 — Applied Machine Learning Projects"
progress: 0
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_lending_club_credit_risk_masterclass]]"
---

---

# Bank Churn Prediction with Neural Networks — Classification, Imbalance, and Retention Modeling

> This note is my full technical record of how I use a **bank customer churn project** to understand core data-science ideas from first principles: binary classification, exploratory data analysis, preprocessing, neural networks, optimization, class imbalance, model evaluation, and business interpretation.
>
> I use this one project as a beginner-friendly anchor to learn both the **math** and the **code** behind predictive modeling. Even though churn prediction is not the same as credit-risk modeling, it teaches many of the same core ideas that appear again in PD models, early-warning systems, fraud models, collections prioritization, and customer-risk analytics inside banks.

---

## The Project at a Glance

**Dataset:** Bank customer churn dataset from Kaggle, loaded in the notebook through `kagglehub`.

**Raw dataset shape:** `10,000 rows × 14 columns`

**Objective:** Predict whether a customer will **exit the bank within the next 6 months**.

**Target variable:** `Exited`

- `0` = customer did not leave
- `1` = customer left

**Why this project matters to me:**

This is a very useful beginner project because it lets me learn the entire supervised-learning workflow on a real banking-style use case:

- how to define the prediction target
- how to inspect and understand features
- how to prepare data for modeling
- how a neural network turns inputs into probabilities
- how metrics like **precision**, **recall**, and **ROC-AUC** actually matter in business
- how class imbalance changes model behavior
- how model choice depends on the decision problem, not only on accuracy

**The notebook pipeline:**

```text
Raw customer dataset
        │
        ▼
Understand columns and target variable
        │
        ▼
EDA (distribution, correlation, churn patterns)
        │
        ▼
Drop irrelevant columns
        │
        ▼
Outlier handling + categorical encoding + scaling
        │
        ▼
Train / test split
        │
        ▼
Baseline neural network (SGD)
        │
        ▼
Improved neural network (Adam)
        │
        ▼
Dropout regularization
        │
        ▼
Hyperparameter tuning
        │
        ▼
SMOTE for class imbalance
        │
        ▼
Compare models and choose final approach
```

---

## Part 1: What the Business Problem Actually Is

### The Concept

A **churn model** predicts the probability that a customer will stop using the bank.

That matters because retaining an existing customer is usually cheaper than acquiring a new one. If the bank can identify customers who are likely to leave, it can intervene early with:

- retention offers
- service recovery
- relationship-manager outreach
- cross-sell or product-bundle strategies
- customer experience improvements

### The Data-Science Framing

This is a **binary classification problem**.

For each customer, I observe a vector of features:

```text
x = [credit score, geography, gender, age, tenure, balance, ...]
```

The model outputs a probability:

```text
P(Exited = 1 | x)
```

That probability is then converted into a decision, such as:

- high-risk churner
- medium-risk churner
- low-risk churner

### Why This Matters for Interviews

This project is not just about churn.

It is also a clean way to explain:

- what a target variable is
- what a feature vector is
- why classification is different from regression
- how a probability becomes a business decision
- why the wrong metric can lead to the wrong model choice

These are foundational ideas across almost all machine-learning roles.

---

## Part 2: Understanding the Dataset Properly

### The Raw Columns

The notebook uses these core fields:

- `RowNumber` — row index
- `CustomerId` — unique customer ID
- `Surname` — customer surname
- `CreditScore` — customer credit score
- `Geography` — customer location
- `Gender` — customer gender
- `Age` — age
- `Tenure` — years with the bank
- `Balance` — account balance
- `NumOfProducts` — number of bank products used
- `HasCrCard` — whether customer has a credit card
- `IsActiveMember` — whether customer is actively using the bank
- `EstimatedSalary` — estimated salary
- `Exited` — churn target

### What I Should Notice Immediately

The notebook shows:

- **10,000 total rows**
- **14 total columns**
- **no missing values**
- target imbalance:
  - `Exited = 0`: **7,963**
  - `Exited = 1`: **2,037**

So the churn rate is about:

```text
2037 / 10000 = 20.37%
```

That means the problem is **imbalanced**, though not extremely so.

### Why This Imbalance Matters

If I build a lazy model that predicts **everyone will stay**, it would already get about **79.63% accuracy**.

So a model can have “good” accuracy while still being bad at finding churners.

That is why this notebook correctly moves attention toward **recall**, **precision**, **ROC-AUC**, and confusion-matrix interpretation.

---

## Part 3: Exploratory Data Analysis (EDA)

EDA is where I stop thinking like a coder and start thinking like an analyst.

The notebook explores:

1. distribution of `CreditScore`
2. active vs inactive members
3. correlation heatmap
4. churn by gender
5. churn by geography
6. age distribution by churn
7. balance distribution by churn
8. number of products vs churn

### Main Observations from the Notebook

The notebook’s EDA suggests:

- most credit scores are concentrated around the **600–700** range
- the split between active and inactive members is almost even
- **Age** has the clearest positive relationship with churn
- **IsActiveMember** has a negative relationship with churn
- female customers show a higher churn rate in this sample
- customers in **Germany** churn more than those in France or Spain
- customers aged **45+** churn more than younger customers
- churned customers tend to have a higher balance
- customers with only **1 product** churn more, while those with **2 products** churn less

### Why EDA Matters

EDA is not just plotting charts.

It helps me form hypotheses:

- maybe inactivity is a churn signal
- maybe region-specific service issues exist
- maybe product penetration improves stickiness
- maybe age segments behave differently
- maybe raw balance alone does not guarantee retention

This is where business questions and modeling begin to connect.

---

## Part 4: Preprocessing — Turning Raw Data into Model Input

This part is extremely important because most model quality problems start before the algorithm.

### Step 1: Remove Irrelevant Columns

The notebook drops:

- `RowNumber`
- `CustomerId`
- `Surname`

### Why These Were Dropped

`RowNumber` and `CustomerId` are effectively identifiers, not meaningful predictors.

`Surname` is not fully unique, but in this project it is treated as non-informative and not useful for action.

That is a sensible first-pass decision.

### Step 2: Outlier Handling

The notebook removes outliers using the **IQR rule** for:

- `CreditScore`
- `Age`
- `Balance`
- `EstimatedSalary`

It reports:

- outliers removed in `CreditScore`: **15**
- outliers removed in `Age`: **359**
- outliers removed in `Balance`: **0**
- outliers removed in `EstimatedSalary`: **0**

After removal, the remaining sample becomes:

```text
10000 - 15 - 359 = 9626 rows
```

### Important Accuracy Note

This is one place where I need to be careful.

The notebook removes outliers **before** the train-test split. In a production-grade workflow, that is not ideal, because preprocessing rules should usually be learned only on the training data and then applied to validation/test data.

Also, in churn modeling, very old ages are not necessarily “bad data.” They may be real customers. So I should not blindly remove outliers just because the IQR rule says they are statistically unusual.

A better practical approach would be:

- inspect whether the values are genuine or data-entry errors
- fit outlier rules on the training set only
- consider **capping / winsorization** instead of deletion
- remember that tree models often need less aggressive outlier treatment than neural nets or distance-based models

### Step 3: Encode Categorical Variables

The notebook encodes:

- `Gender` using `LabelEncoder`
- `Geography` using one-hot encoding with `drop_first=True`

That produces dummy variables such as:

- `Geography_Germany`
- `Geography_Spain`

with one geography category acting as the reference category.

### Why Encoding Is Needed

A neural network cannot directly work with raw text labels like `"Germany"` or `"Female"`.

It needs numbers.

### Step 4: Train-Test Split

The notebook creates:

- **training set:** `(7700, 11)`
- **test set:** `(1926, 11)`

using a stratified split:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

This is good because stratification keeps the churn proportion similar in train and test.

### Step 5: Feature Scaling

The notebook applies `StandardScaler`:

```python
X_scaled = (X - mean) / std
```

More precisely, each feature becomes:

```text
z = (x - μ) / σ
```

where:

- `μ` = feature mean
- `σ` = feature standard deviation

### Why Scaling Matters for Neural Networks

Without scaling, some variables may dominate just because they are numerically larger.

For example:

- `Balance` may be in tens of thousands
- `HasCrCard` is only `0` or `1`

If I do not scale, optimization becomes harder and training may become unstable or slower.

---

## Part 5: The Neural Network from First Principles

This is the heart of the project.

### What a Neural Network Is

A neural network is a function that learns complex relationships between inputs and an output.

For one neuron, the basic calculation is:

```text
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

or in vector form:

```text
z = wᵀx + b
```

Then I apply an activation function.

### ReLU Activation

In hidden layers, the notebook uses **ReLU**:

```text
ReLU(z) = max(0, z)
```

Why ReLU is common:

- simple
- fast to compute
- helps neural nets learn non-linear patterns
- usually works well in practice

### Sigmoid Activation for Binary Classification

The output layer uses **sigmoid**:

```text
σ(z) = 1 / (1 + e^(-z))
```

This converts the output into a probability between 0 and 1.

So if the model outputs `0.82`, I interpret it as:

```text
P(customer will churn) ≈ 82%
```

### The Loss Function: Binary Cross-Entropy

The model is trained using **binary cross-entropy**:

```text
L = -[y log(p) + (1 - y) log(1 - p)]
```

where:

- `y` is the true label (`0` or `1`)
- `p` is the predicted probability of class `1`

This loss punishes confident wrong predictions very strongly.

### How Learning Happens

Training means updating the weights to reduce the loss.

At a high level:

1. do a forward pass
2. compute loss
3. compute gradients using backpropagation
4. update weights using an optimizer

That cycle repeats across many epochs.

### The Baseline Architecture in the Notebook

The first model is:

```python
model_sgd = Sequential()
model_sgd.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model_sgd.add(Dense(32, activation='relu'))
model_sgd.add(Dense(1, activation='sigmoid'))
```

So the architecture is:

```text
11 input features
        │
        ▼
Dense layer: 64 neurons, ReLU
        │
        ▼
Dense layer: 32 neurons, ReLU
        │
        ▼
Dense layer: 1 neuron, Sigmoid
```

This is a fairly standard small feedforward network for tabular binary classification.

---

## Part 6: Baseline Model — Neural Network with SGD

### What SGD Is

The first model uses **Stochastic Gradient Descent (SGD)** with learning rate `0.01`.

The update rule is conceptually:

```text
w_new = w_old - η ∂L/∂w
```

where:

- `η` is the learning rate
- `∂L/∂w` is the gradient of the loss with respect to the weights

### Why SGD Matters

SGD is the classic optimizer. It is simple and foundational.

Even if better optimizers exist, understanding SGD helps me understand the logic of all gradient-based learning.

### Baseline Results

The notebook reports:

- confusion matrix:

```text
[[1486   50]
 [ 204  186]]
```

- precision for churn class: **0.79**
- recall for churn class: **0.48**
- accuracy: **0.87**
- ROC-AUC: **0.8533**

### How to Read This Confusion Matrix

For class `1 = churn`:

- **True Positives (TP)** = 186
- **False Negatives (FN)** = 204
- **False Positives (FP)** = 50
- **True Negatives (TN)** = 1486

So the model misses more churners than it catches:

```text
Recall = TP / (TP + FN) = 186 / (186 + 204) ≈ 0.48
```

This is the key weakness.

### What This Means Business-Wise

If recall is only 48%, the bank is failing to identify more than half of the customers who will actually leave.

That may be unacceptable if the purpose of the model is proactive retention.

---

## Part 7: Why Recall Was Chosen as the Main Metric

This choice is one of the most important parts of the notebook.

### The Logic

The notebook prioritizes **recall** because missing a churner can be more costly than incorrectly flagging a non-churner.

### The Formula

```text
Recall = TP / (TP + FN)
```

This answers:

> Out of all the customers who actually churned, how many did I correctly catch?

### Why Accuracy Alone Is Not Enough

With imbalanced data, accuracy can look good even if the churn class is handled badly.

That is exactly what happens here:

- accuracy is high
- recall for churn is much lower

This is a classic interview point.

Whenever a class is rarer and more important, I should immediately ask:

- do I care more about precision or recall?
- what is the cost of false negatives?
- what is the cost of false positives?

---

## Part 8: Improved Optimizer — Adam

### What Adam Is

The next model uses **Adam**.

Adam is generally better than plain SGD because it adapts learning rates during training and combines ideas from:

- momentum
- adaptive per-parameter learning rates

In practice, Adam often converges faster and more reliably on tabular and neural-network problems.

### Model Setup

The Adam model keeps the same architecture:

- 64-neuron hidden layer
- 32-neuron hidden layer
- sigmoid output

but changes the optimizer to:

```python
Adam(learning_rate=0.001)
```

### Results

The notebook summary states roughly:

- training accuracy: **88.4%**
- validation accuracy: **84.9%**
- precision for churn class: **0.66**
- recall for churn class: **0.54**
- ROC-AUC: **0.8503**

### Interpretation

Compared to SGD:

- recall improves from about **0.48** to **0.54**
- the model catches more churners
- there is some sign of mild overfitting later in training

This is a good example of how **changing only the optimizer** can materially affect results.

---

## Part 9: Dropout — Regularization in Neural Networks

### What Overfitting Means

A model overfits when it learns patterns too specific to the training data and fails to generalize well to new data.

### What Dropout Does

Dropout randomly turns off a fraction of neurons during training.

The notebook uses:

```python
Dropout(0.3)
```

That means roughly 30% of units in that layer are dropped during each training pass.

### Why Dropout Helps

It prevents the network from relying too heavily on a small set of neurons and encourages more robust internal representations.

### Model Structure with Dropout

```python
Dense(64, activation='relu')
Dropout(0.3)
Dense(32, activation='relu')
Dropout(0.3)
Dense(1, activation='sigmoid')
```

### Results

The notebook reports:

- validation accuracy: **86.29%**
- precision for churn class: **0.80**
- recall for churn class: **0.43**
- ROC-AUC: **0.8544**

### Interpretation

This model improves **precision** but reduces **recall**.

That means:

- when it predicts churn, it is more often correct
- but it identifies fewer churners overall

So dropout improves stability and may reduce overfitting, but it does not automatically solve the business objective.

This is a good reminder that **regularization and business utility are not the same thing**.

---

## Part 10: Hyperparameter Tuning

### The Idea

A neural network has many design choices:

- number of neurons
- learning rate
- batch size
- number of epochs

These are **hyperparameters**.

The notebook uses `GridSearchCV` with `KerasClassifier` to search across combinations.

### Hyperparameter Grid Used

The notebook tunes:

- `neurons_1`: `[32, 64, 128]`
- `neurons_2`: `[16, 32]`
- `learning_rate`: `[0.001, 0.01]`
- `batch_size`: `[32, 64]`
- `epochs`: `[20, 50]`

That is:

```text
3 × 2 × 2 × 2 × 2 = 48 combinations
```

With 3-fold cross-validation, the notebook runs:

```text
48 × 3 = 144 fits
```

### Best Hyperparameters Found

The notebook reports:

- `batch_size = 64`
- `epochs = 20`
- `learning_rate = 0.001`
- `neurons_1 = 128`
- `neurons_2 = 32`

### Tuned Model Results

- confusion matrix:

```text
[[1449   87]
 [ 180  210]]
```

- precision for churn class: **0.71**
- recall for churn class: **0.54**
- accuracy: **0.86**
- ROC-AUC: **0.8547**

### Why This Matters

This tuned model has the **best ROC-AUC** among the models compared in the notebook, while keeping a more balanced precision-recall tradeoff than the SMOTE model.

So from a pure discrimination perspective, this is arguably the most balanced neural-network version in the notebook.

---

## Part 11: Class Imbalance and SMOTE

This is one of the most important learning sections in the entire project.

### What Class Imbalance Does

Because churners are fewer than non-churners, the model can become biased toward predicting the majority class.

That often causes:

- high overall accuracy
- lower recall for the minority class

### What SMOTE Is

**SMOTE** stands for **Synthetic Minority Oversampling Technique**.

Instead of simply duplicating minority-class rows, SMOTE generates synthetic examples by interpolating between minority-class neighbors.

### Why SMOTE Is Used

The purpose is to help the model see a more balanced training distribution.

In the notebook:

- original training target distribution was imbalanced
- after SMOTE, both classes become:

```text
0 → 6141
1 → 6141
```

### Important Best Practice

The notebook applies SMOTE only to the **training set**, which is correct.

I should never apply SMOTE to the test set, because that would contaminate evaluation.

### SMOTE Model Architecture

The SMOTE model uses:

- 128 neurons in first hidden layer
- dropout 0.3
- 32 neurons in second hidden layer
- dropout 0.2
- Adam optimizer
- 20 epochs
- batch size 64

### SMOTE Model Results

The notebook reports:

- confusion matrix:

```text
[[1253  283]
 [  98  292]]
```

- precision for churn class: **0.51**
- recall for churn class: **0.75**
- accuracy: **0.80**
- ROC-AUC: **0.8515**

### What Changed

This is the key tradeoff:

- recall improved **a lot**
- precision dropped
- accuracy dropped
- ROC-AUC stayed reasonably strong

### Business Interpretation

This model catches many more churners, but it also flags many more non-churners as at-risk.

That can still be acceptable if:

- outreach cost is low
- the business prioritizes not missing churners
- false positives are tolerable

This is exactly the kind of model-selection tradeoff I need to explain clearly in interviews.

---

## Part 12: Final Model Choice — What I Would Actually Say

The notebook chooses the **SMOTE-based neural network** as the final model because recall is the most important objective.

That decision is defensible.

### The Comparison Table

| Model | Accuracy | Precision (Churn) | Recall (Churn) | ROC-AUC |
|---|---:|---:|---:|---:|
| Neural Network with SGD | 0.87 | 0.79 | 0.48 | 0.8533 |
| Neural Network with Adam | 0.85 | 0.66 | 0.54 | 0.8503 |
| Neural Network with Dropout | 0.86 | 0.80 | 0.43 | 0.8544 |
| Tuned Neural Network | 0.86 | 0.71 | 0.54 | 0.8547 |
| SMOTE Neural Network | 0.80 | 0.51 | **0.75** | 0.8515 |

### The Most Honest Interpretation

If the goal is:

- **maximize churn capture**
- and outreach cost is acceptable
- and false positives are manageable

then the **SMOTE model** is the best business choice.

If the goal is:

- **more balanced classification quality**
- better precision-recall balance
- cleaner overall discrimination

then the **tuned model** is arguably stronger.

### One More Important Practical Idea

The notebook compares models at the default probability threshold of `0.5`.

In practice, I should also consider **threshold tuning**.

That means:

- keep the same model
- change the classification cutoff from `0.50` to something else like `0.35`
- choose the threshold that best matches business cost

Sometimes threshold tuning gives a better business solution than changing the entire model.

That is a very strong interview point.

---

## Part 13: The Code Behind the Project

### Core Preprocessing Pattern

```python
# drop irrelevant columns
data_cleaned = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# encode categories
data_cleaned['Gender'] = le.fit_transform(data_cleaned['Gender'])
data_cleaned = pd.get_dummies(data_cleaned, columns=['Geography'], drop_first=True)

# split
X = data_cleaned.drop(columns=['Exited'])
y = data_cleaned['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Core Neural-Network Pattern

```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)
```

### Evaluation Pattern

```python
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_class))
print(classification_report(y_test, y_pred_class))
print(roc_auc_score(y_test, y_pred))
```

### What I Should Understand in the Code

I should be able to explain each line at a conceptual level:

- `Dense(...)` creates a fully connected layer
- `activation='relu'` introduces non-linearity
- `activation='sigmoid'` converts the output to probability
- `compile(...)` defines how learning happens
- `loss='binary_crossentropy'` defines the objective being minimized
- `fit(...)` performs training
- `predict(...)` outputs probabilities
- thresholding converts probabilities into class labels
- evaluation metrics show whether the model is useful

---

## Part 14: What This Project Teaches Me About Core Data Science

This one project covers a large number of beginner-to-intermediate concepts:

### 1. Problem Framing

Before code, I need to know:

- what the target is
- whether the problem is classification or regression
- what the decision cost really is

### 2. Data Understanding

A model is only as good as feature understanding.

I need to know:

- which variables are identifiers
- which variables are business features
- which variables may leak target information
- which variables need encoding or scaling

### 3. Model Evaluation

A “good” model depends on what the business needs.

- accuracy is not enough
- recall matters when missing positives is costly
- precision matters when acting on positives is expensive
- ROC-AUC measures ranking quality across thresholds

### 4. Optimization and Regularization

This project taught me:

- the role of SGD and Adam
- why dropout exists
- why hyperparameters matter
- why training dynamics matter, not just final accuracy

### 5. Imbalanced Learning

This project also teaches:

- why minority classes are difficult
- what SMOTE does
- why oversampling changes precision-recall tradeoffs
- why business context must guide model selection

---

## Part 15: How This Connects Back to Banking and Quant Work

Even though this is a churn problem, it transfers strongly to banking analytics.

### Direct Banking Relevance

Banks care about churn because:

- deposit attrition affects funding stability
- product attrition affects cross-sell revenue
- inactive customers are cheaper to lose than high-value customers
- retention strategy is a real business problem

### Transfer to Risk and Quant Roles

This project teaches the same technical muscles used in risk modeling:

- binary classification
- class imbalance handling
- threshold setting
- feature preprocessing
- model comparison
- confusion-matrix interpretation
- production-minded thinking about what kind of error matters more

### The Connection to PD Modeling

A PD model also predicts a binary event:

- churn: customer leaves
- PD: borrower defaults

The underlying modeling workflow is similar:

```text
raw data
→ preprocessing
→ encode / scale
→ train model
→ evaluate ranking and classification quality
→ choose threshold / business action
```

The business meaning is different, but the data-science discipline is very similar.

---

## Part 16: What I Would Improve in a Production-Grade Version

This is where I convert a notebook project into stronger professional thinking.

### 1. Use a Proper Train / Validation / Test Split

In the notebook, the test set is also used as validation data during neural-network training.

That is not ideal.

A better setup is:

- **train set** → fit model
- **validation set** → tune architecture and thresholds
- **test set** → final untouched evaluation

### 2. Move All Preprocessing into a Pipeline

For cleaner reproducibility, I would put:

- column dropping
- encoding
- scaling
- any capping or winsorization

into a formal pipeline.

### 3. Revisit Outlier Logic

I would not automatically remove age outliers without checking whether they are genuine customers.

For tabular banking data, deleting rows too quickly can remove useful signal.

### 4. Compare Against Strong Baselines

A neural network should not be used just because it sounds advanced.

I would compare it against:

- logistic regression
- random forest
- XGBoost / LightGBM

On tabular structured data, boosted trees often perform very strongly.

### 5. Consider Class Weights and Threshold Tuning

Before or alongside SMOTE, I would test:

- class-weighted loss
- threshold optimization
- cost-sensitive evaluation

### 6. Add Probability Calibration

If the bank wants reliable probabilities, not just rankings, I would check calibration using:

- calibration curves
- Brier score
- Platt scaling or isotonic regression if needed

### 7. Add Explainability

For business adoption, I would want explainability using:

- SHAP
- permutation importance
- partial dependence / ICE where useful

### 8. Add Monitoring

If deployed, I would monitor:

- score distribution drift
- input-feature drift
- precision / recall over time
- campaign response effectiveness
- threshold stability

This directly connects to the monitoring mindset used in my other notes.

---

## Part 17: Interview-Ready Technical Questions I Should Be Able to Answer

### Why did I scale the data?

Because neural networks train better when features are on comparable scales. Large-scale variables can dominate gradients and slow learning.

### Why use sigmoid in the final layer?

Because the target is binary and sigmoid maps outputs to probabilities in `[0, 1]`.

### Why use binary cross-entropy?

Because it is the standard loss for binary classification and strongly penalizes confident wrong predictions.

### Why did recall matter more than accuracy?

Because the business goal was to identify churners. Missing churners is more costly than getting some false alarms.

### Why did SMOTE improve recall?

Because it balanced the minority class in the training data, making the model pay more attention to churn patterns.

### Why did precision drop under SMOTE?

Because when I make the model more aggressive in identifying churners, it also tends to produce more false positives.

### Why is ROC-AUC useful?

Because it measures ranking ability across thresholds instead of locking me into one classification cutoff.

### Why is dropout useful?

Because it regularizes the network and reduces over-reliance on specific neurons, helping generalization.

### Why is the current notebook not fully production-grade?

Because the test set is reused during model development, outlier treatment is applied before splitting, and the preprocessing is not packaged as a robust reusable pipeline.

---

## Connections to the Rest of the Quant OS Brain

This note is not a credit-risk note, but it is a strong **applied machine-learning foundation note** for the rest of the brain.

- [[1_lending_club_credit_risk_masterclass|1_full_pd_model]] — This note and the PD note both solve binary classification problems. The business event differs, but the workflow of preprocessing, training, evaluation, and threshold-based action is structurally similar.

- [[1_lending_club_credit_risk_masterclass|2_monitoring_model]] — If this churn model were deployed, I would monitor feature drift, score drift, and recall decay over time using the same production mindset described in the monitoring note.

- [[1_lending_club_credit_risk_masterclass|3_lgd_ead_model_rewritten]] — That note explains severity and exposure after default. This churn note is different in objective, but it reinforces general machine-learning ideas like feature handling, target design, and evaluation tradeoffs.

- [[1_lending_club_credit_risk_masterclass|4_ecl_cecl_stress_testing_rewritten]] — The ECL note is about converting risk components into money. This note is about converting probability estimates into retention action. Both are examples of how modeling supports business decisions.

---

## Key Concepts Summary

| Concept | What It Is | Where It Appears in This Project |
|---|---|---|
| **Binary Classification** | Predicting one of two classes | `Exited = 0/1` |
| **Target Imbalance** | One class is much less frequent than the other | 20.37% churn vs 79.63% non-churn |
| **Feature Scaling** | Standardizing variables to comparable ranges | `StandardScaler()` |
| **One-Hot Encoding** | Turning categories into dummy columns | `Geography_Germany`, `Geography_Spain` |
| **Label Encoding** | Converting a binary text category into 0/1 | `Gender` |
| **Dense Layer** | Fully connected neural-network layer | 64-neuron and 32-neuron hidden layers |
| **ReLU** | Hidden-layer activation | `max(0, z)` |
| **Sigmoid** | Output activation for binary probability | final churn probability |
| **Binary Cross-Entropy** | Loss function for binary classification | training objective |
| **SGD** | Basic gradient-based optimizer | baseline model |
| **Adam** | Adaptive optimizer | improved training stability |
| **Dropout** | Regularization by randomly dropping neurons | 0.3 dropout layers |
| **Recall** | Fraction of actual positives correctly found | key business metric for churn |
| **Precision** | Fraction of predicted positives that are truly positive | affected strongly by SMOTE tradeoff |
| **ROC-AUC** | Threshold-independent ranking quality | model comparison metric |
| **SMOTE** | Minority-class synthetic oversampling | improved churn recall to 0.75 |
| **Hyperparameter Tuning** | Searching model design choices systematically | 48 combinations, 144 CV fits |
| **Threshold Tuning** | Choosing a probability cutoff based on business cost | stronger production decision step |
| **Data Leakage** | Using information from outside the training process improperly | outlier removal before split, test reused for validation |

---

*This note is Version 1.0. It is my beginner-friendly machine-learning foundation note for tabular binary classification in a banking setting. It also acts as a bridge from generic data science into more specialized bank analytics and risk-modeling workflows.*
