---
title: "Twitter Sentiment Analysis with NLP — Text Cleaning, Vectorization, and Multi-Class Classification"
date: 2026-03-17
tags:
  - twitter-sentiment
  - NLP
  - natural-language-processing
  - text-classification
  - multi-class-classification
  - countvectorizer
  - tf-idf
  - random-forest
  - text-preprocessing
  - tokenization
  - stopwords
  - lemmatization
  - feature-engineering
  - model-evaluation
  - data-science
  - customer-analytics
cluster: "03 — Applied Machine Learning Projects"
progress: 0
links:
  - "[[Tharun-Kumar-Gajula]]"
  - "[[1_lending_club_credit_risk_masterclass]]"
  - "[[5_bank_churn_neural_networks_masterclass]]"
  - "[[6_employee_retention_tree_models_masterclass]]"
  - "[[7_socio_economic_household_classification_masterclass]]"
---

---

# Twitter Sentiment Analysis with NLP — Text Cleaning, Vectorization, and Multi-Class Classification

> This note is my full technical record of how I use a **Twitter sentiment analysis project** to understand natural language processing from first principles.
>
> I use this project to learn how raw text becomes machine-learning input, how text cleaning changes the information available to a model, how vectorization works, how multi-class classification is evaluated, and how sentiment models connect to real business decisions.
>
> Even though this is not a credit-risk project, it still teaches me a lot of core ideas that matter later in risk analytics: unstructured data handling, noisy real-world inputs, feature extraction, class imbalance, model evaluation, and practical interpretation.

---

## The Project at a Glance

**Dataset:** Airline-related tweets labeled with sentiment

**Raw dataset shape:** `14,640 rows × 15 columns`

**Target variable:** `airline_sentiment`

- `negative`
- `neutral`
- `positive`

**Modeling dataset used in the notebook:** only two columns were retained for modeling:

- `text`
- `airline_sentiment`

**Text feature space size after vectorization:** `10,987` unique token features

**Main objective:** predict the sentiment of a tweet from its text.

This is a **multi-class classification** problem, not a binary classification problem.

### Why this project matters to me

This is an excellent beginner NLP project because it teaches me:

- how text data differs from tabular data
- why text must be converted into numbers before modeling
- how cleaning choices affect signal and noise
- the difference between **CountVectorizer** and **TF-IDF**
- how multi-class classification metrics are interpreted
- why class imbalance matters in sentiment tasks
- why a model can show good accuracy but still struggle on minority classes
- how NLP projects are connected to business use cases such as customer-experience analytics, complaints monitoring, brand monitoring, service quality tracking, and early-warning intelligence

---

## The Full Pipeline I Built

```text
Raw tweet dataset
        │
        ▼
Understand columns and target labels
        │
        ▼
EDA on airlines, sentiments, and negative reasons
        │
        ▼
Keep only text + sentiment for modeling
        │
        ▼
Clean text (HTML, contractions, numbers)
        │
        ▼
Tokenize text
        │
        ▼
Remove stopwords
        │
        ▼
Lemmatize words
        │
        ▼
Rejoin processed tokens into text strings
        │
        ▼
Convert text into numeric vectors
        │        ├── CountVectorizer
        │        └── TF-IDF
        ▼
Train / test split
        │
        ▼
Random Forest classifier
        │
        ▼
Cross-validation + tuning n_estimators
        │
        ▼
Classification report + confusion matrix + feature importance
        │
        ▼
Compare CountVectorizer vs TF-IDF
```

---

## Part 1: What the Business Problem Actually Is

At the highest level, I want a model that takes a tweet and predicts whether the tweet expresses:

- a **negative** opinion
- a **neutral** statement
- a **positive** opinion

In notation form, I want the model to learn something like:

```text
P(Y = negative | text)
P(Y = neutral  | text)
P(Y = positive | text)
```

and then assign the class with the highest predicted probability.

### Why this matters in practice

A sentiment model is useful because firms receive a massive amount of unstructured customer feedback through:

- social media posts
- app reviews
- email complaints
- call-center transcripts
- survey comments
- chatbot logs

If I can classify sentiment automatically, I can use that output for:

- complaint escalation
- customer-service triage
- brand monitoring
- product issue tracking
- service quality dashboards
- trend detection by airline, route, or issue type

### Why this matters for my broader analytics learning

This project is not about credit risk directly, but the deeper workflow is very transferable.

In credit-risk work, I may later see unstructured information in:

- collections notes
- underwriting comments
- fraud review text
- customer complaint text
- servicing interactions
- call transcripts
- dispute narratives

So this project helps me build intuition for how text becomes features and how models learn from language.

---

## Part 2: Understanding the Data

The notebook begins with a dataset of `14,640` tweets and `15` columns.

The important variables mentioned in the notebook include:

- `tweet_id` — unique tweet identifier
- `airline_sentiment` — target label
- `airline_sentiment_confidence` — confidence in the label
- `negativereason` — reason for negative sentiment when applicable
- `negativereason_confidence` — confidence in the negative reason label
- `airline` — airline being discussed
- `text` — the actual tweet text
- `retweet_count`
- `tweet_created`
- `tweet_location`
- `user_timezone`

### What I notice from the initial overview

The notebook identifies several important facts:

1. The data contains both **categorical** and **numerical** columns.
2. Several columns have substantial missing values, especially reason-related and location-related fields.
3. The sentiment target has three classes: `negative`, `neutral`, and `positive`.
4. Negative tweets dominate the dataset.
5. Some columns are useful for EDA but not necessary for the first text model.

### Why only two columns were kept for modeling

The notebook simplifies the modeling problem by keeping only:

- `text`
- `airline_sentiment`

That means the final model is learning sentiment **only from language**, not from metadata like airline name, time, location, or label confidence.

This is a good beginner design because it isolates the core NLP problem:

> Can I predict sentiment from text alone?

---

## Part 3: Class Distribution and Why It Matters

From the model outputs, the stratified test set contains:

- `1,835` negative tweets
- `620` neutral tweets
- `473` positive tweets

Because the split is stratified, these proportions reflect the full dataset reasonably well.

### Approximate class shares in the full dataset

Using the test-set supports, the overall class distribution is roughly:

- **Negative:** `62.7%`
- **Neutral:** `21.2%`
- **Positive:** `16.1%`

This matters a lot.

If one class is much more common than the others, a model can look strong overall while still doing a weak job on the smaller classes.

That is exactly what happens here:

- the model is strongest on **negative** tweets
- it struggles more on **neutral** tweets
- positive tweets are classified better than neutral in some cases, but still not as strongly as the negative class

### Why sentiment imbalance is common

This dataset is about airline-related tweets. People often go to social media when they are upset about:

- delays
- cancellations
- baggage issues
- customer-service problems
- long waiting times

So a negative skew is not surprising.

---

## Part 4: Exploratory Data Analysis and What It Tells Me

Before modeling, the notebook performs EDA around:

- percentage of tweets by airline
- overall sentiment distribution
- sentiment distribution by airline
- negative reasons
- word clouds for negative and positive tweets

### Main insights from the EDA

The notebook’s analysis suggests:

- some airlines receive more tweet volume than others
- negative sentiment is the largest class by far
- customer-service problems are one of the strongest drivers of negative tweets
- operational issues such as delays, cancellations, luggage problems, and waiting times appear repeatedly
- positive tweets contain words such as gratitude, appreciation, and praise

### What EDA is doing here mathematically

EDA itself is not a model, but it helps me inspect the empirical distribution of the data.

For example, if I compute the proportion of tweets in class `k`, I am estimating:

```text
p_k = count(class = k) / total observations
```

If I compute the distribution of negative reasons, I am simply counting frequencies and comparing them.

That may sound simple, but it is important because it tells me:

- whether the target is imbalanced
- whether some sub-groups dominate the data
- whether a few issues explain most complaints
- whether the modeling problem is realistic and meaningful

### One accuracy note I want to keep in mind

The notebook text mentions some airline-specific wording in the narrative that is not consistently supported by the displayed unique-airline output. So for my own notes, I focus on the robust conclusion:

- negative customer experience themes dominate
- complaint reasons are informative
- the text contains strong business signal

---

## Part 5: Why Text Cannot Go Directly into a Standard Machine-Learning Model

Most machine-learning models expect numeric input such as:

- income
- age
- balance
- transaction count
- utilization ratio

But a tweet is raw text, for example:

```text
@airline my flight was delayed again and nobody helped me
```

A model cannot directly multiply or split on that sentence. So I need to convert language into numbers.

That is the central idea of NLP feature extraction.

The notebook does this through several preprocessing steps first, and then through vectorization.

---

## Part 6: Text Cleaning and Preprocessing

The notebook applies the following sequence.

### Step 1: Keep only the necessary columns

```python
df_cleaned = df[['text', 'airline_sentiment']].copy()
```

This reduces the project to the text-classification core.

---

### Step 2: Remove HTML tags

```python
def remove_html(text):
    return re.sub(r'<.*?>', '', text)
```

#### Why this matters

Sometimes text contains markup like:

```html
<b>delayed</b>
```

That markup is not real language meaning. It is formatting noise.

Mathematically, removing noise can improve the signal-to-noise ratio in the feature space.

---

### Step 3: Expand contractions

```python
def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"n't", " not", phrase)
    ...
```

#### Why this matters

`can't` and `can not` should represent the same meaning.

If I do not standardize them, the vectorizer may treat them as different tokens. That creates unnecessary fragmentation.

This is a normalization step.

---

### Step 4: Remove numbers

```python
def remove_numbers(text):
    return re.sub(r'\d+', '', text)
```

#### Why this matters

Numbers in tweets are sometimes useful, but in many beginner sentiment projects they are mostly treated as noise.

Examples:

- flight numbers
- timestamps
- booking references
- counts

The assumption in this notebook is that numbers do not carry as much stable sentiment information as the words do.

This is not always true, but it is a reasonable simplification for a baseline model.

---

### Step 5: Tokenization

```python
df_cleaned['tokens'] = df_cleaned['text'].apply(lambda x: [token.text for token in nlp(x)])
```

A **token** is usually a word or word-like unit.

Example:

```text
"flight delayed again"
```

becomes approximately:

```text
["flight", "delayed", "again"]
```

#### Why tokenization matters

Tokenization is how I break text into manageable units that can later be counted or weighted.

---

### Step 6: Stopword removal

```python
stop_words = nlp.Defaults.stop_words
df_cleaned['tokens'] = df_cleaned['tokens'].apply(
    lambda tokens: [word for word in tokens if word.lower() not in stop_words]
)
```

**Stopwords** are very common words such as:

- the
- is
- are
- and
- to
- of

#### Why this matters

These words often carry little sentiment signal by themselves.

By removing them, I reduce the dimensionality of the text space and focus more on content-bearing words such as:

- delay
- cancel
- thanks
- terrible
- helpful

This can improve efficiency and sometimes model quality.

---

### Step 7: Lemmatization

```python
df_cleaned['tokens'] = df_cleaned['tokens'].apply(
    lambda tokens: [nlp(word)[0].lemma_ for word in tokens]
)
```

A **lemma** is a normalized base form of a word.

Examples:

- `delays` → `delay`
- `delayed` → `delay`
- `flying` → `fly`

#### Why this matters

Without lemmatization, similar words are treated as separate features.

That means the model may spread its learning across multiple tokens that really mean almost the same thing.

Lemmatization helps consolidate signal.

---

### Step 8: Join tokens back into processed text

```python
df_cleaned['processed_text'] = df_cleaned['tokens'].apply(lambda tokens: ' '.join(tokens))
```

At this point, each tweet becomes a cleaned text string ready for vectorization.

---

## Part 7: How Text Becomes Numbers

The notebook compares two approaches:

1. **CountVectorizer**
2. **TF-IDF Vectorizer**

Both transform text into a matrix with shape:

```text
14,640 × 10,987
```

That means:

- each row = one tweet
- each column = one vocabulary term
- each cell = some numeric representation of that term in that tweet

### Why the matrix is so wide

Language has many unique words. Even after cleaning, the vocabulary is large.

This is a common NLP pattern:

- relatively fewer observations than total possible text features
- very sparse matrices
- many zeros

---

## Part 8: CountVectorizer — The Basic Bag-of-Words Idea

### The idea

CountVectorizer builds a **document-term matrix**.

For each tweet and each word in the vocabulary, it stores the count of how many times that word appears.

If the vocabulary is:

```text
[delay, cancel, thank, great]
```

and a tweet is:

```text
"great thank you"
```

its vector might look like:

```text
[0, 0, 1, 1]
```

### Mathematical view

If I let:

- `d` = a tweet
- `t` = a token

then the CountVectorizer value is essentially:

```text
x(d, t) = count of token t in document d
```

This is the simplest bag-of-words representation.

### Why it is called “bag of words”

Because it usually ignores word order and keeps only token presence/frequency.

So these two sentences can look very similar numerically:

- `flight not good`
- `good flight not`

That is one limitation of classical vectorization.

---

## Part 9: TF-IDF — Weighting Rare but Informative Terms More Strongly

TF-IDF stands for **Term Frequency–Inverse Document Frequency**.

It still uses term-level features, but instead of raw counts only, it adjusts words based on how common they are across the entire corpus.

### Intuition

A word that appears in nearly every tweet is less informative.

A word that appears only in some tweets may carry stronger discriminatory value.

### Basic formula

A common simplified form is:

```text
tf-idf(t, d) = tf(t, d) × idf(t)
```

where:

```text
tf(t, d) = frequency of term t in document d
```

and

```text
idf(t) = log(N / df(t))
```

with:

- `N` = total number of documents
- `df(t)` = number of documents containing term `t`

So:

- common words get lower weight
- rarer informative words get higher weight

### Why this can help

Words like `delay`, `cancelled`, `rude`, `thanks`, or `amazing` may matter more than generic words that appear everywhere.

TF-IDF tries to reflect that idea.

---

## Part 10: The Target Encoding Step

The notebook encodes sentiment labels numerically:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

This maps the classes to numbers.

Because LabelEncoder sorts labels alphabetically, the mapping is:

- `0 = negative`
- `1 = neutral`
- `2 = positive`

### Why this is needed

Most scikit-learn classifiers require numeric targets.

The numbers do **not** mean negative < neutral < positive in a numeric distance sense. They are just class IDs.

---

## Part 11: Train-Test Split

The notebook uses:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

### What each argument means

- `test_size=0.2` → 20% of data goes to the test set
- `random_state=42` → makes the split reproducible
- `stratify=y` → preserves class proportions in train and test

### Why stratification matters

Since this dataset is imbalanced, I want train and test to contain roughly the same sentiment mix.

Otherwise, performance estimates can become unstable or misleading.

---

## Part 12: The Model Used — Random Forest Classifier

The notebook uses **RandomForestClassifier** for both vectorization approaches.

### What a random forest is

A random forest is an ensemble of many decision trees.

Each tree is trained on a bootstrap sample of the training data, and each split considers only a random subset of features.

For classification, the forest predicts by majority vote across trees.

### Mathematical intuition

If I have trees:

```text
T1(x), T2(x), ..., TB(x)
```

then the random forest prediction is the class receiving the most votes:

```text
ŷ = mode(T1(x), T2(x), ..., TB(x))
```

### Why random forests are useful

They:

- reduce variance compared with one single decision tree
- can handle nonlinear relationships
- are relatively robust
- provide feature importance

### But why random forests are not always ideal for text

For very high-dimensional sparse text data, linear models such as:

- Logistic Regression
- Linear SVM
- Naive Bayes

are often strong baselines.

So this project is useful pedagogically, but in real NLP practice I should not assume Random Forest is automatically the best model.

---

## Part 13: Cross-Validation and Tuning

The notebook performs 5-fold cross-validation on the training set:

```python
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
```

### What 5-fold cross-validation means

The training data is split into 5 parts.

Then the model is trained 5 times:

- each time, 4 folds are used for training
- 1 fold is used for validation

The accuracy scores are averaged.

### Why this matters

A single train-validation split can be noisy. Cross-validation gives a more stable estimate of expected training performance.

---

## Part 14: CountVectorizer Model Results

### Cross-validation performance

The notebook reports 5-fold accuracy scores of approximately:

```text
[0.758, 0.754, 0.742, 0.757, 0.765]
```

Mean cross-validation accuracy:

```text
0.755
```

So the training-stage validation performance is about **75.5%**.

### Final test-set classification report

For the CountVectorizer-based Random Forest model:

```text
Accuracy = 0.77
```

Class-wise performance:

- **Negative (0)**
  - Precision = `0.81`
  - Recall = `0.90`
  - F1 = `0.85`

- **Neutral (1)**
  - Precision = `0.61`
  - Recall = `0.53`
  - F1 = `0.57`

- **Positive (2)**
  - Precision = `0.73`
  - Recall = `0.57`
  - F1 = `0.64`

Macro average F1:

```text
0.69
```

Weighted average F1:

```text
0.76
```

### What this means

The model is strongest on the dominant negative class.

That is expected because:

- negative tweets are the largest class
- negative tweets often contain strong complaint-related words
- sentiment boundaries for negative tweets may be easier to learn than for neutral tweets

The hardest class is **neutral**.

That also makes sense because neutral language often overlaps with both positive and negative language. A neutral tweet may contain airline terms without a strong emotional marker.

---

## Part 15: TF-IDF Model Results

### Cross-validation performance

The TF-IDF version reports 5-fold accuracy scores of approximately:

```text
[0.756, 0.756, 0.746, 0.757, 0.754]
```

Mean cross-validation accuracy:

```text
0.754
```

So the training-stage validation result is almost the same as CountVectorizer.

### Final test-set classification report

For the TF-IDF-based Random Forest model:

```text
Accuracy = 0.75
```

Class-wise performance:

- **Negative (0)**
  - Precision = `0.77`
  - Recall = `0.95`
  - F1 = `0.85`

- **Neutral (1)**
  - Precision = `0.65`
  - Recall = `0.38`
  - F1 = `0.48`

- **Positive (2)**
  - Precision = `0.76`
  - Recall = `0.50`
  - F1 = `0.60`

Macro average F1:

```text
0.64
```

Weighted average F1:

```text
0.73
```

### What this means

The TF-IDF model is very aggressive in catching negative tweets, shown by the **0.95 recall** for the negative class.

But it does worse on neutral tweets than the CountVectorizer model.

So in this notebook, **CountVectorizer + Random Forest** is the slightly better overall baseline.

---

## Part 16: Comparing CountVectorizer vs TF-IDF

### Side-by-side summary

| Metric | CountVectorizer + RF | TF-IDF + RF |
|---|---:|---:|
| Mean CV accuracy | 75.5% | 75.4% |
| Test accuracy | 77% | 75% |
| Negative F1 | 0.85 | 0.85 |
| Neutral F1 | 0.57 | 0.48 |
| Positive F1 | 0.64 | 0.60 |
| Weighted F1 | 0.76 | 0.73 |

### My interpretation

In this notebook:

- both methods are broadly comparable
- CountVectorizer performs a little better overall
- TF-IDF increases recall on negative tweets, but that comes with weaker neutral detection

### Why CountVectorizer may work slightly better here

A possible reason is that complaint-related keywords are already highly informative in raw frequency form. Words such as:

- delay
- cancel
- hold
- bag
- customer
- thanks
- awesome

may be so directly tied to sentiment that simple counts work well enough.

---

## Part 17: Evaluation Metrics — What They Mean and Why I Must Know Them

For interviews, I need to understand these clearly.

### 1. Accuracy

```text
Accuracy = correct predictions / total predictions
```

This is useful, but by itself it can hide weakness on minority classes.

In this dataset, predicting negative well boosts overall accuracy because negative is the largest class.

---

### 2. Precision

For a given class:

```text
Precision = TP / (TP + FP)
```

It answers:

> Of everything the model predicted as this class, how much was correct?

If positive precision is low, many tweets predicted as positive are actually not positive.

---

### 3. Recall

```text
Recall = TP / (TP + FN)
```

It answers:

> Of all the true examples of this class, how many did the model find?

If neutral recall is low, the model is missing many truly neutral tweets.

---

### 4. F1 Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

This balances precision and recall.

That is why F1 is often very useful in imbalanced classification.

---

### 5. Macro average

Macro average gives equal weight to each class.

So it tells me how well the model performs across classes **without letting the majority class dominate**.

---

### 6. Weighted average

Weighted average uses class frequencies as weights.

That means large classes influence it more.

In this dataset, weighted metrics are strongly influenced by the negative class.

---

## Part 18: Why Neutral Sentiment Is Hard

The notebook shows that neutral is the weakest class.

That is very common in sentiment modeling.

### Reasons neutral is difficult

1. Neutral language often lacks strong emotion words.
2. Neutral tweets may share vocabulary with both positive and negative tweets.
3. The boundary between neutral and mildly negative is often subjective.
4. Some tweets are factual but still imply dissatisfaction.
5. Annotation noise may exist in human-labeled sentiment datasets.

### Example intuition

A tweet like:

```text
flight delayed 30 mins waiting at gate
```

might be labeled negative.

But another tweet like:

```text
boarding now at gate 24 for delayed departure
```

might sound more neutral depending on context.

That overlap makes the class boundary fuzzy.

---

## Part 19: Feature Importance in a Text Model

The notebook extracts feature importance from the Random Forest.

### What feature importance means here

It estimates which terms are most useful for decision splits in the trees.

Words associated with:

- delays
- cancellations
- customer service
- appreciation
- gratitude

become important because they help separate the sentiment classes.

### Caution about interpretation

Feature importance in tree models is useful, but I should not overinterpret it as a causal explanation.

A word being important means:

> the model found this token useful for splitting the data

It does **not** automatically mean the word itself causes sentiment.

---

## Part 20: Important Methodological Caveats I Need to Know

This part is very important for interviews because it shows maturity.

### Caveat 1: Vectorization is fit before the train-test split

In the notebook, both CountVectorizer and TF-IDF are fit on the full processed dataset **before** the split:

```python
X = count_vectorizer.fit_transform(df_cleaned['processed_text'])
```

and similarly for TF-IDF.

### Why this is a problem

The vocabulary and document statistics are being learned from the full dataset, including observations that later end up in the test set.

That means the test set is not fully isolated.

This is a mild form of **data leakage**.

### Cleaner production approach

The correct sequence is:

1. split raw text into train and test
2. fit the vectorizer on **training text only**
3. transform the training text
4. transform the test text using the fitted training vectorizer

This makes evaluation more honest.

---

### Caveat 2: Random Forest is not usually the strongest default for sparse text

The notebook uses Random Forest, which is okay for learning.

But for classic text classification, stronger baseline choices often include:

- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM

And for more modern setups:

- pretrained embeddings
- LSTM/GRU sequence models
- transformer models such as BERT

So I should treat this notebook as a strong introductory NLP exercise, not as the final word on best text-model architecture.

---

### Caveat 3: Only unigram bag-of-words features are used

The notebook uses standard vectorizers in a simple way.

That means phrase structure and local word order are mostly lost.

So:

- `not good`
- `good`

may not be handled as intelligently as a more advanced model would handle them.

Adding **n-grams** such as bigrams could help.

---

### Caveat 4: Metadata is ignored

The notebook intentionally keeps only text and sentiment.

That is good for learning the NLP core, but in practice I might improve performance with additional features such as:

- airline
- retweet count
- presence of mentions
- tweet length
- punctuation intensity
- hashtag count
- whether the tweet contains complaint keywords

---

## Part 21: A Cleaner Production-Style Pipeline I Should Remember

If I were rebuilding this more rigorously, I would do this:

```text
Raw text + target
        │
        ▼
Train-test split on raw text
        │
        ▼
Fit text cleaner / vectorizer on training data only
        │
        ▼
Transform train and test separately
        │
        ▼
Use stratified CV inside training only
        │
        ▼
Tune multiple model families
        │
        ▼
Evaluate once on held-out test set
        │
        ▼
Inspect class-wise metrics and error cases
```

### Better scikit-learn design

A `Pipeline` is ideal here because it prevents leakage and makes the workflow reproducible.

Example skeleton:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=3)),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced'))
])
```

This is the kind of structure I should remember for interviews.

---

## Part 22: How I Would Explain the Math in an Interview

If someone asks me for the math, I should answer at a practical level.

### 1. Bag-of-words math

Each document is converted into a vector of token counts:

```text
x_d = [count(w1 in d), count(w2 in d), ..., count(wV in d)]
```

where `V` is the vocabulary size.

---

### 2. TF-IDF math

Each term gets weighted by how frequent it is in the document and how rare it is across all documents:

```text
tf-idf(t, d) = tf(t, d) × log(N / df(t))
```

So common words get lower influence and rare informative words get higher influence.

---

### 3. Random Forest math intuition

Each tree partitions feature space using rules such as:

```text
if token_delay_weight > threshold → go left
else → go right
```

The forest averages across many such trees by majority vote.

---

### 4. Multi-class classification idea

In a three-class problem, the model ultimately assigns one of the three labels.

Evaluation is then done one class at a time and also in aggregate.

That is why class-wise precision, recall, and F1 are so important.

---

## Part 23: Code Logic I Should Be Able to Explain

If I am asked to walk through the code, this is the flow I should explain:

### Data preparation

```python
df_cleaned = df[['text', 'airline_sentiment']].copy()
```

Keep the modeling problem simple.

### Cleaning functions

```python
remove_html()
decontracted()
remove_numbers()
```

Normalize text and remove obvious noise.

### Tokenization and lemmatization

```python
nlp = spacy.load('en_core_web_sm')
```

Use spaCy to break text into tokens and reduce words to their lemmas.

### Vectorization

```python
CountVectorizer()
TfidfVectorizer()
```

Convert text into sparse numeric matrices.

### Label encoding

```python
LabelEncoder()
```

Convert string labels to numeric classes.

### Split and model

```python
train_test_split(..., stratify=y)
RandomForestClassifier(...)
```

Train on one part of the data and evaluate on held-out data.

### Evaluation

```python
classification_report()
confusion_matrix()
```

Check not only overall accuracy but also per-class performance.

---

## Part 24: Business Interpretation of the Results

A model like this can help summarize large volumes of customer feedback into structured intelligence.

### Example practical uses

- identify sudden increases in negative sentiment
- compare service perception across airlines or product lines
- route negative tweets for faster customer-service action
- identify recurring complaint themes such as delays or baggage issues
- track how operational disruptions affect customer reaction

### What the current model already shows

The text clearly contains useful signal.

Even with a relatively simple classical NLP pipeline, the model reaches about:

- **77% test accuracy** with CountVectorizer + Random Forest
- strong performance on negative tweets
- reasonable but not strong performance on positive and neutral tweets

That means customer dissatisfaction language is especially detectable.

---

## Part 25: How This Connects to My Other Brain Notes

This project connects nicely to the other notes I built.

### Connection to PD modeling

In PD models, I work with structured variables such as utilization, delinquencies, and bureau features.

Here, I work with unstructured text.

But the core workflow is still familiar:

- define target
- inspect data quality
- preprocess inputs
- split train and test
- choose features
- fit model
- evaluate performance
- interpret outputs
- think about production use and leakage

### Connection to model monitoring

If this sentiment model were in production, I would still monitor:

- class distribution drift
- vocabulary drift
- missing text or malformed text rate
- average tweet length drift
- prediction mix drift
- degradation in precision / recall by class

That is the same model-risk mindset as in my monitoring note.

### Connection to churn and retention projects

The churn and employee-retention projects taught me tabular classification.

This project teaches me text classification.

Together, they help me understand that machine learning is not only about one model family. It is about how I represent information and connect it to a decision problem.

---

## Part 26: What I Would Improve Next

If I continue this project, the most meaningful improvements would be:

### 1. Fix leakage completely

Split the raw text before fitting the vectorizer.

### 2. Try stronger classical NLP baselines

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

### 3. Add n-grams

Use bigrams such as:

- `not good`
- `customer service`
- `late flight`
- `lost bag`

### 4. Handle class imbalance explicitly

Possible options:

- class weights
- threshold analysis
- resampling where appropriate

### 5. Add more features

- tweet length
- exclamation count
- number of mentions
- airline name
- sentiment lexicon counts

### 6. Perform error analysis

Review misclassified neutral and positive tweets to understand where the model is failing.

### 7. Try modern NLP methods

A transformer-based model would likely perform better than bag-of-words + Random Forest if implemented carefully.

---

## Part 27: Interview-Ready Questions I Should Be Able to Answer from This Note

### Why did I remove stopwords?

To reduce common low-information words and focus more on tokens that carry sentiment signal.

### Why did I lemmatize?

To map related word forms to a shared base form and reduce unnecessary vocabulary fragmentation.

### Difference between CountVectorizer and TF-IDF?

CountVectorizer uses raw token counts. TF-IDF adjusts counts by term rarity across the corpus, downweighting very common terms.

### Why was neutral harder to classify?

Because neutral language overlaps with both positive and negative language and often lacks strong emotional markers.

### Why is accuracy alone not enough?

Because the data is imbalanced. Strong performance on the majority negative class can hide weaker performance on neutral and positive classes.

### What leakage exists in the notebook?

The vectorizer is fit on the full dataset before the train-test split, so vocabulary information from the future test set leaks into feature construction.

### Why is stratified splitting important?

It preserves sentiment class proportions in train and test, which is especially important when classes are imbalanced.

### Is Random Forest the best NLP model here?

Not necessarily. It is useful for learning, but text classification often performs very well with Logistic Regression, Naive Bayes, Linear SVM, or modern transformer models.

---

## Part 28: Final Takeaways I Want to Retain

1. **Text must be converted into numbers before standard ML models can use it.**
2. **Cleaning choices directly affect the vocabulary and the signal the model sees.**
3. **CountVectorizer and TF-IDF are classical but powerful text representations.**
4. **This is a multi-class classification problem, so class-wise metrics matter a lot.**
5. **Negative tweets are easiest to detect because the class is larger and the language is often more explicit.**
6. **Neutral sentiment is the hardest class because the boundary is fuzzy.**
7. **Vectorizer fitting must happen after the train-test split to avoid leakage.**
8. **Random Forest works as a learning baseline, but stronger NLP baselines should also be tested.**
9. **Business value comes from turning unstructured feedback into structured monitoring and action.**
10. **This project strengthens my understanding of NLP, classification, evaluation, and production thinking all at once.**

---

## Key Concepts Summary

- **NLP:** turning language into structured features that a model can use
- **Tokenization:** splitting text into words or word-like units
- **Stopword removal:** dropping very common words with low information value
- **Lemmatization:** reducing words to their base form
- **CountVectorizer:** bag-of-words counts
- **TF-IDF:** term weighting based on rarity and informativeness
- **Sparse matrix:** a matrix with many zeros, common in text data
- **Multi-class classification:** predicting one of more than two classes
- **Precision / Recall / F1:** class-sensitive performance metrics
- **Macro average:** equal weight to each class
- **Weighted average:** weight by class frequency
- **Data leakage:** allowing information from test data to influence training or preprocessing
- **Random Forest:** ensemble of decision trees using bagging and random feature subsampling

---

This note gives me a complete beginner-friendly view of how a classical NLP sentiment-analysis pipeline works from raw tweet text all the way to model evaluation and practical business interpretation.
