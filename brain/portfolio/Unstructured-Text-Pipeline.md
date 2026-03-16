---
title: "NLP Pipeline & Twitter Sentiment Analysis"
date: "2026-03-19"
summary: "Engineered a Natural Language Processing pipeline (77% acc) demonstrating foundational text classification techniques used in adverse media screening."
tags: ["Python", "SpaCy", "TF-IDF", "Random Forest", "NLP", "AML"]
cluster: "Quantitative Portfolio"
---

# Unstructured Text & Sentiment Analysis

In modern risk management, unstructured text (like news articles, financial reports, or social media) is a critical data source for Adverse Media Screening and AML (Anti-Money Laundering) compliance. This project builds the foundational Natural Language Processing (NLP) architecture required to automatically ingest, process, and classify volatile text data.

## Project Architecture
- **Objective:** Process raw, unstructured Twitter text data to predict brand sentiment, acting as a proxy for automated reputational risk monitoring.
- **Core Methodology:** Built a comprehensive NLP preprocessing pipeline utilizing regex and `spaCy` for tokenization, lemmatization, and stop-word removal. Transformed the cleaned text into mathematical vectors evaluating both `CountVectorizer` and `TF-IDF` strategies.
- **Modeling:** Trained and hyperparameter-tuned a Random Forest classifier. The optimized model achieved **77% overall accuracy**, with a critical **95% Recall** in identifying negative sentiment via TF-IDF, proving its utility as a risk-flagging engine.

*(Text preprocessing functions, TF-IDF matrix generation, and word cloud visualizations will be added here).*

---

## 1. Data Ingestion & Exploratory Analysis

The pipeline ingested a dataset of 14,640 tweets directed at six major US airlines. Before applying NLP transformations, baseline distributions were analyzed to understand the inherent data bias:

- **Sentiment Skew:** The data was heavily imbalanced toward negative sentiment (>60% of all records). This is typical in risk-monitoring datasets, where complaints or adverse events are disproportionately represented compared to neutral operations.
- **Root Cause Extraction:** Analysis of the `negativereason` feature revealed "Customer Service Issues" (approx. 3,000 mentions), "Late Flights", and "Cancelled Flights" as the primary drivers of reputational damage.
- **Brand Exposure:** United Airlines held the largest share of mentions (>25%), tightly correlated with the highest volume of negative sentiment.

---

## 2. Text Preprocessing (The NLP Pipeline)

Machine Learning algorithms cannot process raw text. A strict preprocessing pipeline was engineered using Regular Expressions (`re`) and the `spaCy` (`en_core_web_sm`) library to strip noise and normalize the data.

1. **Noise Reduction:** - Stripped HTML tags and URLs.
   - Removed numerical digits to prevent the model from memorizing specific flight numbers or dates.
2. **De-contraction:** Mapped conversational contractions to their formal equivalents (e.g., "won't" $\rightarrow$ "will not", "I'm" $\rightarrow$ "I am") to preserve semantic meaning.
3. **Tokenization & Stop-word Removal:** Split sentences into individual tokens and removed low-value English stop-words (e.g., "the", "is", "at") using `spaCy`'s default vocabulary.
4. **Lemmatization:** Transformed words back to their base dictionary form (e.g., "flying", "flew" $\rightarrow$ "fly") using `spaCy`'s advanced linguistic models, drastically reducing the dimensionality of the final vocabulary.

---

## 3. Mathematical Vectorization 

To feed the cleaned text into a Random Forest ensemble, the words were mapped into a high-dimensional mathematical space (14,640 documents $\times$ 10,987 unique vocabulary features). Two distinct strategies were evaluated:

### Strategy A: CountVectorizer (Bag of Words)
Constructs a matrix where each cell represents the raw frequency of a word in a specific tweet. 
- *Result:* Created a sparse matrix. While computationally simple, it gives disproportionate weight to frequently occurring words that may lack predictive signal.

### Strategy B: TF-IDF (Term Frequency-Inverse Document Frequency)
A superior quantitative approach that scales word counts by how rare they are across the entire dataset. It highlights words that are unique to specific sentiments.
$$W_{i,j} = tf_{i,j} \times \log\left(\frac{N}{df_i}\right)$$
- $tf_{i,j}$: Frequency of term $i$ in document $j$.
- $N$: Total number of documents.
- $df_i$: Number of documents containing term $i$.

---

## 4. Modeling & Hyperparameter Optimization

A `RandomForestClassifier` was selected for its robustness against overfitting in high-dimensional, sparse datasets (like text matrices).

- **Hyperparameter Tuning:** The `n_estimators` (number of trees) parameter was systematically tuned via cross-validation. Misclassification error was plotted against tree count, revealing that the model stabilized optimally at **110 estimators**.
- **Model Evaluation (CountVectorizer):** - **Accuracy:** 77%
  - **Negative Sentiment Precision/Recall:** 0.81 / 0.90 (Excellent capture rate for the dominant class).
- **Model Evaluation (TF-IDF):** - **Accuracy:** 75%
  - **Negative Sentiment Recall:** **0.95**. The TF-IDF model proved exceptionally sensitive to negative text, making it highly effective for risk-flagging.

---

## 5. Interview Talking Points: Application to AML & Risk

- **From Airlines to Adverse Media:** While trained on airline data, this exact mathematical pipeline (Regex cleanup $\rightarrow$ `spaCy` Lemmatization $\rightarrow$ TF-IDF $\rightarrow$ Random Forest) is how quantitative teams ingest Bloomberg terminal feeds or Reuters articles to flag entities for Adverse Media in KYC/AML checks.
- **Prioritizing Recall in Risk Models:** In the context of risk management, the TF-IDF model's **95% Recall for negative sentiment** is the most critical metric. Missing a negative event (False Negative) carries severe regulatory or reputational penalties, whereas a False Positive simply triggers a manual compliance review. 
- **Handling Structural Imbalance:** Identified that the model struggled with Precision on the "Neutral" class (0.65) due to the heavy negative skew. Addressed this conceptually by recommending future iterations utilize SMOTE or algorithm-level class weighting (`class_weight='balanced'`) to penalize minority class errors.