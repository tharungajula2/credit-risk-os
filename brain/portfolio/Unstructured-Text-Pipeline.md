---
title: "NLP Pipeline & Twitter Sentiment Analysis"
date: "2026-03-15"
summary: "Classification pipeline (77% acc) demonstrating foundational techniques used in adverse media screening."
tags: ["Python", "NLTK", "TF-IDF", "NLP", "AML"]
---

# Unstructured Text & Sentiment Analysis

In modern risk management, unstructured text (like news articles or social media) is a critical data source for Adverse Media Screening and AML (Anti-Money Laundering) compliance. This project builds the foundational NLP skills required for those tasks.

## Project Architecture
- **Objective:** Process raw, unstructured Twitter text data to predict brand sentiment for airlines.
- **Core Methodology:** Built a comprehensive Natural Language Processing (NLP) pipeline. Utilized `SpaCy` and `NLTK` for tokenization, lemmatization, and stop-word removal. Transformed text into mathematical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
- **Modeling:** Managed class imbalance using SMOTE and trained a Random Forest classifier, achieving **77% accuracy** on highly volatile social media text.

*(Text preprocessing functions, TF-IDF matrix generation, and word cloud visualizations will be added here).*