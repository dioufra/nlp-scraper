# 📰 NLP-Enriched News Intelligence Platform

This project is a full pipeline that scrapes recent articles from [Al Jazeera](https://www.aljazeera.com), processes their content using Natural Language Processing (NLP), classifies them by topic using semantic embeddings, analyzes sentiment, detects environmental scandals, and stores the results in a SQL database. It is built in Python and integrates modern NLP techniques using SpaCy and scikit-learn.

## 🚀 Features

- ✅ **Web Scraper** for Al Jazeera using their GraphQL API
- ✅ **Article Storage** in SQLite via SQLAlchemy ORM
- ✅ **Text Preprocessing**: cleaning, tokenization, stopword removal, stemming
- ✅ **Topic Classification** using word embeddings and Logistic Regression
- ✅ **Sentiment Analysis** using NLTK's VADER
- ✅ **Scandal Detection** using semantic similarity to environment-related keywords
- ✅ **Concurrency** for faster scraping
- ✅ **Learning Curve Visualization** to evaluate classifier performance

## 🗂 Project Structure
```
project
.
├── data
│   └── ...
├── nlp_enriched_news.py
├── requirements.txt
├── README.md
├── results
│   ├── training_model.py
│   ├── enhanced_news.csv
│   └── learning_curves.png
└── scraper_news.py
```

## 🧠 How It Works

### 1. Scraper (`nlp_enriched_news.py`)
- Collects up to 400 articles via Al Jazeera’s GraphQL API.
- Extracts `url`, `date`, `headline`, and `body` for each article.
- Filters and stores articles newer than 90 days into a SQLite database.

### 2. Topic Classifier (`topic_classifier.py`)
- Preprocesses the news text using SpaCy + NLTK.
- Converts the text into dense vectors using SpaCy’s `en_core_web_lg` embeddings.
- Trains a Logistic Regression classifier.
- Evaluates performance with a learning curve.
- Saves the classifier and vectorizer with `pickle`.

#### ❓ What is Overfitting?
Overfitting happens when a model learns the training data too well, including noise and irrelevant details. As a result, the model performs poorly on unseen data. We mitigate this by validating on separate data and plotting a learning curve to assess generalization.

### 3. Sentiment Analysis
- Uses NLTK's VADER sentiment intensity analyzer to compute sentiment scores (positive, negative, or neutral).
- Helps understand the tone of the article's content.

### 4. Scandal Detection
- Uses SpaCy's `en_core_web_lg` embeddings to represent sentences semantically.
- Compares each sentence mentioning a company to a list of environment-related keywords.
- Computes cosine similarity between sentence vectors and keyword vectors.
- Flags articles with high similarity (> 0.75) as potential environmental scandals.

#### 🔹 Why `en_core_web_lg` Embeddings?
This SpaCy model provides 300-dimensional word vectors trained on a large corpus. It enables fine-grained semantic similarity computation and improves the performance of both topic classification and scandal detection.

#### 🔹 Why Cosine Similarity?
Cosine similarity is effective for comparing direction of vectors in high-dimensional space regardless of magnitude. It's widely used in NLP tasks for semantic comparison due to its simplicity and interpretability.

## 📦 Requirements
* requests
* beautifulsoup4
* sqlalchemy
* spacy
* nltk
* scikit-learn
* matplotlib
* pandas
* numpy

## Installation
1. Clone the repository
2. Create and set up the environment:
```bash
conda create --name <env> --file requirements.txt
python -m spacy download en_core_web_lg
python -m nltk.downloader punkt wordnet stopwords vader_lexicon
```

## Usage
1. Data Collection
Run the scraper to collect at least 300 articles:
```python
python scraper_news.py
```

2. Topic classifier
Run the classifier to train a topic detector model
```python
python topic_classifier.py
```

3. NLP Processing
Run the NLP pipeline on collected articles:
```bash
python nlp_enriched_news.py
```
