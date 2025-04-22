import os
import re
import string
import pickle
import spacy  # type: ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import learning_curve


# Initialize NLP components
nlp = spacy.load("en_core_web_lg")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# Preprocessor
def preprocess_text(text):
    """
    Preprocess a given text for natural language processing.

    This function removes URLs, punctuation, stopwords,
    lowercases all words, and applies stemming.

    :param text: Input raw text
    :type text: str
    :return: Preprocessed text
    :rtype: str
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return " ".join(stemmed_tokens)


# Classifier building
def train_topic_classifier(train_data_path, test_data_path):
    """Train and save a topic classification model"""
    # Load datasets
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # Preprocess text
    train_df["processed_text"] = train_df["Text"].apply(preprocess_text)
    test_df["processed_text"] = test_df["Text"].apply(preprocess_text)

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df["processed_text"])
    X_test = vectorizer.transform(test_df["processed_text"])
    y_train = train_df["Category"]
    y_test = test_df["Category"]

    # Train classifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Plot learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        classifier,
        X_train,
        y_train,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training score")
    plt.plot(
        train_sizes, np.mean(test_scores, axis=1), "o-", label="Cross-validation score"
    )
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.savefig("../results/learning_curves.png")
    plt.close()

    # Save model
    os.makedirs("../results", exist_ok=True)

    with open("../results/topic_classifier.pkl", "wb") as f:
        pickle.dump({"classifier": classifier, "vectorizer": vectorizer}, f)

    return classifier, vectorizer


if __name__ == "__main__":
    train_topic_classifier("./bbc_news_train.csv", "./bbc_news_tests.csv")
