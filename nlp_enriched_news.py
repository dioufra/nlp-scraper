# ┌────────────────────────────────────┐
# │           Imports & Setup          │
# └────────────────────────────────────┘

import re
import os
import spacy  # type: ignore
import string
import pickle
import numpy as np
import pandas as pd


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sqlalchemy import create_engine
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

from results.training_model import train_topic_classifier


# ┌────────────────────────────────────┐
# │       NLP Components               │
# └────────────────────────────────────┘
nlp = spacy.load("en_core_web_lg")
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ┌────────────────────────────────────┐
# │       Disasters Keywords           │
# └────────────────────────────────────┘
DISASTER_KEYWORDS = [
    "pollution",
    "contamination",
    "spill",
    "leak",
    "hazardous",
    "toxic",
    "deforestation",
    "emission",
    "carbon",
    "waste",
    "chemical",
    "oil spill",
    "radiation",
    "environmental damage",
    "ecological disaster",
    "climate change",
    "global warming",
    "extinction",
    "habitat loss",
    "industrial accident",
]

# Precompute keyword embeddings
keyword_embeddings = [nlp(keyword).vector for keyword in DISASTER_KEYWORDS]


# ┌────────────────────────────────────┐
# │         Entities Detection         │
# └────────────────────────────────────┘
def get_companies(text):
    """
    Detect entities of a given text

    Given a text return a list of all orginations given a text.

        >>> EXAMPLE TO ADD LATER

    :param text: text to be preprocessed
    :type text: string
    :return: a list of ORG
    :rtype: list
    """

    doc = nlp(text)

    return list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))


# ┌────────────────────────────────────┐
# │            Preprocessor            │
# └────────────────────────────────────┘
def preprocess_text(text):
    """
    Preprocess a given text for natural language processing

    Given a text return a list of the temmed tokens composing the text
    with the stop words removed and the text lowercased.

    :param text: text to be preprocessed
    :type text: string
    :return: a list of stemmed world
    :rtype: list
    """

    print("Text preprocessing ...")

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


def analyze_sentiment(text):
    """Analyze sentiment using NLTK's VADER"""
    # Lowercase
    text = text.lower()

    # Keep essential punctuation
    text = re.sub(r"(?<!\w)[@#](?!\w)", "", text)  # Remove handles/hashtags

    scores = sia.polarity_scores(text)
    compound_score = scores["compound"]

    if compound_score >= 0.05:
        return "positive", compound_score
    elif compound_score <= -0.05:
        return "negative", compound_score
    else:
        return "neutral", compound_score


# ┌────────────────────────────────────┐
# │          Dectect Scandal           │
# └────────────────────────────────────┘
def detect_scandal(text, entities):
    """Detect environmental scandal mentions for specific organizations"""
    if not entities:
        return 0.0

    # Process sentences
    sentences = sent_tokenize(text)

    # Scan each sentence
    for sent in sentences:
        # Check if sentence contains any of our entities
        for entity in entities:
            if entity in sent:
                # Check for scandal in this sentence
                sent_embedding = nlp(sent).vector
                if not sent_embedding.any():
                    continue

                # Calculate similarity with scandal keywords
                similarities = [
                    np.dot(sent_embedding, kw_embedding)
                    / (np.linalg.norm(sent_embedding) * np.linalg.norm(kw_embedding))
                    for kw_embedding in keyword_embeddings
                ]

                max_similarity = max(similarities) if similarities else 0.0

                # If scandal detected, print alert
                if max_similarity > 0.75:
                    print(f"Environmental scandal detected for {entity}")
                    # print(f"Relevant sentence: '{sent}'\n")

    return max_similarity


# ┌────────────────────────────────────┐
# │          Process Articles          │
# └────────────────────────────────────┘
def process_article(row, classifier, vectorizer):
    """Process a single news article through the NLP pipeline"""
    result = {
        "uuid": row.get("uuid", ""),
        "url": row.get("url", ""),
        "date": row.get("date", ""),
        "headline": row.get("headline", ""),
        "body": row.get("body", ""),
    }

    # Combine headline and body for analysis
    full_text = f"{row.get('headline', '')}. {row.get('body', '')}"

    print(f"\nEnriching {row.get('url', '')}:")

    # Entity detection
    print("\n---------- Detect entities ----------")
    orgs = get_companies(full_text)
    result["org"] = ", ".join(orgs)
    print(
        f"Detected {len(orgs)} companies which are {', '.join(orgs) if orgs else 'None'}"
    )

    # Topic detection
    print("\n---------- Topic detection ----------")
    processed_text = preprocess_text(full_text)
    X = vectorizer.transform([processed_text])
    topic = classifier.predict(X)[0]
    result["topic"] = topic
    print(f"The topic of the article is: {topic}")

    # Sentiment analysis
    print("\n---------- Sentiment analysis ----------")
    sentiment, score = analyze_sentiment(full_text)
    result["sentiment"] = sentiment
    result["sentiment_score"] = score
    print(f"The article {result['headline']} has a {sentiment} sentiment")

    # Scandal detection
    print("\n---------- Scandal detection ----------")
    scandal_score = detect_scandal(full_text, orgs)
    result["scandal_distance"] = scandal_score
    print(f"Environmental scandal score: {scandal_score:.2f}")

    return result


# ┌────────────────────────────────────┐
# │             Load data              │
# └────────────────────────────────────┘
def load_data(db_path):
    engine = create_engine(db_path)

    df = pd.read_sql_table("article", con=engine)

    return df.sort_values(by="date", ascending=False)


# ┌────────────────────────────────────┐
# │                Main                │
# └────────────────────────────────────┘
def main():
    # Load data
    df = load_data("sqlite:///data/articles.db")

    # Train or load topic detector
    detector_path = "results/topic_classifier.pkl"
    if not os.path.exists(detector_path):
        print("Training topic classifier...")
        train_topic_classifier("./data/bbc_news_train.csv", "./data/topic_test.csv")

    with open(detector_path, "rb") as f:
        detector = pickle.load(f)
        classifier = detector["classifier"]
        vectorizer = detector["vectorizer"]

    # Process articles without progress bar
    print(f"Processing {len(df)} articles...")
    results = []
    for _, row in df.iterrows():
        try:
            result = process_article(row, classifier, vectorizer)
            results.append(result)
        except Exception as e:
            print(f"Error processing article {row.get('url', '')}: {str(e)}")

    # Post-processing
    enriched_df = pd.DataFrame(results)
    enriched_df["top_10"] = False
    if len(enriched_df) > 0:
        top_indices = enriched_df["scandal_distance"].nlargest(10).index
        enriched_df.loc[top_indices, "top_10"] = True

    # Save results
    os.makedirs("results", exist_ok=True)
    enriched_df.to_csv("results/enhanced_news.csv", index=False)
    print(
        f"Processing complete. Saved {len(enriched_df)} results to results/enhanced_news.csv"
    )


if __name__ == "__main__":
    main()
