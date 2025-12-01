"""
Naive Bayes training using TweetEval (sentiment).
English-only comments.
"""

import os
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class NBTraining:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        print("\n[NB] Loading TweetEval (sentiment)...")
        ds = load_dataset("tweet_eval", "sentiment")

        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]

        print(f"[NB] Dataset loaded: {len(train_texts)} training tweets.")
        print("\n[NB] Example tweets:")
        for i in range(5):
            print(f"- ({train_labels[i]}) {train_texts[i]}")

        print("\n[NB] Vectorizing...")
        vectorizer = CountVectorizer(lowercase=True)
        X = vectorizer.fit_transform(train_texts)
        y = train_labels

        print("[NB] Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print("[NB] Training Naive Bayes...")
        model = MultinomialNB()
        model.fit(X_train, y_train)

        print("[NB] Evaluating model...")
        preds = model.predict(X_test)
        print(classification_report(y_test, preds))

        print("[NB] Saving model + vectorizer...")
        joblib.dump(model, os.path.join(self.output_dir, "nb_model.pkl"))
        joblib.dump(vectorizer, os.path.join(self.output_dir, "nb_vectorizer.pkl"))

        print("\n[NB] Training complete.\n")
