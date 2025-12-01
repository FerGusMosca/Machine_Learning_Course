"""
Naive Bayes testing module.
English-only comments.
"""

import os
import joblib


class NBTesting:
    def __init__(self, models_dir):
        self.models_dir = models_dir

    def run(self):
        model_path = os.path.join(self.models_dir, "nb_model.pkl")
        vec_path = os.path.join(self.models_dir, "nb_vectorizer.pkl")

        if not os.path.exists(model_path):
            print("[NB] ERROR: Model not trained yet.")
            return

        print("\nEnter a tweet to classify:")
        tweet = input("Tweet >> ").strip()

        print("[NB] Loading model...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)

        print("[NB] Vectorizing input...")
        X = vectorizer.transform([tweet])

        print("[NB] Predicting...")
        pred = model.predict(X)[0]

        label_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        print(f"\nResult: {label_map[pred]}\n")
