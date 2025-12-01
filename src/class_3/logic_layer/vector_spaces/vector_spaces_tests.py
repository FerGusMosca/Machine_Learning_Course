import os
import joblib
import numpy as np
from sklearn.preprocessing import normalize


class VSTest:
    """
    Vector Space testing for sentiment tweets.
    Cosine similarity against class-centroids.
    FAST MODE.
    """

    LABEL_NAMES = {
        0: "negative",
        1: "neutral",
        2: "positive"
    }

    def __init__(self, model_dir):
        self.vectorizer = joblib.load(os.path.join(model_dir, "vs_vectorizer.pkl"))
        self.class_vectors = joblib.load(os.path.join(model_dir, "vs_class_vectors.pkl"))

    def run_test(self, text):
        # Vectorize input tweet
        vec = self.vectorizer.transform([text]).toarray()
        vec = normalize(vec, norm='l2')

        sims = {}

        for lbl, cv in self.class_vectors.items():
            cv = cv.ravel()           # flatten centroid
            sim = float(np.dot(vec, cv))
            sims[lbl] = sim

        best_class = max(sims, key=sims.get)

        print("\n[VS] Input tweet:")
        print(text)

        print("\n[VS] Cosine similarities:")
        for k, v in sims.items():
            label = self.LABEL_NAMES[k]
            print(f"  {label:8s} → {v:.4f}")

        print(f"\n[VS] Predicted sentiment → {self.LABEL_NAMES[best_class]}\n")
