import os
import joblib
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import classification_report


class VSTraining:
    """
    Fast Vector Space training using TweetEval (sentiment).
    English-only comments.
    Uses TF-IDF + class centroids (very fast).
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):

        print("\n[VS] Loading TweetEval (sentiment)...")
        ds = load_dataset("tweet_eval", "sentiment")

        texts = ds["train"]["text"]
        labels = ds["train"]["label"]

        print(f"[VS] Dataset loaded: {len(texts)} tweets.")

        print("\n[VS] Example tweets:")
        for i in range(5):
            print(f"- ({labels[i]}) {texts[i]}")

        print("\n[VS] Building TF-IDF matrix...")
        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=20000,
            ngram_range=(1, 1)
        )
        X = vectorizer.fit_transform(texts)
        X_norm = normalize(X, norm='l2')

        print("[VS] Building centroid vectors per class...")
        class_vectors = {}
        labels_np = np.array(labels)

        for lbl in np.unique(labels_np):
            centroid = X_norm[labels_np == lbl].mean(axis=0)
            class_vectors[int(lbl)] = centroid

        # -------------------------------------------------
        #    EVALUATION (classification_report)
        # -------------------------------------------------
        print("\n[VS] Evaluating model on training set...")

        label_names = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }

        preds = []
        for i in range(X_norm.shape[0]):
            vec = X_norm[i]
            sims = {}
            for lbl, cv in class_vectors.items():
                sims[lbl] = float(vec @ cv.T)
            best = max(sims, key=sims.get)
            preds.append(best)

        print(classification_report(
            labels_np,
            preds,
            target_names=[label_names[i] for i in sorted(label_names.keys())]
        ))

        print("[VS] Saving artifacts...")
        joblib.dump(vectorizer, os.path.join(self.output_dir, "vs_vectorizer.pkl"))
        joblib.dump(class_vectors, os.path.join(self.output_dir, "vs_class_vectors.pkl"))

        print("\n[VS] Training complete (FAST MODE).\n")
