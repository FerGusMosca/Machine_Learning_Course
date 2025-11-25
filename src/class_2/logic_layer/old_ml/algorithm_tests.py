import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, make_circles

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc
)

from joblib import load


class AlgorithmsTests:

    def __init__(self, models_dir="./trained_models/"):
        self.models_dir = models_dir

    # ---------------------------------------------------------
    # Load test data (same split as training)
    # ---------------------------------------------------------
    def load_test_data(self):
        X_raw, y_raw = make_circles(
            n_samples=2000,
            noise=0.30,
            factor=0.20,
            random_state=42
        )

        df = pd.DataFrame(X_raw, columns=["x1", "x2"])
        df["target"] = y_raw


        X = df.drop(columns=["target"])
        y = df["target"]

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )
        return X_test, y_test

    # ---------------------------------------------------------
    # Load trained models + metadata
    # ---------------------------------------------------------
    def load_models(self):
        models = {}
        for name in ["logistic_regression", "svm_linear", "decision_tree_weak", "knn_weak"]:
            obj = load(f"{self.models_dir}/{name}.pkl")
            models[name] = {
                "model": obj["model"],
                "metadata": obj["metadata"]
            }
        return models

    # ---------------------------------------------------------
    # Apply scaler from metadata
    # ---------------------------------------------------------
    def scale_test_data(self, X_test, metadata):
        numeric_cols = metadata["numeric_cols"]
        scaler_mean = metadata["scaler_mean"]
        scaler_scale = metadata["scaler_scale"]

        X_scaled = X_test.copy()
        X_scaled[numeric_cols] = (X_scaled[numeric_cols] - scaler_mean) / scaler_scale
        return X_scaled

    # ---------------------------------------------------------
    # F1-score barplot
    # ---------------------------------------------------------
    def plot_f1_scores(self, y_test, preds_dict):
        scores = {name: f1_score(y_test, preds) for name, preds in preds_dict.items()}

        plt.figure(figsize=(7, 5))
        sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette="viridis")
        plt.title("F1-Score Comparison")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")
        plt.show()

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------
    def plot_confusion_matrix(self, y_test, preds_dict):
        for name, preds in preds_dict.items():
            cm = confusion_matrix(y_test, preds)

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix – {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()

    # ---------------------------------------------------------
    # ROC Curve
    # ---------------------------------------------------------
    def plot_roc(self, y_test, models, X_test):
        plt.figure(figsize=(8, 6))

        for name, m in models.items():
            model = m["model"]
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ---------------------------------------------------------
    # Precision–Recall Curve
    # ---------------------------------------------------------
    def plot_precision_recall(self, y_test, models, X_test):
        plt.figure(figsize=(8, 6))

        for name, m in models.items():
            model = m["model"]
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except:
                continue

            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            plt.plot(recall, precision, label=name)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ---------------------------------------------------------
    # MAIN
    # ---------------------------------------------------------
    def run(self):
        print("\n🔍 Running Algorithm Tests...\n")

        # Load test data
        X_test, y_test = self.load_test_data()

        # Load models + metadata
        models = self.load_models()

        # Use metadata from any model (scaler is the same)
        any_meta = next(iter(models.values()))["metadata"]

        # Scale test data
        X_test_scaled = self.scale_test_data(X_test, any_meta)

        # Predictions
        preds_dict = {
            name: m["model"].predict(X_test_scaled)
            for name, m in models.items()
        }

        print("\n==============================")
        print("🔎 DEBUG — TEST SET SUMMARY")
        print("==============================")
        print("X_test shape:", X_test.shape)
        print("y_test distribution:", y_test.value_counts().to_dict())

        print("\n==============================")
        print("🔎 DEBUG — MODEL PREDICTIONS OVERVIEW")
        print("==============================")

        for name, preds in preds_dict.items():
            print(f"\n▶ Model: {name}")
            print("  Unique preds:", set(preds))
            print("  Count preds:", pd.Series(preds).value_counts().to_dict())
            print("  F1:", f1_score(y_test, preds))

        print("\n==============================")
        print("🔎 DEBUG — SAMPLE COMPARISON")
        print("==============================")

        debug_df = pd.DataFrame({"y_test": y_test.values[:20]})
        for name, preds in preds_dict.items():
            debug_df[name] = preds[:20]

        print(debug_df)
        print("\n==============================\n")

        # PLOTS
        self.plot_f1_scores(y_test, preds_dict)
        self.plot_confusion_matrix(y_test, preds_dict)
        self.plot_roc(y_test, models, X_test_scaled)
        self.plot_precision_recall(y_test, models, X_test_scaled)

        print("\n✅ All test plots completed.\n")
