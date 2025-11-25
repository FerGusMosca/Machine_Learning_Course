# ===== neural_network_tests.py =====
# All comments MUST be in English.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_circles
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split

from keras.src.saving import load_model


class NeuralNetworkTests:

    def __init__(self, models_dir="./trained_models_nn/"):
        self.models_dir = models_dir

    # ---------------------------------------------------------
    # Load EXACT SAME DATASET as training
    # ---------------------------------------------------------
    @staticmethod
    def load_test_data():
        X, y = make_circles(
            n_samples=2000,
            factor=0.2,
            noise=0.25,
            random_state=42
        )
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df[["x1", "x2"]],
            df["target"],
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        return X_test, y_test

    # ---------------------------------------------------------
    # Apply SAME SCALING used in training
    # ---------------------------------------------------------
    @staticmethod
    def scale_test_data(X_test, metadata):
        mean = np.array(metadata["scaler_mean"])
        scale = np.array(metadata["scaler_scale"])
        return (X_test - mean) / scale

    # ---------------------------------------------------------
    def plot_confusion(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title("NN Confusion Matrix")
        plt.show()

    def plot_roc(self, y_test, y_prob):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC={auc_score:.2f}")
        plt.plot([0,1],[0,1],'k--')
        plt.legend()
        plt.title("ROC Curve")
        plt.show()

    def plot_pr(self, y_test, y_prob):
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(6,5))
        plt.plot(recall, precision)
        plt.title("Precision–Recall Curve")
        plt.show()

    @staticmethod
    def plot_prediction_scatter(X_test, y_pred):
        plt.figure(figsize=(7, 6))
        plt.scatter(
            X_test["x1"], X_test["x2"],
            c=y_pred,
            cmap="coolwarm",
            alpha=0.6
        )
        plt.title("NN Predictions Scatter Plot (0 / 1)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    # ---------------------------------------------------------
    # MAIN
    # ---------------------------------------------------------
    def run(self):
        print("\n🔍 Running Neural Network Tests...\n")

        # Load correct test data
        X_test, y_test = self.load_test_data()

        # Load saved metadata
        with open(os.path.join(self.models_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        # Load model
        model = load_model(os.path.join(self.models_dir, "neural_network_model.keras"))

        # Scale
        X_test_scaled = self.scale_test_data(X_test, metadata)

        # Predictions
        y_prob = model.predict(X_test_scaled).ravel()
        y_pred = (y_prob > 0.5).astype(int)



        print("\n📘 Classification Report:")
        print(classification_report(y_test, y_pred))

        print("\n📘 First 20 predictions:")
        print(pd.DataFrame({
            "y_test": y_test.values[:20],
            "y_pred": y_pred[:20],
            "prob": y_prob[:20],
        }))

        # PLOTS
        self.plot_confusion(y_test, y_pred)
        self.plot_roc(y_test, y_prob)
        self.plot_pr(y_test, y_prob)

        # Scatter plot of predictions
        self.plot_prediction_scatter(X_test, y_pred)

        print("\n✅ Done — Neural Network evaluation.\n")
