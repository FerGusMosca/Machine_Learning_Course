# ===== nn_data_exploration.py =====
# All comments MUST be in English.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA


class NNDataExploration:

    @staticmethod
    def load_dataset():
        X, y = make_circles(
            n_samples=2000,
            factor=0.2,
            noise=0.25,
            random_state=42
        )
        df = pd.DataFrame(X, columns=["x1", "x2"])
        df["target"] = y
        return df

    @staticmethod
    def show_basic_info(df):
        print("\n=== NN Dataset Overview ===")
        print(df.head())
        print("\nShape:", df.shape)
        print("\nColumns:", df.columns.tolist())
        print("\nTarget distribution:\n", df["target"].value_counts())

    @staticmethod
    def plot_target_distribution(df):
        df["target"].value_counts().plot(kind="bar", color=["skyblue", "salmon"])
        plt.title("Target Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.grid(True, axis="y")
        plt.show()

    @staticmethod
    def plot_scatter(df):
        plt.figure(figsize=(6, 5))
        plt.scatter(df["x1"], df["x2"], c=df["target"], cmap="coolwarm", alpha=0.7)
        plt.title("Circles Dataset — Scatter Plot")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    @staticmethod
    def plot_histograms(df):
        for c in ["x1", "x2"]:
            plt.figure(figsize=(6, 4))
            df[df.target == 0][c].hist(alpha=0.6, label="Class 0")
            df[df.target == 1][c].hist(alpha=0.6, label="Class 1")
            plt.legend()
            plt.title(f"Histogram — {c}")
            plt.show()

    @staticmethod
    def plot_pca(df):
        X = df[["x1", "x2"]]
        y = df["target"]
        pca = PCA(n_components=2)
        comps = pca.fit_transform(X)

        plt.scatter(comps[:, 0], comps[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        plt.title("PCA Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()

    @staticmethod
    def run():
        df = NNDataExploration.load_dataset()
        NNDataExploration.show_basic_info(df)
        NNDataExploration.plot_target_distribution(df)
        NNDataExploration.plot_histograms(df)
        NNDataExploration.plot_scatter(df)
        NNDataExploration.plot_pca(df)
