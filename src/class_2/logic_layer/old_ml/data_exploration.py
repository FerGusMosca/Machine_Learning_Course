import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA


class DataExploration:

    # ==============================
    # Load NON-LINEAR dataset
    # ==============================
    @staticmethod
    def load_dataset():
        X, y = make_moons(
            n_samples=2000,
            noise=0.25,
            random_state=42
        )

        df = pd.DataFrame(X, columns=["feature1", "feature2"])
        df["target"] = y
        return df

    # ==============================
    # Basic info
    # ==============================
    @staticmethod
    def show_basic_info(df):
        print("\n=== Dataset Overview ===")
        print(df.head())
        print("\nShape:", df.shape)
        print("\nColumns:", list(df.columns))

    # ==============================
    # Target Distribution
    # ==============================
    @staticmethod
    def plot_target_distribution(df):
        df["target"].value_counts().plot(kind="bar")
        plt.title("Target Distribution (0 / 1)")
        plt.xlabel("target")
        plt.ylabel("count")
        plt.show()

    # ==============================
    # Boxplots
    # ==============================
    @staticmethod
    def plot_boxplots_by_target(df):
        for f in ["feature1", "feature2"]:
            plt.figure(figsize=(6, 4))
            df.boxplot(column=f, by="target")
            plt.title(f"{f} by class")
            plt.suptitle("")
            plt.xlabel("target")
            plt.ylabel(f)
            plt.show()

    # ==============================
    # Histograms
    # ==============================
    @staticmethod
    def plot_histograms(df):
        for f in ["feature1", "feature2"]:
            plt.figure(figsize=(6, 4))
            df[df["target"] == 0][f].hist(alpha=0.6, label="class 0")
            df[df["target"] == 1][f].hist(alpha=0.6, label="class 1")
            plt.title(f"Histogram: {f}")
            plt.legend()
            plt.xlabel(f)
            plt.ylabel("count")
            plt.show()

    # ==============================
    # Scatter Plot 2D
    # ==============================
    @staticmethod
    def plot_scatter(df):
        plt.figure(figsize=(6, 5))
        plt.scatter(
            df["feature1"],
            df["feature2"],
            c=df["target"],
            cmap="coolwarm",
            alpha=0.7
        )
        plt.xlabel("feature1")
        plt.ylabel("feature2")
        plt.title("Scatter Plot (colored by target)")
        plt.show()

    # ==============================
    # PCA 2D visualization
    # ==============================
    @staticmethod
    def plot_pca(df):
        X = df.drop(columns=["target"])
        y = df["target"]

        pca = PCA(n_components=2)
        comps = pca.fit_transform(X)

        plt.figure(figsize=(6, 5))
        plt.scatter(comps[:, 0], comps[:, 1], c=y, cmap="coolwarm", alpha=0.7)
        plt.title("PCA 2D Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    # ==============================
    # Run full exploration
    # ==============================
    @staticmethod
    def run():
        # 🔥 Dataset imposible para ML clásico
        X, y = make_circles(
            n_samples=2000,
            noise=0.30,  # Ruido grande → rompe TODO ML clásico
            factor=0.20,  # Círculo interno súper chico
            random_state=42
        )

        df = pd.DataFrame(X, columns=["feature1", "feature2"])
        df["target"] = y

        DataExploration.show_basic_info(df)
        DataExploration.plot_target_distribution(df)
        DataExploration.plot_boxplots_by_target(df)
        DataExploration.plot_histograms(df)
        DataExploration.plot_scatter(df)
        DataExploration.plot_pca(df)
