import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA


class DataExploration:

    # ==============================
    # Load dataset
    # ==============================
    @staticmethod
    def load_dataset():
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
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
        plt.title("Target Distribution (0 = malignant, 1 = benign)")
        plt.xlabel("target")
        plt.ylabel("count")
        plt.show()

    # ==============================
    # Boxplots
    # ==============================
    @staticmethod
    def plot_boxplots_by_target(df):
        features_to_plot = [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area"
        ]

        for f in features_to_plot:
            plt.figure(figsize=(6, 4))
            df.boxplot(column=f, by="target")
            plt.title(f"{f} by class (0=malignant, 1=benign)")
            plt.suptitle("")
            plt.xlabel("target")
            plt.ylabel(f)
            plt.show()

    # ==============================
    # Histograms
    # ==============================
    @staticmethod
    def plot_histograms(df):
        features_to_plot = ["mean radius", "mean texture"]

        for f in features_to_plot:
            plt.figure(figsize=(6, 4))
            df[df["target"] == 0][f].hist(alpha=0.6, label="malignant")
            df[df["target"] == 1][f].hist(alpha=0.6, label="benign")
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
            df["mean radius"],
            df["mean texture"],
            c=df["target"],
            cmap="coolwarm",
            alpha=0.7
        )
        plt.xlabel("mean radius")
        plt.ylabel("mean texture")
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
        df = DataExploration.load_dataset()

        # Basic overview
        DataExploration.show_basic_info(df)

        # Class balance
        DataExploration.plot_target_distribution(df)

        # Stats vis
        DataExploration.plot_boxplots_by_target(df)
        DataExploration.plot_histograms(df)

        # Scatter
        DataExploration.plot_scatter(df)

        # PCA
        DataExploration.plot_pca(df)
