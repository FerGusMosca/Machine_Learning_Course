import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


class AlgorithmTraining:

    def __init__(self, csv_path, target_col, output_dir="./trained_models/"):
        self.csv_path = csv_path
        self.target_col = target_col
        self.output_dir = output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # ------------------------------------------------------------
    # Utility — Save a model + metadata
    # ------------------------------------------------------------
    def _save_model(self, model, name, metadata=None):
        file_path = os.path.join(self.output_dir, f"{name}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump({"model": model, "metadata": metadata}, f)
        print(f"💾 Saved → {file_path}")

    # ------------------------------------------------------------
    # Utility — Dataset stats
    # ------------------------------------------------------------
    def _print_dataset_stats(self, df):
        print("\n====================================================")
        print("📊 DATASET STATISTICS")
        print("====================================================")
        print("\n▶ Shape:", df.shape)
        print("\n▶ Head:\n", df.head())
        print("\n▶ Null values:\n", df.isna().sum())
        print("\n▶ Class distribution:", df[self.target_col].value_counts(normalize=True))
        print("\n▶ Correlation (numeric):\n", df.corr())

    # ------------------------------------------------------------
    # Utility — GridSearch wrapper
    # ------------------------------------------------------------
    def _train_with_gridsearch(self, model, params, X_train, y_train):
        gs = GridSearchCV(model, params, cv=5, n_jobs=-1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_, gs.best_params_, gs.best_score_

    # ------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------
    def run(self):
        # 1) Load dataset
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        self._print_dataset_stats(df)

        # 2) Split into features/labels
        features = [c for c in df.columns if c != self.target_col]
        X = df[features]
        y = df[self.target_col]

        # 3) Standardize numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X.loc[:, numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # 4) Train/test split — ALWAYS reproducible
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )

        print("\n====================================================")
        print("📦 TRAIN/TEST SPLIT")
        print("====================================================")
        print("Train Shape:", X_train.shape)
        print("Test  Shape:", X_test.shape)

        # 5) Models + hyperparameters
        models = {
            "logistic_regression": {
                "model": LogisticRegression(max_iter=2000),
                "params": {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs"]},
            },
            "svm": {
                "model": SVC(probability=True),
                "params": {
                    "kernel": ["rbf", "linear"],
                    "C": [0.1, 1, 10],
                    "gamma": ["scale", "auto"],
                },
            },
            "decision_tree": {
                "model": DecisionTreeClassifier(),
                "params": {
                    "max_depth": [3, 5, 7, 10],
                    "criterion": ["gini", "entropy"],
                    "min_samples_split": [2, 5, 10],
                },
            },
            "knn": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                },
            },
        }

        # 6) Train each model
        print("\n====================================================")
        print("🤖 TRAINING MODELS")
        print("====================================================")

        for name, cfg in models.items():
            print(f"\n🔹 Training {name}...")

            best_model, best_params, train_score = self._train_with_gridsearch(
                cfg["model"], cfg["params"], X_train, y_train
            )

            print("   ✔ Best Params:", best_params)
            print("   ✔ CV Train Accuracy:", train_score)

            y_hat = best_model.predict(X_test)
            print("\n📘 Classification Report:")
            print(classification_report(y_test, y_hat))

            print("📘 Confusion Matrix:")
            print(confusion_matrix(y_test, y_hat))

            # 7) Save model INCLUDING scaler info
            metadata = {
                "best_params": best_params,
                "train_score_cv": train_score,
                "features": X.columns.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "numeric_cols": numeric_cols.tolist()
            }

            self._save_model(best_model, name, metadata)

        print("\n====================================================")
        print("🏁 TRAINING FINISHED — MODELS + SCALER SAVED")
        print("====================================================")
