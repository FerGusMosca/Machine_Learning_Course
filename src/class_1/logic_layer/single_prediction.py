import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from joblib import load


class SinglePrediction:

    def __init__(self, models_dir="./trained_models/"):
        self.models_dir = models_dir

    def load_models(self):
        models = {}
        for name in ["logistic_regression", "svm", "decision_tree", "knn"]:
            obj = load(f"{self.models_dir}/{name}.pkl")
            models[name] = {
                "model": obj["model"],
                "metadata": obj["metadata"]
            }
        return models

    def scale_patient(self, patient_df, metadata):
        numeric_cols = metadata["numeric_cols"]
        scaler_mean = metadata["scaler_mean"]
        scaler_scale = metadata["scaler_scale"]

        df = patient_df.copy()
        df[numeric_cols] = (df[numeric_cols] - scaler_mean) / scaler_scale
        return df

    def run(self):
        print("\n🧬 SINGLE-PATIENT DIAGNOSIS\n")

        # Load dataset (same as training)
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        # Select a random patient
        idx = np.random.randint(0, len(df))
        patient = df.iloc[idx:idx+1]

        print("📌 Patient index:", idx)
        print("\n📋 Patient attributes:\n")
        print(patient.T)

        models = self.load_models()
        any_meta = next(iter(models.values()))["metadata"]

        # scale with training scaler
        patient_scaled = self.scale_patient(patient, any_meta)

        print("\n==============================")
        print("🔮 MODEL PREDICTIONS")
        print("==============================")

        for name, m in models.items():
            model = m["model"]

            pred = model.predict(patient_scaled)[0]

            label = "🟢 BENIGN (0)" if pred == 0 else "🔴 MALIGNANT (1)"

            # probability if available
            try:
                prob = model.predict_proba(patient_scaled)[0][1]
                print(f"\n{name}: {label} — Confidence: {prob:.3f}")
            except:
                print(f"\n{name}: {label} — (no probability available)")

        print("\n==============================")
        print("✔️ True diagnosis in dataset:", "BENIGN (0)" if y[idx] == 0 else "MALIGNANT (1)")
        print("==============================\n")
