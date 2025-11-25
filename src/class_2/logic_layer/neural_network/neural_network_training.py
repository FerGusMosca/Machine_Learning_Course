# ===== neural_network_training.py =====
# All comments MUST be in English.

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks


class NeuralNetworkTraining:

    def __init__(self, output_dir="./trained_models_nn/"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # ------------------------------------------------------------
    # Build Neural Network
    # ------------------------------------------------------------
    def _build_model(self, input_dim):
        he = tf.keras.initializers.HeNormal()
        l2 = regularizers.l2(1e-4)

        model = models.Sequential([
            layers.Input(shape=(input_dim,)),

            layers.Dense(64, activation="relu", kernel_initializer=he, kernel_regularizer=l2),
            layers.BatchNormalization(),
            layers.Dropout(0.30),

            layers.Dense(32, activation="relu", kernel_initializer=he, kernel_regularizer=l2),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(16, activation="relu", kernel_initializer=he, kernel_regularizer=l2),

            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("\n====== NN ARCHITECTURE ======")
        model.summary()
        return model

    # ------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------
    def run(self):
        print("\n🤖 TRAINING NEURAL NETWORK")

        # -------------------------------------------
        # 1) FIXED DATASET — ALWAYS THE SAME
        # -------------------------------------------
        X_raw, y_raw = make_circles(
            n_samples=2000,
            noise=0.20,
            factor=0.30,
            random_state=42
        )

        df = pd.DataFrame(X_raw, columns=["x1", "x2"])
        df["target"] = y_raw

        X = df[["x1", "x2"]]
        y = df["target"]

        # -------------------------------------------
        # 2) FIXED TRAIN/TEST SPLIT — IDENTICAL FOR TRAIN/TEST
        # -------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            shuffle=True
        )

        # -------------------------------------------
        # 3) SCALING
        # -------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -------------------------------------------
        # 4) Build NN
        # -------------------------------------------
        model = self._build_model(input_dim=X_train_scaled.shape[1])

        # -------------------------------------------
        # 5) Train NN
        # -------------------------------------------
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
            ]
        )

        # -------------------------------------------
        # 6) Evaluation
        # -------------------------------------------
        loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"\n🔥 Test Accuracy: {acc:.4f}")

        # -------------------------------------------
        # 7) Save: model + metadata
        # -------------------------------------------
        metadata = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "train_test_seed": 42,
        }

        model.save(f"{self.output_dir}/neural_network_model.keras")
        print(f"💾 Saved NN → {self.output_dir}/neural_network_model.keras")

        with open(f"{self.output_dir}/metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)
        print(f"💾 Saved metadata → {self.output_dir}/metadata.pkl")

        print("\n🏁 DONE — Neural Network Training\n")
        return history
