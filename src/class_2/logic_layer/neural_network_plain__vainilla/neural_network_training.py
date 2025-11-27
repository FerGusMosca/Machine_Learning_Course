# ===== neural_network_training.py =====
# Fully functional script: trains a neural network on make_circles
# and automatically saves the model + architecture plot.
# All comments in English.

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from tensorflow.keras.utils import plot_model  # To draw the model


class NeuralNetworkTraining:

    def __init__(self, output_dir="./trained_models_nn/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _build_model(self, input_dim):
        """
        Builds the neural network with the exact architecture you showed:
        Input → 64 → BN → Dropout → 32 → BN → Dropout → 16 → 1 (sigmoid)
        """
        he = tf.keras.initializers.HeNormal()
        l2 = regularizers.l2(1e-4)

        model = models.Sequential([
            layers.Input(shape=(input_dim,), name="input"),

            layers.Dense(64, activation="relu", kernel_initializer=he,
                         kernel_regularizer=l2, name="dense_64"),
            layers.BatchNormalization(name="bn_64"),
            layers.Dropout(0.30, name="dropout_64"),

            layers.Dense(32, activation="relu", kernel_initializer=he,
                         kernel_regularizer=l2, name="dense_32"),
            layers.BatchNormalization(name="bn_32"),
            layers.Dropout(0.25, name="dropout_32"),

            layers.Dense(16, activation="relu", kernel_initializer=he,
                         kernel_regularizer=l2, name="dense_16"),

            layers.Dense(1, activation="sigmoid", name="output")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("\n====== NEURAL NETWORK ARCHITECTURE ======")
        model.summary()

        # Draw and save network diagram
        plot_path = os.path.join(self.output_dir, "neural_network_architecture.png")
        plot_model(
            model,
            to_file=plot_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',      # Vertical layout
            expand_nested=False,
            dpi=200
        )
        print(f"Network diagram saved → {plot_path}")

        # Optional: show the image immediately (great for notebooks or scripts)
        try:
            img = plt.imread(plot_path)
            plt.figure(figsize=(10, 14))
            plt.imshow(img)
            plt.axis("off")
            plt.title("Neural Network Architecture", fontsize=16)
            plt.show()
        except Exception as e:
            print("Could not display image (probably running without GUI).")

        return model

    def run(self):
        print("\nTRAINING NEURAL NETWORK ON MAKE_CIRCLES DATASET\n")

        # 1. Dataset (always the same for reproducibility)
        X_raw, y_raw = make_circles(n_samples=2000, noise=0.20, factor=0.30, random_state=42)
        X = pd.DataFrame(X_raw, columns=["x1", "x2"])
        y = pd.Series(y_raw, name="target")

        # 2. Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # 3. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # 4. Build model (this will also draw it)
        model = self._build_model(input_dim=X_train_scaled.shape[1])

        # 5. Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=60,
            batch_size=32,
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
            ]
        )

        # 6. Evaluation
        loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"\nTest Accuracy: {acc:.4f} — Test Loss: {loss:.4f}")

        # 7. Save model + scaler metadata
        model_path = os.path.join(self.output_dir, "neural_network_model.keras")
        model.save(model_path)
        print(f"Model saved → {model_path}")

        metadata = {
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "random_state": 42
        }
        with open(os.path.join(self.output_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        print("\nDONE — Everything saved in folder:", self.output_dir)
        return history


# =================================================================
# Run the whole pipeline
# =================================================================
if __name__ == "__main__":
    trainer = NeuralNetworkTraining()
    trainer.run()