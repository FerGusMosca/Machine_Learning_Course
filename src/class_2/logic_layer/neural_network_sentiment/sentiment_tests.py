# sentiment_tests.py
# Evaluación completa y ejemplos perfectos (funciona sí o sí)

import os
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

class SentimentTests:

    def __init__(self, models_dir="./trained_models_sentiment/"):
        self.models_dir = models_dir

    def run(self):
        print("\nEVALUACIÓN FINAL DEL MODELO DE SENTIMIENTO\n")

        # Cargar metadata y modelo
        with open(os.path.join(self.models_dir, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)

        vocab_size = metadata["vocab_size"]
        max_len = metadata["max_len"]

        model = load_model(os.path.join(self.models_dir, "sentiment_model.h5"))

        # Test set
        (_, _), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
        X_test = pad_sequences(X_test, maxlen=max_len)

        # Predicciones
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob > 0.5).astype(int)

        print("Classification Report")
        print(classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"]))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negativo", "Positivo"],
                    yticklabels=["Negativo", "Positivo"])
        plt.ylabel("Real")
        plt.xlabel("Predicción")
        plt.title("Matriz de Confusión")
        plt.show()

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],"k--")
        plt.legend()
        plt.title("Curva ROC")
        plt.show()

        # EJEMPLOS EN VIVO (AHORA SÍ DAN BIEN)
        print("\nEJEMPLOS EN VIVO".ljust(60))
        print("-" * 60)

        word_index = tf.keras.datasets.imdb.get_word_index()
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2

        def encode(text):
            seq = [word_index.get(w, 2) for w in text.lower().split()]
            return pad_sequences([seq], maxlen=max_len)

        ejemplos = [
            "this movie is amazing and inspiring",
            "terrible movie, worst acting ever",
            "not bad but very slow",
            "absolutely fantastic",
            "i hated every second of it",
            "best film of the year",
            "I am hungry",
            "It could have been better for sure"


        ]

        for texto in ejemplos:
            prob = float(model.predict(encode(texto), verbose=0)[0][0])
            label = "POSITIVO" if prob > 0.5 else "NEGATIVO"
            print(f"{texto:45} → {prob:.3f} → {label}")

        print("\nT")


if __name__ == "__main__":
    SentimentTests().run()