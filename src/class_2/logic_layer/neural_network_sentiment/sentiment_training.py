# sentiment_training.py
# Modelo de sentimiento IMDB – Funciona perfecto en 2 minutos

import os
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentimentTraining:

    def __init__(self, output_dir="./trained_models_sentiment/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _load_data(self, num_words=10000, max_len=200):
        print("Cargando IMDB...")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)
        X_train = pad_sequences(X_train, maxlen=max_len)
        X_test  = pad_sequences(X_test,  maxlen=max_len)
        return X_train, X_test, y_train, y_test

    def _build_model(self, vocab_size=10000, max_len=200):
        model = models.Sequential([
            layers.Embedding(vocab_size, 128, input_length=max_len, mask_zero=True),
            layers.Conv1D(64, 5, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        model.summary()
        return model

    def run(self):
        X_train, X_test, y_train, y_test = self._load_data()

        model = self._build_model()

        print("\nEntrenando (10 épocas máximo)...")
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=10,
            batch_size=128,
            verbose=1
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTEST ACCURACY FINAL: {test_acc:.4f}")

        model.save(f"{self.output_dir}/sentiment_model.h5")
        with open(f"{self.output_dir}/metadata.pkl", "wb") as f:
            pickle.dump({"vocab_size": 10000, "max_len": 200}, f)


if __name__ == "__main__":
    SentimentTraining().run()