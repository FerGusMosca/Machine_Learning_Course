# English-only comments.
import numpy as np
import re

class CBOWTesting:
    """
    Testing class for CBOW embeddings.
    Given context words, predicts the most likely center word.
    """

    def __init__(self, model_path="./trained_cbow_light.npz"):
        print("[CBOW] Loading model...")
        data = np.load(model_path, allow_pickle=True)

        self.W1 = data["W1"]
        self.W2 = data["W2"]
        self.vocab = list(data["vocab"])
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        print("[CBOW] Model loaded.")

    # --------------------------------------------------------------

    def tokenize(self, text):
        return re.findall(r"[a-zA-Z']+", text.lower())

    # --------------------------------------------------------------

    def predict_center_word(self, context_words):
        """
        context_words: list[str]  e.g. ["happy", "am", "learning"]
        Returns top predicted center word.
        """

        idxs = []
        for w in context_words:
            if w not in self.word_to_idx:
                print(f"[CBOW] WARNING: '{w}' not in vocabulary.")
                return None
            idxs.append(self.word_to_idx[w])

        # ----------------------------------------------------------
        # CBOW forward pass:
        #   1) sum embeddings of context words: hidden = sum(W1[ctx])
        #   2) output scores = hidden @ W2
        #   3) softmax → predicted word
        # ----------------------------------------------------------
        hidden = np.sum(self.W1[idxs], axis=0)  # shape (embed_dim,)
        scores = hidden @ self.W2               # shape (vocab_size,)

        probs = np.exp(scores) / np.sum(np.exp(scores))
        best_idx = np.argmax(probs)

        return self.vocab[best_idx]

    # --------------------------------------------------------------

    def run(self):
        text = input("Enter context words (e.g. 'happy am learning'): ")
        ctx = self.tokenize(text)

        if len(ctx) == 0:
            print("[CBOW] No valid words.")
            return

        pred = self.predict_center_word(ctx)

        print("\n[CBOW] Predicted CENTER WORD:", pred)
