# FILE: src/class_3/logic_layer/embeddings/cbow_model.py
# English-only comments.

import numpy as np
import re
from collections import Counter


class CBOWModel:
    """
    Minimal CBOW model (NumPy) to learn word embeddings
    and predict the missing center word from context.
    """

    def __init__(self, window_size=2, embed_dim=30, lr=0.025):
        self.window = window_size
        self.embed_dim = embed_dim
        self.lr = lr
        self.vocab = None
        self.w2i = None
        self.i2w = None
        self.W1 = None
        self.W2 = None
        self.training_data = None

    # ------------------------------------------------------
    def _tokenize(self, text):
        return re.findall(r"[a-zA-Z']+", text.lower())

    # ------------------------------------------------------
    def build_vocab(self, texts):
        tokens = []
        for t in texts:
            tokens.extend(self._tokenize(t))

        vocab = sorted(list(set(tokens)))
        self.vocab = vocab
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = {i: w for w, i in self.w2i.items()}

        V = len(vocab)
        self.W1 = np.random.randn(V, self.embed_dim) * 0.01
        self.W2 = np.random.randn(self.embed_dim, V) * 0.01

        # Build training samples
        training = []
        for i in range(len(tokens)):
            context = []
            for j in range(i - self.window, i + self.window + 1):
                if j != i and 0 <= j < len(tokens):
                    context.append(tokens[j])
            if context:
                training.append((context, tokens[i]))

        self.training_data = training
        print(f"[CBOW] Vocabulary size = {V}")
        print(f"[CBOW] Training samples = {len(training)}")

    # ------------------------------------------------------
    def _one_hot(self, idx):
        v = np.zeros(len(self.vocab))
        v[idx] = 1
        return v

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    # ------------------------------------------------------
    def train(self, epochs=3):
        if self.training_data is None:
            raise Exception("Call build_vocab(texts) first.")

        for ep in range(epochs):
            loss = 0
            for ctx_words, target in self.training_data:

                ctx_idxs = [self.w2i[w] for w in ctx_words]
                ctx_vec = np.mean(self.W1[ctx_idxs], axis=0)  # (embed_dim,)

                logits = ctx_vec @ self.W2
                probs = self._softmax(logits)

                target_idx = self.w2i[target]
                y = self._one_hot(target_idx)

                loss += -np.log(probs[target_idx] + 1e-12)

                # Backprop
                dlogits = probs - y
                dW2 = np.outer(ctx_vec, dlogits)
                dctx = dlogits @ self.W2.T
                dctx = dctx / len(ctx_idxs)

                # Update
                self.W2 -= self.lr * dW2
                for idx in ctx_idxs:
                    self.W1[idx] -= self.lr * dctx

            print(f"[CBOW] Epoch {ep+1}/{epochs} — loss={loss:.4f}")

    # ------------------------------------------------------
    def predict(self, context_words, top_k=5):
        """Predict missing word from context."""
        context = [w for w in context_words if w in self.w2i]
        if not context:
            return []

        idxs = [self.w2i[w] for w in context]
        ctx_vec = np.mean(self.W1[idxs], axis=0)

        probs = self._softmax(ctx_vec @ self.W2)
        top = probs.argsort()[::-1][:top_k]

        return [(self.i2w[i], float(probs[i])) for i in top]
