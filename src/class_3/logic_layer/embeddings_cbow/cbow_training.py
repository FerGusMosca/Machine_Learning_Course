# English-only comments.
import numpy as np
import re

from src.class_3.logic_layer.embeddings_cbow.cbow import CBOWModel


class CBOWTraining:
    """
    Ultra-light CBOW training (fast demo version).
    Uses a tiny internal corpus so training finishes in seconds.
    """
    def __init__(self, output_path="./trained_cbow_light.npz"):
        self.output_path = output_path

    def run(self):
        print("[CBOW] Training LIGHT version (tiny corpus)...")

        # ---------------------------------------------------------
        # 1) Mini-corpus (internal, no downloads)
        # ---------------------------------------------------------
        texts = [
            "I am happy because I am learning",
            "machine learning is fun and powerful",
            "word embeddings capture semantic meaning",
            "the teacher drinks water every morning",
            "I love reading books in the afternoon",
            "the cat sits on the warm window",
            "dogs are loyal and friendly animals",
            "he is writing a new interesting book",
            "they are playing football outside",
            "the sun is shining brightly today",
            "I feel tired but also motivated",
            "she enjoys cooking pasta for dinner",
            "the children are learning to read",
            "I am reading about neural networks",
            "this movie is amazing and inspiring",
            "the weather today is cold and windy",
            "students are studying for the exam",
            "I drink coffee every single morning",
            "the city is full of beautiful lights",
            "she works hard and never gives up",
            "the river flows quietly through town",
            "I am working on my machine learning project",
            "birds are singing outside my window",
            "the dog is happy because he is eating",
            "this course is fun and I enjoy it",
            "the forest is calm during the evening",
            "he drinks tea while reading the news",
            "the computer runs faster after the update",
            "I am testing word embeddings right now",
            "we are planning a trip next summer",
            "the team is training for the tournament",
            "she is learning Spanish and French",
            "the flowers bloom beautifully in spring",
            "I always learn something new each day",
            "they enjoy walking by the beach",
            "the child is drawing a colorful picture",
            "I practice coding every morning",
            "he is listening to relaxing music",
            "the library is quiet and peaceful",
            "machine learning models learn patterns",
            "the teacher explains everything clearly",
            "I like drinking tea before bed",
            "the mountain looks beautiful at sunrise",
            "we are cooking dinner together",
            "the dog chases the ball happily",
            "I enjoy learning new languages",
            "the book contains useful information",
            "she drinks coffee while studying",
            "the classroom is full of energy",
            "we are improving our machine learning skills"
        ]

        print(f"[CBOW] Corpus size: {len(texts)} sentences")

        # ---------------------------------------------------------
        # 2) Preprocess → list of tokens per text
        # ---------------------------------------------------------
        processed = []
        for t in texts:
            toks = re.findall(r"[a-zA-Z']+", t.lower())
            processed.append(" ".join(toks))

        # ---------------------------------------------------------
        # 3) Create CBOW model
        # ---------------------------------------------------------
        model = CBOWModel(
            window_size=2,   # tiny window
            embed_dim=30,    # small dimension
            lr=0.05          # slightly higher learning rate
        )

        model.build_vocab(processed)
        print(f"[CBOW] Vocabulary size: {len(model.vocab)}")

        # ---------------------------------------------------------
        # 4) Tiny training loop (VERY FAST)
        # ---------------------------------------------------------
        model.train(epochs=30)   # fast, corpus tiny

        # ---------------------------------------------------------
        # 5) Save output
        # ---------------------------------------------------------
        np.savez(self.output_path,
                 W1=model.W1,
                 W2=model.W2,
                 vocab=model.vocab)

        print(f"[CBOW] Saved model → {self.output_path}")
        print("[CBOW] Training COMPLETE (fast version).")
