from datasets import load_dataset
from collections import Counter, defaultdict
import re


class TrigramAutocomplete:
    """
    Trigram-based autocomplete.
    Uses: P(w3 | w1, w2)
    If only one word is entered → falls back to bigram.
    """

    def __init__(self):
        print("[ACOMP-TRI] Loading AG_NEWS corpus...")
        ds = load_dataset("ag_news")

        print("[ACOMP-TRI] Building corpus...")
        texts = [t["text"].lower() for t in ds["train"]]

        tokens = []
        for t in texts:
            toks = re.findall(r"[a-zA-Z']+", t.lower())
            tokens.extend(toks)

        print("[ACOMP-TRI] Training trigrams...")
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = defaultdict(Counter)

        # Build both bigrams & trigrams
        for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
            self.bigram_counts[w2][w3] += 1
            self.trigram_counts[(w1, w2)][w3] += 1

        print(f"[ACOMP-TRI] Done. Trigrams learned: {sum(len(v) for v in self.trigram_counts.values())}")

    # --------------------------------------------------------

    def autocomplete(self, text, top_k=5):
        """
        Autocomplete using last TWO words.
        Example:
            'the teacher drinks' → use ('teacher', 'drinks')
        """
        if not text.strip():
            return []

        tokens = text.strip().lower().split()

        # Not enough tokens → fallback to bigram
        if len(tokens) == 1:
            last = tokens[-1]
            if last not in self.bigram_counts:
                return []
            next_words = self.bigram_counts[last]
            total = sum(next_words.values())
            suggestions = [(w, c / total) for w, c in next_words.items()]
            suggestions.sort(key=lambda x: x[1], reverse=True)
            return suggestions[:top_k]

        # Trigram case
        w1, w2 = tokens[-2], tokens[-1]
        key = (w1, w2)

        if key not in self.trigram_counts:
            return []  # no context found

        next_words = self.trigram_counts[key]
        total = sum(next_words.values())

        suggestions = [
            (w, c / total) for w, c in next_words.items()
        ]

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]
