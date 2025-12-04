from datasets import load_dataset
from collections import Counter, defaultdict
import re


class NGramAutocomplete:
    """
    Bigram-based autocomplete using AG_NEWS corpus.
    Uses: P(next_word | last_word)
    """

    def __init__(self):
        print("[ACOMP] Loading AG_NEWS corpus...")
        ds = load_dataset("ag_news")

        print("[ACOMP] Building corpus...")
        texts = [t["text"].lower() for t in ds["train"]]

        # Tokenize
        corpus_tokens = []
        for t in texts:
            toks = re.findall(r"[a-zA-Z']+", t.lower())
            corpus_tokens.extend(toks)

        print("[ACOMP] Training bigrams...")
        self.bigram_counts = defaultdict(Counter)

        for w1, w2 in zip(corpus_tokens, corpus_tokens[1:]):
            self.bigram_counts[w1][w2] += 1

        print(f"[ACOMP] Done. Bigrams learned: {sum(len(v) for v in self.bigram_counts.values())}")

    # --------------------------------------------------------

    def autocomplete(self, text, top_k=5):
        """
        Autocomplete using only the LAST word of the full phrase.
        Example: text='the teacher drinks' -> use 'drinks'
        """
        if not text.strip():
            return []

        tokens = text.strip().lower().split()
        last = tokens[-1]

        if last not in self.bigram_counts:
            return []

        next_words = self.bigram_counts[last]
        total = sum(next_words.values())

        suggestions = [
            (w, c / total) for w, c in next_words.items()
        ]

        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]
