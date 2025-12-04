from wordfreq import top_n_list
from collections import Counter


class AutoCorrect:
    """
    Autocorrect using wordfreq corpus (Google Books + Twitter + Wikipedia).
    Windows-compatible.
    """

    def __init__(self):
        print("[AC] Loading wordfreq English vocabulary...")

        # Load ~50,000 most common English words
        self.vocab = top_n_list("en", 50000)

        # Lowercase & set
        self.vocab = [w.lower() for w in self.vocab]
        self.vocab_set = set(self.vocab)

        # Fake frequency (wordfreq doesn't expose frequency directly here)
        self.freq = Counter(self.vocab)

        print(f"[AC] Loaded {len(self.vocab)} words.")

    # -----------------------------------------------------

    def edit_distance(self, w1, w2):
        """Simple Levenshtein-like distance."""
        m, n = len(w1), len(w2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                delete = dp[i-1][j] + 1
                insert = dp[i][j-1] + 1
                replace = dp[i-1][j-1] + (0 if w1[i-1] == w2[j-1] else 2)
                dp[i][j] = min(delete, insert, replace)

        return dp[m][n]

    # -----------------------------------------------------

    def suggest(self, word, max_dist=2, top_k=5):
        """
        Suggest corrections based on edit distance.
        """
        word = word.lower()
        L = len(word)

        candidates = []

        # speed trick: filter by word length
        for w in self.vocab:
            if abs(len(w) - L) > max_dist:
                continue

            d = self.edit_distance(word, w)
            if d <= max_dist:
                candidates.append((w, d))

        candidates.sort(key=lambda x: x[1])
        return candidates[:top_k]
