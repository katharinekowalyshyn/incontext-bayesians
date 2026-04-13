"""Graph definitions for the competing-structures experiment.

Grid (4x4, 16 nodes) lives in utils.py.
This file adds Ring (12-node cycle of months).
"""

import numpy as np

MONTHS = [
    "january", "february", "march", "april",
    "may", "june", "july", "august",
    "september", "october", "november", "december",
]

MONTH_COLORS = [
    "#1f77b4", "#aec7e8", "#ffbb78", "#ff7f0e",
    "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
]

MONTH_TO_COLOR = {m: c for m, c in zip(MONTHS, MONTH_COLORS)}


class Ring:
    """12-node ring (cycle) graph.  Each node connects to its two cyclic neighbors."""

    def __init__(self, words=MONTHS):
        self.words = words
        self.n = len(words)
        self.word_to_idx = {w: i for i, w in enumerate(words)}

    def get_valid_next_words(self, word):
        idx = self.word_to_idx[word]
        return [self.words[(idx - 1) % self.n], self.words[(idx + 1) % self.n]]

    def generate_sequence(self, seq_len, start_word=None):
        if start_word is None:
            start_word = np.random.choice(self.words)
        word = start_word
        sequence = [word]
        while len(sequence) < seq_len:
            word = np.random.choice(self.get_valid_next_words(word))
            sequence.append(word)
        return sequence

    def generate_batch(self, seq_len):
        """One sequence starting from each node."""
        return [self.generate_sequence(seq_len, start_word=w) for w in self.words]

    def build_adjacency_matrix(self):
        A = np.zeros((self.n, self.n))
        for i in range(self.n):
            A[i, (i - 1) % self.n] = 1
            A[i, (i + 1) % self.n] = 1
        return A
