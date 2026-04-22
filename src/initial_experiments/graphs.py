"""Graph definitions for the competing-structures experiment.

Grid (4x4, 16 nodes) lives in utils.py.
This file defines Ring variants:

  MONTHS          - original months vocabulary (strong pretrained sequential prior)
  RING_WORDS      - semantically neutral 12-word vocabulary, fully disjoint from grid
  RING_WORDS_OVERLAP - same ring topology but 3 words (rock, sand, box) are shared
                    with the grid vocabulary, so token identity alone cannot
                    disambiguate which graph a position belongs to
"""

import numpy as np

# ── Month ring ─────────────────────────────────────────────────────────────────

# Natural ordering: the in-context ring matches the model's semantic prior
# (Jan→Feb→...→Dec). Expect fast, clean learning because semantic prior helps.
MONTHS = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December",
]

# Permuted ordering: interleave months offset by 6 so that no two naturally-
# adjacent months are neighbors in the in-context ring.
# Ring: Jan→Aug→Mar→Oct→May→Dec→Jul→Feb→Sep→Apr→Nov→Jun→(Jan)
# Minimum natural distance between any in-context-adjacent pair = 5 months.
# This creates a direct conflict between the model's semantic prior and the
# structure defined by the random walk.
MONTHS_PERMUTED = [
    "January", "August", "March", "October",
    "May", "December", "July", "February",
    "September", "April", "November", "June",
]

MONTH_COLORS_NATURAL = [
    "#1f77b4", "#aec7e8", "#ffbb78", "#ff7f0e",
    "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
]

MONTH_COLORS = MONTH_COLORS_NATURAL  # backwards-compat alias
MONTH_TO_COLOR = {m: c for m, c in zip(MONTHS, MONTH_COLORS_NATURAL)}


# ── Neutral ring: fully disjoint from grid ─────────────────────────────────────
# No semantic ordering or associations. All words are concrete nouns with no
# obvious sequential relationship to each other or to the grid vocabulary.

# RING_WORDS = [
#     "candle", "brick", "fern", "lamp",
#     "dust",   "wool",  "reef", "vine",
#     "jar",    "chalk", "marsh", "prism",
# ]

RING_WORDS = [
    "apple", "bird", "car", "egg",
    "house", "milk", "plane", "opera",
    "box", "sand", "sun", "mango",
    "rock", "math", "code", "phone",
]

RING_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]

RING_WORD_TO_COLOR = {w: c for w, c in zip(RING_WORDS, RING_COLORS)}


# ── Overlap ring: 3 words shared with the grid ─────────────────────────────────
# "rock", "sand", and "box" also appear in the 4x4 grid vocabulary.
# When the model sees one of these tokens in a mixed context, it cannot
# determine the source graph from token identity alone — it must use
# surrounding context.  This tests whether the model uses structural
# in-context evidence rather than token-identity shortcuts.
#
# Grid positions of shared words:
#   rock  → grid row 3, col 0  (neighbors: math, box)
#   sand  → grid row 1, col 1  (neighbors: bird, milk, box, sun)  [corrected from utils]
#   box   → grid row 2, col 0  (neighbors: house, sand, rock)
#
# Ring positions of shared words (defined by RING_WORDS_OVERLAP order below):
#   rock  → between vine and sand
#   sand  → between rock and box
#   box   → between sand and prism

RING_WORDS_OVERLAP = [
    "candle", "brick", "fern", "lamp",
    "dust",   "wool",  "reef", "vine",
    "rock",   "sand",  "box",  "prism",
]

OVERLAP_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#a6d854", "#ffd92f", "#e5c494", "#8da0cb",
]

OVERLAP_WORD_TO_COLOR = {w: c for w, c in zip(RING_WORDS_OVERLAP, OVERLAP_COLORS)}

# Words that appear in both the grid and the overlap ring
SHARED_WORDS = {"rock", "sand", "box"}


# ── Ring class ─────────────────────────────────────────────────────────────────

class Ring:
    """12-node ring (cycle) graph.  Each node connects to its two cyclic neighbors.

    Default vocabulary is RING_WORDS (semantically neutral, disjoint from grid).
    Pass words=MONTHS for the original month ring, or words=RING_WORDS_OVERLAP
    for the shared-vocabulary variant.
    """

    def __init__(self, words=RING_WORDS):
        self.words = list(words)
        self.n = len(self.words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}

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

class Hamiltonian:
    """Hamiltonian graph.
    
    - defined to be a complete graph with n nodes, where each node is connected to every other node.
    - each node is a word in the vocabulary.
    
    """
    def __init__(self, words=RING_WORDS):
        self.words = list(words)
        self.n = len(self.words)
        self.word_to_idx = {w: i for i, w in enumerate(self.words)}

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