"""Deterministic graph definitions over the shared 16-word vocabulary.

The ``GridGraph`` and ``RingGraph`` APIs follow the graph classes in
``src/initial_experiments``: each graph exposes ``words``,
``get_valid_next_words(word)``, ``generate_sequence(...)``, and
``build_adjacency_matrix()``.  The key difference is that label order is fixed
instead of shuffled at construction time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .vocabulary import WORDS, validate_vocabulary


@dataclass
class UndirectedGraph:
    """Small undirected graph over a fixed word vocabulary."""

    name: str
    words: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    is_uniform: bool = False

    def __post_init__(self) -> None:
        self.words = validate_vocabulary(self.words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        missing = set(self.words) - set(self.adjacency)
        if missing:
            raise ValueError(f"{self.name} adjacency missing words: {sorted(missing)}")
        for word, neighbors in self.adjacency.items():
            if word not in self.word_to_idx:
                raise ValueError(f"{self.name} has OOV adjacency key: {word}")
            for neighbor in neighbors:
                if neighbor not in self.word_to_idx:
                    raise ValueError(f"{self.name} has OOV neighbor: {neighbor}")
                if not self.is_uniform and word not in self.adjacency[neighbor]:
                    raise ValueError(f"{self.name} edge is not symmetric: {word}, {neighbor}")

    @property
    def n(self) -> int:
        return len(self.words)

    def get_valid_next_words(self, word: str) -> list[str]:
        return list(self.adjacency[word])

    def build_adjacency_matrix(self) -> np.ndarray:
        A = np.zeros((self.n, self.n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.word_to_idx[neighbor]
                A[i, j] = 1
        return A

    def transition_distribution(self, word: str) -> np.ndarray:
        """Uniform random-walk distribution from ``word`` over all 16 words."""

        probs = np.zeros(self.n, dtype=float)
        if self.is_uniform:
            probs[:] = 1.0 / self.n
            return probs

        neighbors = self.get_valid_next_words(word)
        if not neighbors:
            raise ValueError(f"{self.name} has no outgoing neighbors for {word}.")
        for neighbor in neighbors:
            probs[self.word_to_idx[neighbor]] = 1.0 / len(neighbors)
        return probs

    def generate_sequence(
        self,
        seq_len: int,
        start_word: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Generate a random walk; adapted from the initial graph classes."""

        rng = np.random.default_rng() if rng is None else rng
        if seq_len <= 0:
            return []
        word = start_word if start_word is not None else str(rng.choice(self.words))
        sequence = [word]
        while len(sequence) < seq_len:
            if self.is_uniform:
                word = str(rng.choice(self.words))
            else:
                word = str(rng.choice(self.get_valid_next_words(word)))
            sequence.append(word)
        return sequence


class GridGraph(UndirectedGraph):
    """4-by-4 grid graph.  Adapted from ``initial_experiments.sanity_check.Grid``."""

    def __init__(self, words: Iterable[str] = WORDS, rows: int = 4, cols: int = 4):
        vocab = validate_vocabulary(tuple(words))
        if rows * cols != len(vocab):
            raise ValueError(f"Grid shape {rows}x{cols} does not match {len(vocab)} words.")

        grid = np.array(vocab).reshape(rows, cols).tolist()
        adjacency: dict[str, tuple[str, ...]] = {}
        for i, word in enumerate(vocab):
            row, col = i // cols, i % cols
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    neighbors.append(grid[r][c])
            adjacency[word] = tuple(neighbors)

        super().__init__("grid", vocab, adjacency)
        self.rows = rows
        self.cols = cols
        self.grid = grid
        self.word_to_row = {word: i // cols for i, word in enumerate(vocab)}
        self.word_to_col = {word: i % cols for i, word in enumerate(vocab)}


class RingGraph(UndirectedGraph):
    """16-node cycle graph.  Adapted from ``initial_experiments.graphs.Ring``."""

    def __init__(self, words: Iterable[str] = WORDS):
        vocab = validate_vocabulary(tuple(words))
        adjacency = {
            word: (vocab[(i - 1) % len(vocab)], vocab[(i + 1) % len(vocab)])
            for i, word in enumerate(vocab)
        }
        super().__init__("ring", vocab, adjacency)


class ChainGraph(UndirectedGraph):
    """16-node path graph in vocabulary order."""

    def __init__(self, words: Iterable[str] = WORDS):
        vocab = validate_vocabulary(tuple(words))
        adjacency: dict[str, tuple[str, ...]] = {}
        for i, word in enumerate(vocab):
            neighbors = []
            if i > 0:
                neighbors.append(vocab[i - 1])
            if i < len(vocab) - 1:
                neighbors.append(vocab[i + 1])
            adjacency[word] = tuple(neighbors)
        super().__init__("chain", vocab, adjacency)


class StarGraph(UndirectedGraph):
    """Star graph with the first vocabulary item as the hub."""

    def __init__(self, words: Iterable[str] = WORDS):
        vocab = validate_vocabulary(tuple(words))
        hub = vocab[0]
        adjacency = {hub: tuple(vocab[1:])}
        adjacency.update({word: (hub,) for word in vocab[1:]})
        super().__init__("star", vocab, adjacency)
        self.hub = hub


class UniformGraph(UndirectedGraph):
    """No-structure null hypothesis with uniform transitions over all words."""

    def __init__(self, words: Iterable[str] = WORDS):
        vocab = validate_vocabulary(tuple(words))
        adjacency = {word: tuple(vocab) for word in vocab}
        super().__init__("uniform", vocab, adjacency, is_uniform=True)


def build_candidate_graphs(words: Iterable[str] = WORDS) -> dict[str, UndirectedGraph]:
    """Return the default graph-hypothesis set keyed by graph name."""

    vocab = validate_vocabulary(tuple(words))
    graphs: list[UndirectedGraph] = [
        GridGraph(vocab),
        RingGraph(vocab),
        ChainGraph(vocab),
        StarGraph(vocab),
        UniformGraph(vocab),
    ]
    return {graph.name: graph for graph in graphs}
