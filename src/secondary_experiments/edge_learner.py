"""Bayesian-style undirected edge-learning baseline.

Unlike the ideal Bayesian graph observer, this baseline does not know the
candidate graph family.  It treats each unordered word pair as a possible edge
and updates the edge score symmetrically whenever either transition direction is
observed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from .vocabulary import WORDS, validate_vocabulary


class UndirectedEdgeLearner:
    """Learn likely undirected edges from observed transitions."""

    def __init__(
        self,
        words: Sequence[str] = WORDS,
        edge_prior_prob: float = 0.2,
        edge_prior_strength: float = 2.0,
        alpha: float = 0.1,
    ):
        if not 0.0 < edge_prior_prob < 1.0:
            raise ValueError("edge_prior_prob must be in (0, 1).")
        if edge_prior_strength <= 0.0:
            raise ValueError("edge_prior_strength must be positive.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive so predictions are nonzero.")

        self.words = validate_vocabulary(words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.edge_prior_prob = float(edge_prior_prob)
        self.edge_prior_strength = float(edge_prior_strength)
        self.a0 = self.edge_prior_strength * self.edge_prior_prob
        self.b0 = self.edge_prior_strength * (1.0 - self.edge_prior_prob)
        self.alpha = float(alpha)
        self.counts: Counter[tuple[str, str]] = Counter()

    def _edge_key(self, word_a: str, word_b: str) -> tuple[str, str]:
        if word_a == word_b:
            raise ValueError("Self-edges are not part of the edge learner.")
        if word_a not in self.word_to_idx or word_b not in self.word_to_idx:
            raise KeyError(f"Unknown word pair: {word_a}, {word_b}")
        return tuple(sorted((word_a, word_b)))

    def copy(self) -> "UndirectedEdgeLearner":
        other = UndirectedEdgeLearner(
            words=self.words,
            edge_prior_prob=self.edge_prior_prob,
            edge_prior_strength=self.edge_prior_strength,
            alpha=self.alpha,
        )
        other.counts = self.counts.copy()
        return other

    def update(self, current_word: str, next_word: str) -> None:
        """Update the undirected edge count for ``{current_word, next_word}``."""

        if current_word == next_word:
            return
        self.counts[self._edge_key(current_word, next_word)] += 1

    def update_context(self, context: Sequence[str]) -> None:
        """Update on transitions inside ``context`` only.

        For context length L, this consumes transitions
        ``w_1 -> w_2`` through ``w_{L-1} -> w_L`` and does not consume the
        held-out transition ``w_L -> w_{L+1}``.
        """

        for current_word, next_word in zip(context[:-1], context[1:]):
            self.update(current_word, next_word)

    def edge_score(self, word_a: str, word_b: str) -> float:
        if word_a == word_b:
            return 0.0
        return self.a0 + self.counts[self._edge_key(word_a, word_b)]

    def edge_probability(self, word_a: str, word_b: str) -> float:
        if word_a == word_b:
            return 0.0
        count = self.counts[self._edge_key(word_a, word_b)]
        return float((self.a0 + count) / (self.a0 + self.b0 + count))

    def predict_array(self, current_word: str) -> np.ndarray:
        """Return a normalized distribution over all words."""

        if current_word not in self.word_to_idx:
            raise KeyError(f"Unknown current word: {current_word}")

        scores = np.full(len(self.words), self.alpha, dtype=float)
        for i, word in enumerate(self.words):
            if word == current_word:
                continue
            scores[i] += self.edge_score(current_word, word)
        return scores / scores.sum()

    def predict_next(self, current_word: str) -> dict[str, float]:
        probs = self.predict_array(current_word)
        return {word: float(prob) for word, prob in zip(self.words, probs)}

    def edge_posterior(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for i, word_a in enumerate(self.words):
            for word_b in self.words[i + 1:]:
                out[(word_a, word_b)] = self.edge_probability(word_a, word_b)
        return out

    def top_edges(self, k: int = 10) -> list[tuple[str, str, float]]:
        posterior = self.edge_posterior()
        ranked = sorted(posterior.items(), key=lambda item: item[1], reverse=True)
        return [(a, b, float(prob)) for (a, b), prob in ranked[:k]]


def fit_edge_learner(
    context: Sequence[str],
    words: Sequence[str] = WORDS,
    edge_prior_prob: float = 0.2,
    edge_prior_strength: float = 2.0,
    alpha: float = 0.1,
) -> UndirectedEdgeLearner:
    learner = UndirectedEdgeLearner(
        words=words,
        edge_prior_prob=edge_prior_prob,
        edge_prior_strength=edge_prior_strength,
        alpha=alpha,
    )
    learner.update_context(context)
    return learner
