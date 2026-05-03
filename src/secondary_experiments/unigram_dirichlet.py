"""Global unigram Dirichlet–multinomial baseline.

Populates ``unigram_distribution`` in ``experiment.baseline_rows_for_sequence`` and is the
``unigram`` component of ``mixture_analysis.BASELINES``.  Smoothing ``alpha`` matches
``ExperimentConfig.alpha`` (same pseudocount as the cache baseline).

See class docstring for the prediction rule.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .config import DEFAULT_CONFIG
from .vocabulary import WORDS, validate_vocabulary


class UnigramDirichletMultinomialBaseline:
    """Predict next token using global counts (ignores current token). Does the model just pick the word
    that has appeared most often in context?

    p(y | context) = (count(y) + alpha) / (N + V * alpha)
    """

    def __init__(self, alpha: float = DEFAULT_CONFIG.alpha, words: Sequence[str] = WORDS):
        if alpha < 0:
            raise ValueError(f"alpha must be nonnegative, got {alpha}.")
        self.alpha = float(alpha)
        self.words = validate_vocabulary(words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.vocab_size = len(self.words)

    def next_token_distribution(self, context: Sequence[str]) -> np.ndarray:
        if not context:
            raise ValueError("Context must contain at least one word.")

        counts = np.zeros(self.vocab_size, dtype=float)

        # Count all tokens in context (unigram counts)
        for w in context:
            if w in self.word_to_idx:
                counts[self.word_to_idx[w]] += 1.0

        smoothed = counts + self.alpha
        denom = smoothed.sum()

        if denom == 0.0:
            return np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=float)

        return smoothed / denom

    def named_next_token_distribution(self, context: Sequence[str]) -> dict[str, float]:
        probs = self.next_token_distribution(context)
        return {word: float(prob) for word, prob in zip(self.words, probs)}