"""Local transition-count cache baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import DEFAULT_CONFIG
from .vocabulary import WORDS, validate_vocabulary


class CacheBaseline:
    """Predict from earlier same-context transitions out of the current token."""

    def __init__(self, alpha: float = DEFAULT_CONFIG.alpha, words: Sequence[str] = WORDS):
        if alpha < 0:
            raise ValueError(f"alpha must be nonnegative, got {alpha}.")
        self.alpha = float(alpha)
        self.words = validate_vocabulary(words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}

    def next_token_distribution(self, context: Sequence[str]) -> np.ndarray:
        if not context:
            raise ValueError("Context must contain at least one word.")

        current_word = context[-1]
        counts = np.zeros(len(self.words), dtype=float)
        for w_from, w_to in zip(context[:-1], context[1:]):
            if w_from == current_word:
                counts[self.word_to_idx[w_to]] += 1.0

        smoothed = counts + self.alpha
        denom = smoothed.sum()
        if denom == 0.0:
            return np.full(len(self.words), 1.0 / len(self.words), dtype=float)
        return smoothed / denom

    def named_next_token_distribution(self, context: Sequence[str]) -> dict[str, float]:
        probs = self.next_token_distribution(context)
        return {word: float(prob) for word, prob in zip(self.words, probs)}
