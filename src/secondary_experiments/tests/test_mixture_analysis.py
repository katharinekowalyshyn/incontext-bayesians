"""Tests for mixture-of-baselines analysis."""

from __future__ import annotations

import unittest

import numpy as np

from src.secondary_experiments.mixture_analysis import fit_by_context_length, run_mixture_analysis
from src.secondary_experiments.vocabulary import WORDS


def dist_dict(values):
    arr = np.asarray(values, dtype=float)
    arr = arr / arr.sum()
    return {word: float(prob) for word, prob in zip(WORDS, arr)}


class MixtureAnalysisTest(unittest.TestCase):
    def test_fit_prefers_matching_baseline(self) -> None:
        p_bayes = np.array([0.7] + [0.3 / 15] * 15)
        p_edge = np.full(16, 1 / 16)
        p_cache = np.array([0.1, 0.6] + [0.3 / 14] * 14)
        p_sem = np.array([0.2, 0.2, 0.4] + [0.2 / 13] * 13)
        p_uni = np.full(16, 1 / 16)
        rows = []
        for seed in range(3):
            rows.append({
                "context_length": 10,
                "llm_distribution": dist_dict(p_bayes),
                "bayes_distribution": dist_dict(p_bayes),
                "edge_learner_distribution": dist_dict(p_edge),
                "cache_distribution": dist_dict(p_cache),
                "unigram_distribution": dist_dict(p_uni),
                "semantic_prior_distribution": dist_dict(p_sem),
            })
        fit = fit_by_context_length(rows, WORDS, alpha=1.0, n_steps=400, lr=0.05)[0]
        self.assertGreater(fit["weights"]["ideal_bayes"], 0.7)
        self.assertLess(fit["kl_llm_mixture"], fit["kl_llm_individual"]["cache"])


if __name__ == "__main__":
    unittest.main()
