"""Tests for the undirected edge-learning baseline."""

from __future__ import annotations

import unittest

import numpy as np

from src.secondary_experiments.cache_baseline import CacheBaseline
from src.secondary_experiments.edge_learner import UndirectedEdgeLearner


class EdgeLearnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.learner = UndirectedEdgeLearner()

    def test_output_probabilities_sum_to_one_and_are_nonzero(self) -> None:
        probs = self.learner.predict_array("apple")
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)
        self.assertTrue(np.all(probs > 0.0))

    def test_update_increases_forward_prediction(self) -> None:
        before = self.learner.predict_next("apple")["bird"]
        self.learner.update("apple", "bird")
        after = self.learner.predict_next("apple")["bird"]
        self.assertGreater(after, before)

    def test_update_increases_reverse_prediction(self) -> None:
        before = self.learner.predict_next("bird")["apple"]
        self.learner.update("apple", "bird")
        after = self.learner.predict_next("bird")["apple"]
        self.assertGreater(after, before)

    def test_cache_does_not_receive_symmetric_update(self) -> None:
        cache_before = CacheBaseline().next_token_distribution(["bird"])
        cache_after = CacheBaseline()
        cache_after.next_token_distribution(["bird"])

        context = ["apple", "bird"]
        cache = CacheBaseline()
        after_context = cache.next_token_distribution(context)

        word_to_idx = cache.word_to_idx
        self.assertAlmostEqual(
            cache_before[word_to_idx["apple"]],
            after_context[word_to_idx["apple"]],
            places=12,
        )

    def test_no_self_edges_but_self_prediction_is_smoothed(self) -> None:
        self.learner.update("apple", "apple")
        self.assertEqual(self.learner.edge_probability("apple", "apple"), 0.0)
        self.assertGreater(self.learner.predict_next("apple")["apple"], 0.0)


if __name__ == "__main__":
    unittest.main()
