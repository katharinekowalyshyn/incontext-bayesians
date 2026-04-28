"""Checks for the noisy Bayesian graph-observer likelihood."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.secondary_experiments.bayesian_observer import BayesianGraphObserver
from src.secondary_experiments.graphs import build_candidate_graphs


EPSILON = 0.05


class BayesianObserverLikelihoodTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = build_candidate_graphs()
        self.observer = BayesianGraphObserver(graphs=self.graphs, epsilon=EPSILON)
        self.words = self.observer.words
        self.n_words = len(self.words)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}

    def test_transition_distributions_sum_to_one(self) -> None:
        for graph_name in self.observer.names:
            for current_word in self.words:
                probs = self.observer.graph_transition_distribution(graph_name, current_word)
                self.assertEqual(probs.shape, (self.n_words,))
                self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)
                self.assertTrue(np.all(probs > 0.0))

    def test_invalid_transitions_get_noise_floor(self) -> None:
        expected = EPSILON / self.n_words
        self.assertAlmostEqual(expected, 0.003125, places=12)

        for graph_name, graph in self.graphs.items():
            if graph.is_uniform:
                continue
            for current_word in self.words:
                neighbors = set(graph.get_valid_next_words(current_word))
                invalid_words = [word for word in self.words if word not in neighbors]
                self.assertTrue(invalid_words, f"{graph_name}:{current_word} has no invalid transitions")
                probs = self.observer.graph_transition_distribution(graph_name, current_word)
                for invalid_word in invalid_words:
                    actual = probs[self.word_to_idx[invalid_word]]
                    self.assertAlmostEqual(actual, expected, places=12)
                    self.assertGreater(actual, 0.0)

    def test_valid_transitions_get_graph_mass_plus_noise_floor(self) -> None:
        noise_floor = EPSILON / self.n_words

        for graph_name, graph in self.graphs.items():
            if graph.is_uniform:
                continue
            for current_word in self.words:
                neighbors = graph.get_valid_next_words(current_word)
                degree = len(neighbors)
                expected = (1.0 - EPSILON) / degree + noise_floor
                probs = self.observer.graph_transition_distribution(graph_name, current_word)
                for neighbor in neighbors:
                    actual = probs[self.word_to_idx[neighbor]]
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_uniform_hypothesis_is_independent_of_epsilon(self) -> None:
        for epsilon in (0.0, EPSILON, 0.9):
            observer = BayesianGraphObserver(graphs=self.graphs, epsilon=epsilon)
            for current_word in self.words:
                probs = observer.graph_transition_distribution("uniform", current_word)
                expected = np.full(self.n_words, 1.0 / self.n_words)
                np.testing.assert_allclose(probs, expected, rtol=0.0, atol=1e-12)

    def test_invalid_transition_lowers_but_does_not_zero_posterior(self) -> None:
        prior = BayesianGraphObserver(graphs=self.graphs, epsilon=EPSILON).posterior(["apple"])
        posterior = self.observer.posterior(["apple", "egg"])

        self.assertNotIn("egg", self.graphs["grid"].get_valid_next_words("apple"))
        self.assertLess(posterior["grid"], prior["grid"])
        self.assertGreater(posterior["grid"], 0.0)
        self.assertTrue(math.isfinite(math.log(posterior["grid"])))
        self.assertAlmostEqual(sum(posterior.values()), 1.0, places=12)

    def test_sequence_loglik_uses_log_probabilities(self) -> None:
        context = ["apple", "egg"]
        expected_prob = EPSILON / self.n_words
        expected_logprob = math.log(expected_prob)
        self.assertAlmostEqual(
            self.observer.sequence_loglik("grid", context),
            expected_logprob,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
