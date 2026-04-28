"""Bayesian graph-observer baseline for the secondary experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .config import DEFAULT_CONFIG
from .graphs import UndirectedGraph, build_candidate_graphs
from .vocabulary import WORDS, validate_vocabulary


def _logsumexp(x: np.ndarray) -> float:
    """Numerically stable log-sum-exp, copied from the initial Bayesian model."""

    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


class BayesianGraphObserver:
    """Posterior over graph hypotheses plus a next-token mixture distribution."""

    def __init__(
        self,
        graphs: Mapping[str, UndirectedGraph] | None = None,
        epsilon: float = DEFAULT_CONFIG.epsilon,
        prior: Mapping[str, float] | None = None,
        words: Sequence[str] = WORDS,
    ):
        self.words = validate_vocabulary(words)
        self.graphs = dict(build_candidate_graphs(self.words) if graphs is None else graphs)
        self.names = tuple(self.graphs.keys())
        self.epsilon = float(epsilon)
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}

        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}.")
        self._validate_graphs()
        self.log_prior = self._make_log_prior(prior)

    def _validate_graphs(self) -> None:
        for name, graph in self.graphs.items():
            if tuple(graph.words) != self.words:
                raise ValueError(f"{name} does not use the shared 16-word vocabulary.")

    def _make_log_prior(self, prior: Mapping[str, float] | None) -> dict[str, float]:
        if prior is None:
            return {name: -np.log(len(self.names)) for name in self.names}

        missing = set(self.names) - set(prior)
        extra = set(prior) - set(self.names)
        if missing or extra:
            raise ValueError(f"Prior keys mismatch. Missing={missing}, extra={extra}")

        probs = np.array([prior[name] for name in self.names], dtype=float)
        if np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError("Prior probabilities must be nonnegative and sum positive.")
        probs = probs / probs.sum()
        return {name: float(np.log(prob)) for name, prob in zip(self.names, probs)}

    def graph_transition_distribution(self, graph_name: str, current_word: str) -> np.ndarray:
        """Noisy random-walk kernel over all 16 words for one graph hypothesis."""

        graph = self.graphs[graph_name]
        if graph.is_uniform:
            return np.full(len(self.words), 1.0 / len(self.words), dtype=float)

        p_graph = graph.transition_distribution(current_word)
        p_noise = np.full(len(self.words), 1.0 / len(self.words), dtype=float)
        return (1.0 - self.epsilon) * p_graph + self.epsilon * p_noise

    def transition_logprob(self, graph_name: str, w_from: str, w_to: str) -> float:
        probs = self.graph_transition_distribution(graph_name, w_from)
        return float(np.log(probs[self.word_to_idx[w_to]]))

    def sequence_loglik(self, graph_name: str, context: Sequence[str]) -> float:
        if len(context) < 2:
            return 0.0
        return float(
            sum(
                self.transition_logprob(graph_name, context[i], context[i + 1])
                for i in range(len(context) - 1)
            )
        )

    def log_joint(self, context: Sequence[str]) -> np.ndarray:
        return np.array(
            [
                self.log_prior[name] + self.sequence_loglik(name, context)
                for name in self.names
            ],
            dtype=float,
        )

    def log_posterior(self, context: Sequence[str]) -> np.ndarray:
        log_joint = self.log_joint(context)
        return log_joint - _logsumexp(log_joint)

    def posterior(self, context: Sequence[str]) -> dict[str, float]:
        probs = np.exp(self.log_posterior(context))
        return {name: float(prob) for name, prob in zip(self.names, probs)}

    def next_token_distribution(self, context: Sequence[str]) -> np.ndarray:
        """Bayesian posterior-predictive distribution over all 16 graph words."""

        if not context:
            raise ValueError("Context must contain at least one word.")

        current_word = context[-1]
        posterior = np.exp(self.log_posterior(context))
        mixture = np.zeros(len(self.words), dtype=float)
        for graph_weight, graph_name in zip(posterior, self.names):
            mixture += graph_weight * self.graph_transition_distribution(graph_name, current_word)
        return mixture / mixture.sum()

    def named_next_token_distribution(self, context: Sequence[str]) -> dict[str, float]:
        probs = self.next_token_distribution(context)
        return {word: float(prob) for word, prob in zip(self.words, probs)}
