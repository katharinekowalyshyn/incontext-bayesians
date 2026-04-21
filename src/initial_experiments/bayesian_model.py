"""Simplified Bayesian model for identifying graph structure from a random walk.

Given a sequence of tokens x = (w_1, ..., w_T) and a set of candidate graphs
{G_1, ..., G_K}, we compute the posterior p(G_k | x) assuming x was generated
by a uniform random walk on some G_k.

Under a uniform random walk on G:
    p(w_{t+1} | w_t, G) = 1 / deg_G(w_t)   if (w_t, w_{t+1}) is an edge,
                       = 0                  otherwise.

So the log-likelihood of a walk is:
    log p(x | G) = sum_{t=1}^{T-1} log p(w_{t+1} | w_t, G)

For numerical safety (and so sequences with tokens outside G's vocabulary
still get a finite score), invalid transitions are assigned a small
smoothing probability `eps` rather than 0. With eps=0 the model is exact.

The log posterior odds between two candidates (Eq. 4 in Bigelow et al. 2025)
are:
    log p(G_1 | x) / p(G_2 | x)  =  b_12  +  [log p(x|G_1) - log p(x|G_2)]
where b_12 = log p(G_1)/p(G_2) is a prior log-odds term (default 0 = flat).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class GraphLikelihood:
    """Wraps a graph object with a `get_valid_next_words(word) -> list[str]` method
    (matches the Grid and Ring classes in this repo) and provides a uniform
    random-walk likelihood over token sequences.
    """

    name: str
    graph: object  # duck-typed: needs .words and .get_valid_next_words(word)
    eps: float = 1e-8  # smoothing for invalid / OOV transitions

    def __post_init__(self):
        self.vocab = set(self.graph.words)

    def transition_logprob(self, w_from: str, w_to: str) -> float:
        """log p(w_to | w_from, G) under a uniform random walk on G."""
        if w_from not in self.vocab or w_to not in self.vocab:
            return np.log(self.eps)
        neighbors = self.graph.get_valid_next_words(w_from)
        if w_to in neighbors:
            return -np.log(len(neighbors))
        return np.log(self.eps)

    def sequence_loglik(self, walk: Sequence[str]) -> float:
        """log p(walk | G). Ignores the first token (we don't model the start
        distribution — it cancels under any uniform prior over start nodes)."""
        return float(sum(
            self.transition_logprob(walk[t], walk[t + 1])
            for t in range(len(walk) - 1)
        ))

    def sequence_loglik_curve(self, walk: Sequence[str]) -> np.ndarray:
        """Cumulative log p(walk[:t+1] | G) for t = 1..T-1. Useful for tracing
        how evidence accumulates with context length."""
        steps = np.array([
            self.transition_logprob(walk[t], walk[t + 1])
            for t in range(len(walk) - 1)
        ])
        return np.cumsum(steps)


class BayesianGraphClassifier:
    """Classify a random walk among K candidate graphs.

    Usage:
        clf = BayesianGraphClassifier([
            GraphLikelihood("grid", Grid()),
            GraphLikelihood("ring", Ring()),
        ])
        clf.classify(walk)          # -> ("grid", posterior_probs_array)
        clf.log_odds(walk, "grid", "ring")   # -> scalar log p(grid|x)/p(ring|x)
    """

    def __init__(
        self,
        candidates: Iterable[GraphLikelihood],
        log_prior: dict[str, float] | None = None,
    ):
        self.candidates = list(candidates)
        self.names = [c.name for c in self.candidates]
        if log_prior is None:
            log_prior = {c.name: 0.0 for c in self.candidates}
        self.log_prior = log_prior

    def log_joint(self, walk: Sequence[str]) -> np.ndarray:
        """log p(G_k) + log p(x | G_k) for each candidate k."""
        return np.array([
            self.log_prior.get(c.name, 0.0) + c.sequence_loglik(walk)
            for c in self.candidates
        ])

    def log_posterior(self, walk: Sequence[str]) -> np.ndarray:
        """Normalised log p(G_k | x) across candidates."""
        lj = self.log_joint(walk)
        return lj - _logsumexp(lj)

    def posterior(self, walk: Sequence[str]) -> np.ndarray:
        return np.exp(self.log_posterior(walk))

    def classify(self, walk: Sequence[str]) -> tuple[str, np.ndarray]:
        """Return (predicted_graph_name, posterior_probs)."""
        post = self.posterior(walk)
        return self.names[int(np.argmax(post))], post

    def log_odds(self, walk: Sequence[str], target: str, other: str) -> float:
        """log p(target | x) / p(other | x), i.e. Eq. 4 of Bigelow et al. 2025.

        Returns b + log p(x|target)/p(x|other), where b is the prior log-odds.
        """
        t = next(c for c in self.candidates if c.name == target)
        o = next(c for c in self.candidates if c.name == other)
        b = self.log_prior.get(target, 0.0) - self.log_prior.get(other, 0.0)
        return b + t.sequence_loglik(walk) - o.sequence_loglik(walk)

    def log_odds_curve(
        self, walk: Sequence[str], target: str, other: str
    ) -> np.ndarray:
        """Log-odds as a function of context length T = 1..len(walk)-1.
        Useful for visualising the sigmoidal phase transition."""
        t = next(c for c in self.candidates if c.name == target)
        o = next(c for c in self.candidates if c.name == other)
        b = self.log_prior.get(target, 0.0) - self.log_prior.get(other, 0.0)
        return b + t.sequence_loglik_curve(walk) - o.sequence_loglik_curve(walk)


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from graphs import Ring, RING_WORDS
    from sanity_check import Grid, WORDS as GRID_WORDS, set_seed

    set_seed(0)
    grid = Grid() # Grid()
    ring = Grid() # Ring(words=RING_WORDS)

    print(f'grid =  {grid.words}')
    print(f'grid2 =  {ring.words}')

    clf = BayesianGraphClassifier([
        GraphLikelihood("grid", grid),
        GraphLikelihood("ring", grid),
    ])

    for true_name, g in [("grid", grid), ("ring", ring)]:
        walk = g.generate_sequence(seq_len=14000)
        pred, post = clf.classify(walk)
        lo = clf.log_odds(walk, "grid", "ring")
        print(
            f"true={true_name:4s}  predicted={pred:4s}  "
            f"p(grid|x)={post[0]:.3f}  p(ring|x)={post[1]:.3f}  "
            f"log-odds(grid/ring)={lo:+.2f}"
        )
