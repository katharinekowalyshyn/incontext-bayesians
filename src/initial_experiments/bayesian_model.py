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

import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from graphs import Ring
from sanity_check import Grid, WORDS as GRID_WORDS, set_seed


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


# ── Demo / sanity check ───────────────────────────────────────────────────────
#
# We build two graphs over the same 16-word vocabulary:
#   - a 4x4 grid (Grid()), and
#   - a 16-node ring wired in the natural GRID_WORDS order (Ring(GRID_WORDS)).
#
# Because the vocabularies are identical, the classifier cannot cheat by
# falling back to log(eps) for out-of-vocab tokens -- it has to distinguish
# the two graphs purely by edge structure.  Expected signs of the
# log-odds(grid / ring) curve:
#   * walk sampled from the grid  -> log-odds climbs positive
#   * walk sampled from the ring  -> log-odds climbs negative

def hamiltonian_ring_order(grid: Grid) -> list[str]:
    """Return the grid's words traced along a Hamiltonian cycle.

    Uses a standard "boustrophedon + return column" construction: walk
    down column 0, snake through columns 1..cols-1 from the bottom row
    upward, then close the cycle along row 0.  Every consecutive pair
    (including the wrap-around) is a grid edge, so every ring edge of
    ``Ring(words=hamiltonian_ring_order(grid))`` is also a grid edge.

    Requires ``min(rows, cols) >= 2`` and at least one of rows/cols to be
    even — a rectangular grid graph with both dimensions odd has no
    Hamiltonian cycle (standard bipartite-colouring argument).  The
    produced cycle is validated against ``grid.get_valid_next_words`` so
    any bug in the construction would raise rather than silently return
    a wrong ordering.
    """
    rows, cols = grid.rows, grid.cols
    if min(rows, cols) < 2:
        raise ValueError(f"Grid {rows}x{cols} is too small for a Hamiltonian cycle.")
    if rows % 2 == 1 and cols % 2 == 1:
        raise ValueError(
            f"Grid {rows}x{cols} has no Hamiltonian cycle "
            f"(requires at least one even dimension)."
        )

    # Orient so the "down-column-0" leg has an even length.  If rows is
    # odd we transpose; the Grid's row/col labels are swapped internally
    # but the final list of words is the same cycle.
    if rows % 2 == 1:
        r_dim, c_dim = cols, rows
        def lookup(r: int, c: int) -> str:
            return grid.grid[c][r]
    else:
        r_dim, c_dim = rows, cols
        def lookup(r: int, c: int) -> str:
            return grid.grid[r][c]

    coords: list[tuple[int, int]] = [(r, 0) for r in range(r_dim)]
    for i, r in enumerate(range(r_dim - 1, 0, -1)):
        col_iter = range(1, c_dim) if i % 2 == 0 else range(c_dim - 1, 0, -1)
        coords.extend((r, c) for c in col_iter)
    coords.extend((0, c) for c in range(c_dim - 1, 0, -1))

    words = [lookup(r, c) for r, c in coords]

    # Validate: every cyclically-consecutive pair must be an actual grid edge.
    n = len(words)
    assert n == rows * cols, f"cycle visits {n} != {rows*cols} nodes"
    assert len(set(words)) == n, "cycle repeats a node"
    for i, w in enumerate(words):
        nxt = words[(i + 1) % n]
        if nxt not in grid.get_valid_next_words(w):
            raise AssertionError(
                f"hamiltonian_ring_order produced a non-grid edge: {w} -> {nxt}"
            )
    return words


def posterior_predictive_valid_neighbor_prob(
    classifier: "BayesianGraphClassifier",
    walk: Sequence[str],
    target_graph: object,
) -> np.ndarray:
    """Bayesian mixture's predicted P(next token ∈ target_graph neighbors).

    For each prefix ``x_{1..t+1}`` (t = 0..T-2), computes

        Σ_k  p(G_k | x_{1..t+1}) · |N_target(w_t) ∩ N_{G_k}(w_t)| / |N_{G_k}(w_t)|

    where ``N_G(w)`` are the neighbors of ``w`` in graph ``G``.  This is
    the direct analogue of the LLM metric in
    ``vocabulary_tl_experiment.py`` (sum of softmax probabilities over
    ground-truth neighbor tokens), so the two curves live on the same
    [0, 1] axis and can be overlaid.
    """
    target_vocab = set(target_graph.words)
    probs = np.empty(len(walk) - 1)
    for t in range(len(walk) - 1):
        prefix = walk[: t + 1]
        post = classifier.posterior(prefix)
        w_t = walk[t]
        target_nbrs = (
            set(target_graph.get_valid_next_words(w_t))
            if w_t in target_vocab else set()
        )
        p_valid = 0.0
        for k, cand in enumerate(classifier.candidates):
            if w_t not in cand.vocab:
                continue
            nbrs_k = cand.graph.get_valid_next_words(w_t)
            if not nbrs_k:
                continue
            p_valid += post[k] * sum(1 for n in nbrs_k if n in target_nbrs) / len(nbrs_k)
        probs[t] = p_valid
    return probs


def log_prob_distance_curve(
    classifier: "BayesianGraphClassifier",
    walk: Sequence[str],
    name_a: str,
    name_b: str,
) -> np.ndarray:
    """Per-context-length distance |log p(walk[:t+1] | a) - log p(walk[:t+1] | b)|.

    Returns a length T-1 array; element t is

        | log p(walk[:t+2] | G_a)  -  log p(walk[:t+2] | G_b) |

    This is the absolute value of the (flat-prior) log-odds curve and
    measures how distinguishable the two hypotheses are given the
    evidence seen so far: 0 means the two models agree perfectly on the
    walk's likelihood; large values mean one model is exponentially more
    plausible than the other.

    Any monotone-increasing trend is lower-bounded by the per-step KL
    divergence between the two transition kernels, averaged over the
    walk's stationary distribution — so the slope of this curve is a
    direct empirical estimate of that KL on the observed walk.
    """
    a = next(c for c in classifier.candidates if c.name == name_a)
    b = next(c for c in classifier.candidates if c.name == name_b)
    diff = a.sequence_loglik_curve(walk) - b.sequence_loglik_curve(walk)
    return np.abs(diff)


# ── Complexity-weighted log-prior over candidate graphs ──────────────────────
#
# Checkpoint-2 §4.3 commits to the **MST description-length** proxy as the
# complexity term in the Upgrade prior:
#
#     log p(G_k)  ∝  b_0 − λ · C_MST(G_k)
#     C_MST(G)   =  |E_MST(G)| · ⌈log_2 |V|⌉       bits
#
# ``mst_log_prior`` below is the primary implementation of that choice; it is
# the one the main fit results of §5 should use.
#
# We also include a secondary helper, ``dirichlet_energy_log_prior``,
# that re-expresses C(G_k) as the normalised Dirichlet energy E_G(H_∞) of
# Llama's asymptotic class-mean representations (Thm 5.1 of Park et al.,
# 2025).  It is intended for the §4.3 appendix "alternative proxies evaluated
# qualitatively", NOT for the reported fits.

def _mst_edge_count(A: np.ndarray) -> int:
    """Number of edges in a minimum spanning tree of the (connected) graph
    whose unweighted symmetric adjacency matrix is ``A``.

    For a connected graph on |V| nodes a spanning tree has exactly
    |V|−1 edges, regardless of edge weights — so for the unweighted
    graphs we use in this project, this collapses to ``|V| − 1``
    whenever the graph is connected.  We verify connectivity and
    fall back to counting edges per connected component when the
    graph is disconnected, so the helper is safe for arbitrary A.
    """
    A = np.asarray(A)
    n = A.shape[0]
    if n == 0:
        return 0

    # Iterative BFS to enumerate connected components.
    seen = np.zeros(n, dtype=bool)
    comps = 0
    for start in range(n):
        if seen[start]:
            continue
        comps += 1
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(A[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
    # A spanning forest has n - (#components) edges.
    return n - comps


def mst_complexity_bits(A: np.ndarray) -> float:
    """``C_MST(G) = |E_MST(G)| · ⌈log_2 |V|⌉``  (Checkpoint-2 §4.3, v1).

    NOTE: as of the two-condition 16-node refactor, this helper is kept only
    for backward compatibility.  Any connected graph on |V| nodes has an MST
    with exactly |V|−1 edges, so ``mst_complexity_bits`` depends only on |V|
    and cannot distinguish a 16-node grid from a 16-node ring.  The Upgrade
    prior now uses :func:`edge_complexity_bits` instead.
    """
    A = np.asarray(A)
    n = A.shape[0]
    if n <= 1:
        return 0.0
    bits_per_vertex = int(np.ceil(np.log2(n)))
    return float(_mst_edge_count(A) * bits_per_vertex)


def edge_complexity_bits(A: np.ndarray) -> float:
    """``C_edges(G) = |E(G)| · ⌈log_2 |V|⌉``  (Checkpoint-2 §4.3, v2).

    Description length of the full unweighted edge list: each of ``|E|``
    undirected edges is encoded by its two endpoint indices, and every
    index takes ``⌈log_2 |V|⌉`` bits.  Ignoring the shared factor of 2
    (absorbed into λ) this gives ``|E| · ⌈log_2 |V|⌉``.

    This is the MDL complexity the Upgrade prior uses after the two-
    condition refactor:

        4×4 grid (|V|=16, |E|=24)   →   24 · 4 = 96 bits
        16-cycle ring (|V|=16, |E|=16) → 16 · 4 = 64 bits

    — the 32-bit asymmetry is what identifies λ.
    """
    A = np.asarray(A)
    n = A.shape[0]
    if n <= 1:
        return 0.0
    # Count undirected edges (A is symmetric so divide by 2; self-loops are
    # assumed absent).
    n_edges = float(A.sum()) / 2.0
    bits_per_vertex = int(np.ceil(np.log2(n)))
    return float(n_edges * bits_per_vertex)


def mst_log_prior(
    adjacencies: dict[str, np.ndarray],
    lam: float = 1.0,
    b0: float = 0.0,
    centre: bool = True,
    complexity: str = "edges",
) -> dict[str, float]:
    """Checkpoint-2 §4 Upgrade prior:  b_k = b_0 − λ · C(G_k).

    The function name is retained for backward compatibility, but the default
    complexity measure is now the full-edge-list MDL ``C_edges`` (v2 of the
    prior; see :func:`edge_complexity_bits`).  Pass ``complexity="mst"`` to
    recover the original MST-based prior (v1) — this is only useful when the
    candidate graphs have different numbers of vertices.

    Args
    ----
    adjacencies : ``{graph_name: A}`` — symmetric binary adjacency of each
        candidate graph.
    lam : learned λ ≥ 0 penalty weight.  λ = 0 collapses to a flat
        (shared-b_0) prior, which is exactly the Baseline of §3 — the
        Upgrade then degenerates to the Baseline, as required by the
        §4.2 unit-test contract ("with λ=0 the upgrade degenerates to
        a shared-b baseline").
    b0 : shared bias.  Fit jointly with λ in the Upgrade; set to 0
        here if you only care about relative prior odds between
        candidates (only differences matter in the BayesianGraphClassifier).
    centre : if True, shift so ``max(log_prior) = 0``.  Purely cosmetic;
        only relative values affect the posterior.
    complexity : ``"edges"`` (default, v2) or ``"mst"`` (v1 backward-compat).

    Returns
    -------
    ``{name: b_k}`` suitable for ``BayesianGraphClassifier(..., log_prior=...)``.
    Simpler graphs (smaller bit-cost) receive a less-negative prior.
    """
    if complexity == "edges":
        fn = edge_complexity_bits
    elif complexity == "mst":
        fn = mst_complexity_bits
    else:
        raise ValueError(f"unknown complexity measure: {complexity!r}")
    bits = {name: fn(A) for name, A in adjacencies.items()}
    raw = {name: b0 - lam * c for name, c in bits.items()}
    if centre:
        offset = max(raw.values())
        raw = {name: v - offset for name, v in raw.items()}
    return raw


def _normalized_dirichlet_energy(H: np.ndarray, A: np.ndarray) -> float:
    """E_G(H) / Tr(H_c^T D H_c). Matches pca_analysis.dirichlet_energy(..., normalize=True)."""
    A = np.asarray(A, dtype=np.float64)
    deg = A.sum(axis=1)
    # E_G(H) = Σ_{ij} A_{ij} ||h_i - h_j||^2 = 2 Tr(H^T L H).
    L = np.diag(deg) - A
    E = float(np.trace(H.T @ L @ H))
    # Centre rows by their degree-weighted mean before computing the denominator
    # (same convention as pca_analysis.py).
    w = deg / max(deg.sum(), 1e-12)
    mu = (w[:, None] * H).sum(axis=0, keepdims=True)
    Hc = H - mu
    denom = float(np.trace(Hc.T @ (deg[:, None] * Hc)))
    if denom <= 1e-12:
        return float("nan")
    return E / denom


def dirichlet_energy_log_prior(
    adjacencies: dict[str, np.ndarray],
    class_means: np.ndarray,
    beta: float = 1.0,
    centre: bool = True,
) -> dict[str, float]:
    """**APPENDIX-ONLY** alternative to ``mst_log_prior``.

    NOT the prior the main §5 fits should use — the Checkpoint-2
    submission commits to the MST description-length proxy
    (``mst_log_prior``) for the Upgrade.  This helper implements the
    alternative "Park-et-al. style" complexity term

        log p(G_k) ∝ −β · E_G(H_∞)

    where E_G is the normalised Dirichlet energy of Llama's asymptotic
    class-mean representations under adjacency A (Thm 5.1 of
    Park et al., 2025).  It is intended for the §4.3 appendix
    comparison of alternative complexity proxies — not for the
    reported fits.

    Args
    ----
    adjacencies : ``{graph_name: A}`` — each ``A`` an (n,n) binary matrix
        on the *same* word ordering as ``class_means``.
    class_means : (n, d) array of asymptotic LLM class means H_∞.  In
        practice this is ``class_means_by_T[T_max]`` from
        ``pca_analysis.run_with_model(...)``.
    beta : scale on the complexity penalty.  β=0 collapses to a flat
        prior; larger β more aggressively rewards low-energy graphs.
    centre : if True (default), subtract the mean log-prior across
        candidates so the maximum entry is 0.  Purely cosmetic — only
        relative values matter for the posterior.

    Returns
    -------
    ``{name: log_prior_value}`` suitable for passing as
    ``log_prior=`` to ``BayesianGraphClassifier``.  Graphs whose edges
    align with the representation geometry (small E_G) get a larger
    log-prior; misaligned graphs get a smaller one.
    """
    energies = {
        name: _normalized_dirichlet_energy(class_means, A)
        for name, A in adjacencies.items()
    }
    raw = {name: -beta * e for name, e in energies.items()}
    if centre:
        offset = max(raw.values())
        raw = {name: v - offset for name, v in raw.items()}
    return raw


def plot_log_prob_distance(
    distances: dict[str, np.ndarray],
    out_path: str,
    title: str = (
        "Distinguishability of ideal-Bayesian hypotheses vs. context length\n"
        "|log p(x | grid) − log p(x | Ham-ring)|"
    ),
    colors: dict[str, str] | None = None,
) -> None:
    """Plot |log-prob distance| per context length for one or more walks.

    Uses log-x and log-y axes: the linear-in-log-log regime indicates
    a power-law evidence accumulation rate.  A pure Bayesian observer
    on an ergodic walk has per-step KL > 0, so on linear axes the
    distance grows linearly in T; log-log makes it easier to see
    deviations from that expectation at small T.
    """
    default_colors = {"grid": "#1565C0", "ring": "#C62828"}
    if colors is not None:
        default_colors.update(colors)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in distances.items():
        T = np.arange(1, len(curve) + 1)
        ax.plot(
            T, curve,
            color=default_colors.get(name, "gray"),
            lw=2, label=f"walk sampled from {name.capitalize()}",
        )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Context length T")
    ax.set_ylabel("|log p(x | grid) − log p(x | Ham-ring)|")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_log_odds_curves(
    curves: dict[str, np.ndarray],
    out_path: str,
    title: str = (
        "Ideal Bayesian log-odds vs. context length\n"
        "(shared 16-word vocabulary, different edge structure)"
    ),
    colors: dict[str, str] | None = None,
    log_x: bool = False,
) -> None:
    """Plot log-odds(grid/ring) curves for one or more walks.

    `curves` is a mapping from a walk label (e.g. "grid", "ring") to its
    per-step log-odds array (as returned by `log_odds_curve`).  One curve
    is drawn per entry; `out_path` is written as a 150-dpi PNG.  Set
    ``log_x=True`` to put the context-length axis on a log scale.
    """
    default_colors = {"grid": "#1565C0", "ring": "#C62828"}
    if colors is not None:
        default_colors.update(colors)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, curve in curves.items():
        T = len(curve) + 1  # curve is length T-1 (skips the first token)
        expected = "> 0" if name == "grid" else "< 0" if name == "ring" else None
        label = f"walk sampled from {name.capitalize()}"
        if expected is not None:
            label += f"  (expect {expected})"
        ax.plot(
            range(1, T), curve,
            color=default_colors.get(name, "gray"),
            lw=2, label=label,
        )

    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("Context length T")
    ax.set_ylabel("log p(G=grid | x) / p(G=ring | x)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3, which="both")
    if log_x:
        ax.set_xscale("log")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_llm_accuracy_curve(
    json_path: str,
    rho: float = 0.0,
    graph: str = "grid",
    min_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``P(next ∈ valid neighbors)`` curve from vocabulary_tl results.

    Returns (context_lengths, means, sems) sorted by context length.
    Eval points with fewer than ``min_samples`` sequences are dropped.
    """
    with open(json_path) as f:
        raw = json.load(f)
    per_len = raw[str(rho)][graph]
    lengths, means, sems = [], [], []
    for L, vals in per_len.items():
        if len(vals) < min_samples:
            continue
        arr = np.asarray(vals, dtype=float)
        lengths.append(int(L))
        means.append(arr.mean())
        sems.append(arr.std() / max(np.sqrt(len(arr)), 1))
    order = np.argsort(lengths)
    return (
        np.asarray(lengths)[order],
        np.asarray(means)[order],
        np.asarray(sems)[order],
    )


def plot_llm_bayesian_overlay(
    llm_json_path: str,
    bayesian_log_odds: dict[str, np.ndarray],
    out_path: str,
    llm_rho: float = 0.0,
    llm_graph: str = "grid",
    llm_label: str = "Llama-3.1-8B (neutral_disjoint, ρ=0)",
) -> None:
    """Two-panel view, both log-x: LLM accuracy curve + ideal-Bayesian log-odds.

    Left panel  — LLM ``P(next ∈ grid neighbors)`` for pure-grid (ρ=0).
    Right panel — Ideal Bayesian ``log p(grid|x) / p(Ham-ring|x)`` for
                  walks sampled from each graph.  Shows the evidence-
                  accumulation trajectory of the optimal observer.

    We deliberately do *not* overlay the two on the same y-axis: because
    every Hamiltonian-ring edge is also a grid edge, the Bayesian's
    posterior-predictive ``P(next ∈ grid nbrs)`` on a grid walk is
    identically 1.0 (both hypotheses predict only grid neighbors).  The
    informative ideal-observer signal is therefore in the log-odds
    trajectory, not the predictive accuracy.
    """
    L, mean, sem = load_llm_accuracy_curve(llm_json_path, rho=llm_rho, graph=llm_graph)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: LLM accuracy ──
    ax = axes[0]
    ax.errorbar(
        L, mean, yerr=sem, fmt="o-", color="#1976D2", lw=2, ms=5, capsize=3,
        label=llm_label,
    )
    ax.axhline(1.0, color="k", lw=0.7, alpha=0.4, label="ideal upper bound")
    ax.set_xscale("log")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("P(next token ∈ grid neighbors)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("LLM accuracy on pure-grid walks")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # ── Panel 2: Bayesian log-odds ──
    ax = axes[1]
    colors = {"grid": "#1565C0", "ring": "#C62828"}
    for name, curve in bayesian_log_odds.items():
        T = np.arange(1, len(curve) + 1)
        expected = "> 0" if name == "grid" else "< 0" if name == "ring" else None
        label = f"walk from {name.capitalize()}"
        if expected is not None:
            label += f" (expect {expected})"
        ax.plot(T, curve, color=colors.get(name, "gray"), lw=2, label=label)
    ax.axhline(0, color="k", lw=0.7, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("log p(G=grid | x) / p(G=Ham-ring | x)")
    ax.set_title("Ideal Bayesian log-odds\n({grid, Hamiltonian-ring} candidates)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Llama-3.1-8B vs. ideal Bayesian observer  "
        "(shared 16-word vocab; Ham-ring ⊂ grid edges)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    set_seed(0)
    grid = Grid()
    ring = Ring(words=hamiltonian_ring_order(grid))

    # Edge-set diagnostics: every Ham-ring edge IS a grid edge, so the only
    # way to distinguish the two hypotheses is when a grid walk uses a
    # "chord" edge (grid edge not on the cycle).  There are 8 such chords.
    grid_edges = {
        frozenset((w, n)) for w in grid.words for n in grid.get_valid_next_words(w)
    }
    ring_edges = {
        frozenset((w, n)) for w in ring.words for n in ring.get_valid_next_words(w)
    }
    print(f"grid vocab ({len(grid.words)}): {grid.words}")
    print(f"ring vocab ({len(ring.words)}): {ring.words}")
    print(
        f"|E_grid|={len(grid_edges)}  |E_ring|={len(ring_edges)}  "
        f"|E_grid ∩ E_ring|={len(grid_edges & ring_edges)}  "
        f"|E_grid \\ E_ring|={len(grid_edges - ring_edges)} (grid-only chord edges)"
    )

    clf = BayesianGraphClassifier([
        GraphLikelihood("grid", grid),
        GraphLikelihood("ring", ring),
    ])

    T = 2000  # match the longest LLM eval context
    walks = {
        "grid": grid.generate_sequence(seq_len=T),
        "ring": ring.generate_sequence(seq_len=T),
    }
    curves = {
        name: clf.log_odds_curve(walk, "grid", "ring")
        for name, walk in walks.items()
    }

    for name, curve in curves.items():
        pred, post = clf.classify(walks[name])
        print(
            f"true={name:4s}  predicted={pred:4s}  "
            f"p(grid|x)={post[0]:.3f}  p(ring|x)={post[1]:.3f}  "
            f"log-odds(grid/ring)={curve[-1]:+.2f}"
        )

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    # Log-odds-only plot on log-x (all context lengths visible on one figure).
    log_odds_path = os.path.join(results_dir, "bayesian_model_sanity_check.png")
    plot_log_odds_curves(
        curves, log_odds_path,
        title=(
            "Ideal Bayesian log-odds vs. context length\n"
            "(grid vs. Hamiltonian-ring — shared 16-word vocab, Ham-ring ⊂ grid edges)"
        ),
        log_x=True,
    )
    print(f"\nSaved {log_odds_path}")

    # Distinguishability distance (abs log-prob gap) for both walks.
    distances = {
        name: log_prob_distance_curve(clf, walk, "grid", "ring")
        for name, walk in walks.items()
    }
    distance_path = os.path.join(results_dir, "bayesian_log_prob_distance.png")
    plot_log_prob_distance(distances, distance_path)
    print(f"Saved {distance_path}")

    llm_json = os.path.join(results_dir, "vocabulary_tl", "neutral_disjoint.json")
    overlay_path = os.path.join(results_dir, "bayesian_llm_overlay.png")
    plot_llm_bayesian_overlay(
        llm_json_path=llm_json,
        bayesian_log_odds=curves,
        out_path=overlay_path,
        llm_rho=0.0,
        llm_graph="grid",
        llm_label="Llama-3.1-8B\n(neutral_disjoint, ρ=0, pure grid)",
    )
    print(f"Saved {overlay_path}")

    # ── sign sanity checks ──
    grid_final = curves["grid"][-1]
    ring_final = curves["ring"][-1]
    assert grid_final > 0, (
        f"Grid walk should favour grid but log-odds(grid/ring)={grid_final:+.2f}"
    )
    assert ring_final < 0, (
        f"Ring walk should favour ring but log-odds(grid/ring)={ring_final:+.2f}"
    )
    print("PASS: signs match expectations. Implementation is correct.")
