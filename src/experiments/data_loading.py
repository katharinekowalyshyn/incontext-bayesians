"""Shared data-loading, splits, and graph helpers for the Bayesian fits.

This module is consumed by:
  - src/experiments/fit_baseline.py  (Tim, Milestone 2)
  - src/experiments/fit_upgrade.py   (Dan,  Milestones 3 & 4)

Design notes
------------
* We fit against ``src/initial_experiments/results/vocabulary_tl/{condition}.json``
  which stores ``results[ρ][graph][L] = list of per-walk accuracies`` with
  accuracy = P(next token ∈ valid graph neighbors) from a Llama-3.1-8B forward
  pass.  See ``vocabulary_tl_experiment.py`` for the generator.

* ``p0`` is estimated empirically per (condition, ρ, graph) from the earliest
  context lengths, following Checkpoint-2 §3.1.

* Train/val/test split is deterministic by walk index.  With 16 walks we use
  8/4/4 exactly per Checkpoint-2 §2.2; with 12 walks we use the proportional
  6/3/3.  The split is applied inside each (condition, ρ, graph) cell so a fit
  never sees held-out walks.

* ``mst_complexity_bits`` is imported from the primary model module so all of
  the repo agrees on the definition of ``C_MST(G)``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
INITIAL = os.path.join(REPO, "src", "initial_experiments")
if INITIAL not in sys.path:
    sys.path.insert(0, INITIAL)

from bayesian_model import edge_complexity_bits  # noqa: E402
from graphs import (  # noqa: E402
    RING_DISJOINT_16, RING_OVERLAP_16,
)
from sanity_check import Grid  # noqa: E402

DATA_DIR = os.path.join(INITIAL, "results", "vocabulary_tl")
RESULTS_ROOT = os.path.join(HERE, "results")

# Both graphs have 16 nodes so every (condition, ρ) cell produces 16 walks.
GRID_N = 16
RING_N = 16

CONDITIONS: list[str] = [
    "disjoint",
    "overlap",
]

CONDITION_RING_WORDS: dict[str, list[str]] = {
    "disjoint": list(RING_DISJOINT_16),
    "overlap":  list(RING_OVERLAP_16),
}


# ─── Data loading ────────────────────────────────────────────────────────────

@dataclass
class Cell:
    """All per-walk observations for one (condition, ρ, graph) triple.

    Attributes
    ----------
    condition, rho, graph : identifiers
    L   : sorted array of context lengths that have ≥1 observation
    acc : (n_walks, len(L)) array of accuracies; NaN where a walk had no token
          on this graph at that L (happens for mixed ρ because which tokens
          fall in the last ``segment_len`` window is stochastic).
    n_walks : number of walks in this cell (rows of ``acc``).
    p0 : empirical p̂(N ≤ p0_threshold) across train walks (set after split).
    """

    condition: str
    rho: float
    graph: str
    L: np.ndarray
    acc: np.ndarray
    n_walks: int
    p0: float | None = None
    train_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    val_idx:   np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    test_idx:  np.ndarray = field(default_factory=lambda: np.zeros(0, int))


def load_condition(condition: str, data_dir: str = DATA_DIR) -> dict:
    """Return ``{ρ: {graph: {L: list[float]}}}`` with numeric keys."""
    path = os.path.join(data_dir, f"{condition}.json")
    with open(path) as f:
        raw = json.load(f)
    return {
        float(rho_str): {
            graph: {int(L): list(vals) for L, vals in accs.items()}
            for graph, accs in data.items()
        }
        for rho_str, data in raw.items()
    }


def to_cells(condition: str, data: dict | None = None, *,
             graphs: Iterable[str] = ("grid", "ring"),
             min_walks: int = 1,
             data_dir: str = DATA_DIR) -> list[Cell]:
    """Pack a condition's raw dict into one :class:`Cell` per (ρ, graph).

    Rows are aligned across L by walk index (row k = walk k across all L).  A
    cell with ``n_walks < min_walks`` is skipped.
    """
    if data is None:
        data = load_condition(condition, data_dir)

    cells: list[Cell] = []
    for rho in sorted(data.keys()):
        for graph in graphs:
            per_L = data[rho].get(graph, {})
            if not per_L:
                continue
            Ls = sorted(per_L.keys())
            # Determine n_walks = max across L (some L may have <max if
            # truncated; pad with NaN).
            n_walks = max((len(v) for v in per_L.values()), default=0)
            if n_walks < min_walks:
                continue
            acc = np.full((n_walks, len(Ls)), np.nan)
            for j, L in enumerate(Ls):
                vals = per_L[L]
                acc[: len(vals), j] = vals
            cells.append(Cell(
                condition=condition,
                rho=float(rho),
                graph=graph,
                L=np.asarray(Ls, dtype=float),
                acc=acc,
                n_walks=n_walks,
            ))
    return cells


# ─── Train/val/test split ────────────────────────────────────────────────────

def split_walks(n_walks: int, *, ratios: tuple[float, float, float] = (0.5, 0.25, 0.25)
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic split over walk indices.

    Matches Checkpoint-2 §2.2 at n_walks=16 (8/4/4) and degrades proportionally.
    """
    if n_walks <= 0:
        z = np.zeros(0, dtype=int)
        return z, z, z
    n_train = int(round(n_walks * ratios[0]))
    n_val   = int(round(n_walks * ratios[1]))
    n_test  = n_walks - n_train - n_val
    if n_test < 0:
        n_train += n_test
        n_test = 0
    idx = np.arange(n_walks)
    return (idx[:n_train],
            idx[n_train:n_train + n_val],
            idx[n_train + n_val:n_train + n_val + n_test])


def apply_split(cell: Cell, ratios: tuple[float, float, float] = (0.5, 0.25, 0.25)
                ) -> Cell:
    cell.train_idx, cell.val_idx, cell.test_idx = split_walks(cell.n_walks, ratios=ratios)
    return cell


# ─── p0 estimation (Checkpoint-2 §3.1) ───────────────────────────────────────

def estimate_p0(cell: Cell, *, max_L: float = 100.0,
                walk_idx: np.ndarray | None = None) -> float:
    """Empirical pre-transition accuracy.

    Averaged over all (walk, L) pairs with L ≤ ``max_L`` and a non-NaN value.
    If no such pair exists (e.g. because the minimum L is above ``max_L``),
    falls back to the minimum-L column mean.
    """
    if walk_idx is None:
        walk_idx = cell.train_idx if cell.train_idx.size else np.arange(cell.n_walks)
    if walk_idx.size == 0:
        return float("nan")
    L_mask = cell.L <= max_L
    sub = cell.acc[np.ix_(walk_idx, L_mask)] if L_mask.any() else \
          cell.acc[walk_idx, :1]
    vals = sub[~np.isnan(sub)]
    if vals.size == 0:
        vals = cell.acc[walk_idx, 0]
        vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


# ─── Graph adjacencies + MST complexity ──────────────────────────────────────

def _cycle_adjacency(n: int) -> np.ndarray:
    """Unweighted adjacency of an ``n``-node cycle."""
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, (i - 1) % n] = 1.0
        A[i, (i + 1) % n] = 1.0
    return A


def graph_adjacency(graph: str, *, condition: str = "neutral_disjoint") -> np.ndarray:
    """Adjacency matrix for the underlying graph structure.

    Sizes are fixed to the Checkpoint-2 spec (``GRID_N=16``, ``RING_N=12``)
    rather than read from the current ``graphs.py`` vocabularies.  ``condition``
    is plumbed through only for forward-compatibility with vocabulary-aware
    helpers (e.g. colour maps).
    """
    if graph == "grid":
        return Grid().build_adjacency_matrix().astype(float)
    if graph == "ring":
        return _cycle_adjacency(RING_N)
    raise ValueError(f"unknown graph: {graph!r}")


def c_mst(graph: str, condition: str = "disjoint") -> float:
    """Graph complexity (bits) used by the Upgrade prior.

    Historically ``C_MST(G) = |E_MST(G)| · ⌈log₂|V|⌉`` (Checkpoint-2 §4.3 v1),
    but that measure only depends on |V| for connected graphs and can't
    distinguish a 16-node ring from a 16-node grid.  After the
    two-condition refactor we use the full edge-list MDL instead:

        C_edges(G) = |E(G)| · ⌈log₂|V|⌉

    The helper keeps its old name for backward-compatibility in callers
    that already imported ``c_mst``; only the definition changed.
    """
    return edge_complexity_bits(graph_adjacency(graph, condition=condition))


# ─── Observations → flat arrays for optimisation ─────────────────────────────

def flatten(cell: Cell, walk_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (N, y) flattened over selected walks, dropping NaNs."""
    sub = cell.acc[walk_idx, :]
    L_col = np.broadcast_to(cell.L, sub.shape)
    mask = ~np.isnan(sub)
    return L_col[mask].astype(float), sub[mask].astype(float)


# ─── Misc utilities ──────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == "__main__":
    # quick smoke test: print C_MST for grid and ring, and a one-line summary
    # of each condition's cells.
    print(f"C_MST(grid) = {c_mst('grid'):.2f} bits")
    print(f"C_MST(ring) = {c_mst('ring'):.2f} bits")
    for cond in CONDITIONS:
        try:
            cells = to_cells(cond)
        except FileNotFoundError as err:
            print(f"{cond}: no data ({err})")
            continue
        for cell in cells:
            apply_split(cell)
            cell.p0 = estimate_p0(cell)
            print(f"{cond:>18}  ρ={cell.rho:.2f}  {cell.graph:>4}  "
                  f"n_walks={cell.n_walks:>2}  L={len(cell.L):>2}  p0={cell.p0:.3f}")
