"""Representation-level analysis of in-context grid/ring walks.

Reproduces and extends Park et al. (ICLR 2025, "In-context learning of
representations"):

    1. PCA of Llama-3.1-8B class-mean activations at multiple context
       lengths (T=200, 400, 1400), using the paper's sliding window of
       Nw=50 tokens and a per-snapshot PCA basis (optionally a shared
       basis for trajectory analysis).

    2. Normalized Dirichlet energy E_G(H) vs. context length, where
       E_G(H) = Σ_{ij} A_{ij} ||h_i - h_j||^2 and H is the matrix of
       class-mean activations. The paper predicts E_G should *decrease*
       as context grows and the representations re-organize to match
       the in-context graph.

    3. Paper Figure 3 reproduction for the ``months_permuted`` condition:
       PC1/PC2 shows centroids ordered along the natural calendar ring
       (pretrained semantic prior), while PC3/PC4 shows them ordered
       along the in-context permuted ring.

    4. Theoretical spectral embedding (Thm 5.1): eigenvectors
       (u_2, u_3, ...) of the graph Laplacian L_G = D - A, which
       minimize the Dirichlet energy subject to a unit-norm constraint.
       We plot these as a reference for what the top PCs "should" look
       like once the representations have fully re-organized, and
       overlay a Procrustes-aligned copy on the PCA scatter.

Two entry points (see ``--help``):

    python src/initial_experiments/pca_analysis.py
        # → cached "after" PCA (PC1/PC2 + PC3/PC4) from
        #   iclr_induction-main/01_reproduce.py artefacts, plus
        #   spectral-embedding reference figures (no GPU needed).

    python src/initial_experiments/pca_analysis.py --with-model [--condition C]
        # → full pipeline:
        #     * per-snapshot PCA at T=200/400/1400 (paper Fig 2),
        #     * Dirichlet-energy curve (paper Fig 4),
        #     * shared-basis 3-phase trajectory plot,
        #     * Fig 3 reproduction if --condition months_permuted.
        # Requires Llama-3.1-8B (~16 GB GPU RAM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sanity_check import WORDS as GRID_WORDS, set_seed, WORD_TO_COLOR
from graphs import (
    MONTHS, MONTHS_PERMUTED, MONTH_TO_COLOR,
    RING_WORDS, RING_WORDS_OVERLAP,
    RING_WORD_TO_COLOR, OVERLAP_WORD_TO_COLOR,
    Ring,
)

# Hamiltonian-cycle constructor is shared with bayesian_model.py; reuse.
from bayesian_model import hamiltonian_ring_order  # noqa: E402


# ── Canonical graphs ──────────────────────────────────────────────────────────

class CanonicalGrid:
    """Non-shuffling 4×4 grid matching ``iclr_induction-main/utils.Grid``.

    The ICLR repo's Grid lays ``WORDS`` out in list order
    (``apple, bird, car, egg`` in row 0, etc.), whereas
    ``sanity_check.Grid`` shuffles on init. Our cached activations come
    from the former, so we need the former's adjacency whenever we draw
    edges over a PCA scatter or compute theoretical quantities.
    """

    def __init__(self, words: Sequence[str] = GRID_WORDS, rows: int = 4, cols: int = 4):
        assert rows * cols == len(words), (
            f"Grid {rows}x{cols}={rows*cols} does not match {len(words)} words"
        )
        self.words = list(words)
        self.rows = rows
        self.cols = cols
        # Mirror sanity_check.Grid's attributes so hamiltonian_ring_order works.
        self.grid = np.array(self.words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(self.words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(self.words)}

    def build_adjacency_matrix(self) -> np.ndarray:
        n = len(self.words)
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            r, c = divmod(i, self.cols)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < self.rows and 0 <= c2 < self.cols:
                    A[i, r2 * self.cols + c2] = 1
        return A

    def get_valid_next_words(self, word: str) -> list[str]:
        i = self.words.index(word)
        r, c = divmod(i, self.cols)
        nbrs: list[str] = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < self.rows and 0 <= c2 < self.cols:
                nbrs.append(self.words[r2 * self.cols + c2])
        return nbrs


def _generate_walk_on_graph(graph, seq_len: int) -> list[str]:
    """Uniform random walk starting from a random node."""
    cur = str(np.random.choice(graph.words))
    seq = [cur]
    while len(seq) < seq_len:
        cur = str(np.random.choice(graph.get_valid_next_words(cur)))
        seq.append(cur)
    return seq


# ── Paths & paper-aligned constants ───────────────────────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(REPO_ROOT, "iclr_induction-main", "results", "reproduce", "data")
OUT_DIR = os.path.join(HERE, "results")

# Paper: class means are computed over a sliding window of Nw=50 tokens,
# snapshotted at T=200, 400, 1400.  The corresponding window is the
# trailing 50 positions of the first T tokens (i.e. [T-Nw, T)).
PAPER_WINDOW_SIZE = 50                     # Nw
PAPER_SNAPSHOT_T = (200, 400, 1400)        # Fig 2 / Fig 3 panels

# For Dirichlet-energy curves we sample many T's on a log scale.
DIRICHLET_ENERGY_TS = (
    60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1400,
)


# ── Condition registry ────────────────────────────────────────────────────────
#
# Mirrors the registry in vocabulary_tl_experiment.py so we can do a
# head-to-head comparison between LLM behavioural accuracy and the
# representation-level quantities computed here.

def _make_grid_condition():
    g = CanonicalGrid(words=list(GRID_WORDS))
    return {
        "graph": g,
        "walk_words": list(GRID_WORDS),
        "adjacency": g.build_adjacency_matrix(),
        "word_to_color": WORD_TO_COLOR,
        "label": "4x4 grid",
        "color": "#1976D2",
    }


def _make_ring_condition(words, word_to_color, label, color):
    g = Ring(words=list(words))
    return {
        "graph": g,
        "walk_words": list(words),
        "adjacency": g.build_adjacency_matrix(),
        "word_to_color": word_to_color,
        "label": label,
        "color": color,
    }


# Deferred-construction factory so that CanonicalGrid etc. are defined above.
def get_conditions() -> dict[str, dict]:
    return {
        "grid":             _make_grid_condition(),
        "months_natural":   _make_ring_condition(
            MONTHS, MONTH_TO_COLOR,
            "Months — natural order",  "#1565C0",
        ),
        "months_permuted":  _make_ring_condition(
            MONTHS_PERMUTED, MONTH_TO_COLOR,
            "Months — permuted",       "#AD1457",
        ),
        "neutral_disjoint": _make_ring_condition(
            RING_WORDS, RING_WORD_TO_COLOR,
            "Neutral ring — disjoint", "#2E7D32",
        ),
        "neutral_overlap":  _make_ring_condition(
            RING_WORDS_OVERLAP, OVERLAP_WORD_TO_COLOR,
            "Neutral ring — overlap",  "#E65100",
        ),
    }

# Exploratory "before / during / after" windows (trajectory mode).
PHASE_WINDOWS_DEFAULT: dict[str, tuple[int, int]] = {
    "before": (10, 100),       # pretrained-dominated regime
    "during": (500, 700),      # structure emerging
    "after":  (1200, 1400),    # structure fully formed
}


# ── Section 1: PCA + sliding-window class means ───────────────────────────────

def compute_top_k_pca(class_means: np.ndarray, k: int = 4) -> np.ndarray:
    """Top-k PCA directions of `class_means` (shape [n_words, d]).

    Output shape: [k, d], row-major — row i is PCi.
    """
    centered = class_means - class_means.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:k]


def compute_class_means_np(
    activations: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
) -> np.ndarray:
    """Per-word mean activation. Shape: [len(words), d_model].

    `activations[i]` is the residual stream at position `i`, with the
    corresponding token `tokens[i]`. Rows for words that never appear
    are returned as zero vectors (matching the cached artefacts and
    ``utils.compute_class_means``).
    """
    assert len(activations) == len(tokens), (
        f"activations ({len(activations)}) and tokens ({len(tokens)}) must align"
    )
    tokens = list(tokens)
    d = activations.shape[1]
    means = np.zeros((len(words), d), dtype=activations.dtype)
    for j, w in enumerate(words):
        idxs = [i for i, t in enumerate(tokens) if t == w]
        if idxs:
            means[j] = activations[idxs].mean(axis=0)
    return means


def class_means_sliding(
    activations: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
    T: int,
    window: int = PAPER_WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper-aligned class means at context length T.

    Uses the trailing ``window`` positions of the first ``T`` tokens,
    i.e. positions ``[T - window, T)``.

    Returns (means, present_mask):
        means        : [len(words), d_model] — rows for un-seen words
                        are zeros (caller should mask them out before
                        fitting PCA or computing Dirichlet energy,
                        otherwise the origin-at-zero points will
                        dominate).
        present_mask : [len(words)] bool array, True iff that word
                        appeared at least once in the window.
    """
    if T > len(activations):
        raise ValueError(f"T={T} exceeds sequence length {len(activations)}")
    if T - window < 0:
        raise ValueError(f"T={T} is shorter than window Nw={window}")
    sl_acts = activations[T - window:T]
    sl_tokens = list(tokens[T - window:T])
    means = compute_class_means_np(sl_acts, sl_tokens, words)
    present = np.array([w in sl_tokens for w in words], dtype=bool)
    return means, present


# ── Section 2: Spectral embedding (Thm 5.1) ───────────────────────────────────

def laplacian_spectral_embedding(
    A: np.ndarray, k: int = 4, normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Graph-Laplacian eigenvectors u_1, ..., u_n (ascending eigenvalue).

    Park et al. (Thm 5.1) show that when class means minimize Dirichlet
    energy subject to orthonormality they are exactly (up to rotation)
    the top non-trivial eigenvectors of the graph Laplacian
    L_G = D - A.  So the top PCs of the class means should match
    (u_2, ..., u_{k+1}) as context length → ∞.

    Args:
        A: Symmetric adjacency matrix, shape [n, n].
        k: Number of non-trivial eigenvectors to return.
        normalized: If True, use the symmetric normalized Laplacian
            L_sym = I - D^{-1/2} A D^{-1/2} (common for clustering).
            Default (raw Laplacian) matches the paper's derivation.

    Returns:
        (eigvals, eigvecs) where
            eigvals : [k] — the eigenvalues λ_2, ..., λ_{k+1} in
                ascending order (skipping λ_1 = 0).
            eigvecs : [n, k] — columns are the corresponding
                eigenvectors, each unit-norm.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), f"A must be square, got {A.shape}"
    d = A.sum(axis=1)
    if normalized:
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        L = np.eye(n) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    else:
        L = np.diag(d) - A
    L = (L + L.T) / 2  # symmetrize against round-off
    eigvals, eigvecs = np.linalg.eigh(L)
    # Drop the trivial eigenvalue (≈0) corresponding to the constant
    # vector; ascending order from np.linalg.eigh already puts it first.
    return eigvals[1:1 + k], eigvecs[:, 1:1 + k]


def procrustes_align(
    source: np.ndarray, target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Best rigid-plus-scale map `source -> target` that minimizes
    Frobenius error.

    Both inputs have shape [n, k] and are assumed already centered to
    their respective means.  Returns (aligned, scale) where
    ``aligned = scale * source @ R`` and R is a k×k orthogonal matrix.

    Used to place a spectral embedding into the coordinate system of a
    PCA scatter (which has an arbitrary sign/rotation).
    """
    assert source.shape == target.shape, (
        f"shape mismatch: {source.shape} vs {target.shape}"
    )
    s_c = source - source.mean(0, keepdims=True)
    t_c = target - target.mean(0, keepdims=True)
    # Optimal rotation via SVD of the cross-covariance.
    U, _, Vt = np.linalg.svd(t_c.T @ s_c, full_matrices=False)
    R = U @ Vt          # k × k orthogonal
    # Optimal scale = trace(Σ) / ||source||_F^2 .
    # Σ is the singular-value diagonal of t_c.T @ s_c; trace is its sum.
    num = np.trace((t_c.T @ s_c) @ R.T)
    den = (s_c ** 2).sum()
    scale = num / den if den > 0 else 1.0
    aligned = scale * (s_c @ R.T) + target.mean(0, keepdims=True)
    return aligned, float(scale)


# ── Section 3: Dirichlet energy ───────────────────────────────────────────────

def dirichlet_energy(H: np.ndarray, A: np.ndarray, normalize: bool = True) -> float:
    """E_G(H) = (1/2) Σ_{ij} A_{ij} ||h_i - h_j||^2 = Tr(H^T L H).

    Args:
        H: [n, d] matrix of node representations (class means).
        A: [n, n] symmetric adjacency.
        normalize: If True, divide by the "degree-weighted variance"
            Σ_i deg_i · ||h_i - mean(H)||^2 = Tr((H-μ)^T D (H-μ)).  This
            yields a unit-free number in [0, λ_max(L_sym)] that does
            not depend on the overall scale of H, matching the "fraction
            of energy" quantity plotted in Park et al. Fig 4.
    """
    H = np.asarray(H, dtype=float)
    A = np.asarray(A, dtype=float)
    n = H.shape[0]
    assert A.shape == (n, n), f"A shape {A.shape} != ({n},{n})"
    D = np.diag(A.sum(axis=1))
    L = D - A
    # Tr(H^T L H) = (1/2) Σ A_ij ||h_i - h_j||^2 since A is symmetric.
    e = float(np.trace(H.T @ L @ H))
    if not normalize:
        return e
    H_c = H - H.mean(axis=0, keepdims=True)
    denom = float(np.trace(H_c.T @ D @ H_c))
    return e / denom if denom > 0 else float("nan")


def dirichlet_energy_curve(
    activations: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
    A: np.ndarray,
    Ts: Sequence[int] = DIRICHLET_ENERGY_TS,
    window: int = PAPER_WINDOW_SIZE,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalized Dirichlet energy E_G(H_T) at each requested T.

    For each T in ``Ts`` we compute class means on the trailing ``window``
    tokens of the first T positions, then return E_G on those means.

    Returns (Ts_kept, energies) — any T < window is silently skipped.
    """
    Ts_kept, energies = [], []
    for T in Ts:
        if T - window < 0 or T > len(activations):
            continue
        H, present = class_means_sliding(activations, tokens, words, T, window=window)
        # Drop absent words: zero-rows would make Dirichlet energy
        # meaningless (they'd appear as a single point at the origin).
        if present.sum() < 2:
            continue
        H_p = H[present]
        A_p = np.asarray(A)[np.ix_(present, present)]
        energies.append(dirichlet_energy(H_p, A_p, normalize=normalize))
        Ts_kept.append(T)
    return np.asarray(Ts_kept), np.asarray(energies)


# ── Section 4: Plotting ───────────────────────────────────────────────────────

def _edges_from_adjacency(A: np.ndarray) -> list[tuple[int, int]]:
    n = A.shape[0]
    return [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j]]


def _draw_scatter(
    ax: plt.Axes,
    projected: np.ndarray,                # [n_words, 2]
    words: Sequence[str],
    A: np.ndarray,                        # [n_words, n_words] — edges to draw
    word_to_color: dict[str, str],
    pc_x: int = 1, pc_y: int = 2,
    title: str | None = None,
    show_labels: bool = True,
    edge_style: dict | None = None,
):
    """One PCA scatter panel: adjacency edges + centroids, labelled by word.

    `words[i]` is the token label of row i in `projected` *and* row i
    in `A` — the caller is responsible for making them aligned.
    """
    edge_style = {"color": "gray", "alpha": 0.35, "linestyle": "--",
                  "linewidth": 0.6, **(edge_style or {})}
    for i, j in _edges_from_adjacency(A):
        ax.plot(
            [projected[i, 0], projected[j, 0]],
            [projected[i, 1], projected[j, 1]],
            **edge_style,
        )
    for i, w in enumerate(words):
        ax.scatter(
            projected[i, 0], projected[i, 1],
            color=word_to_color.get(w, "gray"), s=140, marker="*",
            edgecolors="black", linewidths=0.5, zorder=5,
        )
        if show_labels:
            ax.annotate(
                w, (projected[i, 0], projected[i, 1]),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
            )
    ax.set_xlabel(f"PC{pc_x}")
    ax.set_ylabel(f"PC{pc_y}")
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)


def plot_pca_two_planes(
    class_means: np.ndarray,
    words: Sequence[str],
    grid: CanonicalGrid,
    out_path: str,
    phase_label: str = "after context",
    pca_dirs: np.ndarray | None = None,
    overlay_spectral: bool = False,
) -> np.ndarray:
    """PC1/PC2 and PC3/PC4 scatters side-by-side for a single phase.

    If ``overlay_spectral`` is True, we overlay the grid's spectral
    embedding (u_2, u_3) on the PC1/PC2 panel and (u_4, u_5) on the
    PC3/PC4 panel, Procrustes-aligned to the PCA centroids.
    """
    if pca_dirs is None:
        pca_dirs = compute_top_k_pca(class_means, k=4)
    elif pca_dirs.shape[0] < 4:
        raise ValueError(f"need 4 PCA dirs, got {pca_dirs.shape[0]}")

    projected = class_means @ pca_dirs.T  # [n_words, 4]
    A = grid.build_adjacency_matrix()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _draw_scatter(
        axes[0], projected[:, :2], words, A, WORD_TO_COLOR,
        pc_x=1, pc_y=2, title=f"PC1 vs PC2 — {phase_label}",
    )
    _draw_scatter(
        axes[1], projected[:, 2:4], words, A, WORD_TO_COLOR,
        pc_x=3, pc_y=4, title=f"PC3 vs PC4 — {phase_label}",
    )
    if overlay_spectral:
        _, specs = laplacian_spectral_embedding(A, k=5)  # u2..u6
        aligned_12, _ = procrustes_align(specs[:, :2], projected[:, :2])
        aligned_34, _ = procrustes_align(specs[:, 2:4], projected[:, 2:4])
        for ax, aligned in [(axes[0], aligned_12), (axes[1], aligned_34)]:
            ax.scatter(
                aligned[:, 0], aligned[:, 1],
                marker="o", facecolors="none", edgecolors="#d81b60",
                s=200, linewidths=1.5, zorder=4,
                label="spectral embedding (Procrustes-aligned)",
            )
            for i, j in _edges_from_adjacency(A):
                ax.plot(
                    [aligned[i, 0], aligned[j, 0]],
                    [aligned[i, 1], aligned[j, 1]],
                    color="#d81b60", alpha=0.25, linewidth=0.6,
                )
            ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Llama-3.1-8B class-mean activations, layer 26 "
        "(Park et al. ICLR 2025 Fig 2 extended to PC3/PC4)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pca_dirs


def plot_pca_phase_grid(
    class_means_by_phase: dict[str, np.ndarray],
    pca_dirs: np.ndarray,
    words: Sequence[str],
    A: np.ndarray,
    word_to_color: dict[str, str],
    out_path: str,
) -> None:
    """3-phase × 2-plane PCA scatters on a *shared* PCA basis.

    Using the same basis for every phase lets you track each word
    centroid's trajectory from "before" (pretrained dominated) to
    "after" (graph-structured).
    """
    A = np.asarray(A)
    phases = list(class_means_by_phase)
    n_phases = len(phases)
    fig, axes = plt.subplots(
        n_phases, 2, figsize=(11, 5.2 * n_phases), squeeze=False,
    )
    for row, phase in enumerate(phases):
        projected = class_means_by_phase[phase] @ pca_dirs.T
        _draw_scatter(
            axes[row, 0], projected[:, :2], words, A, word_to_color,
            pc_x=1, pc_y=2, title=f"PC1 vs PC2 — {phase}",
            show_labels=(row == 0),
        )
        _draw_scatter(
            axes[row, 1], projected[:, 2:4], words, A, word_to_color,
            pc_x=3, pc_y=4, title=f"PC3 vs PC4 — {phase}",
            show_labels=(row == 0),
        )
    fig.suptitle(
        "Grid-structure emergence in Llama-3.1-8B representations\n"
        "(shared PCA basis across phases; layer 26 class means)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_snapshots_paper(
    class_means_by_T: dict[int, np.ndarray],
    words: Sequence[str],
    A: np.ndarray,
    word_to_color: dict[str, str],
    out_path: str,
    title_prefix: str = "Llama-3.1-8B class-mean PCA",
    overlay_spectral: bool = False,
    present_mask_by_T: dict[int, np.ndarray] | None = None,
) -> None:
    """Paper-aligned Fig 2: per-snapshot PCA, columns are T's, rows are (PC1/PC2, PC3/PC4).

    Each column fits its *own* PCA on that snapshot's class means
    (re-orients the axes independently for each T), so the reader sees
    how the low-d structure "opens up" as context accumulates.

    If ``present_mask_by_T`` is given, rows where the mask is False
    (word unseen in that snapshot's window) are dropped before PCA
    and from plotting — otherwise their zero-rows would pin the
    origin and collapse the scatter.
    """
    Ts = sorted(class_means_by_T)
    n_T = len(Ts)
    fig, axes = plt.subplots(
        2, n_T, figsize=(4.2 * n_T, 8.5), squeeze=False,
    )
    A = np.asarray(A)
    spec_full = None
    if overlay_spectral:
        _, spec_full = laplacian_spectral_embedding(A, k=5)

    for col, T in enumerate(Ts):
        H_full = class_means_by_T[T]
        if present_mask_by_T is not None and T in present_mask_by_T:
            present = present_mask_by_T[T]
        else:
            # fall back to non-zero-row detection
            present = np.linalg.norm(H_full, axis=1) > 0
        n_present = int(present.sum())
        if n_present < 3:
            for row in range(2):
                ax = axes[row, col]
                ax.text(0.5, 0.5, f"T={T}: only {n_present} words seen",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue
        H = H_full[present]
        words_p = [w for w, ok in zip(words, present) if ok]
        A_p = A[np.ix_(present, present)]
        pca_dirs = compute_top_k_pca(H, k=4)
        proj = H @ pca_dirs.T
        for row, (pc_lo, pc_hi) in enumerate([(0, 2), (2, 4)]):
            ax = axes[row, col]
            n_missing = len(words) - n_present
            tag = "" if n_missing == 0 else f"  ({n_missing} words unseen)"
            _draw_scatter(
                ax, proj[:, pc_lo:pc_hi], words_p, A_p, word_to_color,
                pc_x=pc_lo + 1, pc_y=pc_lo + 2,
                title=f"T = {T}  —  PC{pc_lo+1} vs PC{pc_lo+2}{tag}",
                show_labels=(col == 0),
            )
            if overlay_spectral:
                spec_p = spec_full[present]
                aligned, _ = procrustes_align(spec_p[:, pc_lo:pc_hi], proj[:, pc_lo:pc_hi])
                ax.scatter(
                    aligned[:, 0], aligned[:, 1],
                    marker="o", facecolors="none", edgecolors="#d81b60",
                    s=180, linewidths=1.2, zorder=4,
                )
                for i, j in _edges_from_adjacency(A_p):
                    ax.plot(
                        [aligned[i, 0], aligned[j, 0]],
                        [aligned[i, 1], aligned[j, 1]],
                        color="#d81b60", alpha=0.2, linewidth=0.5,
                    )
    fig.suptitle(
        f"{title_prefix}  —  per-snapshot PCA basis, Nw={PAPER_WINDOW_SIZE} sliding window",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dirichlet_energy_curve(
    Ts: np.ndarray,
    energies: np.ndarray,               # mean curve, shape [n_T]
    out_path: str,
    title: str = "Normalized Dirichlet energy vs. context length",
    label: str = "grid",
    color: str = "#1565C0",
    extra_curves: dict | None = None,
    stds: np.ndarray | None = None,     # optional ±1σ band for main curve
) -> None:
    """Plot E_G(H_T) / (H-centered degree-weighted variance) vs. T on log-x.

    `extra_curves` maps label -> (Ts, energies, color) OR
    label -> (Ts, energies, color, stds) for overlaying multiple
    graph hypotheses / conditions with optional ±1σ bands.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    def _plot_one(t, e, c, lab, s=None, marker="o"):
        ax.plot(t, e, f"{marker}-", color=c, lw=2, ms=5, label=lab)
        if s is not None:
            ax.fill_between(t, e - s, e + s, color=c, alpha=0.18, lw=0)

    _plot_one(Ts, energies, color, f"E_{label}", s=stds)
    if extra_curves:
        markers = iter(["s", "^", "D", "v", "P", "X"])
        for lab, tup in extra_curves.items():
            t2, e2, c2 = tup[0], tup[1], tup[2]
            s2 = tup[3] if len(tup) > 3 else None
            _plot_one(t2, e2, c2, f"E_{lab}", s=s2, marker=next(markers))
    ax.set_xscale("log")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Normalized Dirichlet energy  E_G(H) / Tr(H_c^T D H_c)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_vs_accuracy(
    Ts: np.ndarray,                         # [n_T]
    energies_per_walk: np.ndarray,          # [n_walks, n_T]
    accuracies_per_walk: np.ndarray,        # [n_walks, n_T]
    out_path: str,
    condition: str,
    color: str = "#1565C0",
) -> None:
    """Scatter of Llama accuracy vs. normalised Dirichlet energy.

    Each marker is one (walk, context-length) pair.  Markers are
    coloured by T on a log-scale colormap so the reader can see the
    trajectory: as T grows, points should move toward (low E, high P).

    Also computes per-T Spearman + Pearson correlations across walks.
    """
    n_walks, n_T = energies_per_walk.shape
    assert accuracies_per_walk.shape == (n_walks, n_T)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: scatter coloured by T ──
    ax = axes[0]
    # Repeat T along walks so each row=walk, col=T has the right T label
    T_mat = np.broadcast_to(np.asarray(Ts), (n_walks, n_T))
    sc = ax.scatter(
        energies_per_walk.ravel(), accuracies_per_walk.ravel(),
        c=T_mat.ravel(), cmap="viridis", norm=plt.matplotlib.colors.LogNorm(),
        s=40, edgecolors="black", linewidths=0.3, alpha=0.85,
    )
    # Connect per-walk trajectories with thin lines so the reader sees the sweep.
    for w in range(n_walks):
        ax.plot(
            energies_per_walk[w], accuracies_per_walk[w],
            color="gray", alpha=0.3, lw=0.6, zorder=1,
        )
    # Per-T means (bold markers).
    e_mean = energies_per_walk.mean(axis=0)
    a_mean = accuracies_per_walk.mean(axis=0)
    ax.plot(
        e_mean, a_mean,
        "*-", color=color, ms=12, lw=2, mec="black", mew=0.6, zorder=5,
        label="per-T mean",
    )
    cb = fig.colorbar(sc, ax=ax, label="Context length T")
    cb.ax.tick_params(labelsize=8)
    ax.set_xlabel("Normalized Dirichlet energy  E_G(H_T)")
    ax.set_ylabel("Llama  P(next token ∈ valid neighbors)")
    ax.set_title(f"Representation alignment vs. behavioural accuracy\n"
                 f"({condition}; each dot = one walk × one T)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # ── Right: per-T correlation bars ──
    ax = axes[1]
    try:
        from scipy.stats import spearmanr, pearsonr
        have_scipy = True
    except ImportError:
        have_scipy = False
    pearson, spearman = [], []
    for i in range(n_T):
        e_i = energies_per_walk[:, i]
        a_i = accuracies_per_walk[:, i]
        if np.std(e_i) == 0 or np.std(a_i) == 0 or n_walks < 3:
            pearson.append(np.nan); spearman.append(np.nan); continue
        if have_scipy:
            pearson.append(pearsonr(e_i, a_i).statistic)
            spearman.append(spearmanr(e_i, a_i).statistic)
        else:
            pearson.append(float(np.corrcoef(e_i, a_i)[0, 1]))
            spearman.append(np.nan)
    x = np.arange(n_T)
    width = 0.38
    ax.bar(x - width/2, pearson, width, label="Pearson", color="#1976D2")
    ax.bar(x + width/2, spearman, width, label="Spearman", color="#D81B60")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in Ts], rotation=45, fontsize=8)
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Correlation between E_G and accuracy across walks")
    ax.set_title("Cross-walk correlation of E_G with Llama accuracy")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spectral_embedding_reference(
    graphs: dict[str, tuple[Sequence[str], np.ndarray, dict[str, str]]],
    out_path: str,
    title: str = "Graph Laplacian spectral embeddings (theoretical prediction)",
) -> None:
    """Standalone reference figure: u_2/u_3 and u_4/u_5 for each named graph.

    Args:
        graphs: mapping name -> (words, A, word_to_color).  The rows of
            A and the entries of words must align.
    """
    names = list(graphs)
    n = len(names)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.5 * n), squeeze=False)
    for row, name in enumerate(names):
        words, A, colors = graphs[name]
        _, specs = laplacian_spectral_embedding(A, k=5)
        _draw_scatter(
            axes[row, 0], specs[:, :2], words, A, colors,
            pc_x=2, pc_y=3, title=f"{name}:  u_2 vs u_3",
            edge_style=dict(color="gray", alpha=0.5, linestyle="-",
                            linewidth=0.8),
        )
        _draw_scatter(
            axes[row, 1], specs[:, 2:4], words, A, colors,
            pc_x=4, pc_y=5, title=f"{name}:  u_4 vs u_5",
            edge_style=dict(color="gray", alpha=0.5, linestyle="-",
                            linewidth=0.8),
        )
    fig.suptitle(title, fontsize=11, y=1.00)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_months_fig3(
    class_means: np.ndarray,
    months_permuted: Sequence[str],          # in-context (walk-defining) order
    months_natural: Sequence[str],           # calendar (semantic) order
    month_to_color: dict[str, str],
    out_path: str,
    T: int,
    present: np.ndarray | None = None,
) -> None:
    """Paper Fig 3 reproduction for ``months_permuted``:

      * Left panel (PC1/PC2): edges drawn along the *natural* calendar
        ring Jan→Feb→…→Dec→Jan.  If the pretrained semantic prior
        dominates the top two PCs this looks like a clean ring.

      * Right panel (PC3/PC4): edges drawn along the *in-context*
        permuted ring.  If the in-context structure has taken over the
        3rd/4th PCs this looks like a clean ring too.

    Assumes ``class_means[i]`` is aligned to ``months_permuted[i]``.
    """
    assert len(class_means) == len(months_permuted) == len(months_natural), (
        f"need 12 months in each input, got {len(class_means)}/"
        f"{len(months_permuted)}/{len(months_natural)}"
    )
    n = len(months_permuted)
    if present is None:
        present = np.linalg.norm(class_means, axis=1) > 0
    if present.sum() < 3:
        raise ValueError(
            f"Only {int(present.sum())} months seen — not enough for a PCA plot."
        )

    months_permuted_p = [m for m, ok in zip(months_permuted, present) if ok]
    class_means_p = class_means[present]
    pca_dirs = compute_top_k_pca(class_means_p, k=4)
    proj = class_means_p @ pca_dirs.T  # [n_p, 4]
    n_p = len(months_permuted_p)

    # Adjacency along the in-context permuted ring, restricted to present months.
    idx_in_perm = {m: i for i, m in enumerate(months_permuted)}
    perm_idx_p = [idx_in_perm[m] for m in months_permuted_p]
    A_perm = np.zeros((n_p, n_p), dtype=np.int8)
    for i, pi in enumerate(perm_idx_p):
        for pj in [(pi - 1) % n, (pi + 1) % n]:
            if pj in perm_idx_p:
                j = perm_idx_p.index(pj)
                A_perm[i, j] = 1

    # Adjacency along the natural calendar ring, rows aligned to the
    # *present-restricted* permuted-order rows.
    idx_in_nat = {m: i for i, m in enumerate(months_natural)}
    nat_idx_p = [idx_in_nat[m] for m in months_permuted_p]
    A_nat = np.zeros((n_p, n_p), dtype=np.int8)
    for i, ni in enumerate(nat_idx_p):
        for nj in [(ni - 1) % n, (ni + 1) % n]:
            if nj in nat_idx_p:
                j = nat_idx_p.index(nj)
                A_nat[i, j] = 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _draw_scatter(
        axes[0], proj[:, :2], months_permuted_p, A_nat, month_to_color,
        pc_x=1, pc_y=2,
        title="PC1 vs PC2 — edges along natural calendar ring\n"
              "(pretrained semantic prior)",
        edge_style=dict(color="#2e7d32", alpha=0.6, linestyle="-", linewidth=1.0),
    )
    _draw_scatter(
        axes[1], proj[:, 2:4], months_permuted_p, A_perm, month_to_color,
        pc_x=3, pc_y=4,
        title="PC3 vs PC4 — edges along in-context permuted ring\n"
              "(re-organised representation)",
        edge_style=dict(color="#1565c0", alpha=0.6, linestyle="-", linewidth=1.0),
    )
    fig.suptitle(
        f"Fig 3 reproduction: months_permuted  "
        f"(T={T} tokens, Llama-3.1-8B layer 26)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Section 5: Entry points ───────────────────────────────────────────────────

def run_from_cached() -> None:
    """PC1/PC2+PC3/PC4 + spectral-embedding reference for cached "after"-context data."""
    pca_npz = os.path.join(CACHE_DIR, "pca.npz")
    seq_json = os.path.join(CACHE_DIR, "sequence.json")
    if not (os.path.exists(pca_npz) and os.path.exists(seq_json)):
        raise FileNotFoundError(
            f"Cached PCA artefacts not found under {CACHE_DIR}. "
            "Run iclr_induction-main/01_reproduce.py first, or pass --with-model."
        )
    data = np.load(pca_npz)
    class_means = data["class_means"]  # [16, 4096]
    print(f"Loaded cached class_means {class_means.shape} from {pca_npz}")

    grid = CanonicalGrid(words=list(GRID_WORDS))
    A_grid = grid.build_adjacency_matrix()

    # PC1/PC2 + PC3/PC4, with spectral-embedding overlay.
    out_path = os.path.join(OUT_DIR, "pca_after_pc1234.png")
    plot_pca_two_planes(
        class_means, list(GRID_WORDS), grid, out_path,
        phase_label="after context (cached, last 200 positions)",
        overlay_spectral=True,
    )
    print(f"Saved {out_path}")

    # Standalone spectral-embedding reference for grid and Hamiltonian ring.
    ham_order = hamiltonian_ring_order(grid)
    # Build ham-ring adjacency in the canonical GRID_WORDS order.
    word_idx = {w: i for i, w in enumerate(GRID_WORDS)}
    n = len(GRID_WORDS)
    A_ham = np.zeros((n, n), dtype=np.int8)
    for k, w in enumerate(ham_order):
        nxt = ham_order[(k + 1) % n]
        i, j = word_idx[w], word_idx[nxt]
        A_ham[i, j] = A_ham[j, i] = 1

    ref_path = os.path.join(OUT_DIR, "spectral_embedding_reference.png")
    plot_spectral_embedding_reference(
        graphs={
            "4×4 grid":           (list(GRID_WORDS), A_grid, WORD_TO_COLOR),
            "Hamiltonian ring":   (list(GRID_WORDS), A_ham,  WORD_TO_COLOR),
        },
        out_path=ref_path,
        title=("Theoretical PCA prediction from Thm 5.1 of Park et al.\n"
               "(eigenvectors u_2..u_5 of the graph Laplacian)"),
    )
    print(f"Saved {ref_path}")

    # Also dump the Dirichlet energy of the cached means under both
    # candidate adjacencies — a one-number sanity check that the
    # cached representation aligns with the grid and not the ring.
    e_grid = dirichlet_energy(class_means, A_grid, normalize=True)
    e_ham = dirichlet_energy(class_means, A_ham,  normalize=True)
    print(f"Dirichlet energy (cached class means):")
    print(f"  under 4×4 grid       : E_normalized = {e_grid:.4f}")
    print(f"  under Hamiltonian rng: E_normalized = {e_ham:.4f}")
    print(f"  (lower => means align with that graph's structure)")


def _build_token_map(model, words: Sequence[str]) -> dict[str, int]:
    """Return {word: first_token_id}.  Prefixed space matches Llama BPE."""
    mapping: dict[str, int] = {}
    multi = []
    for w in words:
        ids = model.tokenizer.encode(" " + w, add_special_tokens=False)
        mapping[w] = ids[0]
        if len(ids) > 1:
            multi.append((w, ids))
    if multi:
        print(f"  WARNING: multi-token words (using first token): "
              f"{[w for w, _ in multi]}")
    return mapping


def _per_position_accuracy(
    probs: np.ndarray,                   # [seq_len-1, vocab]
    sequence: Sequence[str],             # len = seq_len
    graph,
    tok_map: dict[str, int],
) -> np.ndarray:
    """accuracy[t] = P(tokens[t+1] ∈ valid_nbrs(sequence[t])) under probs[t]."""
    n = len(sequence) - 1
    acc = np.zeros(n, dtype=np.float64)
    for t in range(n):
        valid = graph.get_valid_next_words(sequence[t])
        acc[t] = sum(
            float(probs[t, tok_map[nb]]) for nb in valid if nb in tok_map
        )
    return acc


def _window_mean_accuracy(
    per_pos_acc: np.ndarray, Ts: Sequence[int], window: int,
) -> np.ndarray:
    """Trailing-window mean of per-position accuracy at each T."""
    out = np.zeros(len(Ts), dtype=np.float64)
    for k, T in enumerate(Ts):
        lo, hi = max(0, T - window), min(len(per_pos_acc), T)
        if hi <= lo:
            out[k] = np.nan
        else:
            out[k] = float(per_pos_acc[lo:hi].mean())
    return out


def _run_one_walk(
    model,
    graph,
    walk_words: Sequence[str],
    tok_map: dict[str, int],
    seq_len: int,
    hook_name: str,
    with_logits: bool,
) -> dict:
    """Generate a single walk, run one forward pass, return
    activations + sequence + per-position accuracy."""
    import torch

    sequence = _generate_walk_on_graph(graph, seq_len)
    text = " " + " ".join(sequence)
    tokens = model.tokenizer(text, return_tensors="pt").input_ids.to(model.cfg.device)
    with torch.no_grad():
        if with_logits:
            logits, cache = model.run_with_cache(
                tokens, names_filter=[hook_name],
            )
            probs = torch.softmax(logits[0, 1:, :], dim=-1).cpu().numpy()
        else:
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            probs = None
    acts = cache[hook_name][0, 1:, :].cpu().numpy()
    if len(acts) != len(sequence):
        raise RuntimeError(
            f"Activation length {len(acts)} != sequence length {len(sequence)}. "
            "Multi-token vocabulary word(s) broke alignment."
        )
    per_pos_acc = (
        _per_position_accuracy(probs, sequence, graph, tok_map)
        if probs is not None else None
    )
    return {"sequence": sequence, "acts": acts, "per_pos_acc": per_pos_acc}


def run_with_model(
    seq_len: int = 1400,
    layer: int = 26,
    seed: int = 42,
    condition: str = "grid",
    snapshot_Ts: Sequence[int] = PAPER_SNAPSHOT_T,
    window: int = PAPER_WINDOW_SIZE,
    phase_windows: dict[str, tuple[int, int]] = PHASE_WINDOWS_DEFAULT,
    n_walks: int = 1,
    measure_accuracy: bool = True,
) -> None:
    """Full paper-aligned pipeline. Loads Llama-3.1-8B; requires GPU.

    Supported conditions are listed in ``get_conditions()``.

    When ``n_walks > 1`` we repeat the whole pipeline for multiple random
    starting walks, then aggregate:

      * Dirichlet-energy curves are reported as mean ± 1σ across walks.
      * Per-snapshot class means are averaged across walks before PCA
        (this is what Park et al. effectively do in 01_reproduce.py —
        they run one walk per starting node and average class means).
      * If ``measure_accuracy`` is on, we also extract per-position Llama
        accuracy (P(next ∈ valid nbrs)) and emit an energy-vs-accuracy
        scatter (see ``plot_energy_vs_accuracy``).
    """
    import torch
    from transformer_lens import HookedTransformer

    conditions = get_conditions()
    if condition not in conditions:
        raise ValueError(
            f"Unknown condition {condition!r}. "
            f"Available: {sorted(conditions.keys())}"
        )
    cfg = conditions[condition]
    graph = cfg["graph"]
    walk_words = cfg["walk_words"]
    A_primary = cfg["adjacency"]
    word_to_color = cfg["word_to_color"]

    print(
        f"Running condition='{condition}' with seq_len={seq_len}, "
        f"layer={layer}, n_walks={n_walks}, measure_accuracy={measure_accuracy}"
    )
    print("Loading Llama-3.1-8B (may take a minute on first run)...")
    model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Llama-3.1-8B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=os.environ.get("HF_HOME", None),
    )
    hook_name = f"blocks.{layer}.hook_resid_pre"
    tok_map = _build_token_map(model, walk_words)

    Ts_energy = np.array(DIRICHLET_ENERGY_TS, dtype=int)
    # Pre-compute an Ham-ring adjacency for the grid condition.
    A_ham = None
    if condition == "grid":
        ham_order = hamiltonian_ring_order(graph)
        word_idx = {w: i for i, w in enumerate(walk_words)}
        n = len(walk_words)
        A_ham = np.zeros((n, n), dtype=np.int8)
        for k, w in enumerate(ham_order):
            nxt = ham_order[(k + 1) % n]
            i, j = word_idx[w], word_idx[nxt]
            A_ham[i, j] = A_ham[j, i] = 1

    # ── Outer loop: one walk at a time, accumulate aggregates ───────────────
    energies_per_walk = np.zeros((n_walks, len(Ts_energy)), dtype=np.float64)
    energies_ham_per_walk = (
        np.zeros((n_walks, len(Ts_energy))) if A_ham is not None else None
    )
    acc_per_walk = (
        np.zeros((n_walks, len(Ts_energy))) if measure_accuracy else None
    )
    # For PCA: sum class means then divide; and count how many walks had each word.
    class_means_sum: dict[int, np.ndarray] = {}
    present_counts: dict[int, np.ndarray] = {}

    last_sequence = None
    last_acts = None

    for w_idx in range(n_walks):
        set_seed(seed + w_idx)
        walk = _run_one_walk(
            model, graph, walk_words, tok_map, seq_len, hook_name,
            with_logits=measure_accuracy,
        )
        sequence = walk["sequence"]
        acts = walk["acts"]
        last_sequence, last_acts = sequence, acts

        # (a) Dirichlet-energy curve
        Ts_a, energies = dirichlet_energy_curve(
            acts, sequence, walk_words, A_primary,
            Ts=Ts_energy, window=window, normalize=True,
        )
        if len(Ts_a) != len(Ts_energy):
            raise RuntimeError(
                f"dirichlet_energy_curve dropped T values "
                f"({len(Ts_a)} vs {len(Ts_energy)}) — check seq_len/window "
                f"vs DIRICHLET_ENERGY_TS."
            )
        energies_per_walk[w_idx] = energies
        if A_ham is not None:
            _, e_ham = dirichlet_energy_curve(
                acts, sequence, walk_words, A_ham,
                Ts=Ts_energy, window=window, normalize=True,
            )
            energies_ham_per_walk[w_idx] = e_ham

        # (b) Trailing-window LLM accuracy at the same T's
        if acc_per_walk is not None:
            acc_per_walk[w_idx] = _window_mean_accuracy(
                walk["per_pos_acc"], Ts_energy, window,
            )

        # (c) Class means at snapshot T's
        for T in snapshot_Ts:
            H, present = class_means_sliding(
                acts, sequence, walk_words, T, window=window,
            )
            if T not in class_means_sum:
                class_means_sum[T] = np.zeros_like(H)
                present_counts[T] = np.zeros(len(walk_words), dtype=int)
            # Only accumulate rows that were actually observed this walk.
            class_means_sum[T][present] += H[present]
            present_counts[T][present] += 1

        print(
            f"  walk {w_idx+1}/{n_walks}: "
            f"E_primary(T=1400)={energies[-1]:.3f}"
            + (f", acc(T=1400)={acc_per_walk[w_idx, -1]:.3f}"
               if acc_per_walk is not None else "")
        )

    # ── Aggregate across walks ──────────────────────────────────────────────
    class_means_by_T: dict[int, np.ndarray] = {}
    present_by_T: dict[int, np.ndarray] = {}
    for T in snapshot_Ts:
        counts = present_counts[T]
        present = counts > 0
        mean_H = np.zeros_like(class_means_sum[T])
        mean_H[present] = class_means_sum[T][present] / counts[present, None]
        class_means_by_T[T] = mean_H
        present_by_T[T] = present

    energies_mean = energies_per_walk.mean(axis=0)
    energies_std = energies_per_walk.std(axis=0, ddof=1) if n_walks >= 2 else None

    # ── (1) Paper-aligned Fig 2 PCA snapshots (averaged class means) ────────
    paper_fig2_path = os.path.join(
        OUT_DIR, f"pca_snapshots_{condition}_paper.png"
    )
    plot_pca_snapshots_paper(
        class_means_by_T, walk_words, A_primary, word_to_color,
        paper_fig2_path,
        title_prefix=(
            f"Llama-3.1-8B class-mean PCA ({condition}) — "
            f"avg over {n_walks} walks"
        ),
        overlay_spectral=True,
        present_mask_by_T=present_by_T,
    )
    print(f"Saved {paper_fig2_path}")

    # ── (2) Dirichlet-energy curve with ±1σ band ────────────────────────────
    extra = {}
    if energies_ham_per_walk is not None:
        extra["Ham-ring"] = (
            Ts_energy,
            energies_ham_per_walk.mean(axis=0),
            "#C62828",
            energies_ham_per_walk.std(axis=0, ddof=1) if n_walks >= 2 else None,
        )
    energy_path = os.path.join(OUT_DIR, f"dirichlet_energy_{condition}.png")
    plot_dirichlet_energy_curve(
        Ts_energy, energies_mean, energy_path,
        title=(
            f"Normalized Dirichlet energy vs. context length ({condition})\n"
            f"mean ± 1σ over {n_walks} walks  —  "
            "expect E_G(H_T) to decrease as reps re-organise"
        ),
        label=condition, color=cfg["color"], extra_curves=extra,
        stds=energies_std,
    )
    print(f"Saved {energy_path}")

    # ── (2b) Energy-vs-accuracy scatter ─────────────────────────────────────
    if acc_per_walk is not None and n_walks >= 1:
        scatter_path = os.path.join(
            OUT_DIR, f"energy_vs_accuracy_{condition}.png"
        )
        plot_energy_vs_accuracy(
            Ts_energy, energies_per_walk, acc_per_walk,
            scatter_path, condition=condition, color=cfg["color"],
        )
        print(f"Saved {scatter_path}")

    # ── (3) Shared-basis 3-phase trajectory (uses the *last* walk) ──────────
    class_means_by_phase: dict[str, np.ndarray] = {}
    if last_acts is not None:
        for phase, (start, stop) in phase_windows.items():
            if stop > seq_len:
                continue
            cm = compute_class_means_np(
                last_acts[start:stop], last_sequence[start:stop], walk_words,
            )
            class_means_by_phase[phase] = cm
    if class_means_by_phase:
        stacked = np.concatenate(list(class_means_by_phase.values()), axis=0)
        pca_dirs_shared = compute_top_k_pca(stacked, k=4)
        traj_path = os.path.join(
            OUT_DIR, f"pca_before_during_after_{condition}.png"
        )
        plot_pca_phase_grid(
            class_means_by_phase, pca_dirs_shared, walk_words,
            A_primary, word_to_color, traj_path,
        )
        print(f"Saved {traj_path}")

    # ── (4) Fig 3 reproduction for months_permuted ──────────────────────────
    if condition == "months_permuted":
        T_fig3 = max([T for T in snapshot_Ts if T <= seq_len] or [seq_len])
        fig3_path = os.path.join(OUT_DIR, "pca_months_permuted_fig3.png")
        plot_months_fig3(
            class_means_by_T[T_fig3],
            months_permuted=list(MONTHS_PERMUTED),
            months_natural=list(MONTHS),
            month_to_color=MONTH_TO_COLOR,
            out_path=fig3_path,
            T=T_fig3,
            present=present_by_T[T_fig3],
        )
        print(f"Saved {fig3_path}")

    # ── Persist raw artefacts for re-plotting / cross-condition overlay ─────
    out_npz = os.path.join(OUT_DIR, f"pca_pipeline_{condition}.npz")
    save_kwargs = dict(
        condition=np.array(condition),
        walk_words=np.array(walk_words),
        n_walks=np.array(n_walks),
        seq_len=np.array(seq_len),
        layer=np.array(layer),
        Ts=Ts_energy,
        energies_mean=energies_mean,
        energies_per_walk=energies_per_walk,
        A_primary=A_primary,
    )
    if energies_std is not None:
        save_kwargs["energies_std"] = energies_std
    if energies_ham_per_walk is not None:
        save_kwargs["energies_ham_per_walk"] = energies_ham_per_walk
    if acc_per_walk is not None:
        save_kwargs["acc_per_walk"] = acc_per_walk
    for T, H in class_means_by_T.items():
        save_kwargs[f"class_means_T{T}"] = H
        save_kwargs[f"present_T{T}"] = present_by_T[T]
    np.savez(out_npz, **save_kwargs)
    print(f"Saved {out_npz}")


# ── Sigmoid N* helpers ────────────────────────────────────────────────────────

def n_star_from_sigmoid(b: float, gamma: float, alpha: float) -> float:
    """Solve ``b + γ · N^(1-α) = 0`` for N.

    Returns the critical context length ``N*`` at which the Bigelow et
    al. (2025) posterior σ(b + γ N^(1-α)) crosses 0.5.  Requires
    b < 0, γ > 0, α ∈ (0, 1) — the domain the Baseline reparameterises
    into (Checkpoint-2 §3.2).
    """
    if gamma <= 0 or not (0 < alpha < 1):
        return float("nan")
    if b >= 0:
        return 0.0
    return float((-b / gamma) ** (1.0 / (1.0 - alpha)))


def _load_sigmoid_json(path: str) -> dict[str, dict]:
    """Parse a sigmoid-fit JSON produced by Tim's Milestone-2 fit script.

    Accepted schemas:

        1. Flat:    {"b": ..., "gamma": ..., "alpha": ...}
           → interpreted as applying to every condition.

        2. Nested:  {"grid": {"b": ..., ...}, "months_permuted": {...}, ...}
           → per-condition N*.

    Returns a dict {condition: {"b", "gamma", "alpha"}}; an empty string
    key ``""`` means "applies to all conditions".
    """
    with open(path) as f:
        raw = json.load(f)
    if set(raw.keys()) & {"b", "gamma", "alpha"}:
        return {"": {k: float(raw[k]) for k in ("b", "gamma", "alpha")}}
    out = {}
    for cond, sub in raw.items():
        if not isinstance(sub, dict):
            continue
        try:
            out[cond] = {k: float(sub[k]) for k in ("b", "gamma", "alpha")}
        except KeyError:
            continue
    return out


def _annotate_n_star(ax, N_star: float, label: str = "", color: str = "k") -> None:
    if not np.isfinite(N_star) or N_star <= 0:
        return
    ax.axvline(
        N_star, color=color, linestyle=":", lw=1.2, alpha=0.7,
    )
    ymin, ymax = ax.get_ylim()
    ax.text(
        N_star, ymin + 0.97 * (ymax - ymin),
        f"  N*={N_star:.0f}" + (f"  ({label})" if label else ""),
        color=color, fontsize=7, rotation=90,
        ha="left", va="top", alpha=0.8,
    )


# ── Cross-condition E_G overlay (reads cached npz files) ──────────────────────

def run_overlay(
    conditions: Sequence[str] | None = None,
    sigmoid_json: str | None = None,
) -> None:
    """Read cached ``pca_pipeline_{condition}.npz`` files and produce:

      * ``dirichlet_energy_overlay.png`` — E_G vs T for every condition
        on one figure (mean + ±1σ band if n_walks ≥ 2).
      * ``energy_vs_accuracy_overlay.png`` — per-T mean accuracy vs
        mean E_G across conditions (only conditions that logged
        accuracy are included).

    When ``sigmoid_json`` is provided, we also draw a dotted vertical
    line at the sigmoid inflection point N* = (−b/γ)^(1/(1−α)) for each
    condition (Milestone 8: "overlay Dirichlet energy drop with
    sigmoid inflection point").
    """
    if conditions is None:
        conditions = list(get_conditions().keys())
    colors = {c: cfg["color"] for c, cfg in get_conditions().items()}
    labels = {c: cfg["label"] for c, cfg in get_conditions().items()}

    sigmoid_by_cond = _load_sigmoid_json(sigmoid_json) if sigmoid_json else {}

    # ── E_G overlay ──
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False
    for c in conditions:
        p = os.path.join(OUT_DIR, f"pca_pipeline_{c}.npz")
        if not os.path.exists(p):
            print(f"  skipping {c}: {p} not found")
            continue
        any_data = True
        data = np.load(p, allow_pickle=True)
        Ts = data["Ts"]
        # Back-compat: old single-walk runs stored just `energies`.
        if "energies_mean" in data.files:
            em = data["energies_mean"]
        else:
            em = data["energies"]
        es = data["energies_std"] if "energies_std" in data.files else None
        color = colors.get(c, "#444")
        label = labels.get(c, c)
        ax.plot(Ts, em, "o-", color=color, lw=2, ms=5, label=label)
        if es is not None:
            ax.fill_between(Ts, em - es, em + es, color=color, alpha=0.15, lw=0)
    if not any_data:
        raise RuntimeError(
            "No pca_pipeline_*.npz files found. Run `--with-model` first."
        )
    # N* annotations from the sigmoid fit (if provided).
    if sigmoid_by_cond:
        # Need a limit on the y-axis before we axvline, so set xlim/ylim first.
        ax.set_xscale("log")
        fig.canvas.draw()  # realise axis ranges
        for c in conditions:
            if c not in sigmoid_by_cond and "" not in sigmoid_by_cond:
                continue
            p = os.path.join(OUT_DIR, f"pca_pipeline_{c}.npz")
            if not os.path.exists(p):
                continue
            params = sigmoid_by_cond.get(c, sigmoid_by_cond.get("", {}))
            N_star = n_star_from_sigmoid(**params)
            _annotate_n_star(ax, N_star, label=c, color=colors.get(c, "#444"))
    ax.set_xscale("log")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Normalized Dirichlet energy  E_G(H)")
    ax.set_title(
        "Representation alignment E_G vs. context length\n"
        + ("(paper predicts drop at sigmoid inflection N*)"
           if sigmoid_by_cond
           else "(paper predicts decrease as representations re-organise)")
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(OUT_DIR, "dirichlet_energy_overlay.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out1}")

    # ── Energy-vs-accuracy overlay (mean across walks per T) ──
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    any_data = False
    for c in conditions:
        p = os.path.join(OUT_DIR, f"pca_pipeline_{c}.npz")
        if not os.path.exists(p):
            continue
        data = np.load(p, allow_pickle=True)
        if "acc_per_walk" not in data.files:
            continue
        any_data = True
        Ts = data["Ts"]
        em = (
            data["energies_mean"] if "energies_mean" in data.files
            else data["energies"]
        )
        am = data["acc_per_walk"].mean(axis=0)
        color = colors.get(c, "#444")
        ax.plot(
            em, am, "o-", color=color, lw=1.5, ms=6,
            label=labels.get(c, c),
        )
        # Label a few T values along each trajectory.
        for k in range(0, len(Ts), max(1, len(Ts) // 4)):
            ax.annotate(
                f"T={int(Ts[k])}", (em[k], am[k]),
                fontsize=7, color=color, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )
    if any_data:
        ax.set_xlabel("Normalized Dirichlet energy  E_G(H)")
        ax.set_ylabel("Llama  P(next ∈ valid neighbors) (trailing-window mean)")
        ax.set_title(
            "Behavioural accuracy vs. representation alignment\n"
            "(low E_G + high accuracy = both representation and "
            "behaviour track the in-context graph)"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out2 = os.path.join(OUT_DIR, "energy_vs_accuracy_overlay.png")
        fig.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved {out2}")
    plt.close(fig)


# ── Per-layer Dirichlet-energy sweep ─────────────────────────────────────────
#
# Checkpoint-2 §5.4 and Milestone 7 commit to "dirichlet energy of the
# residual stream activations at each layer" — the main pipeline only
# hits one layer (default 26, matching the ICLR cache).  This function
# does multiple layers in a single forward pass using TransformerLens'
# multi-hook cache, so the incremental GPU cost is roughly
# ``O(#layers × seq_len × d_model)`` CPU RAM per walk (tens of MB).

def run_layer_sweep(
    condition: str,
    layers: Sequence[int],
    seq_len: int = 1400,
    seed: int = 42,
    n_walks: int = 8,
    window: int = PAPER_WINDOW_SIZE,
) -> None:
    """Per-layer E_G(H_T) sweep (Checkpoint-2 §5.4 / Milestone 7).

    Fires a fresh Llama load (standalone from the main pipeline so it
    can be run after ``run_with_model``).  For every walk we extract
    residual-stream activations at every layer in ``layers`` via a
    single forward pass, then compute the normalised Dirichlet-energy
    curve at each layer.  Output:

      * ``dirichlet_energy_by_layer_{condition}.png`` — one curve per
        layer (mean ± 1σ over walks).
      * ``dirichlet_energy_by_layer_{condition}.npz`` — per-walk,
        per-layer energies for downstream annotation / sigmoid overlay.
    """
    import torch
    from transformer_lens import HookedTransformer

    conditions = get_conditions()
    if condition not in conditions:
        raise ValueError(
            f"Unknown condition {condition!r}. Available: "
            f"{sorted(conditions.keys())}"
        )
    cfg = conditions[condition]
    graph = cfg["graph"]
    walk_words = cfg["walk_words"]
    A_primary = cfg["adjacency"]
    layers = list(layers)

    print(
        f"Layer sweep — condition={condition}, layers={layers}, "
        f"n_walks={n_walks}, seq_len={seq_len}"
    )
    print("Loading Llama-3.1-8B...")
    model = HookedTransformer.from_pretrained_no_processing(
        "meta-llama/Llama-3.1-8B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        cache_dir=os.environ.get("HF_HOME", None),
    )
    hook_names = [f"blocks.{L}.hook_resid_pre" for L in layers]
    tok_map = _build_token_map(model, walk_words)

    Ts = np.array(DIRICHLET_ENERGY_TS, dtype=int)
    # energies_per_walk[layer_idx, walk_idx, T_idx]
    energies = np.zeros((len(layers), n_walks, len(Ts)), dtype=np.float64)

    for w_idx in range(n_walks):
        set_seed(seed + w_idx)
        sequence = _generate_walk_on_graph(graph, seq_len)
        text = " " + " ".join(sequence)
        tokens = model.tokenizer(text, return_tensors="pt").input_ids.to(
            model.cfg.device
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        for li, (L, h) in enumerate(zip(layers, hook_names)):
            acts = cache[h][0, 1:, :].cpu().numpy()
            if len(acts) != len(sequence):
                raise RuntimeError(
                    f"Activation length {len(acts)} != sequence length "
                    f"{len(sequence)} at layer {L}."
                )
            _, e = dirichlet_energy_curve(
                acts, sequence, walk_words, A_primary,
                Ts=Ts, window=window, normalize=True,
            )
            energies[li, w_idx] = e
        print(f"  walk {w_idx+1}/{n_walks}: E @ (T=1400, "
              f"layer={layers[-1]}) = {energies[-1, w_idx, -1]:.3f}")
        # Free cache between walks.
        del cache

    # ── Plot: one curve per layer with ±1σ band ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("viridis")
    for li, L in enumerate(layers):
        em = energies[li].mean(axis=0)
        es = energies[li].std(axis=0, ddof=1) if n_walks >= 2 else None
        color = cmap(li / max(1, len(layers) - 1))
        ax.plot(Ts, em, "o-", color=color, lw=2, ms=4, label=f"layer {L}")
        if es is not None:
            ax.fill_between(Ts, em - es, em + es, color=color, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Normalized Dirichlet energy  E_G(H_T)")
    ax.set_title(
        f"Per-layer residual-stream E_G vs. context length ({condition})\n"
        f"{n_walks} walks — expect deeper layers to align earlier"
    )
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = os.path.join(
        OUT_DIR, f"dirichlet_energy_by_layer_{condition}.png"
    )
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")

    # ── Persist per-walk per-layer energies ──────────────────────────────────
    out_npz = os.path.join(
        OUT_DIR, f"dirichlet_energy_by_layer_{condition}.npz"
    )
    np.savez(
        out_npz,
        condition=np.array(condition),
        layers=np.array(layers),
        Ts=Ts,
        energies_per_walk_per_layer=energies,
        energies_mean_per_layer=energies.mean(axis=1),
        energies_std_per_layer=(
            energies.std(axis=1, ddof=1) if n_walks >= 2 else
            np.zeros_like(energies.mean(axis=1))
        ),
    )
    print(f"Saved {out_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--with-model", action="store_true",
        help="Run Llama-3.1-8B (requires GPU + ~16 GB VRAM). Without this "
             "flag, we only replot cached artefacts + spectral reference.",
    )
    parser.add_argument(
        "--overlay", action="store_true",
        help="Don't run a model; read every cached pca_pipeline_*.npz and "
             "produce cross-condition E_G and E_G-vs-accuracy overlay plots.",
    )
    parser.add_argument(
        "--condition", default="grid",
        choices=sorted(get_conditions().keys()),
        help="Experimental condition for --with-model. 'grid' reproduces "
             "paper Fig 2+4; 'months_permuted' additionally produces Fig 3; "
             "the remaining ring conditions mirror vocabulary_tl_experiment.py.",
    )
    parser.add_argument("--seq-len", type=int, default=1400)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-walks", type=int, default=1,
        help="Number of independent random walks to run for the condition. "
             ">1 gives mean±σ Dirichlet-energy bands and averaged class means.",
    )
    parser.add_argument(
        "--no-accuracy", action="store_true",
        help="Skip per-position softmax accuracy extraction "
             "(faster, but no energy-vs-accuracy scatter).",
    )
    parser.add_argument(
        "--sigmoid-json", default=None,
        help="Path to a JSON with fitted Baseline sigmoid parameters "
             "{b, gamma, alpha} (flat) or {condition: {b, gamma, alpha}} "
             "(nested).  When provided in --overlay mode, we annotate the "
             "sigmoid inflection point N* = (-b/γ)^(1/(1-α)) on each curve.",
    )
    parser.add_argument(
        "--layer-sweep", action="store_true",
        help="Run a per-layer Dirichlet-energy sweep for --condition using "
             "--layers (comma-separated). Produces dirichlet_energy_by_layer_"
             "{cond}.png. Independent of --with-model's main pipeline.",
    )
    parser.add_argument(
        "--layers", default="8,16,20,24,26,28",
        help="Comma-separated layer indices for --layer-sweep (default: "
             "'8,16,20,24,26,28').",
    )
    args = parser.parse_args()

    if args.layer_sweep:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        run_layer_sweep(
            condition=args.condition, layers=layers,
            seq_len=args.seq_len, seed=args.seed, n_walks=args.n_walks,
        )
    elif args.overlay:
        run_overlay(sigmoid_json=args.sigmoid_json)
    elif args.with_model:
        run_with_model(
            seq_len=args.seq_len, layer=args.layer, seed=args.seed,
            condition=args.condition,
            n_walks=args.n_walks,
            measure_accuracy=not args.no_accuracy,
        )
    else:
        run_from_cached()


if __name__ == "__main__":
    main()
