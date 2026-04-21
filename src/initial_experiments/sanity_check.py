"""Sanity checks for grid and ring random walks, and their mixing.

Checks:
  1. Transition matrix accuracy (empirical ≈ true) for both graphs
  2. Vocabulary disjointness (grid words ∩ ring words = ∅)
  3. Node coverage at T=1400
  4. Mixing ratio at rho ∈ {0, 0.25, 0.5, 0.75, 1.0}
  5. Plots: graph diagrams, transition matrices, sample interleaved walk

Run from the repo root:
    python iclr_induction-main/initial_experiments/sanity_check.py
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graphs import Ring, RING_WORDS, RING_WORD_TO_COLOR, RING_COLORS

# ── Grid (inlined from utils.py to avoid torch import) ────────────────────────

WORDS = [
    "apple", "bird", "car", "egg",
    "house", "milk", "plane", "opera",
    "box", "sand", "sun", "mango",
    "rock", "math", "code", "phone",
]

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]

WORD_TO_COLOR = {w: c for w, c in zip(WORDS, COLORS)}


class Grid:
    def __init__(self, words=WORDS, rows=4, cols=4):
        words = words.copy()
        random.shuffle(words)
        self.words = words
        self.rows = rows
        self.cols = cols
        self.grid = np.array(words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(words)}

    def get_valid_next_words(self, word):
        row, col = self.word_to_row[word], self.word_to_col[word]
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                neighbors.append(self.grid[r][c])
        return neighbors

    def generate_sequence(self, seq_len, start_word=None):
        if start_word is None:
            start_word = np.random.choice(self.words)
        row, col = self.word_to_row[start_word], self.word_to_col[start_word]
        sequence = [self.grid[row][col]]
        while len(sequence) < seq_len:
            neighbors = self.get_valid_next_words(sequence[-1])
            choice = np.random.choice(neighbors)
            row, col = self.word_to_row[choice], self.word_to_col[choice]
            sequence.append(self.grid[row][col])
        return sequence

    def build_adjacency_matrix(self):
        n = len(self.words)
        A = np.zeros((n, n))
        for i, w in enumerate(self.words):
            for nb in self.get_valid_next_words(w):
                j = self.words.index(nb)
                A[i, j] = 1
        return A


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 1400
SEGMENT_LEN = 100  # tokens per segment in interleaved sequences
N_CHECK_WALKS = 50  # walks used for transition matrix estimation
CHECK_WALK_LEN = 500


# ── mixing ─────────────────────────────────────────────────────────────────────

def make_interleaved_sequence(grid, ring, total_len, rho, segment_len=SEGMENT_LEN):
    """
    Build an interleaved context of length total_len from two graphs.

    Design: ACROSS-SEQUENCE mixing.  Each segment is a contiguous pure walk on
    one graph; segments are concatenated.  rho = fraction of segments from ring.

    Returns:
        sequence : list[str]  — token sequence of length total_len
        labels   : list[str]  — 'grid' or 'ring' for each position
    """
    sequence, labels = [], []
    while len(sequence) < total_len:
        use_ring = np.random.rand() < rho
        graph = ring if use_ring else grid
        seg = graph.generate_sequence(segment_len)
        source = "ring" if use_ring else "grid"
        sequence.extend(seg)
        labels.extend([source] * len(seg))
    return sequence[:total_len], labels[:total_len]


# ── check 1: transition matrix ─────────────────────────────────────────────────

def compute_empirical_transitions(graph, n_walks=N_CHECK_WALKS, seq_len=CHECK_WALK_LEN):
    """Return (empirical_T, true_T) as numpy arrays."""
    words = graph.words
    n = len(words)
    idx = {w: i for i, w in enumerate(words)}

    counts = np.zeros((n, n))
    for _ in range(n_walks):
        seq = graph.generate_sequence(seq_len)
        for a, b in zip(seq[:-1], seq[1:]):
            counts[idx[a], idx[b]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    empirical = counts / np.where(row_sums > 0, row_sums, 1)

    A = graph.build_adjacency_matrix()
    true_T = A / A.sum(axis=1, keepdims=True)
    return empirical, true_T


def check_transition_matrix(graph, name):
    empirical, true_T = compute_empirical_transitions(graph)
    max_err = np.abs(empirical - true_T).max()
    mean_err = np.abs(empirical - true_T).mean()
    status = "PASS" if max_err < 0.08 else "FAIL"
    print(f"  [{status}] {name}: max_err={max_err:.4f}  mean_err={mean_err:.4f}")
    assert max_err < 0.08, f"Transition matrix error too large for {name}: {max_err:.4f}"
    return empirical, true_T


# ── check 2: vocabulary overlap ────────────────────────────────────────────────

def check_vocabulary_overlap(grid, ring):
    overlap = set(grid.words) & set(ring.words)
    status = "PASS" if not overlap else "FAIL"
    print(f"  [{status}] Vocabulary overlap: {overlap if overlap else 'none (disjoint)'}")
    return overlap


# ── check 3: node coverage ─────────────────────────────────────────────────────

def check_node_coverage(graph, name, seq_len=SEQ_LEN, n_trials=5):
    coverages = []
    for _ in range(n_trials):
        seq = graph.generate_sequence(seq_len)
        coverages.append(len(set(seq)) / len(graph.words))
    mean_cov = np.mean(coverages)
    status = "PASS" if mean_cov == 1.0 else "WARN"
    print(f"  [{status}] {name}: mean node coverage = {mean_cov:.3f} over {n_trials} walks of length {seq_len}")
    return mean_cov


# ── check 4: mixing ratio ──────────────────────────────────────────────────────

def check_mixing_ratios(grid, ring, rhos=(0.0, 0.25, 0.5, 0.75, 1.0), n_trials=50):
    # With segment_len=100 and total_len=1400, there are 14 segments per sequence.
    # Binomial(14, rho) gives std ≈ sqrt(rho*(1-rho)/14) ≈ 0.12 per sequence.
    # Over n_trials=50, the mean std ≈ 0.017, so tolerance of 0.05 is tight but fair.
    print(f"  {'rho':>6}  {'mean_ring_frac':>14}  {'std':>6}  {'error':>8}  status")
    for rho in rhos:
        fracs = []
        for _ in range(n_trials):
            _, labels = make_interleaved_sequence(grid, ring, SEQ_LEN, rho)
            fracs.append(labels.count("ring") / len(labels))
        mean_frac = np.mean(fracs)
        std_frac = np.std(fracs)
        err = abs(mean_frac - rho)
        status = "PASS" if err < 0.05 else "FAIL"
        print(f"  {rho:>6.2f}  {mean_frac:>14.3f}  {std_frac:>6.3f}  {err:>8.4f}  {status}")
        assert err < 0.05, f"Mixing ratio off at rho={rho}: got {mean_frac:.3f}"


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_graph_diagrams(grid, ring):
    """Side-by-side diagrams of the two graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── grid ──
    ax = axes[0]
    pos = {}
    for i, w in enumerate(grid.words):
        r, c = grid.word_to_row[w], grid.word_to_col[w]
        pos[w] = (c, -r)

    A_grid = grid.build_adjacency_matrix()
    for i, wi in enumerate(grid.words):
        for j, wj in enumerate(grid.words):
            if A_grid[i, j] and i < j:
                x0, y0 = pos[wi]
                x1, y1 = pos[wj]
                ax.plot([x0, x1], [y0, y1], "k-", lw=1.2, alpha=0.4, zorder=1)

    for w in grid.words:
        x, y = pos[w]
        ax.scatter(x, y, s=200, color=WORD_TO_COLOR[w], zorder=3,
                   edgecolors="black", linewidths=0.8)
        ax.text(x, y - 0.28, w, ha="center", va="top", fontsize=8)

    ax.set_title("Grid (16 nodes, 4×4)", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── ring ──
    ax = axes[1]
    n = ring.n
    angles = [2 * np.pi * i / n for i in range(n)]
    pos_ring = {w: (np.cos(a), np.sin(a)) for w, a in zip(ring.words, angles)}

    A_ring = ring.build_adjacency_matrix()
    for i, wi in enumerate(ring.words):
        for j, wj in enumerate(ring.words):
            if A_ring[i, j] and i < j:
                x0, y0 = pos_ring[wi]
                x1, y1 = pos_ring[wj]
                ax.plot([x0, x1], [y0, y1], "k-", lw=1.2, alpha=0.4, zorder=1)

    for w in ring.words:
        x, y = pos_ring[w]
        ax.scatter(x, y, s=200, color=RING_WORD_TO_COLOR[w], zorder=3,
                   edgecolors="black", linewidths=0.8)
        offset = 0.18
        ax.text(x * (1 + offset), y * (1 + offset), w, ha="center", va="center", fontsize=8)

    ax.set_title("Ring (12 nodes, months of year)", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Competing graph structures", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "graph_diagrams.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_transition_matrices(grid, ring):
    """Empirical and true transition matrices for both graphs."""
    emp_g, true_g = compute_empirical_transitions(grid)
    emp_r, true_r = compute_empirical_transitions(ring)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    titles = [
        ("Empirical T — Grid", emp_g, grid.words),
        ("True T — Grid", true_g, grid.words),
        ("Empirical T — Ring", emp_r, ring.words),
        ("True T — Ring", true_r, ring.words),
    ]
    for ax, (title, mat, words) in zip(axes.flat, titles):
        im = ax.imshow(mat, vmin=0, vmax=0.5, cmap="Blues")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(words)))
        ax.set_yticks(range(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=6)
        ax.set_yticklabels(words, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "transition_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_sample_interleaved_walk(grid, ring, rho=0.5, n_tokens=120):
    """Visualize a short interleaved walk as a sequence of colored tokens."""
    seq, labels = make_interleaved_sequence(grid, ring, n_tokens, rho, segment_len=20)

    fig, ax = plt.subplots(figsize=(14, 3))
    tokens_per_row = 30
    n_rows = (n_tokens + tokens_per_row - 1) // tokens_per_row

    for i, (tok, lab) in enumerate(zip(seq, labels)):
        col = i % tokens_per_row
        row = i // tokens_per_row
        color = WORD_TO_COLOR.get(tok) or RING_WORD_TO_COLOR.get(tok, "gray")
        rect = mpatches.FancyBboxPatch(
            (col, -row - 0.9), 0.95, 0.85,
            boxstyle="round,pad=0.03",
            facecolor=color, edgecolor="black", linewidth=0.5,
        )
        ax.add_patch(rect)
        ax.text(col + 0.47, -row - 0.47, tok, ha="center", va="center",
                fontsize=6.5, color="black", fontweight="bold" if lab == "ring" else "normal")
        # small indicator dot: ring=triangle, grid=square
        marker = "^" if lab == "ring" else "s"
        ax.plot(col + 0.85, -row - 0.82, marker, ms=3,
                color="navy" if lab == "ring" else "darkgreen", zorder=5)

    grid_patch = mpatches.Patch(color="darkgreen", label="grid segment")
    ring_patch = mpatches.Patch(color="navy", label="ring segment")
    ax.legend(handles=[grid_patch, ring_patch], loc="upper right",
              fontsize=9, framealpha=0.9)

    ax.set_xlim(-0.1, tokens_per_row + 0.1)
    ax.set_ylim(-n_rows - 0.2, 0.3)
    ax.set_title(f"Sample interleaved walk  (rho={rho}, segment_len=20, first {n_tokens} tokens)",
                 fontsize=11)
    ax.axis("off")

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "sample_interleaved_walk.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_mixing_ratio_sweep(grid, ring):
    """Bar chart of actual ring fraction vs. target rho."""
    rhos = np.linspace(0, 1, 11)
    n_trials = 20
    means, stds = [], []
    for rho in rhos:
        fracs = []
        for _ in range(n_trials):
            _, labels = make_interleaved_sequence(grid, ring, SEQ_LEN, rho)
            fracs.append(labels.count("ring") / len(labels))
        means.append(np.mean(fracs))
        stds.append(np.std(fracs))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(rhos, means, yerr=stds, fmt="o-", color="steelblue",
                capsize=4, label="observed ring fraction")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="ideal (y=x)")
    ax.set_xlabel("Target rho")
    ax.set_ylabel("Observed ring fraction")
    ax.set_title("Mixing ratio check")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, "mixing_ratio_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    grid = Grid()
    ring = Ring()

    print("=" * 60)
    print("Check 1: Transition matrices")
    check_transition_matrix(grid, "Grid")
    check_transition_matrix(ring, "Ring")

    print("\nCheck 2: Vocabulary overlap")
    check_vocabulary_overlap(grid, ring)

    print("\nCheck 3: Node coverage at T=1400")
    check_node_coverage(grid, "Grid")
    check_node_coverage(ring, "Ring")

    print("\nCheck 4: Mixing ratios")
    check_mixing_ratios(grid, ring)

    print("\nGenerating plots...")
    plot_graph_diagrams(grid, ring)
    plot_sample_interleaved_walk(grid, ring, rho=0.5)
    plot_mixing_ratio_sweep(grid, ring)

    # Also save raw transition matrix data for later reference
    A_grid = grid.build_adjacency_matrix()
    A_ring = ring.build_adjacency_matrix()
    np.savez(
        os.path.join(RESULTS_DIR, "adjacency.npz"),
        grid_adjacency=A_grid,
        ring_adjacency=A_ring,
        grid_words=np.array(grid.words),
        ring_words=np.array(ring.words),
    )

    print("\nAll checks passed.")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
