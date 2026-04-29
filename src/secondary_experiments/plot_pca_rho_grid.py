"""Make a 2-row × N-column PCA figure comparing all mixing ratios at T=1400.

Row 0: PC1 vs PC2
Row 1: PC3 vs PC4
Columns: ρ = 0 (pure grid) → 0.2 → … → 0.8 → 1.0 (pure ring), left to right.

Each panel uses its own PCA basis (computed from that condition's class means),
which matches the paper convention and is most interpretable per-condition.

Usage:
    # plain scatter
    python -m src.secondary_experiments.plot_pca_rho_grid

    # with ideal graph edges overlaid (second figure)
    python -m src.secondary_experiments.plot_pca_rho_grid --with-structure

    # both at once, different T
    python -m src.secondary_experiments.plot_pca_rho_grid --with-structure --T 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Graph adjacency colours
_GRID_EDGE_COLOR  = "#1565C0"   # deep blue
_RING_EDGE_COLOR  = "#B71C1C"   # deep red

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Ordered columns: (rho, npz_path, column_label)
# ρ=0 → pure grid; ρ=1 → pure ring.
_LADDER: list[tuple[float, str, str]] = [
    (0.0, "all_graphs/grid/pca_grid.npz",                       "ρ=0\n(pure grid)"),
    (0.2, "mix_grid80_ring20/pca_mix_grid80_ring20.npz",        "ρ=0.2"),
    (0.3, "mix_grid70_ring30/pca_mix_grid70_ring30.npz",        "ρ=0.3"),
    (0.4, "mix_grid60_ring40/pca_mix_grid60_ring40.npz",        "ρ=0.4"),
    (0.5, "mix_grid50_ring50/pca_mix_grid50_ring50.npz",        "ρ=0.5"),
    (0.6, "mix_grid40_ring60/pca_mix_grid40_ring60.npz",        "ρ=0.6"),
    (0.7, "mix_grid30_ring70/pca_mix_grid30_ring70.npz",        "ρ=0.7"),
    (0.8, "mix_grid20_ring80/pca_mix_grid20_ring80.npz",        "ρ=0.8"),
    (1.0, "all_graphs/ring/pca_ring.npz",                       "ρ=1\n(pure ring)"),
]

# Colour for each word — same palette used by the rest of the pipeline.
WORDS = (
    "apple", "bird",  "car",   "egg",
    "house", "milk",  "plane", "opera",
    "box",   "sand",  "sun",   "mango",
    "rock",  "math",  "code",  "phone",
)
COLORS = (
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
)
WORD_TO_COLOR = dict(zip(WORDS, COLORS))


def _load_npz(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return dict(d)


def _compute_pca(H: np.ndarray, k: int = 4) -> np.ndarray:
    """Top-k PCA directions (rows of returned matrix)."""
    centered = H.astype(np.float64) - H.astype(np.float64).mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:k]


def _get_adjacencies() -> dict[str, np.ndarray]:
    """Return {'grid': A_grid, 'ring': A_ring} over the 16-word WORDS vocabulary."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.secondary_experiments.graphs import GridGraph, RingGraph
    return {
        "grid": GridGraph(WORDS).build_adjacency_matrix(),
        "ring": RingGraph(WORDS).build_adjacency_matrix(),
    }


def _draw_edges(
    ax: plt.Axes,
    projected: np.ndarray,   # (n_present, ≥2)
    words_p: list[str],
    A: np.ndarray,           # (n_present, n_present) adjacency
    xi: int,
    yi: int,
    color: str,
    lw: float = 1.2,
    alpha: float = 0.55,
    zorder: int = 2,
) -> None:
    """Draw edges from adjacency matrix A onto (xi, yi) projected coordinates."""
    n = len(words_p)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                ax.plot(
                    [projected[i, xi], projected[j, xi]],
                    [projected[i, yi], projected[j, yi]],
                    color=color, lw=lw, alpha=alpha,
                    solid_capstyle="round", zorder=zorder,
                )


def _draw_panel(
    ax: plt.Axes,
    projected: np.ndarray,    # (n_present, 4)
    words_p: list[str],
    pc_x: int,                # 1-indexed
    pc_y: int,
    title: str,
    show_ylabel: bool,
    # Optional ideal-structure overlays: list of (A_present, color)
    edge_overlays: list[tuple[np.ndarray, str]] | None = None,
) -> None:
    xi, yi = pc_x - 1, pc_y - 1

    # Draw ideal edges beneath the scatter points.
    if edge_overlays:
        for A_p, color in edge_overlays:
            _draw_edges(ax, projected, words_p, A_p, xi, yi, color=color)

    for i, word in enumerate(words_p):
        ax.scatter(
            projected[i, xi],
            projected[i, yi],
            color=WORD_TO_COLOR.get(word, "#888888"),
            s=100,
            marker="*",
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        ax.annotate(
            word,
            (projected[i, xi], projected[i, yi]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.3),
        )
    ax.set_xlabel(f"PC{pc_x}", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(f"PC{pc_y}", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.2)


def make_rho_grid_figure(
    T: int = 1400,
    results_dir: Path = RESULTS_DIR,
    out_path: Path | None = None,
    with_structure: bool = False,
) -> Path:
    # Only keep columns whose NPZ file exists.
    available = [
        (rho, rel, label)
        for rho, rel, label in _LADDER
        if (results_dir / rel).exists()
    ]
    if not available:
        raise FileNotFoundError(
            f"No NPZ files found under {results_dir}. "
            "Run `python -m src.secondary_experiments.run_pca_all` first."
        )

    # Pre-load full adjacency matrices (16×16, WORDS order) once if needed.
    adj_full: dict[str, np.ndarray] = _get_adjacencies() if with_structure else {}

    n_cols = len(available)
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.4 * n_cols, 5.5),
        squeeze=False,
    )

    for col, (rho, rel_path, label) in enumerate(available):
        npz = _load_npz(results_dir / rel_path)
        key_H = f"class_means_T{T}"
        key_p = f"present_T{T}"

        if key_H not in npz:
            for row in range(2):
                axes[row, col].text(
                    0.5, 0.5, f"T={T}\nnot in NPZ",
                    ha="center", va="center",
                    transform=axes[row, col].transAxes, fontsize=8,
                )
                axes[row, col].set_axis_off()
            continue

        H_full    = npz[key_H].astype(np.float64)
        present   = npz[key_p].astype(bool)
        H         = H_full[present]
        words_p   = [w for w, ok in zip(WORDS, present) if ok]
        n_present = present.sum()

        if n_present < 4:
            for row in range(2):
                axes[row, col].text(
                    0.5, 0.5, f"T={T}\nonly {n_present} words",
                    ha="center", va="center",
                    transform=axes[row, col].transAxes, fontsize=8,
                )
                axes[row, col].set_axis_off()
            continue

        pca_dirs  = _compute_pca(H, k=4)
        projected = H @ pca_dirs.T          # (n_present, 4)

        # Build per-panel edge overlays (sub-adjacency restricted to present words).
        edge_overlays: list[tuple[np.ndarray, str]] | None = None
        if with_structure and adj_full:
            edge_overlays = []
            if rho <= 0.0:
                # Pure grid: show only grid edges
                overlays_spec = [("grid", _GRID_EDGE_COLOR)]
            elif rho >= 1.0:
                # Pure ring: show only ring edges
                overlays_spec = [("ring", _RING_EDGE_COLOR)]
            else:
                # Mixed: show both; draw the minority graph more transparent
                # by using thinner lines (handled inside _draw_edges via alpha).
                overlays_spec = [
                    ("grid", _GRID_EDGE_COLOR),
                    ("ring", _RING_EDGE_COLOR),
                ]
            for graph_name, color in overlays_spec:
                A_full = adj_full[graph_name]
                # Restrict to present words only.
                A_p = A_full[np.ix_(present, present)]
                edge_overlays.append((A_p, color))

        for row, (pc_x, pc_y) in enumerate([(1, 2), (3, 4)]):
            n_missing = len(WORDS) - n_present
            col_title = label if row == 0 else ""
            if col_title and n_missing:
                col_title += f"\n({n_missing} unseen)"
            _draw_panel(
                axes[row, col],
                projected,
                words_p,
                pc_x=pc_x,
                pc_y=pc_y,
                title=col_title,
                show_ylabel=(col == 0),
                edge_overlays=edge_overlays,
            )

    # Row labels on the left-most column.
    for row, (pc_x, pc_y) in enumerate([(1, 2), (3, 4)]):
        axes[row, 0].set_ylabel(f"PC{pc_y}  /  PC{pc_x}–PC{pc_y}", fontsize=8)

    # Legend for structure figure.
    if with_structure:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=_GRID_EDGE_COLOR, lw=1.5, label="grid edges"),
            Line2D([0], [0], color=_RING_EDGE_COLOR,  lw=1.5, label="ring edges"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            fontsize=8,
            framealpha=0.85,
            bbox_to_anchor=(0.5, -0.02),
        )

    suffix = "_structure" if with_structure else ""
    title_note = " + ideal graph edges" if with_structure else ""
    fig.suptitle(
        f"Llama 3.1 8B — class-mean PCA at T={T}  ·  all mixing ratios"
        f"{title_note}  (layer 26, Nw=50)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = results_dir / f"pca_rho_grid_T{T}{suffix}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


def make_rho_grid_figure_split(
    T: int = 1400,
    results_dir: Path = RESULTS_DIR,
    out_path: Path | None = None,
) -> Path:
    """4-row × N-column figure: grid edges on top half, ring edges on bottom half.

    Rows 0–1: PC1/2 and PC3/4 with only grid edges (blue).
    Rows 2–3: PC1/2 and PC3/4 with only ring edges (red).
    Each column is one mixing ratio.
    """
    available = [
        (rho, rel, label)
        for rho, rel, label in _LADDER
        if (results_dir / rel).exists()
    ]
    if not available:
        raise FileNotFoundError(f"No NPZ files found under {results_dir}.")

    adj_full = _get_adjacencies()
    n_cols   = len(available)

    # 4 rows: [grid PC1/2, grid PC3/4, ring PC1/2, ring PC3/4]
    ROW_SPEC = [
        (0, (1, 2), "grid", _GRID_EDGE_COLOR),
        (1, (3, 4), "grid", _GRID_EDGE_COLOR),
        (2, (1, 2), "ring", _RING_EDGE_COLOR),
        (3, (3, 4), "ring", _RING_EDGE_COLOR),
    ]

    fig, axes = plt.subplots(
        4, n_cols,
        figsize=(2.4 * n_cols, 10.5),
        squeeze=False,
    )

    # Horizontal divider between the two blocks.
    from matplotlib.lines import Line2D
    for col_ax in axes[2]:
        col_ax.spines["top"].set_linewidth(2.0)
        col_ax.spines["top"].set_color("#444444")

    for col, (rho, rel_path, label) in enumerate(available):
        npz       = _load_npz(results_dir / rel_path)
        key_H     = f"class_means_T{T}"
        key_p     = f"present_T{T}"

        if key_H not in npz:
            for row in range(4):
                axes[row, col].text(0.5, 0.5, f"T={T}\nnot in NPZ",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=8)
                axes[row, col].set_axis_off()
            continue

        H_full    = npz[key_H].astype(np.float64)
        present   = npz[key_p].astype(bool)
        H         = H_full[present]
        words_p   = [w for w, ok in zip(WORDS, present) if ok]
        n_present = present.sum()

        if n_present < 4:
            for row in range(4):
                axes[row, col].text(0.5, 0.5, f"T={T}\nonly {n_present} words",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes, fontsize=8)
                axes[row, col].set_axis_off()
            continue

        pca_dirs  = _compute_pca(H, k=4)
        projected = H @ pca_dirs.T

        for row_idx, (pc_x, pc_y), graph_name, color in [
            (r, pc, g, c) for r, pc, g, c in ROW_SPEC
        ]:
            # Only draw the edges that belong to the current half.
            # For pure-grid columns: skip ring rows (no ring signal).
            # For pure-ring columns: skip grid rows (no grid signal).
            if rho == 0.0 and graph_name == "ring":
                edge_overlays = None
            elif rho == 1.0 and graph_name == "grid":
                edge_overlays = None
            else:
                A_p = adj_full[graph_name][np.ix_(present, present)]
                edge_overlays = [(A_p, color)]

            # Column title only on row 0 and row 2 (top of each block).
            if row_idx == 0:
                n_missing = len(WORDS) - n_present
                col_title = label + (f"\n({n_missing} unseen)" if n_missing else "")
            elif row_idx == 2:
                col_title = label   # repeat ρ label at top of ring block
            else:
                col_title = ""

            _draw_panel(
                axes[row_idx, col],
                projected, words_p,
                pc_x=pc_x, pc_y=pc_y,
                title=col_title,
                show_ylabel=(col == 0),
                edge_overlays=edge_overlays,
            )

    # Y-axis labels for the left-most column.
    row_ylabels = ["PC1 vs PC2", "PC3 vs PC4", "PC1 vs PC2", "PC3 vs PC4"]
    for row_idx, ylabel in enumerate(row_ylabels):
        axes[row_idx, 0].set_ylabel(ylabel, fontsize=8)

    # Block labels on the far left.
    for block_row, block_label, color in [(0, "Grid edges →", _GRID_EDGE_COLOR),
                                           (2, "Ring edges →", _RING_EDGE_COLOR)]:
        fig.text(
            0.005, 1 - (block_row / 4) - 0.12,
            block_label,
            rotation=90, va="center", ha="center",
            fontsize=9, color=color, fontweight="bold",
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], color=_GRID_EDGE_COLOR, lw=1.5, label="ideal grid edges"),
        Line2D([0], [0], color=_RING_EDGE_COLOR,  lw=1.5, label="ideal ring edges"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=8, framealpha=0.85, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Llama 3.1 8B — class-mean PCA at T={T}  ·  all mixing ratios\n"
        f"Top: grid structure overlay   ·   Bottom: ring structure overlay   "
        f"(layer 26, Nw=50)",
        fontsize=11, y=1.005,
    )
    fig.tight_layout()

    if out_path is None:
        out_path = results_dir / f"pca_rho_grid_T{T}_split.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--T", type=int, default=1400,
        help="Context-length snapshot to show (default 1400; must be in 200/400/1400).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output PNG path (default: results/pca_rho_grid_T{T}[_structure].png).",
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Root results directory to scan.",
    )
    parser.add_argument(
        "--with-structure", action="store_true",
        help="Also produce the combined-edges overlay figure (_structure.png).",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Produce the 4-row split figure: grid edges on top, ring on bottom.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rd = Path(args.results_dir)

    # Plain scatter (always).
    make_rho_grid_figure(T=args.T, results_dir=rd,
                         out_path=Path(args.out) if args.out else None,
                         with_structure=False)

    # Combined-overlay version.
    if args.with_structure:
        make_rho_grid_figure(T=args.T, results_dir=rd,
                             out_path=None, with_structure=True)

    # Split 4-row version.
    if args.split:
        make_rho_grid_figure_split(T=args.T, results_dir=rd)


if __name__ == "__main__":
    main()
