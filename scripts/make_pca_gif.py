#!/usr/bin/env python3
"""Build an animated GIF from a secondary-experiment ``pca_*.npz`` snapshot file.

Each frame is **horizontal**: PC1/2 (left) and PC3/4 (right) for one context
length T, then the GIF steps through T ∈ snapshot times in the NPZ.

Example (grid condition, for README):

    python scripts/make_pca_gif.py \\
        --npz src/secondary_experiments/results/all_graphs/grid/pca_grid.npz \\
        --out docs/readme_pca_evolution.gif

Mixed runs (OR grid + ring edges), matching ``run_pca_analysis_mixed`` when
grid dominated:

    python scripts/make_pca_gif.py \\
        --npz src/secondary_experiments/results/mix_grid30_ring70/pca_mix_grid30_ring70.npz \\
        --graph grid --overlay ring \\
        --out docs/readme_pca_mix.gif

Requires: numpy, matplotlib, Pillow (same conda env as other experiments).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.secondary_experiments.graphs import build_candidate_graphs  # noqa: E402
from src.secondary_experiments.pca_analysis import (  # noqa: E402
    load_pca_npz,
    write_pca_evolution_gif,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--npz",
        type=Path,
        default=REPO_ROOT / "src/secondary_experiments/results/all_graphs/grid/pca_grid.npz",
        help="Path to pca_*.npz from run_pca / run_pca_all",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/readme_pca_evolution.gif",
        help="Output .gif path",
    )
    p.add_argument(
        "--graph",
        default=None,
        choices=["grid", "ring", "chain", "star", "uniform"],
        help="Primary graph for edge drawing. Default: true_graph from the npz if valid, else grid.",
    )
    p.add_argument(
        "--overlay",
        action="append",
        default=[],
        choices=["grid", "ring", "chain", "star", "uniform"],
        help="Additional graph(s) to OR into edge drawing (repeat flag for several).",
    )
    p.add_argument("--duration-ms", type=int, default=900, help="Frame duration in ms")
    p.add_argument("--dpi", type=int, default=125)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = args.npz if args.npz.is_absolute() else REPO_ROOT / args.npz
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if not npz_path.is_file():
        raise SystemExit(f"NPZ not found: {npz_path}")

    result = load_pca_npz(npz_path)
    graph_map = build_candidate_graphs()

    if args.graph is not None:
        primary = args.graph
    elif result.true_graph in graph_map:
        primary = result.true_graph
    else:
        primary = "grid"

    overlay_graphs = None
    if args.overlay:
        overlay_graphs = {name: graph_map[name] for name in args.overlay}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = write_pca_evolution_gif(
        result,
        graph_map[primary],
        out_path,
        overlay_graphs=overlay_graphs,
        duration_ms=args.duration_ms,
        dpi=args.dpi,
    )
    print(f"Wrote {written} ({len(result.class_means_by_T)} frames)")


if __name__ == "__main__":
    main()
