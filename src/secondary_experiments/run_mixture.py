"""CLI for Bayesian mixture-of-baselines analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.secondary_experiments.mixture_analysis import run_mixture_analysis
else:
    from .mixture_analysis import run_mixture_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to llm_results.json containing all four baseline distributions.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=1.2,
        help="Symmetric Dirichlet prior alpha on mixture weights.",
    )
    parser.add_argument("--n-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--no-smooth", action="store_true", help="Skip lambda(t)=softmax(a+b log t).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Running mixture analysis on {args.input}")
    print(f"  dirichlet_alpha={args.dirichlet_alpha}")
    print(f"  smooth_context_model={not args.no_smooth}")
    paths = run_mixture_analysis(
        input_path=args.input,
        out_dir=args.out_dir,
        alpha=args.dirichlet_alpha,
        n_steps=args.n_steps,
        lr=args.lr,
        smooth=not args.no_smooth,
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
