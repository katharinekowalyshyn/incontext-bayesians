"""CLI entrypoint for secondary PCA/representation analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.secondary_experiments.config import DEFAULT_CONFIG
    from src.secondary_experiments.pca_analysis import (
        DEFAULT_ENERGY_T,
        DEFAULT_LAYER,
        DEFAULT_SNAPSHOT_T,
        DEFAULT_WINDOW,
        run_pca_analysis,
    )
else:
    from .config import DEFAULT_CONFIG
    from .pca_analysis import (
        DEFAULT_ENERGY_T,
        DEFAULT_LAYER,
        DEFAULT_SNAPSHOT_T,
        DEFAULT_WINDOW,
        run_pca_analysis,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_name)
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG.device,
        choices=["cuda", "mps", "cpu"],
        help="Device for TransformerLens. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--true-graphs", nargs="+", default=["grid"])
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.seq_len)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONFIG.seeds))
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--snapshot-Ts", type=int, nargs="+", default=list(DEFAULT_SNAPSHOT_T))
    parser.add_argument("--energy-Ts", type=int, nargs="+", default=list(DEFAULT_ENERGY_T))
    parser.add_argument("--out-dir", default=str(DEFAULT_CONFIG.output_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        DEFAULT_CONFIG,
        model_name=args.model,
        device=args.device,
        seq_len=args.seq_len,
        seeds=tuple(args.seeds),
        true_graphs=tuple(args.true_graphs),
        output_dir=Path(args.out_dir),
    )
    print(f"Running secondary PCA analysis with {config.model_name}...")
    print(f"  device={config.device or 'auto'}")
    print(f"  true_graphs={config.true_graphs}")
    print(f"  seq_len={config.seq_len} layer={args.layer} window={args.window}")
    print(f"  snapshot_Ts={tuple(args.snapshot_Ts)}")
    print(f"  energy_Ts={tuple(args.energy_Ts)}")
    print(f"  seeds={config.seeds}")
    paths = run_pca_analysis(
        config=config,
        true_graphs=args.true_graphs,
        layer=args.layer,
        snapshot_Ts=tuple(args.snapshot_Ts),
        energy_Ts=tuple(args.energy_Ts),
        window=args.window,
    )
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
