"""CLI entrypoint for secondary PCA/representation analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

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
        run_pca_analysis_mixed,
    )
else:
    from .config import DEFAULT_CONFIG
    from .pca_analysis import (
        DEFAULT_ENERGY_T,
        DEFAULT_LAYER,
        DEFAULT_SNAPSHOT_T,
        DEFAULT_WINDOW,
        run_pca_analysis,
        run_pca_analysis_mixed,
    )


def parse_mix_spec(spec: str | None) -> dict[str, float] | None:
    if spec is None:
        return None
    result: dict[str, float] = {}
    for part in spec.split(","):
        if ":" not in part:
            raise ValueError(f"Bad mix component {part!r}; expected name:weight.")
        name, weight = part.split(":", 1)
        result[name.strip()] = float(weight)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA / representation analysis for secondary graph experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_name)
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG.device,
        choices=["cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype (default: float16).",
    )
    # Pure-graph mode
    parser.add_argument(
        "--true-graphs",
        nargs="+",
        default=["grid"],
        help="One or more pure true graphs to analyse (ignored when --mix is given).",
    )
    # Mixed mode
    parser.add_argument(
        "--mix",
        default=None,
        help="Mixed transition ratios, e.g. 'grid:70.0,ring:30.0'. "
             "Overrides --true-graphs.",
    )
    parser.add_argument(
        "--mix-name",
        default=None,
        help="Label used in output filenames for a mixed run (e.g. mix_grid70_ring30).",
    )
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
    _dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    config = replace(
        DEFAULT_CONFIG,
        model_name=args.model,
        device=args.device,
        dtype=_dtype_map[args.dtype],
        seq_len=args.seq_len,
        seeds=tuple(args.seeds),
        true_graphs=tuple(args.true_graphs),
        output_dir=Path(args.out_dir),
    )

    mix_ratios = parse_mix_spec(args.mix)

    if mix_ratios is not None:
        mix_name = args.mix_name or (
            "mix_" + "_".join(f"{k}{int(round(v))}" for k, v in mix_ratios.items())
        )
        print(f"Running secondary PCA analysis (mixed) with {config.model_name}...")
        print(f"  mix={dict(mix_ratios)} → {mix_name}")
        print(f"  seq_len={config.seq_len}  layer={args.layer}  window={args.window}")
        print(f"  snapshot_Ts={tuple(args.snapshot_Ts)}")
        print(f"  seeds={config.seeds}")
        paths = run_pca_analysis_mixed(
            config=config,
            mix_ratios=mix_ratios,
            mix_name=mix_name,
            layer=args.layer,
            snapshot_Ts=tuple(args.snapshot_Ts),
            energy_Ts=tuple(args.energy_Ts),
            window=args.window,
        )
    else:
        print(f"Running secondary PCA analysis with {config.model_name}...")
        print(f"  true_graphs={config.true_graphs}")
        print(f"  seq_len={config.seq_len}  layer={args.layer}  window={args.window}")
        print(f"  snapshot_Ts={tuple(args.snapshot_Ts)}")
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
