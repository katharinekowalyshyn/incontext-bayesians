"""CLI entrypoint for secondary graph-baseline experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.secondary_experiments.config import DEFAULT_CONFIG
    from src.secondary_experiments.experiment import run_baseline_only, run_with_llm, save_json
else:
    from .config import DEFAULT_CONFIG
    from .experiment import run_baseline_only, run_with_llm, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Run only baselines.")
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_name)
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG.device,
        choices=["cuda", "mps", "cpu"],
        help="Device for TransformerLens. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype (default: float16, ~16 GB for Llama-3.1-8B).",
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_CONFIG.epsilon)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.seq_len)
    parser.add_argument("--eval-lengths", type=int, nargs="+", default=list(DEFAULT_CONFIG.eval_lengths))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONFIG.seeds))
    parser.add_argument("--true-graphs", nargs="+", default=list(DEFAULT_CONFIG.true_graphs))
    parser.add_argument(
        "--mix",
        default=None,
        help="Mixed transition ratios, e.g. 'grid:80,ring:20'. Overrides --true-graphs.",
    )
    parser.add_argument(
        "--mix-name",
        default=None,
        help="Label used in outputs for a mixed run.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_CONFIG.output_dir))
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def parse_mix_spec(spec: str | None) -> tuple[tuple[str, float], ...] | None:
    if spec is None:
        return None
    pairs = []
    for part in spec.split(","):
        if ":" not in part:
            raise ValueError(f"Bad mix component {part!r}; expected name:weight.")
        name, weight = part.split(":", 1)
        pairs.append((name.strip(), float(weight)))
    return tuple(pairs)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
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
        epsilon=args.epsilon,
        alpha=args.alpha,
        seq_len=args.seq_len,
        eval_lengths=tuple(args.eval_lengths),
        seeds=tuple(args.seeds),
        true_graphs=tuple(args.true_graphs),
        mix_ratios=parse_mix_spec(args.mix),
        mix_name=args.mix_name,
        output_dir=out_dir,
    )

    if args.skip_llm:
        print("Running baseline-only secondary experiment...")
        print(f"  true_graphs={config.true_graphs}")
        if config.mix_ratios is not None:
            print(f"  mix={dict(config.mix_ratios)} name={config.mix_name}")
        print(f"  seq_len={config.seq_len} eval_lengths={config.eval_lengths}")
        print(f"  seeds={config.seeds}")
        rows = run_baseline_only(config)
        stem = "baseline_results"
    else:
        print(f"Running full secondary experiment with {config.model_name}...")
        print(f"  device={config.device or 'auto'}")
        print(f"  true_graphs={config.true_graphs}")
        if config.mix_ratios is not None:
            print(f"  mix={dict(config.mix_ratios)} name={config.mix_name}")
        print(f"  seq_len={config.seq_len} eval_lengths={config.eval_lengths}")
        print(f"  seeds={config.seeds}")
        rows = run_with_llm(config)
        stem = "llm_results"

    print(f"Saving {len(rows)} rows...")
    json_path = save_json(rows, out_dir / f"{stem}.json")
    print(f"Saved {json_path}")

    if not args.no_plots:
        if __package__ in {None, ""}:
            from src.secondary_experiments.plotting import make_all_plots
        else:
            from .plotting import make_all_plots

        paths = make_all_plots(rows, out_dir)
        for path in paths:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
