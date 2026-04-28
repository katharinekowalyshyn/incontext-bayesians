"""Run secondary experiments sequentially, one output folder per graph."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.secondary_experiments.config import DEFAULT_CONFIG
else:
    from .config import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_CONFIG.true_graphs))
    parser.add_argument("--no-pure", action="store_true", help="Do not run pure graph experiments.")
    parser.add_argument(
        "--mixes",
        nargs="*",
        default=[],
        help="Mixed transition ratios to run, e.g. grid:80,ring:20 chain:50,star:50.",
    )
    parser.add_argument("--out-root", default=str(DEFAULT_CONFIG.output_dir / "sequential"))
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_name)
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG.device,
        choices=["cuda", "mps", "cpu"],
        help="Device for TransformerLens. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.seq_len)
    parser.add_argument("--eval-lengths", type=int, nargs="+", default=list(DEFAULT_CONFIG.eval_lengths))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONFIG.seeds))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_CONFIG.epsilon)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--include-pca", action="store_true")
    parser.add_argument("--pca-layer", type=int, default=26)
    parser.add_argument("--pca-window", type=int, default=50)
    return parser.parse_args()


def mix_folder_name(spec: str) -> str:
    return "mix_" + spec.replace(":", "").replace(",", "_").replace(".", "p")


def _append_option(cmd: list[str], flag: str, value) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def run_command(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = [] if args.no_pure else [("pure", graph, graph) for graph in args.graphs]
    experiments.extend(("mix", spec, mix_folder_name(spec)) for spec in args.mixes)

    print(f"Running {len(experiments)} experiment(s) sequentially.")
    print(f"Output root: {out_root}")
    print(f"Device: {args.device or 'auto'}")

    for idx, (kind, target, folder_name) in enumerate(experiments, start=1):
        print(f"\n[{idx}/{len(experiments)}] Starting {kind} experiment: {target}")
        out_dir = out_root / folder_name

        behavior_cmd = [
            sys.executable,
            "-m",
            "src.secondary_experiments.run_experiment",
            "--out-dir",
            str(out_dir),
            "--model",
            args.model,
            "--seq-len",
            str(args.seq_len),
            "--epsilon",
            str(args.epsilon),
            "--alpha",
            str(args.alpha),
            "--eval-lengths",
            *[str(x) for x in args.eval_lengths],
            "--seeds",
            *[str(x) for x in args.seeds],
        ]
        if kind == "pure":
            behavior_cmd.extend(["--true-graphs", target])
        else:
            behavior_cmd.extend(["--mix", target, "--mix-name", folder_name])
        _append_option(behavior_cmd, "--device", args.device)
        if args.skip_llm:
            behavior_cmd.append("--skip-llm")
        run_command(behavior_cmd)
        print(f"[{idx}/{len(experiments)}] Finished behavior run: {target}")

        if args.include_pca and not args.skip_llm:
            if kind == "mix":
                print(f"Skipping PCA for mixed run {target}: PCA currently expects a single true graph.")
                continue
            pca_cmd = [
                sys.executable,
                "-m",
                "src.secondary_experiments.run_pca",
                "--true-graphs",
                target,
                "--out-dir",
                str(out_dir),
                "--model",
                args.model,
                "--seq-len",
                str(args.seq_len),
                "--layer",
                str(args.pca_layer),
                "--window",
                str(args.pca_window),
                "--seeds",
                *[str(x) for x in args.seeds],
            ]
            _append_option(pca_cmd, "--device", args.device)
            run_command(pca_cmd)
            print(f"[{idx}/{len(experiments)}] Finished PCA run: {target}")

    print("\nAll requested experiments finished.")


if __name__ == "__main__":
    main()
