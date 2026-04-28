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

# Default ρ ladder — ring fraction increases from 0 to 1.
# ρ=0 → pure graph_a;  ρ=1 → pure graph_b.
# Interior points are the same as the primary vocabulary_tl ladder.
DEFAULT_RHO_LADDER: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def rho_to_mix_spec(graph_a: str, graph_b: str, rho: float) -> str:
    """Convert a (graph_a, graph_b, ρ) triple into a run_experiment --mix spec.

    ρ is the fraction of transitions taken from graph_b; (1-ρ) from graph_a.
    Weights are given as percentages rounded to one decimal place.
    """
    pct_b = round(rho * 100, 1)
    pct_a = round((1.0 - rho) * 100, 1)
    return f"{graph_a}:{pct_a},{graph_b}:{pct_b}"


def rho_folder_name(graph_a: str, graph_b: str, rho: float) -> str:
    """Stable folder name for a ρ-ladder cell, e.g. ``mix_grid80_ring20``."""
    pct_b = int(round(rho * 100))
    pct_a = 100 - pct_b
    return f"mix_{graph_a}{pct_a}_{graph_b}{pct_b}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run secondary graph-baseline experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=list(DEFAULT_CONFIG.true_graphs),
        help="Pure-graph conditions to run.",
    )
    parser.add_argument("--no-pure", action="store_true", help="Do not run pure graph experiments.")
    parser.add_argument(
        "--mixes",
        nargs="*",
        default=[],
        help="Individual mixed transition ratios to run, e.g. grid:80,ring:20.",
    )

    # ── ρ-ladder shorthand ────────────────────────────────────────────────────
    ladder_group = parser.add_argument_group("ρ-ladder (shorthand for full mixing sweep)")
    ladder_group.add_argument(
        "--rho-ladder",
        action="store_true",
        help=(
            "Run the full ρ ladder for --ladder-graphs instead of specifying "
            "--mixes manually. Ladder points: "
            + ", ".join(str(r) for r in DEFAULT_RHO_LADDER)
        ),
    )
    ladder_group.add_argument(
        "--ladder-graphs",
        nargs=2,
        metavar=("GRAPH_A", "GRAPH_B"),
        default=["grid", "ring"],
        help="Two graph names to ladder over (default: grid ring). "
             "ρ=0 → pure GRAPH_A; ρ=1 → pure GRAPH_B.",
    )
    ladder_group.add_argument(
        "--rho-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_RHO_LADDER),
        metavar="RHO",
        help="Custom ρ values to use with --rho-ladder (default: full 9-point ladder).",
    )

    parser.add_argument("--out-root", default=str(DEFAULT_CONFIG.output_dir))
    parser.add_argument("--model", default=DEFAULT_CONFIG.model_name)
    parser.add_argument(
        "--device",
        default=DEFAULT_CONFIG.device,
        choices=["cuda", "mps", "cpu"],
        help="Device for TransformerLens. Defaults to cuda, then mps, then cpu.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.seq_len)
    parser.add_argument(
        "--eval-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFIG.eval_lengths),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONFIG.seeds))
    parser.add_argument("--epsilon", type=float, default=DEFAULT_CONFIG.epsilon)
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG.alpha)
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model weight dtype forwarded to every subprocess (default: float16).",
    )
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

    # ── Build the experiment list ─────────────────────────────────────────────
    experiments: list[tuple[str, str, str]] = []

    if not args.no_pure:
        graphs_for_pure = list(args.graphs)
        # When using --rho-ladder, automatically include the two endpoint pure
        # runs (ρ=0 and ρ=1) even if they aren't in --graphs, so the output
        # directory is self-contained.
        if args.rho_ladder:
            for g in args.ladder_graphs:
                if g not in graphs_for_pure:
                    graphs_for_pure.append(g)
        experiments.extend(("pure", graph, graph) for graph in graphs_for_pure)

    # Manual --mixes specs (unchanged behaviour).
    experiments.extend(("mix", spec, mix_folder_name(spec)) for spec in args.mixes)

    # ρ-ladder: expand into one mix experiment per interior ρ value.
    if args.rho_ladder:
        graph_a, graph_b = args.ladder_graphs
        for rho in sorted(set(args.rho_values)):
            if rho <= 0.0 or rho >= 1.0:
                # Endpoints are already covered by the pure-graph runs.
                continue
            spec = rho_to_mix_spec(graph_a, graph_b, rho)
            folder = rho_folder_name(graph_a, graph_b, rho)
            experiments.append(("mix", spec, folder))

    print(f"Running {len(experiments)} experiment(s) sequentially.")
    print(f"Output root: {out_root}")
    print(f"Device: {args.device or 'auto'}")
    if args.rho_ladder:
        graph_a, graph_b = args.ladder_graphs
        interior = [r for r in sorted(set(args.rho_values)) if 0.0 < r < 1.0]
        print(
            f"ρ-ladder: {graph_a}/{graph_b}, "
            f"{len(interior)} interior points: {interior}"
        )

    for idx, (kind, target, folder_name) in enumerate(experiments, start=1):
        print(f"\n[{idx}/{len(experiments)}] Starting {kind} experiment: {target} → {folder_name}/")
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
        behavior_cmd.extend(["--dtype", args.dtype])
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
