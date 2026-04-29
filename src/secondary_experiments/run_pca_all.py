"""Run PCA analysis for every existing result folder in secondary_experiments/results/.

Discovers folders automatically: pure-graph subfolders under ``all_graphs/``
and top-level ``mix_*`` folders.  Skips any folder that already has a
``pca_snapshots_*.png`` file (re-run with ``--force`` to overwrite).

Quick-start:
    python -m src.secondary_experiments.run_pca_all

Background:
    nohup python -m src.secondary_experiments.run_pca_all \\
        > logs/pca_all_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown

Force rerun all:
    python -m src.secondary_experiments.run_pca_all --force

Specific subset:
    python -m src.secondary_experiments.run_pca_all \\
        --only mix_grid50_ring50 grid ring
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.secondary_experiments.config import DEFAULT_CONFIG
    from src.secondary_experiments.pca_analysis import (
        DEFAULT_ENERGY_T,
        DEFAULT_LAYER,
        DEFAULT_SNAPSHOT_T,
        DEFAULT_WINDOW,
    )
else:
    from .config import DEFAULT_CONFIG
    from .pca_analysis import (
        DEFAULT_ENERGY_T,
        DEFAULT_LAYER,
        DEFAULT_SNAPSHOT_T,
        DEFAULT_WINDOW,
    )

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Regex that parses mix folder names like "mix_grid70_ring30" into
# {"grid": 70.0, "ring": 30.0}.
_MIX_FOLDER_RE = re.compile(
    r"mix_([a-z]+)(\d+(?:p\d+)?)_([a-z]+)(\d+(?:p\d+)?)"
)


def _pct_str_to_float(s: str) -> float:
    return float(s.replace("p", "."))


def discover_conditions(results_dir: Path) -> list[dict]:
    """Return a list of condition dicts ready to dispatch.

    Each dict has keys:
        kind        : "pure" | "mix"
        out_dir     : Path to the result folder
        true_graph  : graph name (pure only)
        mix_spec    : "grid:70.0,ring:30.0" (mix only)
        mix_name    : folder name (mix only)
    """
    conditions: list[dict] = []

    # Pure-graph conditions live under all_graphs/{graph_name}/
    all_graphs_dir = results_dir / "all_graphs"
    if all_graphs_dir.is_dir():
        for graph_dir in sorted(all_graphs_dir.iterdir()):
            if graph_dir.is_dir() and (graph_dir / "llm_results.json").exists():
                conditions.append(
                    {
                        "kind": "pure",
                        "out_dir": graph_dir,
                        "true_graph": graph_dir.name,
                    }
                )

    # Mixed conditions are top-level mix_* folders.
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("mix_"):
            continue
        if not (folder / "llm_results.json").exists():
            continue
        m = _MIX_FOLDER_RE.fullmatch(folder.name)
        if m is None:
            print(f"  [warn] could not parse mix spec from folder name: {folder.name}")
            continue
        g_a, pct_a, g_b, pct_b = m.group(1), m.group(2), m.group(3), m.group(4)
        spec = f"{g_a}:{_pct_str_to_float(pct_a)},{g_b}:{_pct_str_to_float(pct_b)}"
        conditions.append(
            {
                "kind": "mix",
                "out_dir": folder,
                "mix_spec": spec,
                "mix_name": folder.name,
            }
        )

    return conditions


def already_done(condition: dict) -> bool:
    out_dir = condition["out_dir"]
    if condition["kind"] == "pure":
        stem = condition["true_graph"]
    else:
        stem = condition["mix_name"]
    return (out_dir / f"pca_snapshots_{stem}.png").exists()


def build_pca_cmd(
    condition: dict,
    args: argparse.Namespace,
) -> list[str]:
    base = [
        sys.executable,
        "-m",
        "src.secondary_experiments.run_pca",
        "--out-dir", str(condition["out_dir"]),
        "--model", args.model,
        "--seq-len", str(args.seq_len),
        "--layer", str(args.layer),
        "--window", str(args.window),
        "--dtype", args.dtype,
        "--seeds", *[str(s) for s in args.seeds],
        "--snapshot-Ts", *[str(t) for t in args.snapshot_Ts],
        "--energy-Ts", *[str(t) for t in args.energy_Ts],
    ]
    if args.device:
        base.extend(["--device", args.device])

    if condition["kind"] == "pure":
        base.extend(["--true-graphs", condition["true_graph"]])
    else:
        base.extend([
            "--mix", condition["mix_spec"],
            "--mix-name", condition["mix_name"],
        ])
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA for every secondary-experiment result folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="FOLDER",
        help="Process only these folder names (e.g. grid mix_grid50_ring50).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if pca_snapshots_*.png already exists.",
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
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_CONFIG.seq_len)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_CONFIG.seeds))
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--snapshot-Ts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SNAPSHOT_T),
    )
    parser.add_argument(
        "--energy-Ts",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENERGY_T),
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="Root directory to scan for result folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    conditions = discover_conditions(results_dir)
    if not conditions:
        print(f"No result folders with llm_results.json found under {results_dir}.")
        return

    if args.only:
        only_set = set(args.only)
        conditions = [
            c for c in conditions
            if (c["true_graph"] if c["kind"] == "pure" else c["mix_name"]) in only_set
        ]
        if not conditions:
            print(f"None of {sorted(only_set)} matched any discovered conditions.")
            return

    total = len(conditions)
    skipped = 0
    print(f"Discovered {total} condition(s) under {results_dir}")

    for idx, cond in enumerate(conditions, start=1):
        name = cond.get("true_graph") or cond.get("mix_name")
        if not args.force and already_done(cond):
            print(f"[{idx}/{total}] SKIP  {name}  (already has pca_snapshots_*.png)")
            skipped += 1
            continue

        cmd = build_pca_cmd(cond, args)
        print(f"\n[{idx}/{total}] {cond['kind'].upper()}  {name}")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[{idx}/{total}] Done  {name}")

    remaining = total - skipped
    print(f"\nFinished.  Ran {remaining} PCA job(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
