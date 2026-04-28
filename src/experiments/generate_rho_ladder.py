"""Generate or patch LLM accuracy data for a subset of ρ values.

``vocabulary_tl_experiment.py`` now runs the full ρ ladder ({0.0, 0.2, 0.3,
0.4, 0.5, 0.6, 0.7, 0.8, 1.0}) by default.  This wrapper is kept for the case
where you only want to re-run a handful of ρ points (e.g. after tweaking
``SEGMENT_LEN`` or fixing a bad seed) without redoing the whole 9-point
ladder — it calls into ``run_condition_rho`` directly and merges the
resulting rows into the existing JSON.

Usage
-----
    # Just overlap, only the three rhos nearest 0.5:
    python src/experiments/generate_rho_ladder.py \\
        --condition overlap --rhos 0.4 0.5 0.6

    # Dry run — list what would be generated without loading the model:
    python src/experiments/generate_rho_ladder.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INITIAL = os.path.abspath(os.path.join(HERE, os.pardir, "initial_experiments"))
if INITIAL not in sys.path:
    sys.path.insert(0, INITIAL)

from data_loading import CONDITIONS  # noqa: E402

# Default ladder = everything vocabulary_tl_experiment.py would run on its own.
DEFAULT_RHO_LADDER = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def _merge_into_json(cond: str, rho: float, cond_results: dict) -> str:
    """Merge a single new ρ-row into the condition's vocabulary_tl JSON."""
    from vocabulary_tl_experiment import DATA_DIR  # noqa: E402
    path = os.path.join(DATA_DIR, f"{cond}.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    # ``cond_results`` from run_condition_rho returns ``{graph: {L: [vals]}}``.
    existing[str(rho)] = {
        graph: {str(L): vals for L, vals in accs.items()}
        for graph, accs in cond_results.items()
    }
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return path


def run(condition: str, rhos: list[float], *, dry_run: bool = False) -> None:
    if dry_run:
        print(f"  [dry-run] {condition}: would run rhos = {rhos}")
        return

    # Imports done lazily so --dry-run and --help don't trigger torch import.
    import vocabulary_tl_experiment as vte  # noqa: E402

    print(f"Loading {vte.MODEL_NAME} via TransformerLens...")
    model = vte.load_model()
    print(f"Model loaded on {model.cfg.device}.\n")

    for i, rho in enumerate(rhos):
        print(f"  [{i + 1}/{len(rhos)}] {condition}  ρ={rho}")
        # Use a seed_offset that doesn't collide with the existing {0, 1, 2}
        # used for ρ ∈ {0, 0.5, 1}.  The ladder uses offsets starting at 10.
        grid_accs, ring_accs, shared_accs = vte.run_condition_rho(
            model, condition, rho, vte.EVAL_LENGTHS, seed_offset=10 + i,
        )
        path = _merge_into_json(condition, rho, {
            "grid": grid_accs, "ring": ring_accs, "shared": shared_accs,
        })
        print(f"  merged   → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument(
        "--rhos", type=float, nargs="+", default=DEFAULT_RHO_LADDER,
        help="ρ values to add to the ladder (default: 0.2..0.8 step 0.1).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not load the model or run inference; just report what would happen.",
    )
    args = parser.parse_args()

    conditions = CONDITIONS if args.condition == "all" else [args.condition]
    rhos = list(args.rhos)

    print(f"{'(dry-run) ' if args.dry_run else ''}"
          f"ρ ladder: {rhos}  ×  conditions: {conditions}")
    for cond in conditions:
        run(cond, rhos, dry_run=args.dry_run)
    print("\nDone.  Re-run fit_baseline.py / fit_upgrade.py to use the new rows.")


if __name__ == "__main__":
    main()
