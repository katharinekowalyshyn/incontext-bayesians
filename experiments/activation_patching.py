"""Residual-stream activation patching between graph-family contexts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prompt_pairs import build_prompt_pairs
from src.interventions.hooks import available_layers
from src.interventions.patching import PatchingSpec, patching_rows_for_pair
from src.metrics.graph_logit_diff import standard_error
from src.secondary_experiments.graphs import build_candidate_graphs
from src.secondary_experiments.vocabulary import build_token_map


@dataclass(frozen=True)
class ActivationPatchingConfig:
    model: str
    clean_graph: str
    corrupt_graph: str
    num_pairs: int
    context_length: int
    seed: int
    output_dir: str
    layers: tuple[int, ...] | None
    positions: tuple[str, ...]
    activation: str
    dtype: str
    device: str | None
    top_k: int
    min_abs_denom: float
    resume: bool


def parse_args() -> ActivationPatchingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--clean_graph", default="grid")
    parser.add_argument("--corrupt_graph", default="ring")
    parser.add_argument("--num_pairs", type=int, default=500)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["final"],
        choices=[
            "final",
            "all",
            "same_token_occurrences",
            "edge_observation_positions",
            "mean_context",
        ],
    )
    parser.add_argument(
        "--activation",
        default="resid_post",
        choices=["resid_pre", "resid_post", "attn_out", "mlp_out"],
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_abs_denom", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return ActivationPatchingConfig(
        model=args.model,
        clean_graph=args.clean_graph,
        corrupt_graph=args.corrupt_graph,
        num_pairs=args.num_pairs,
        context_length=args.context_length,
        seed=args.seed,
        output_dir=args.output_dir,
        layers=None if args.layers is None else tuple(args.layers),
        positions=tuple(args.positions),
        activation=args.activation,
        dtype=args.dtype,
        device=args.device,
        top_k=args.top_k,
        min_abs_denom=args.min_abs_denom,
        resume=args.resume,
    )


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys: set[tuple] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(
                (
                    row["pair_id"],
                    row["layer"],
                    row["position_strategy"],
                    tuple(row["positions"]),
                    row.get("activation", "resid_post"),
                )
            )
    return keys


def write_summary(rows_path: Path, out_path: Path) -> None:
    values: list[float] = []
    excluded = 0
    total = 0
    by_layer: dict[int, list[float]] = {}
    if rows_path.exists():
        with rows_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                row = json.loads(line)
                value = row.get("normalized_effect")
                if value is None:
                    excluded += 1
                    continue
                values.append(float(value))
                by_layer.setdefault(int(row["layer"]), []).append(float(value))
    summary = {
        "num_rows": total,
        "num_usable_normalized_effects": len(values),
        "num_excluded_small_denominator": excluded,
        "mean_normalized_effect": float(np.mean(values)) if values else None,
        "se_normalized_effect": standard_error(values),
        "by_layer": {
            str(layer): {
                "n": len(layer_values),
                "mean": float(np.mean(layer_values)),
                "se": standard_error(layer_values),
            }
            for layer, layer_values in sorted(by_layer.items())
        },
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    config = parse_args()
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(asdict(config), f, indent=2)

    from src.secondary_experiments.llm_inference import load_model

    graphs = build_candidate_graphs()
    model = load_model(config.model, device=config.device, dtype=config.dtype)
    token_map = build_token_map(model)
    layers = tuple(available_layers(model) if config.layers is None else config.layers)
    spec = PatchingSpec(
        layers=layers,
        position_strategies=config.positions,
        activation=config.activation,
        top_k=config.top_k,
        min_abs_denom=config.min_abs_denom,
    )
    pairs = build_prompt_pairs(
        clean_graph=config.clean_graph,
        corrupt_graph=config.corrupt_graph,
        num_pairs=config.num_pairs,
        context_length=config.context_length,
        seed=config.seed,
        graphs=graphs,
    )

    rows_path = out_dir / "rows.jsonl"
    done = existing_keys(rows_path) if config.resume else set()
    for idx, pair in enumerate(pairs, start=1):
        print(f"[patching] pair {idx}/{len(pairs)} final={pair.final_word}")
        rows = patching_rows_for_pair(model, pair, graphs, token_map, spec)
        if done:
            rows = [
                row
                for row in rows
                if (
                    row["pair_id"],
                    row["layer"],
                    row["position_strategy"],
                    tuple(row["positions"]),
                    row.get("activation", "resid_post"),
                )
                not in done
            ]
        append_jsonl(rows_path, rows)

    write_summary(rows_path, out_dir / "summary.json")
    print(f"Saved rows to {rows_path}")
    print(f"Saved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
