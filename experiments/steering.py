"""Graph-difference residual-stream steering experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prompt_pairs import build_prompt_pairs, no_context_sequence
from src.interventions.hooks import make_position_selector, steered_logits, logits_for_sequence
from src.interventions.steering import compute_steering_vectors
from src.metrics.graph_logit_diff import (
    graph_contrast_metrics,
    kl_to_reference_vocab_distribution,
    normalized_effect,
    standard_error,
)
from src.secondary_experiments.graphs import build_candidate_graphs
from src.secondary_experiments.vocabulary import build_token_map


@dataclass(frozen=True)
class SteeringConfig:
    model: str
    source_graph: str
    target_graph: str
    num_train_contexts: int
    num_eval_contexts: int
    context_length: int
    layers: tuple[int, ...]
    alphas: tuple[float, ...]
    seed: int
    output_dir: str
    activation: str
    position_strategy: str
    dtype: str
    device: str | None
    top_k: int
    min_abs_denom: float
    no_context_eval: bool


def parse_args() -> SteeringConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--source_graph", default="grid")
    parser.add_argument("--target_graph", default="ring")
    parser.add_argument("--num_train_contexts", type=int, default=1000)
    parser.add_argument("--num_eval_contexts", type=int, default=500)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--activation",
        default="resid_post",
        choices=["resid_pre", "resid_post", "attn_out", "mlp_out"],
    )
    parser.add_argument(
        "--position_strategy",
        default="final",
        choices=["final", "mean_context", "same_token_occurrences", "edge_observation_positions"],
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_abs_denom", type=float, default=1e-6)
    parser.add_argument("--no_context_eval", action="store_true")
    args = parser.parse_args()
    return SteeringConfig(
        model=args.model,
        source_graph=args.source_graph,
        target_graph=args.target_graph,
        num_train_contexts=args.num_train_contexts,
        num_eval_contexts=args.num_eval_contexts,
        context_length=args.context_length,
        layers=tuple(args.layers),
        alphas=tuple(args.alphas),
        seed=args.seed,
        output_dir=args.output_dir,
        activation=args.activation,
        position_strategy=args.position_strategy,
        dtype=args.dtype,
        device=args.device,
        top_k=args.top_k,
        min_abs_denom=args.min_abs_denom,
        no_context_eval=args.no_context_eval,
    )


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_vectors(path: Path, vector_set) -> None:
    payload = {
        "source_graph": vector_set.source_graph,
        "target_graph": vector_set.target_graph,
        "layers": vector_set.layers,
        "position_strategy": vector_set.position_strategy,
        "vectors": {str(k): v.cpu() for k, v in vector_set.vectors.items()},
        "shuffled_vectors": {str(k): v.cpu() for k, v in vector_set.shuffled_vectors.items()},
        "random_vectors": {str(k): v.cpu() for k, v in vector_set.random_vectors.items()},
    }
    torch.save(payload, path)


def position_groups(strategy: str, sequence: tuple[str, ...]) -> list[int]:
    groups = make_position_selector(strategy)(sequence)
    return sorted({pos for group in groups for pos in group})


def rows_for_eval_pair(
    model,
    pair,
    graphs,
    token_map,
    vector_set,
    config: SteeringConfig,
) -> list[dict]:
    source_graph = graphs[config.source_graph]
    target_graph = graphs[config.target_graph]
    source_sequence = tuple(no_context_sequence(pair.final_word) if config.no_context_eval else pair.clean_sequence)
    target_sequence = tuple(no_context_sequence(pair.final_word) if config.no_context_eval else pair.corrupt_sequence)

    source_logits = logits_for_sequence(model, source_sequence, token_map=token_map)
    target_logits = logits_for_sequence(model, target_sequence, token_map=token_map)
    source_metrics = graph_contrast_metrics(
        source_logits,
        source_graph,
        target_graph,
        pair.final_word,
        token_map,
        context_for_seen_edges=target_sequence,
        top_k=config.top_k,
    )
    target_metrics = graph_contrast_metrics(
        target_logits,
        source_graph,
        target_graph,
        pair.final_word,
        token_map,
        context_for_seen_edges=target_sequence,
        top_k=config.top_k,
    )
    source_vocab_dist = source_metrics.vocab_distribution

    base = {
        "model": config.model,
        "seed": config.seed,
        "pair_id": pair.pair_id,
        "source_graph": config.source_graph,
        "target_graph": config.target_graph,
        "context_length": len(target_sequence),
        "train_context_length": config.context_length,
        "final_token": pair.final_word,
        "activation": config.activation,
        "position_strategy": config.position_strategy,
        "no_context_eval": config.no_context_eval,
        "source_sequence": list(source_sequence),
        "target_sequence": list(target_sequence),
        "source_metric": source_metrics.graph_logit_diff,
        "target_metric": target_metrics.graph_logit_diff,
        **source_metrics.to_json_dict(prefix="source_"),
        **target_metrics.to_json_dict(prefix="target_"),
    }

    controls = {
        "real": vector_set.vectors,
        "random_norm_matched": vector_set.random_vectors,
        "shuffled_label": vector_set.shuffled_vectors,
    }
    rows: list[dict] = []
    for layer in config.layers:
        for control_name, vectors in controls.items():
            vector = vectors[int(layer)]
            for eval_direction, sequence, sign, start_metric, end_metric in (
                (
                    "target_plus_source_minus_target",
                    target_sequence,
                    1.0,
                    target_metrics.graph_logit_diff,
                    source_metrics.graph_logit_diff,
                ),
                (
                    "source_minus_source_minus_target",
                    source_sequence,
                    -1.0,
                    source_metrics.graph_logit_diff,
                    target_metrics.graph_logit_diff,
                ),
            ):
                positions = position_groups(config.position_strategy, sequence)
                for alpha in config.alphas:
                    applied_alpha = sign * float(alpha)
                    logits = steered_logits(
                        model,
                        sequence,
                        layer=int(layer),
                        vector=vector,
                        alpha=applied_alpha,
                        positions=positions,
                        activation=config.activation,
                        token_map=token_map,
                    )
                    metrics = graph_contrast_metrics(
                        logits,
                        source_graph,
                        target_graph,
                        pair.final_word,
                        token_map,
                        context_for_seen_edges=sequence,
                        top_k=config.top_k,
                    )
                    norm, usable, denom = normalized_effect(
                        metrics.graph_logit_diff,
                        clean_metric=end_metric,
                        corrupt_metric=start_metric,
                        min_abs_denom=config.min_abs_denom,
                    )
                    rows.append(
                        {
                            **base,
                            "layer": int(layer),
                            "alpha": float(alpha),
                            "applied_alpha": applied_alpha,
                            "control": control_name,
                            "eval_direction": eval_direction,
                            "positions": [int(pos) for pos in positions],
                            "steered_metric": metrics.graph_logit_diff,
                            "normalized_effect": norm,
                            "normalization_denominator": denom,
                            "normalization_usable": usable,
                            "kl_to_clean_source_distribution": kl_to_reference_vocab_distribution(
                                logits,
                                source_vocab_dist,
                                token_map,
                            ),
                            **metrics.to_json_dict(prefix="steered_"),
                        }
                    )
    return rows


def write_summary(rows_path: Path, out_path: Path) -> None:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    total = 0
    with rows_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            value = row.get("normalized_effect")
            if value is None:
                continue
            key = (row["control"], row["eval_direction"], int(row["layer"]))
            grouped.setdefault(key, []).append(float(value))
    summary = {
        "num_rows": total,
        "groups": {
            "|".join(map(str, key)): {
                "n": len(values),
                "mean_normalized_effect": float(np.mean(values)),
                "se_normalized_effect": standard_error(values),
            }
            for key, values in sorted(grouped.items())
            if values
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
    print("[steering] computing train graph-difference vectors...")
    vector_set = compute_steering_vectors(
        model,
        graphs,
        source_graph=config.source_graph,
        target_graph=config.target_graph,
        num_train_contexts=config.num_train_contexts,
        context_length=config.context_length,
        layers=config.layers,
        seed=config.seed,
        token_map=token_map,
        activation=config.activation,
        position_strategy=config.position_strategy,
    )
    save_vectors(out_dir / "steering_vectors.pt", vector_set)

    pairs = build_prompt_pairs(
        clean_graph=config.source_graph,
        corrupt_graph=config.target_graph,
        num_pairs=config.num_eval_contexts,
        context_length=config.context_length,
        seed=config.seed + 100_000,
        graphs=graphs,
    )
    rows_path = out_dir / "rows.jsonl"
    for idx, pair in enumerate(pairs, start=1):
        print(f"[steering] eval pair {idx}/{len(pairs)} final={pair.final_word}")
        append_jsonl(rows_path, rows_for_eval_pair(model, pair, graphs, token_map, vector_set, config))
    write_summary(rows_path, out_dir / "summary.json")
    print(f"Saved rows to {rows_path}")
    print(f"Saved vectors to {out_dir / 'steering_vectors.pt'}")


if __name__ == "__main__":
    main()
