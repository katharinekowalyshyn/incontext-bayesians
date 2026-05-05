"""Attention-head candidate scoring and ablation for graph prompts."""

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

from src.data.prompt_pairs import build_prompt_pairs
from src.interventions.hooks import ablated_head_logits, available_layers, logits_for_sequence, tokens_from_sequence
from src.metrics.graph_logit_diff import graph_contrast_metrics, standard_error
from src.secondary_experiments.graphs import build_candidate_graphs
from src.secondary_experiments.vocabulary import build_token_map


@dataclass(frozen=True)
class HeadAblationConfig:
    model: str
    clean_graph: str
    corrupt_graph: str
    num_pairs: int
    context_length: int
    seed: int
    output_dir: str
    layers: tuple[int, ...] | None
    top_candidate_heads: int
    dtype: str
    device: str | None
    top_k: int
    min_abs_denom: float


def parse_args() -> HeadAblationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--clean_graph", default="grid")
    parser.add_argument("--corrupt_graph", default="ring")
    parser.add_argument("--num_pairs", type=int, default=200)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--top_candidate_heads", type=int, default=32)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_abs_denom", type=float, default=1e-6)
    args = parser.parse_args()
    return HeadAblationConfig(
        model=args.model,
        clean_graph=args.clean_graph,
        corrupt_graph=args.corrupt_graph,
        num_pairs=args.num_pairs,
        context_length=args.context_length,
        seed=args.seed,
        output_dir=args.output_dir,
        layers=None if args.layers is None else tuple(args.layers),
        top_candidate_heads=args.top_candidate_heads,
        dtype=args.dtype,
        device=args.device,
        top_k=args.top_k,
        min_abs_denom=args.min_abs_denom,
    )


def positions_for_same_token(sequence: tuple[str, ...]) -> list[int]:
    current = sequence[-1]
    return [i + 1 for i, word in enumerate(sequence[:-1]) if word == current]


def positions_for_edge_observations(sequence: tuple[str, ...]) -> list[int]:
    current = sequence[-1]
    positions: set[int] = set()
    for idx, (left, right) in enumerate(zip(sequence[:-1], sequence[1:])):
        if left == current or right == current:
            positions.add(idx + 1)
            positions.add(idx + 2)
    return sorted(positions)


def score_heads_for_sequence(model, sequence, layers, token_map) -> list[dict]:
    tokens = tokens_from_sequence(model, sequence, token_map=token_map)
    names = [f"blocks.{layer}.attn.hook_pattern" for layer in layers]
    same_positions = positions_for_same_token(tuple(sequence))
    edge_positions = positions_for_edge_observations(tuple(sequence))
    final_pos = tokens.shape[1] - 1
    rows = []
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names)
    for layer in layers:
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, final_pos, :].detach().float().cpu()
        for head in range(pattern.shape[0]):
            same_score = float(pattern[head, same_positions].sum().item()) if same_positions else 0.0
            edge_score = float(pattern[head, edge_positions].sum().item()) if edge_positions else 0.0
            rows.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    "same_token_attention": same_score,
                    "edge_observation_attention": edge_score,
                    "combined_attention": same_score + edge_score,
                }
            )
    return rows


def select_candidate_heads(score_rows: list[dict], top_k: int) -> list[tuple[int, int]]:
    aggregate: dict[tuple[int, int], list[float]] = {}
    for row in score_rows:
        aggregate.setdefault((row["layer"], row["head"]), []).append(row["combined_attention"])
    ranked = sorted(
        (
            (layer, head, float(np.mean(scores)))
            for (layer, head), scores in aggregate.items()
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    return [(layer, head) for layer, head, _ in ranked[:top_k]]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_summary(rows_path: Path, out_path: Path) -> None:
    grouped: dict[str, list[float]] = {}
    with rows_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("normalized_ablation_effect")
            if value is None:
                continue
            key = f"L{row['layer']}H{row['head']}"
            grouped.setdefault(key, []).append(float(value))
    summary = {
        key: {
            "n": len(values),
            "mean_normalized_ablation_effect": float(np.mean(values)),
            "se": standard_error(values),
        }
        for key, values in sorted(grouped.items())
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
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    token_map = build_token_map(model)
    layers = tuple(available_layers(model) if config.layers is None else config.layers)
    pairs = build_prompt_pairs(
        config.clean_graph,
        config.corrupt_graph,
        config.num_pairs,
        config.context_length,
        config.seed,
        graphs=graphs,
    )

    print("[head] scoring attention candidates...")
    score_rows: list[dict] = []
    for pair in pairs:
        score_rows.extend(score_heads_for_sequence(model, pair.corrupt_sequence, layers, token_map))
    append_jsonl(out_dir / "candidate_scores.jsonl", score_rows)
    candidate_heads = select_candidate_heads(score_rows, config.top_candidate_heads)
    with (out_dir / "candidate_heads.json").open("w") as f:
        json.dump([{"layer": layer, "head": head} for layer, head in candidate_heads], f, indent=2)

    rows_path = out_dir / "rows.jsonl"
    clean_graph = graphs[config.clean_graph]
    corrupt_graph = graphs[config.corrupt_graph]
    for idx, pair in enumerate(pairs, start=1):
        print(f"[head] ablation pair {idx}/{len(pairs)} final={pair.final_word}")
        clean_logits = logits_for_sequence(model, pair.clean_sequence, token_map=token_map)
        corrupt_logits = logits_for_sequence(model, pair.corrupt_sequence, token_map=token_map)
        clean_metrics = graph_contrast_metrics(
            clean_logits,
            clean_graph,
            corrupt_graph,
            pair.final_word,
            token_map,
            context_for_seen_edges=pair.corrupt_sequence,
            top_k=config.top_k,
        )
        corrupt_metrics = graph_contrast_metrics(
            corrupt_logits,
            clean_graph,
            corrupt_graph,
            pair.final_word,
            token_map,
            context_for_seen_edges=pair.corrupt_sequence,
            top_k=config.top_k,
        )
        out_rows = []
        for layer, head in candidate_heads:
            logits = ablated_head_logits(model, pair.corrupt_sequence, layer, head, token_map=token_map)
            metrics = graph_contrast_metrics(
                logits,
                clean_graph,
                corrupt_graph,
                pair.final_word,
                token_map,
                context_for_seen_edges=pair.corrupt_sequence,
                top_k=config.top_k,
            )
            # Positive means ablation reduces the corrupt run's clean-minus-corrupt graph signal.
            denom = clean_metrics.graph_logit_diff - corrupt_metrics.graph_logit_diff
            usable = abs(denom) >= config.min_abs_denom
            norm = None if not usable else (corrupt_metrics.graph_logit_diff - metrics.graph_logit_diff) / denom
            out_rows.append(
                {
                    "model": config.model,
                    "seed": config.seed,
                    "pair_id": pair.pair_id,
                    "clean_graph": config.clean_graph,
                    "corrupt_graph": config.corrupt_graph,
                    "context_length": config.context_length,
                    "final_token": pair.final_word,
                    "layer": int(layer),
                    "head": int(head),
                    "clean_metric": clean_metrics.graph_logit_diff,
                    "corrupt_metric": corrupt_metrics.graph_logit_diff,
                    "ablated_metric": metrics.graph_logit_diff,
                    "normalized_ablation_effect": norm,
                    "normalization_denominator": denom,
                    "normalization_usable": usable,
                    "clean_sequence": list(pair.clean_sequence),
                    "corrupt_sequence": list(pair.corrupt_sequence),
                    **metrics.to_json_dict(prefix="ablated_"),
                }
            )
        append_jsonl(rows_path, out_rows)
    write_summary(rows_path, out_dir / "summary.json")
    print(f"Saved rows to {rows_path}")


if __name__ == "__main__":
    main()
