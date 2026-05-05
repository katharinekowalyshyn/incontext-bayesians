"""Reusable activation-patching routines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from src.data.prompt_pairs import PromptPair
from src.interventions.hooks import logits_and_cache, make_position_selector, patched_logits
from src.metrics.graph_logit_diff import graph_contrast_metrics, normalized_effect
from src.secondary_experiments.graphs import UndirectedGraph
from src.secondary_experiments.vocabulary import WORDS


@dataclass(frozen=True)
class PatchingSpec:
    layers: tuple[int, ...]
    position_strategies: tuple[str, ...] = ("final",)
    activation: str = "resid_post"
    top_k: int = 5
    min_abs_denom: float = 1e-6


def token_frequency_l1(clean_sequence: Sequence[str], corrupt_sequence: Sequence[str]) -> float:
    words = sorted(set(clean_sequence) | set(corrupt_sequence))
    clean_total = max(len(clean_sequence), 1)
    corrupt_total = max(len(corrupt_sequence), 1)
    return float(
        sum(
            abs(clean_sequence.count(word) / clean_total - corrupt_sequence.count(word) / corrupt_total)
            for word in words
        )
    )


def _metric_dict(
    logits: torch.Tensor,
    clean_graph: UndirectedGraph,
    corrupt_graph: UndirectedGraph,
    final_word: str,
    token_map: Mapping[str, int],
    seen_context: Sequence[str],
    prefix: str,
    top_k: int,
) -> dict:
    metrics = graph_contrast_metrics(
        logits=logits,
        clean_graph=clean_graph,
        corrupt_graph=corrupt_graph,
        current_word=final_word,
        token_map=token_map,
        context_for_seen_edges=seen_context,
        words=WORDS,
        top_k=top_k,
    )
    return metrics.to_json_dict(prefix=f"{prefix}_")


def patching_rows_for_pair(
    model,
    pair: PromptPair,
    graphs: Mapping[str, UndirectedGraph],
    token_map: Mapping[str, int],
    spec: PatchingSpec,
) -> list[dict]:
    """Run residual/activation patching for one clean/corrupt prompt pair."""

    clean_graph = graphs[pair.clean_graph]
    corrupt_graph = graphs[pair.corrupt_graph]
    clean_logits, clean_cache = logits_and_cache(
        model,
        pair.clean_sequence,
        layers=spec.layers,
        activation=spec.activation,
        token_map=token_map,
    )
    corrupt_logits, _ = logits_and_cache(
        model,
        pair.corrupt_sequence,
        layers=spec.layers,
        activation=spec.activation,
        token_map=token_map,
    )

    # Align seen/held-out splits to the corrupt evaluation context.  This is the
    # key cache-control question: did the corrupt prompt itself observe the
    # graph edge whose logit is being helped by the intervention?
    clean_metrics = graph_contrast_metrics(
        clean_logits,
        clean_graph,
        corrupt_graph,
        pair.final_word,
        token_map,
        context_for_seen_edges=pair.corrupt_sequence,
        top_k=spec.top_k,
    )
    corrupt_metrics = graph_contrast_metrics(
        corrupt_logits,
        clean_graph,
        corrupt_graph,
        pair.final_word,
        token_map,
        context_for_seen_edges=pair.corrupt_sequence,
        top_k=spec.top_k,
    )

    base = {
        "model": getattr(model, "name", None),
        "seed": pair.seed,
        "pair_id": pair.pair_id,
        "clean_graph": pair.clean_graph,
        "corrupt_graph": pair.corrupt_graph,
        "context_length": len(pair.clean_sequence),
        "final_token": pair.final_word,
        "activation": spec.activation,
        "clean_generation_seed": pair.clean_generation_seed,
        "corrupt_generation_seed": pair.corrupt_generation_seed,
        "token_frequency_l1": token_frequency_l1(pair.clean_sequence, pair.corrupt_sequence),
        "clean_sequence": list(pair.clean_sequence),
        "corrupt_sequence": list(pair.corrupt_sequence),
        "clean_metric": clean_metrics.graph_logit_diff,
        "corrupt_metric": corrupt_metrics.graph_logit_diff,
        **clean_metrics.to_json_dict(prefix="clean_"),
        **corrupt_metrics.to_json_dict(prefix="corrupt_"),
    }

    rows: list[dict] = []
    for strategy in spec.position_strategies:
        selector = make_position_selector(strategy)
        grouped_positions = selector(pair.corrupt_sequence)
        for layer in spec.layers:
            for group_idx, positions in enumerate(grouped_positions):
                patch_logits = patched_logits(
                    model,
                    pair.corrupt_sequence,
                    clean_cache=clean_cache,
                    layer=layer,
                    positions=positions,
                    activation=spec.activation,
                    token_map=token_map,
                )
                patched_metrics = graph_contrast_metrics(
                    patch_logits,
                    clean_graph,
                    corrupt_graph,
                    pair.final_word,
                    token_map,
                    context_for_seen_edges=pair.corrupt_sequence,
                    top_k=spec.top_k,
                )
                norm, usable, denom = normalized_effect(
                    patched_metrics.graph_logit_diff,
                    clean_metrics.graph_logit_diff,
                    corrupt_metrics.graph_logit_diff,
                    min_abs_denom=spec.min_abs_denom,
                )
                row = {
                    **base,
                    "layer": int(layer),
                    "position_strategy": strategy,
                    "position_group_index": int(group_idx),
                    "positions": [int(p) for p in positions],
                    "patched_metric": patched_metrics.graph_logit_diff,
                    "normalized_effect": norm,
                    "normalization_denominator": denom,
                    "normalization_usable": usable,
                    **patched_metrics.to_json_dict(prefix="patched_"),
                }
                rows.append(row)
    return rows

