"""Graph-difference steering vector utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from src.data.prompt_pairs import context_ending_at
from src.interventions.hooks import hook_name, make_position_selector, tokens_from_sequence
from src.secondary_experiments.graphs import UndirectedGraph
from src.secondary_experiments.vocabulary import WORDS


@dataclass(frozen=True)
class SteeringVectorSet:
    """Layer-indexed graph-difference vectors."""

    source_graph: str
    target_graph: str
    layers: tuple[int, ...]
    position_strategy: str
    vectors: dict[int, torch.Tensor]
    shuffled_vectors: dict[int, torch.Tensor]
    random_vectors: dict[int, torch.Tensor]
    train_contexts_per_graph: int


def activation_summary(
    model,
    sequence: Sequence[str],
    layers: Sequence[int],
    activation: str,
    position_strategy: str,
    token_map: Mapping[str, int],
) -> dict[int, torch.Tensor]:
    """Return one summary activation vector per layer for ``sequence``."""

    tokens = tokens_from_sequence(model, sequence, token_map=token_map)
    names = [hook_name(layer, activation) for layer in layers]
    selector = make_position_selector(position_strategy)
    groups = selector(sequence)
    positions = sorted({pos for group in groups for pos in group if 0 <= pos < tokens.shape[1]})
    if not positions:
        positions = [tokens.shape[1] - 1]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        acts = cache[hook_name(layer, activation)][0, positions, :].detach().float().cpu()
        out[int(layer)] = acts.mean(dim=0)
    return out


def collect_graph_activations(
    model,
    graph: UndirectedGraph,
    graph_name: str,
    num_contexts: int,
    context_length: int,
    seed: int,
    layers: Sequence[int],
    activation: str,
    position_strategy: str,
    token_map: Mapping[str, int],
) -> list[dict[int, torch.Tensor]]:
    """Generate graph contexts and collect activation summaries."""

    rng = np.random.default_rng(seed)
    rows: list[dict[int, torch.Tensor]] = []
    for _ in range(num_contexts):
        final_word = str(rng.choice(graph.words))
        generation_seed = int(rng.integers(0, 2**31 - 1))
        sequence = context_ending_at(graph, final_word, context_length, generation_seed)
        rows.append(
            activation_summary(
                model,
                sequence,
                layers=layers,
                activation=activation,
                position_strategy=position_strategy,
                token_map=token_map,
            )
        )
    return rows


def mean_by_layer(activations: Sequence[dict[int, torch.Tensor]], layers: Sequence[int]) -> dict[int, torch.Tensor]:
    if not activations:
        raise ValueError("Need at least one activation row.")
    return {
        int(layer): torch.stack([row[int(layer)] for row in activations], dim=0).mean(dim=0)
        for layer in layers
    }


def shuffled_label_vectors(
    source_acts: Sequence[dict[int, torch.Tensor]],
    target_acts: Sequence[dict[int, torch.Tensor]],
    layers: Sequence[int],
    seed: int,
) -> dict[int, torch.Tensor]:
    """Compute a shuffled-label graph-difference control vector."""

    rng = np.random.default_rng(seed)
    labels = np.array([1] * len(source_acts) + [0] * len(target_acts), dtype=int)
    rng.shuffle(labels)
    all_rows = list(source_acts) + list(target_acts)
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        src = [row[int(layer)] for row, label in zip(all_rows, labels) if label == 1]
        tgt = [row[int(layer)] for row, label in zip(all_rows, labels) if label == 0]
        out[int(layer)] = torch.stack(src).mean(dim=0) - torch.stack(tgt).mean(dim=0)
    return out


def matched_random_vectors(
    vectors: Mapping[int, torch.Tensor],
    seed: int,
) -> dict[int, torch.Tensor]:
    """Random Gaussian controls matched to each real vector's L2 norm."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    out: dict[int, torch.Tensor] = {}
    for layer, vector in vectors.items():
        rand = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
        rand_norm = torch.linalg.vector_norm(rand)
        vector_norm = torch.linalg.vector_norm(vector)
        if float(rand_norm.item()) > 0.0:
            rand = rand * (vector_norm / rand_norm)
        out[int(layer)] = rand
    return out


def compute_steering_vectors(
    model,
    graphs: Mapping[str, UndirectedGraph],
    source_graph: str,
    target_graph: str,
    num_train_contexts: int,
    context_length: int,
    layers: Sequence[int],
    seed: int,
    token_map: Mapping[str, int],
    activation: str = "resid_post",
    position_strategy: str = "final",
) -> SteeringVectorSet:
    """Compute source-minus-target graph-difference vectors on train contexts."""

    source_acts = collect_graph_activations(
        model,
        graphs[source_graph],
        source_graph,
        num_contexts=num_train_contexts,
        context_length=context_length,
        seed=seed + 101,
        layers=layers,
        activation=activation,
        position_strategy=position_strategy,
        token_map=token_map,
    )
    target_acts = collect_graph_activations(
        model,
        graphs[target_graph],
        target_graph,
        num_contexts=num_train_contexts,
        context_length=context_length,
        seed=seed + 202,
        layers=layers,
        activation=activation,
        position_strategy=position_strategy,
        token_map=token_map,
    )
    source_means = mean_by_layer(source_acts, layers)
    target_means = mean_by_layer(target_acts, layers)
    vectors = {
        int(layer): source_means[int(layer)] - target_means[int(layer)]
        for layer in layers
    }
    shuffled = shuffled_label_vectors(source_acts, target_acts, layers, seed=seed + 303)
    random = matched_random_vectors(vectors, seed=seed + 404)
    return SteeringVectorSet(
        source_graph=source_graph,
        target_graph=target_graph,
        layers=tuple(int(layer) for layer in layers),
        position_strategy=position_strategy,
        vectors=vectors,
        shuffled_vectors=shuffled,
        random_vectors=random,
        train_contexts_per_graph=num_train_contexts,
    )

