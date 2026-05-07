"""Graph-neighbor logit and probability metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch

from src.metrics.seen_heldout_edges import EdgeSplit, edge_split_for_context
from src.secondary_experiments.graphs import UndirectedGraph
from src.secondary_experiments.metrics import kl_divergence
from src.secondary_experiments.vocabulary import WORDS, validate_vocabulary


@dataclass(frozen=True)
class GraphContrastMetrics:
    """Metrics for one logit vector under a graph-family contrast."""

    graph_logit_diff: float
    clean_neighbor_mean_logit: float
    corrupt_neighbor_mean_logit: float
    clean_neighbor_prob_mass: float
    corrupt_neighbor_prob_mass: float
    clean_top1_is_neighbor: bool
    corrupt_top1_is_neighbor: bool
    clean_topk_has_neighbor: bool
    corrupt_topk_has_neighbor: bool
    clean_seen_logit_diff: float | None
    clean_heldout_logit_diff: float | None
    corrupt_seen_logit_diff: float | None
    corrupt_heldout_logit_diff: float | None
    clean_seen_prob_mass: float | None
    clean_heldout_prob_mass: float | None
    corrupt_seen_prob_mass: float | None
    corrupt_heldout_prob_mass: float | None
    non_neighbor_prob_mass: float
    vocab_distribution: dict[str, float]

    def to_json_dict(self, prefix: str = "") -> dict:
        data = {
            "graph_logit_diff": self.graph_logit_diff,
            "clean_neighbor_mean_logit": self.clean_neighbor_mean_logit,
            "corrupt_neighbor_mean_logit": self.corrupt_neighbor_mean_logit,
            "clean_neighbor_prob_mass": self.clean_neighbor_prob_mass,
            "corrupt_neighbor_prob_mass": self.corrupt_neighbor_prob_mass,
            "clean_top1_is_neighbor": self.clean_top1_is_neighbor,
            "corrupt_top1_is_neighbor": self.corrupt_top1_is_neighbor,
            "clean_topk_has_neighbor": self.clean_topk_has_neighbor,
            "corrupt_topk_has_neighbor": self.corrupt_topk_has_neighbor,
            "clean_seen_logit_diff": self.clean_seen_logit_diff,
            "clean_heldout_logit_diff": self.clean_heldout_logit_diff,
            "corrupt_seen_logit_diff": self.corrupt_seen_logit_diff,
            "corrupt_heldout_logit_diff": self.corrupt_heldout_logit_diff,
            "clean_seen_prob_mass": self.clean_seen_prob_mass,
            "clean_heldout_prob_mass": self.clean_heldout_prob_mass,
            "corrupt_seen_prob_mass": self.corrupt_seen_prob_mass,
            "corrupt_heldout_prob_mass": self.corrupt_heldout_prob_mass,
            "non_neighbor_prob_mass": self.non_neighbor_prob_mass,
            "vocab_distribution": self.vocab_distribution,
        }
        if not prefix:
            return data
        return {f"{prefix}{key}": value for key, value in data.items()}


def word_indices(words_subset: Sequence[str], word_to_idx: Mapping[str, int]) -> list[int]:
    return [word_to_idx[word] for word in words_subset]


def token_indices(words_subset: Sequence[str], token_map: Mapping[str, int]) -> list[int]:
    return [int(token_map[word]) for word in words_subset]


def _mean_or_none(values: torch.Tensor, idxs: Sequence[int]) -> float | None:
    if not idxs:
        return None
    return float(values[list(idxs)].float().mean().item())


def _sum_or_none(values: torch.Tensor, idxs: Sequence[int]) -> float | None:
    if not idxs:
        return None
    return float(values[list(idxs)].float().sum().item())


def _mean_logit(logits: torch.Tensor, words_subset: Sequence[str], token_map: Mapping[str, int]) -> float:
    idxs = token_indices(words_subset, token_map)
    if not idxs:
        return float("nan")
    return float(logits[idxs].float().mean().item())


def vocab_distribution_from_logits(
    logits: torch.Tensor,
    token_map: Mapping[str, int],
    words: Sequence[str] = WORDS,
) -> dict[str, float]:
    """Softmax distribution renormalized over the controlled 16-word vocabulary."""

    vocab = validate_vocabulary(words)
    probs = torch.softmax(logits.float(), dim=-1)
    ids = torch.tensor([token_map[word] for word in vocab], device=logits.device)
    selected = probs[ids]
    mass = selected.sum()
    if float(mass.item()) <= 0.0:
        arr = np.full(len(vocab), 1.0 / len(vocab), dtype=float)
    else:
        arr = (selected / mass).detach().cpu().numpy()
    return {word: float(prob) for word, prob in zip(vocab, arr)}


def graph_contrast_metrics(
    logits: torch.Tensor,
    clean_graph: UndirectedGraph,
    corrupt_graph: UndirectedGraph,
    current_word: str,
    token_map: Mapping[str, int],
    context_for_seen_edges: Sequence[str],
    words: Sequence[str] = WORDS,
    top_k: int = 5,
) -> GraphContrastMetrics:
    """Compute the primary graph logit difference and edge-split diagnostics."""

    vocab = validate_vocabulary(words)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    clean_split = edge_split_for_context(clean_graph, context_for_seen_edges, current_word, vocab)
    corrupt_split = edge_split_for_context(corrupt_graph, context_for_seen_edges, current_word, vocab)

    clean_neighbors = clean_split.graph_neighbors
    corrupt_neighbors = corrupt_split.graph_neighbors
    clean_mean = _mean_logit(logits, clean_neighbors, token_map)
    corrupt_mean = _mean_logit(logits, corrupt_neighbors, token_map)

    full_probs = torch.softmax(logits.float(), dim=-1)
    graph_token_ids = torch.tensor([token_map[word] for word in vocab], device=logits.device)
    graph_probs = full_probs[graph_token_ids]
    graph_mass = graph_probs.sum()
    renorm_probs = graph_probs / graph_mass if float(graph_mass.item()) > 0.0 else torch.full_like(graph_probs, 1 / len(vocab))

    clean_word_idxs = word_indices(clean_neighbors, word_to_idx)
    corrupt_word_idxs = word_indices(corrupt_neighbors, word_to_idx)
    non_neighbor_idxs = word_indices(
        [word for word in vocab if word not in set(clean_neighbors) and word not in set(corrupt_neighbors)],
        word_to_idx,
    )

    vocab_logits = torch.stack([logits[token_map[word]].float() for word in vocab])
    top_word_local = torch.topk(vocab_logits, k=min(top_k, len(vocab))).indices.detach().cpu().tolist()
    top_words = {vocab[i] for i in top_word_local}
    top1_word = vocab[int(torch.argmax(vocab_logits).item())]

    clean_seen = word_indices(clean_split.seen_neighbors, word_to_idx)
    clean_heldout = word_indices(clean_split.heldout_neighbors, word_to_idx)
    corrupt_seen = word_indices(corrupt_split.seen_neighbors, word_to_idx)
    corrupt_heldout = word_indices(corrupt_split.heldout_neighbors, word_to_idx)

    clean_seen_mean = _mean_or_none(vocab_logits, clean_seen)
    clean_heldout_mean = _mean_or_none(vocab_logits, clean_heldout)
    corrupt_seen_mean = _mean_or_none(vocab_logits, corrupt_seen)
    corrupt_heldout_mean = _mean_or_none(vocab_logits, corrupt_heldout)
    corrupt_all_mean = _mean_or_none(vocab_logits, corrupt_word_idxs)
    clean_all_mean = _mean_or_none(vocab_logits, clean_word_idxs)

    return GraphContrastMetrics(
        graph_logit_diff=clean_mean - corrupt_mean,
        clean_neighbor_mean_logit=clean_mean,
        corrupt_neighbor_mean_logit=corrupt_mean,
        clean_neighbor_prob_mass=float(renorm_probs[clean_word_idxs].sum().item()) if clean_word_idxs else 0.0,
        corrupt_neighbor_prob_mass=float(renorm_probs[corrupt_word_idxs].sum().item()) if corrupt_word_idxs else 0.0,
        clean_top1_is_neighbor=top1_word in set(clean_neighbors),
        corrupt_top1_is_neighbor=top1_word in set(corrupt_neighbors),
        clean_topk_has_neighbor=bool(set(clean_neighbors).intersection(top_words)),
        corrupt_topk_has_neighbor=bool(set(corrupt_neighbors).intersection(top_words)),
        clean_seen_logit_diff=None if clean_seen_mean is None or corrupt_all_mean is None else clean_seen_mean - corrupt_all_mean,
        clean_heldout_logit_diff=None if clean_heldout_mean is None or corrupt_all_mean is None else clean_heldout_mean - corrupt_all_mean,
        corrupt_seen_logit_diff=None if corrupt_seen_mean is None or clean_all_mean is None else clean_all_mean - corrupt_seen_mean,
        corrupt_heldout_logit_diff=None if corrupt_heldout_mean is None or clean_all_mean is None else clean_all_mean - corrupt_heldout_mean,
        clean_seen_prob_mass=_sum_or_none(renorm_probs, clean_seen),
        clean_heldout_prob_mass=_sum_or_none(renorm_probs, clean_heldout),
        corrupt_seen_prob_mass=_sum_or_none(renorm_probs, corrupt_seen),
        corrupt_heldout_prob_mass=_sum_or_none(renorm_probs, corrupt_heldout),
        non_neighbor_prob_mass=float(renorm_probs[non_neighbor_idxs].sum().item()) if non_neighbor_idxs else 0.0,
        vocab_distribution=vocab_distribution_from_logits(logits, token_map, vocab),
    )


def normalized_effect(
    patched_metric: float,
    clean_metric: float,
    corrupt_metric: float,
    min_abs_denom: float = 1e-6,
) -> tuple[float | None, bool, float]:
    """Return normalized patch/steering effect and whether it is usable."""

    denom = float(clean_metric) - float(corrupt_metric)
    if abs(denom) < min_abs_denom:
        return None, False, denom
    return (float(patched_metric) - float(corrupt_metric)) / denom, True, denom


def kl_to_reference_vocab_distribution(
    logits: torch.Tensor,
    reference_distribution: Mapping[str, float],
    token_map: Mapping[str, int],
    words: Sequence[str] = WORDS,
) -> float:
    current = vocab_distribution_from_logits(logits, token_map, words)
    vocab = validate_vocabulary(words)
    return kl_divergence(
        [reference_distribution[word] for word in vocab],
        [current[word] for word in vocab],
    )


def standard_error(values: Sequence[float]) -> float | None:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(arr) < 2:
        return None
    return float(arr.std(ddof=1) / sqrt(len(arr)))
