"""Sequence generation helpers for pure-graph secondary experiments."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np

from .config import DEFAULT_CONFIG
from .graphs import UndirectedGraph, build_candidate_graphs


def set_seed(seed: int) -> np.random.Generator:
    """Set Python and NumPy seeds, adapted from the initial sanity checks."""

    np.random.seed(seed)
    random.seed(seed)
    return np.random.default_rng(seed)


def generate_sequence(
    graph: UndirectedGraph,
    seq_len: int,
    seed: int | None = None,
    start_word: str | None = None,
) -> list[str]:
    """Generate one pure sequence from ``graph``."""

    rng = np.random.default_rng(seed)
    return graph.generate_sequence(seq_len, start_word=start_word, rng=rng)


def source_labels_for_pure_sequence(graph_name: str, seq_len: int) -> list[str]:
    """Return one source graph label per prediction position."""

    return [graph_name] * seq_len


def normalize_mix_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    clean = {name: float(weight) for name, weight in ratios.items() if float(weight) > 0}
    if not clean:
        raise ValueError("At least one positive mix ratio is required.")
    total = sum(clean.values())
    return {name: weight / total for name, weight in clean.items()}


def balanced_source_schedule(ratios: Mapping[str, float], n_steps: int) -> list[str]:
    """Weighted round-robin graph schedule for transition sources.

    For ``{"grid": 80, "ring": 20}``, this produces a schedule close to
    80/20 while spreading ring transitions through the sequence rather than
    placing them in one contiguous block.
    """

    probs = normalize_mix_ratios(ratios)
    names = tuple(probs)
    actual = {name: 0 for name in names}
    schedule: list[str] = []

    for step in range(n_steps):
        target = {name: probs[name] * (step + 1) for name in names}
        name = max(names, key=lambda candidate: target[candidate] - actual[candidate])
        schedule.append(name)
        actual[name] += 1
    return schedule


def generate_mixed_sequence(
    graphs: Mapping[str, UndirectedGraph],
    ratios: Mapping[str, float],
    seq_len: int,
    seed: int | None = None,
    start_word: str | None = None,
) -> tuple[list[str], list[str]]:
    """Generate a sequence whose transition source alternates by a fixed ratio.

    Returns:
        sequence: word sequence of length ``seq_len``.
        source_graphs: length-``seq_len`` labels.  ``source_graphs[t]`` is the
            graph used to choose the next token after ``sequence[t]``.  The
            final label is still supplied so predictions at the final context
            length have a well-defined neighbor-probability target.
    """

    if seq_len <= 0:
        return [], []

    probs = normalize_mix_ratios(ratios)
    missing = set(probs) - set(graphs)
    if missing:
        raise KeyError(f"Unknown graph(s) in mix ratios: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    words = next(iter(graphs.values())).words
    word = start_word if start_word is not None else str(rng.choice(words))
    sequence = [word]
    source_graphs = balanced_source_schedule(probs, seq_len)

    while len(sequence) < seq_len:
        source_name = source_graphs[len(sequence) - 1]
        graph = graphs[source_name]
        if graph.is_uniform:
            word = str(rng.choice(graph.words))
        else:
            word = str(rng.choice(graph.get_valid_next_words(word)))
        sequence.append(word)
    return sequence, source_graphs


def generate_batch(
    graph: UndirectedGraph,
    seq_len: int,
    seeds: Sequence[int],
    start_words: Sequence[str] | None = None,
) -> list[list[str]]:
    """Generate one sequence per seed, optionally aligned to start words."""

    if start_words is not None and len(start_words) != len(seeds):
        raise ValueError("start_words and seeds must have the same length.")

    sequences = []
    for i, seed in enumerate(seeds):
        start_word = None if start_words is None else start_words[i]
        sequences.append(generate_sequence(graph, seq_len, seed=seed, start_word=start_word))
    return sequences


def generate_pure_graph_sequences(
    graphs: Mapping[str, UndirectedGraph] | None = None,
    true_graphs: Sequence[str] = DEFAULT_CONFIG.true_graphs,
    seq_len: int = DEFAULT_CONFIG.seq_len,
    seeds: Sequence[int] = DEFAULT_CONFIG.seeds,
) -> dict[str, dict[int, list[str]]]:
    """Generate synthetic pure-graph sequences for each true graph and seed."""

    graph_map = build_candidate_graphs() if graphs is None else graphs
    out: dict[str, dict[int, list[str]]] = {}
    for graph_name in true_graphs:
        if graph_name not in graph_map:
            raise KeyError(f"Unknown graph: {graph_name}")
        graph = graph_map[graph_name]
        out[graph_name] = {
            seed: generate_sequence(graph, seq_len=seq_len, seed=seed)
            for seed in seeds
        }
    return out
