"""Clean/corrupt random-walk prompt construction.

This module deliberately reuses the secondary experiment graph objects and
their ``generate_sequence`` method.  Prompts keep the existing format:

    [BOS] word_1 word_2 ... word_T

The final word is the current node whose next-token logits are evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from src.secondary_experiments.graphs import UndirectedGraph, build_candidate_graphs
from src.secondary_experiments.vocabulary import WORDS, validate_vocabulary


@dataclass(frozen=True)
class PromptPair:
    """One clean/corrupt pair with matched final current word."""

    pair_id: int
    seed: int
    clean_graph: str
    corrupt_graph: str
    clean_sequence: tuple[str, ...]
    corrupt_sequence: tuple[str, ...]
    final_word: str
    clean_generation_seed: int
    corrupt_generation_seed: int

    def to_json_dict(self) -> dict:
        out = asdict(self)
        out["clean_sequence"] = list(self.clean_sequence)
        out["corrupt_sequence"] = list(self.corrupt_sequence)
        return out


def context_ending_at(
    graph: UndirectedGraph,
    final_word: str,
    context_length: int,
    seed: int,
) -> list[str]:
    """Generate a valid random-walk context whose last token is ``final_word``.

    For undirected graphs, a random walk started at ``final_word`` and reversed
    is still a valid random walk under the same transition support.  This keeps
    all generation delegated to the existing graph abstraction while allowing
    clean/corrupt prompts to share the same current node.
    """

    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if final_word not in graph.words:
        raise ValueError(f"{final_word!r} is not in graph {graph.name!r}.")

    forward = graph.generate_sequence(
        seq_len=context_length,
        start_word=final_word,
        rng=np.random.default_rng(seed),
    )
    return list(reversed(forward))


def build_prompt_pairs(
    clean_graph: str,
    corrupt_graph: str,
    num_pairs: int,
    context_length: int,
    seed: int,
    graphs: Mapping[str, UndirectedGraph] | None = None,
    words: Sequence[str] = WORDS,
) -> list[PromptPair]:
    """Construct clean/corrupt pairs with the same final word."""

    if num_pairs <= 0:
        raise ValueError("num_pairs must be positive.")
    vocab = validate_vocabulary(words)
    graph_map = build_candidate_graphs(vocab) if graphs is None else dict(graphs)
    if clean_graph not in graph_map:
        raise KeyError(f"Unknown clean graph: {clean_graph}")
    if corrupt_graph not in graph_map:
        raise KeyError(f"Unknown corrupt graph: {corrupt_graph}")

    rng = np.random.default_rng(seed)
    pairs: list[PromptPair] = []
    for pair_id in range(num_pairs):
        final_word = str(rng.choice(vocab))
        clean_seed = int(rng.integers(0, 2**31 - 1))
        corrupt_seed = int(rng.integers(0, 2**31 - 1))
        clean_sequence = context_ending_at(
            graph_map[clean_graph],
            final_word=final_word,
            context_length=context_length,
            seed=clean_seed,
        )
        corrupt_sequence = context_ending_at(
            graph_map[corrupt_graph],
            final_word=final_word,
            context_length=context_length,
            seed=corrupt_seed,
        )
        pairs.append(
            PromptPair(
                pair_id=pair_id,
                seed=seed,
                clean_graph=clean_graph,
                corrupt_graph=corrupt_graph,
                clean_sequence=tuple(clean_sequence),
                corrupt_sequence=tuple(corrupt_sequence),
                final_word=final_word,
                clean_generation_seed=clean_seed,
                corrupt_generation_seed=corrupt_seed,
            )
        )
    return pairs


def no_context_sequence(final_word: str, words: Sequence[str] = WORDS) -> tuple[str, ...]:
    """Return the no-walk prompt sequence for stress-test steering."""

    validate_vocabulary(words)
    if final_word not in words:
        raise ValueError(f"{final_word!r} is not in the controlled vocabulary.")
    return (final_word,)

