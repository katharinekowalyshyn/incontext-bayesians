"""Seen-edge versus held-out-edge splits for graph-neighbor metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.secondary_experiments.graphs import UndirectedGraph
from src.secondary_experiments.vocabulary import WORDS, validate_vocabulary


@dataclass(frozen=True)
class EdgeSplit:
    """Word sets induced by context observations for one graph/current word."""

    graph_neighbors: tuple[str, ...]
    seen_neighbors: tuple[str, ...]
    heldout_neighbors: tuple[str, ...]
    non_neighbors: tuple[str, ...]
    observed_edge_neighbors: tuple[str, ...]

    def to_json_dict(self) -> dict:
        return {
            "graph_neighbors": list(self.graph_neighbors),
            "seen_neighbors": list(self.seen_neighbors),
            "heldout_neighbors": list(self.heldout_neighbors),
            "non_neighbors": list(self.non_neighbors),
            "observed_edge_neighbors": list(self.observed_edge_neighbors),
        }


def observed_neighbors_for_token(context: Sequence[str], token: str) -> set[str]:
    """Neighbors that appeared in an observed transition involving ``token``.

    The graph hypotheses are undirected, so either transition direction counts
    as an observed edge involving the final token.
    """

    observed: set[str] = set()
    for left, right in zip(context[:-1], context[1:]):
        if left == token and right != token:
            observed.add(right)
        if right == token and left != token:
            observed.add(left)
    return observed


def edge_split_for_context(
    graph: UndirectedGraph,
    context: Sequence[str],
    current_word: str | None = None,
    words: Sequence[str] = WORDS,
) -> EdgeSplit:
    """Split graph neighbors into seen and held-out sets."""

    vocab = validate_vocabulary(words)
    if not context:
        raise ValueError("context must contain at least one token.")
    current = context[-1] if current_word is None else current_word
    graph_neighbors = tuple(graph.get_valid_next_words(current))
    observed = observed_neighbors_for_token(context, current)
    seen = tuple(word for word in graph_neighbors if word in observed)
    heldout = tuple(word for word in graph_neighbors if word not in observed)
    non_neighbors = tuple(word for word in vocab if word not in set(graph_neighbors))
    return EdgeSplit(
        graph_neighbors=graph_neighbors,
        seen_neighbors=seen,
        heldout_neighbors=heldout,
        non_neighbors=non_neighbors,
        observed_edge_neighbors=tuple(sorted(observed)),
    )

