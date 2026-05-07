"""TransformerLens hook utilities for residual-stream interventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import torch

from src.secondary_experiments.vocabulary import WORDS, build_token_map, validate_vocabulary


SUPPORTED_ACTIVATIONS = {
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.hook_attn_out",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
}


def hook_name(layer: int, activation: str = "resid_post") -> str:
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation {activation!r}; choose one of {sorted(SUPPORTED_ACTIVATIONS)}."
        )
    return SUPPORTED_ACTIVATIONS[activation].format(layer=int(layer))


def tokens_from_sequence(
    model,
    sequence: Sequence[str],
    token_map: Mapping[str, int] | None = None,
    words: Sequence[str] = WORDS,
) -> torch.Tensor:
    """Tokenize with the existing graph prompt convention: BOS + word tokens."""

    vocab = validate_vocabulary(words)
    tok_map = dict(build_token_map(model, vocab) if token_map is None else token_map)
    bos = model.tokenizer.bos_token_id
    input_ids = [bos] + [tok_map[word] for word in sequence]
    if len(input_ids) > model.cfg.n_ctx:
        input_ids = input_ids[: model.cfg.n_ctx]
    return torch.tensor([input_ids], dtype=torch.long, device=model.cfg.device)


def logits_for_sequence(
    model,
    sequence: Sequence[str],
    token_map: Mapping[str, int] | None = None,
    words: Sequence[str] = WORDS,
) -> torch.Tensor:
    tokens = tokens_from_sequence(model, sequence, token_map=token_map, words=words)
    with torch.no_grad():
        return model(tokens)[0, -1, :].detach()


def logits_and_cache(
    model,
    sequence: Sequence[str],
    layers: Sequence[int],
    activation: str = "resid_post",
    token_map: Mapping[str, int] | None = None,
    words: Sequence[str] = WORDS,
):
    """Run a sequence and cache requested activations."""

    tokens = tokens_from_sequence(model, sequence, token_map=token_map, words=words)
    names = [hook_name(layer, activation) for layer in layers]
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=names)
    return logits[0, -1, :].detach(), cache


def patched_logits(
    model,
    corrupt_sequence: Sequence[str],
    clean_cache,
    layer: int,
    positions: Sequence[int],
    activation: str = "resid_post",
    token_map: Mapping[str, int] | None = None,
    words: Sequence[str] = WORDS,
) -> torch.Tensor:
    """Patch clean activation values into a corrupt forward pass."""

    tokens = tokens_from_sequence(model, corrupt_sequence, token_map=token_map, words=words)
    name = hook_name(layer, activation)
    pos = [int(p) for p in positions if 0 <= int(p) < tokens.shape[1]]
    if not pos:
        raise ValueError("No valid token positions supplied for patching.")

    def patch_hook(value, hook):
        value[:, pos, :] = clean_cache[name][:, pos, :].to(value.device, dtype=value.dtype)
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(name, patch_hook)])
    return logits[0, -1, :].detach()


def steered_logits(
    model,
    sequence: Sequence[str],
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    positions: Sequence[int],
    activation: str = "resid_post",
    token_map: Mapping[str, int] | None = None,
    words: Sequence[str] = WORDS,
) -> torch.Tensor:
    """Add ``alpha * vector`` to selected activation positions."""

    tokens = tokens_from_sequence(model, sequence, token_map=token_map, words=words)
    name = hook_name(layer, activation)
    pos = [int(p) for p in positions if 0 <= int(p) < tokens.shape[1]]
    if not pos:
        raise ValueError("No valid token positions supplied for steering.")

    def steer_hook(value, hook):
        direction = vector.to(value.device, dtype=value.dtype)
        value[:, pos, :] = value[:, pos, :] + float(alpha) * direction
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(name, steer_hook)])
    return logits[0, -1, :].detach()


def ablated_head_logits(
    model,
    sequence: Sequence[str],
    layer: int,
    head: int,
    token_map: Mapping[str, int] | None = None,
    words: Sequence[str] = WORDS,
) -> torch.Tensor:
    """Zero one attention head's result vector as a targeted ablation."""

    tokens = tokens_from_sequence(model, sequence, token_map=token_map, words=words)
    name = f"blocks.{int(layer)}.attn.hook_result"

    def ablate_hook(value, hook):
        value[:, :, int(head), :] = 0.0
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=[(name, ablate_hook)])
    return logits[0, -1, :].detach()


def available_layers(model) -> list[int]:
    return list(range(int(model.cfg.n_layers)))


def make_position_selector(
    strategy: str,
) -> Callable[[Sequence[str]], list[list[int]]]:
    """Return grouped token positions for an intervention strategy.

    Positions are token indices in the Transformer input, where 0 is BOS and
    graph word ``sequence[i]`` is at token position ``i + 1``.
    """

    def final(sequence: Sequence[str]) -> list[list[int]]:
        return [[len(sequence)]]

    def all_positions(sequence: Sequence[str]) -> list[list[int]]:
        return [[i] for i in range(1, len(sequence) + 1)]

    def same_token_occurrences(sequence: Sequence[str]) -> list[list[int]]:
        current = sequence[-1]
        positions = [i + 1 for i, word in enumerate(sequence[:-1]) if word == current]
        return [[p] for p in positions] or [[len(sequence)]]

    def edge_observation_positions(sequence: Sequence[str]) -> list[list[int]]:
        current = sequence[-1]
        positions: set[int] = set()
        for idx, (left, right) in enumerate(zip(sequence[:-1], sequence[1:])):
            if left == current or right == current:
                positions.add(idx + 1)
                positions.add(idx + 2)
        return [[p] for p in sorted(positions)] or [[len(sequence)]]

    def mean_context(sequence: Sequence[str]) -> list[list[int]]:
        return [list(range(1, len(sequence) + 1))]

    selectors = {
        "final": final,
        "all": all_positions,
        "same_token_occurrences": same_token_occurrences,
        "edge_observation_positions": edge_observation_positions,
        "mean_context": mean_context,
    }
    if strategy not in selectors:
        raise ValueError(f"Unknown position strategy {strategy!r}; choose one of {sorted(selectors)}.")
    return selectors[strategy]
