"""TransformerLens inference helpers for 16-word graph distributions.

The model-loading and single-forward-pass structure is copied/adapted from
``src/initial_experiments/vocabulary_tl_experiment.py`` and
``src/initial_experiments/mixing_experiment.py``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .config import DEFAULT_CONFIG
from .graphs import UndirectedGraph
from .vocabulary import WORDS, build_token_map, validate_vocabulary


@dataclass
class LLMMeasurements:
    """Per-context-length LLM measurements for one sequence."""

    distributions: dict[int, np.ndarray]
    neighbor_probs: dict[int, float]
    vocab_masses: dict[int, float]


@dataclass
class SemanticPriorTable:
    """LLM no-context semantic prior p(next_word | current_word)."""

    words: tuple[str, ...]
    distributions: dict[str, np.ndarray]
    vocab_masses: dict[str, float]

    def get_semantic_prior_distribution(self, current_word: str) -> dict[str, float]:
        probs = self.distributions[current_word]
        return {word: float(prob) for word, prob in zip(self.words, probs)}

    def distribution_array(self, current_word: str) -> np.ndarray:
        return self.distributions[current_word]

    def to_json_dict(self) -> dict:
        return {
            "words": list(self.words),
            "rows": {
                current_word: {
                    "distribution": self.get_semantic_prior_distribution(current_word),
                    "vocab_mass": float(self.vocab_masses[current_word]),
                }
                for current_word in self.words
            },
        }


def load_model(
    model_name: str = DEFAULT_CONFIG.model_name,
    cache_dir: str | None = None,
    device: str | None = None,
    dtype: str | torch.dtype = "float16",
):
    """Load a TransformerLens model in float16 to halve GPU memory (~16 GB vs ~32 GB)."""

    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)
    if device is None:
        device = default_device()
    dtype = resolve_dtype(dtype)
    print(f"Loading {model_name} on device={device}, dtype={dtype}...")
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        device=device,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    model.eval()
    print(f"Model loaded on {model.cfg.device}.")
    return model


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose one of {sorted(dtype_map)}.")
    return dtype_map[dtype]


def default_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU.
    Removed MPS for now because it seemed worse"""

    if torch.cuda.is_available():
        return "cuda"
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     return "mps"
    return "cpu"


@torch.no_grad()
def semantic_prior_table(
    model,
    words: Sequence[str] = WORDS,
    token_map: Mapping[str, int] | None = None,
) -> SemanticPriorTable:
    """Estimate the pretrained semantic prior over graph words.

    Prompt format is the same token convention used by the main experiment:
    ``[BOS]`` followed by the single graph-word token for the current word.
    The returned distributions are renormalized over the 16 controlled words.
    """

    vocab = validate_vocabulary(words)
    tok_map = dict(build_token_map(model, vocab) if token_map is None else token_map)
    word_token_ids = torch.tensor(
        [tok_map[word] for word in vocab],
        dtype=torch.long,
        device=model.cfg.device,
    )

    distributions: dict[str, np.ndarray] = {}
    vocab_masses: dict[str, float] = {}
    bos = model.tokenizer.bos_token_id

    for current_word in vocab:
        input_ids = [bos, tok_map[current_word]]
        tokens = torch.tensor([input_ids], dtype=torch.long).to(model.cfg.device)
        logits = model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        graph_probs = probs[word_token_ids]
        vocab_mass = float(graph_probs.sum().item())
        if vocab_mass <= 0.0:
            dist = np.full(len(vocab), 1.0 / len(vocab), dtype=float)
        else:
            dist = (graph_probs / vocab_mass).detach().cpu().numpy()
        distributions[current_word] = dist
        vocab_masses[current_word] = vocab_mass

    return SemanticPriorTable(vocab, distributions, vocab_masses)


@torch.no_grad()
def sequence_llm_measurements(
    model,
    sequence: Sequence[str],
    eval_lengths: Sequence[int],
    words: Sequence[str] = WORDS,
    token_map: Mapping[str, int] | None = None,
    neighbor_graph: UndirectedGraph | None = None,
    source_graphs: Sequence[str] | None = None,
    graph_map: Mapping[str, UndirectedGraph] | None = None,
) -> LLMMeasurements:
    """Extract both old neighbor mass and renormalized 16-word distributions.

    This keeps the indexing convention from ``sequence_neighbor_probs`` in the
    initial experiments:

        input_ids = [BOS] + graph_word_token_ids
        probs = softmax(logits[0, 1:, :])
        probs[L - 1] is the prediction after ``sequence[L - 1]``.
    """

    vocab = validate_vocabulary(words)
    tok_map = dict(build_token_map(model, vocab) if token_map is None else token_map)
    word_token_ids = torch.tensor(
        [tok_map[word] for word in vocab],
        dtype=torch.long,
        device=model.cfg.device,
    )

    bos = model.tokenizer.bos_token_id
    input_ids = [bos] + [tok_map[word] for word in sequence]
    if len(input_ids) > model.cfg.n_ctx:
        input_ids = input_ids[: model.cfg.n_ctx]

    tokens = torch.tensor([input_ids], dtype=torch.long).to(model.cfg.device)
    logits = model(tokens)
    probs = torch.softmax(logits[0, 1:, :], dim=-1)

    distributions: dict[int, np.ndarray] = {}
    neighbor_probs: dict[int, float] = {}
    vocab_masses: dict[int, float] = {}

    for L in eval_lengths:
        if L <= 0:
            continue
        if L - 1 >= probs.shape[0] or L > len(sequence):
            continue

        full_vocab_probs = probs[L - 1, word_token_ids]
        vocab_mass = float(full_vocab_probs.sum().item())
        if vocab_mass <= 0.0:
            dist = np.full(len(vocab), 1.0 / len(vocab), dtype=float)
        else:
            dist = (full_vocab_probs / vocab_mass).detach().cpu().numpy()

        distributions[int(L)] = dist
        vocab_masses[int(L)] = vocab_mass

        target_graph = neighbor_graph
        if source_graphs is not None:
            if graph_map is None:
                raise ValueError("graph_map is required when source_graphs is supplied.")
            target_graph = graph_map[source_graphs[L - 1]]

        if target_graph is not None:
            current_word = sequence[L - 1]
            valid = target_graph.get_valid_next_words(current_word)
            neighbor_probs[int(L)] = float(
                sum(probs[L - 1, tok_map[neighbor]].item() for neighbor in valid)
            )

    return LLMMeasurements(distributions, neighbor_probs, vocab_masses)
