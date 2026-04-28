"""Fixed 16-word vocabulary used by all secondary graph hypotheses.

The word list and color palette are copied from
``src/initial_experiments/sanity_check.py`` so the new experiments use the same
controlled vocabulary as the first round of grid experiments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


WORDS: tuple[str, ...] = (
    "apple",
    "bird",
    "car",
    "egg",
    "house",
    "milk",
    "plane",
    "opera",
    "box",
    "sand",
    "sun",
    "mango",
    "rock",
    "math",
    "code",
    "phone",
)

COLORS: tuple[str, ...] = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
)

WORD_TO_COLOR: Mapping[str, str] = dict(zip(WORDS, COLORS))


def validate_vocabulary(words: Sequence[str] = WORDS) -> tuple[str, ...]:
    """Return ``words`` as a tuple after enforcing the 16-word contract."""

    vocab = tuple(words)
    if len(vocab) != 16:
        raise ValueError(f"Expected exactly 16 words, got {len(vocab)}.")
    if len(set(vocab)) != len(vocab):
        raise ValueError("Vocabulary contains duplicate words.")
    return vocab


def resolve_tokenizer(model_or_tokenizer):
    """Accept either a TransformerLens model or a tokenizer-like object."""

    return getattr(model_or_tokenizer, "tokenizer", model_or_tokenizer)


def build_token_map(model_or_tokenizer, words: Sequence[str] = WORDS) -> dict[str, int]:
    """Return ``{word: token_id}`` for the fixed vocabulary.

    This is adapted from ``initial_experiments.vocabulary_tl_experiment``.  It
    intentionally raises on multi-token words because the secondary experiments
    require exact next-token distributions over the 16 graph words.
    """

    tokenizer = resolve_tokenizer(model_or_tokenizer)
    mapping: dict[str, int] = {}
    multi_token: list[tuple[str, list[int]]] = []

    for word in validate_vocabulary(words):
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            mapping[word] = ids[0]
        else:
            multi_token.append((word, list(ids)))

    if multi_token:
        raise ValueError(f"Expected single-token words, got: {multi_token}")
    return mapping
