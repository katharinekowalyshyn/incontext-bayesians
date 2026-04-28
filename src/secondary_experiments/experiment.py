"""Experiment orchestration for secondary graph baselines."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .bayesian_observer import BayesianGraphObserver
from .cache_baseline import CacheBaseline
from .config import DEFAULT_CONFIG, ExperimentConfig
from .graphs import UndirectedGraph, build_candidate_graphs
from .metrics import kl_divergence, mse, pearson_corr
from .sequence_generation import (
    generate_mixed_sequence,
    generate_sequence,
    source_labels_for_pure_sequence,
)
from .vocabulary import WORDS, validate_vocabulary


def distribution_to_dict(words: Sequence[str], probs: Sequence[float]) -> dict[str, float]:
    return {word: float(prob) for word, prob in zip(words, probs)}


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def neighbor_probability(
    graph: UndirectedGraph,
    current_word: str,
    words: Sequence[str],
    probs: Sequence[float],
) -> float:
    """Sum distribution mass on true-graph valid neighbors of ``current_word``."""

    word_to_idx = {word: i for i, word in enumerate(words)}
    prob_arr = np.asarray(probs, dtype=float)
    return float(sum(prob_arr[word_to_idx[word]] for word in graph.get_valid_next_words(current_word)))


def baseline_rows_for_sequence(
    sequence: Sequence[str],
    true_graph: str,
    seed: int,
    graphs: Mapping[str, UndirectedGraph],
    observer: BayesianGraphObserver,
    cache: CacheBaseline,
    eval_lengths: Sequence[int],
    llm_distributions: Mapping[int, np.ndarray] | None = None,
    llm_neighbor_probs: Mapping[int, float] | None = None,
    llm_vocab_masses: Mapping[int, float] | None = None,
    source_graphs: Sequence[str] | None = None,
    words: Sequence[str] = WORDS,
) -> list[dict]:
    """Build result rows for one sequence.

    If ``llm_distributions`` is omitted, rows contain only baseline outputs and
    posterior diagnostics.  This is used by ``--skip-llm`` sanity runs.
    """

    vocab = validate_vocabulary(words)
    rows: list[dict] = []
    llm_distributions = {} if llm_distributions is None else llm_distributions
    llm_neighbor_probs = {} if llm_neighbor_probs is None else llm_neighbor_probs
    llm_vocab_masses = {} if llm_vocab_masses is None else llm_vocab_masses
    if source_graphs is None:
        source_graphs = source_labels_for_pure_sequence(true_graph, len(sequence))
    if len(source_graphs) < len(sequence):
        raise ValueError("source_graphs must contain at least one label per sequence token.")

    for L in eval_lengths:
        if L <= 0 or L > len(sequence):
            continue

        context = list(sequence[:L])
        bayes_dist = observer.next_token_distribution(context)
        cache_dist = cache.next_token_distribution(context)
        posterior = observer.posterior(context)
        source_graph = source_graphs[L - 1]
        neighbor_graph = graphs[source_graph]
        bayes_neighbor_prob = neighbor_probability(
            neighbor_graph, context[-1], vocab, bayes_dist
        )
        cache_neighbor_prob = neighbor_probability(
            neighbor_graph, context[-1], vocab, cache_dist
        )

        row = {
            "true_graph": true_graph,
            "seed": int(seed),
            "context_length": int(L),
            "current_word": context[-1],
            "source_graph": source_graph,
            "bayes_distribution": distribution_to_dict(vocab, bayes_dist),
            "cache_distribution": distribution_to_dict(vocab, cache_dist),
            "bayes_neighbor_prob": bayes_neighbor_prob,
            "cache_neighbor_prob": cache_neighbor_prob,
            "bayesian_posterior": posterior,
        }

        if L in llm_distributions:
            llm_dist = llm_distributions[L]
            row.update(
                {
                    "llm_distribution": distribution_to_dict(vocab, llm_dist),
                    "llm_neighbor_prob": finite_or_none(
                        llm_neighbor_probs.get(L, float("nan"))
                    ),
                    "llm_vocab_mass": finite_or_none(llm_vocab_masses.get(L, float("nan"))),
                    "kl_llm_bayes": kl_divergence(llm_dist, bayes_dist),
                    "kl_llm_cache": kl_divergence(llm_dist, cache_dist),
                    "mse_llm_bayes": mse(llm_dist, bayes_dist),
                    "mse_llm_cache": mse(llm_dist, cache_dist),
                    "corr_llm_bayes": finite_or_none(pearson_corr(llm_dist, bayes_dist)),
                    "corr_llm_cache": finite_or_none(pearson_corr(llm_dist, cache_dist)),
                }
            )

        rows.append(row)
    return rows


def make_mix_name(ratios: Mapping[str, float]) -> str:
    parts = [f"{name}{float(weight):g}" for name, weight in ratios.items()]
    return "mix_" + "_".join(parts)


def config_mix_dict(config: ExperimentConfig) -> dict[str, float] | None:
    if config.mix_ratios is None:
        return None
    return {name: weight for name, weight in config.mix_ratios}


def run_baseline_only(
    config: ExperimentConfig = DEFAULT_CONFIG,
    graphs: Mapping[str, UndirectedGraph] | None = None,
) -> list[dict]:
    """Run the non-LLM part of the experiment for fast infrastructure checks."""

    graph_map = build_candidate_graphs() if graphs is None else dict(graphs)
    observer = BayesianGraphObserver(
        graphs={name: graph_map[name] for name in config.candidate_graphs},
        epsilon=config.epsilon,
    )
    cache = CacheBaseline(alpha=config.alpha)

    rows: list[dict] = []
    mix_ratios = config_mix_dict(config)
    if mix_ratios is not None:
        true_graphs = (config.mix_name or make_mix_name(mix_ratios),)
    else:
        true_graphs = config.true_graphs

    for true_graph in true_graphs:
        print(f"[baseline] true_graph={true_graph}")
        for seed in config.seeds:
            if mix_ratios is None:
                graph = graph_map[true_graph]
                print(f"  seed={seed}: generating pure {true_graph} sequence...")
                sequence = generate_sequence(graph, seq_len=config.seq_len, seed=seed)
                source_graphs = source_labels_for_pure_sequence(true_graph, len(sequence))
            else:
                print(
                    f"  seed={seed}: generating mixed sequence "
                    f"{dict(mix_ratios)} as {true_graph}..."
                )
                sequence, source_graphs = generate_mixed_sequence(
                    graph_map,
                    mix_ratios,
                    seq_len=config.seq_len,
                    seed=seed,
                )
            print(f"  seed={seed}: computing Bayes/cache rows...")
            rows.extend(
                baseline_rows_for_sequence(
                    sequence=sequence,
                    true_graph=true_graph,
                    seed=seed,
                    graphs=graph_map,
                    observer=observer,
                    cache=cache,
                    eval_lengths=config.eval_lengths,
                    source_graphs=source_graphs,
                )
            )
    return rows


def run_with_llm(config: ExperimentConfig = DEFAULT_CONFIG) -> list[dict]:
    """Run the full experiment, including Llama next-token distributions."""

    from .llm_inference import load_model, sequence_llm_measurements
    from .vocabulary import build_token_map

    graph_map = build_candidate_graphs()
    observer = BayesianGraphObserver(
        graphs={name: graph_map[name] for name in config.candidate_graphs},
        epsilon=config.epsilon,
    )
    cache = CacheBaseline(alpha=config.alpha)

    model = load_model(config.model_name, device=config.device)
    token_map = build_token_map(model)

    rows: list[dict] = []
    mix_ratios = config_mix_dict(config)
    if mix_ratios is not None:
        true_graphs = (config.mix_name or make_mix_name(mix_ratios),)
    else:
        true_graphs = config.true_graphs

    for true_graph in true_graphs:
        print(f"[llm] true_graph={true_graph}")
        for seed in config.seeds:
            if mix_ratios is None:
                graph = graph_map[true_graph]
                print(f"  seed={seed}: generating pure {true_graph} sequence...")
                sequence = generate_sequence(graph, seq_len=config.seq_len, seed=seed)
                source_graphs = source_labels_for_pure_sequence(true_graph, len(sequence))
                neighbor_graph = graph
            else:
                print(
                    f"  seed={seed}: generating mixed sequence "
                    f"{dict(mix_ratios)} as {true_graph}..."
                )
                sequence, source_graphs = generate_mixed_sequence(
                    graph_map,
                    mix_ratios,
                    seq_len=config.seq_len,
                    seed=seed,
                )
                neighbor_graph = None
            print(f"  seed={seed}: running LLM forward pass...")
            llm = sequence_llm_measurements(
                model=model,
                sequence=sequence,
                eval_lengths=config.eval_lengths,
                token_map=token_map,
                neighbor_graph=neighbor_graph,
                source_graphs=source_graphs,
                graph_map=graph_map,
            )
            print(f"  seed={seed}: computing Bayes/cache metrics against LLM...")
            rows.extend(
                baseline_rows_for_sequence(
                    sequence=sequence,
                    true_graph=true_graph,
                    seed=seed,
                    graphs=graph_map,
                    observer=observer,
                    cache=cache,
                    eval_lengths=config.eval_lengths,
                    llm_distributions=llm.distributions,
                    llm_neighbor_probs=llm.neighbor_probs,
                    llm_vocab_masses=llm.vocab_masses,
                    source_graphs=source_graphs,
                )
            )
    return rows


def save_json(rows: Sequence[dict], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(list(rows), f, indent=2)
    return out_path
