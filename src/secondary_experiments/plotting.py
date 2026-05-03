"""Plot helpers for secondary graph-baseline outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRAPH_COLORS = {
    "grid": "#1976D2",
    "ring": "#C62828",
    "chain": "#2E7D32",
    "star": "#6A1B9A",
    "uniform": "#616161",
}

BASELINE_STYLES = {
    "bayes": ("#1976D2", "Bayes"),
    "edge_learner": ("#2E7D32", "Edge learner"),
    "cache": ("#C62828", "Cache"),
    "unigram": ("#00838F", "Unigram Dirichlet"),
    "semantic_prior": ("#6A1B9A", "Semantic prior"),
}


def _group_rows(rows: Sequence[dict]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["true_graph"]].append(row)
    return grouped


def _mean_sem_by_length(
    rows: Sequence[dict],
    key: str,
) -> tuple[list[int], list[float], list[float]]:
    values = defaultdict(list)
    for row in rows:
        if key not in row or row[key] is None:
            continue
        value = float(row[key])
        if np.isfinite(value):
            values[int(row["context_length"])].append(value)
    lengths = sorted(values)
    means = [float(np.mean(values[L])) for L in lengths]
    sems = [
        float(np.std(values[L], ddof=1) / np.sqrt(len(values[L])))
        if len(values[L]) > 1 else 0.0
        for L in lengths
    ]
    return lengths, means, sems


def _plot_mean_sem(ax, lengths, means, sems, color: str, label: str, marker: str = "o"):
    ax.plot(lengths, means, f"{marker}-", lw=2, ms=4, color=color, label=label)
    if any(sem > 0 for sem in sems):
        lo = [mean - sem for mean, sem in zip(means, sems)]
        hi = [mean + sem for mean, sem in zip(means, sems)]
        ax.fill_between(lengths, lo, hi, color=color, alpha=0.16, linewidth=0)


def plot_bayesian_posterior(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Plot p(G | context) over context length for each true generating graph."""

    out = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for true_graph, graph_rows in _group_rows(rows).items():
        posterior_by_graph = defaultdict(lambda: defaultdict(list))
        for row in graph_rows:
            L = int(row["context_length"])
            for graph_name, prob in row["bayesian_posterior"].items():
                posterior_by_graph[graph_name][L].append(float(prob))

        fig, ax = plt.subplots(figsize=(8, 5))
        for graph_name, by_len in sorted(posterior_by_graph.items()):
            lengths = sorted(by_len)
            means = [float(np.mean(by_len[L])) for L in lengths]
            sems = [
                float(np.std(by_len[L], ddof=1) / np.sqrt(len(by_len[L])))
                if len(by_len[L]) > 1 else 0.0
                for L in lengths
            ]
            _plot_mean_sem(
                ax,
                lengths,
                means,
                sems,
                color=GRAPH_COLORS.get(graph_name, "gray"),
                label=graph_name,
            )

        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Bayesian posterior probability")
        ax.set_title(f"Posterior over graph hypotheses: true {true_graph}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_path / f"posterior_{true_graph}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    return out


def plot_llm_metric_comparison(
    rows: Sequence[dict],
    out_dir: str | Path,
    metric_prefix: str,
) -> list[Path]:
    """Plot KL/MSE-to-LLM curves for Bayesian observer vs cache baseline."""

    out = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for true_graph, graph_rows in _group_rows(rows).items():
        series = {}
        for baseline in BASELINE_STYLES:
            key = f"{metric_prefix}_llm_{baseline}"
            lengths, means, sems = _mean_sem_by_length(graph_rows, key)
            if lengths:
                series[baseline] = (lengths, means, sems)
        if not series:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for baseline, (lengths, means, sems) in series.items():
            color, label = BASELINE_STYLES[baseline]
            _plot_mean_sem(ax, lengths, means, sems, color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("Context length")
        ax.set_ylabel(metric_prefix.upper())
        ax.set_title(f"{metric_prefix.upper()} to LLM distribution: true {true_graph}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_path / f"{metric_prefix}_to_llm_{true_graph}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    return out


def plot_llm_correlation_comparison(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Plot correlation between LLM distribution and each baseline distribution."""

    out = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for true_graph, graph_rows in _group_rows(rows).items():
        series = {}
        for baseline in BASELINE_STYLES:
            key = f"corr_llm_{baseline}"
            lengths, means, sems = _mean_sem_by_length(graph_rows, key)
            if lengths:
                series[baseline] = (lengths, means, sems)
        if not series:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for baseline, (lengths, means, sems) in series.items():
            color, label = BASELINE_STYLES[baseline]
            _plot_mean_sem(ax, lengths, means, sems, color=color, label=label)
        ax.axhline(0.0, color="gray", lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Pearson correlation with LLM distribution")
        ax.set_title(f"Distribution correlation to LLM: true {true_graph}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_path / f"corr_to_llm_{true_graph}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    return out


def plot_neighbor_probability_comparison(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Plot P(next token in true-graph neighbors) for LLM, Bayes, and Cache."""

    out = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for true_graph, graph_rows in _group_rows(rows).items():
        lengths_l, means_l, sems_l = _mean_sem_by_length(graph_rows, "llm_neighbor_prob")
        lengths_b, means_b, sems_b = _mean_sem_by_length(graph_rows, "bayes_neighbor_prob")
        lengths_c, means_c, sems_c = _mean_sem_by_length(graph_rows, "cache_neighbor_prob")
        lengths_e, means_e, sems_e = _mean_sem_by_length(graph_rows, "edge_learner_neighbor_prob")
        lengths_u, means_u, sems_u = _mean_sem_by_length(graph_rows, "unigram_neighbor_prob")
        if not lengths_b or not lengths_c:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        if lengths_l:
            _plot_mean_sem(ax, lengths_l, means_l, sems_l, color="black", label="LLM")
        _plot_mean_sem(ax, lengths_b, means_b, sems_b, color="#1976D2", label="Bayes")
        if lengths_e:
            _plot_mean_sem(ax, lengths_e, means_e, sems_e, color="#2E7D32", label="Edge learner")
        _plot_mean_sem(ax, lengths_c, means_c, sems_c, color="#C62828", label="Cache")
        if lengths_u:
            _plot_mean_sem(ax, lengths_u, means_u, sems_u, color="#00838F", label="Unigram Dirichlet")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Context length")
        ax.set_ylabel("P(next token in true-graph neighbors)")
        ax.set_title(f"Neighbor probability comparison: true {true_graph}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_path / f"neighbor_probability_{true_graph}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    return out


def plot_llm_vocab_mass(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Plot raw LLM probability mass assigned to the 16-word task vocabulary."""

    out = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for true_graph, graph_rows in _group_rows(rows).items():
        lengths, means, sems = _mean_sem_by_length(graph_rows, "llm_vocab_mass")
        if not lengths:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        _plot_mean_sem(ax, lengths, means, sems, color="#424242", label="LLM vocab mass")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Context length")
        ax.set_ylabel("P(next token in 16-word task vocabulary)")
        ax.set_title(f"LLM task-vocabulary mass: true {true_graph}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_path / f"llm_vocab_mass_{true_graph}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    return out


def plot_closest_baseline(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    """Plot fraction of seeds closest to each baseline by KL at each context length."""

    out = []
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for true_graph, graph_rows in _group_rows(rows).items():
        labels_by_length = defaultdict(list)
        for row in graph_rows:
            label = row.get("closest_baseline_kl")
            if label is not None:
                labels_by_length[int(row["context_length"])].append(label)
        if not labels_by_length:
            continue

        lengths = sorted(labels_by_length)
        fig, ax = plt.subplots(figsize=(8, 5))
        for baseline, (color, label) in BASELINE_STYLES.items():
            frac = [
                labels_by_length[L].count(baseline) / len(labels_by_length[L])
                for L in lengths
            ]
            ax.plot(lengths, frac, "o-", lw=2, ms=4, color=color, label=label)
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Context length")
        ax.set_ylabel("Fraction closest by KL")
        ax.set_title(f"Closest baseline by KL: true {true_graph}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = out_path / f"closest_baseline_kl_{true_graph}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(path)

    return out


def make_all_plots(rows: Sequence[dict], out_dir: str | Path) -> list[Path]:
    paths = []
    print("Generating Bayesian posterior plots...")
    paths.extend(plot_bayesian_posterior(rows, out_dir))
    print("Generating LLM vocabulary-mass plots...")
    paths.extend(plot_llm_vocab_mass(rows, out_dir))
    print("Generating neighbor-probability plots...")
    paths.extend(plot_neighbor_probability_comparison(rows, out_dir))
    print("Generating KL-to-LLM plots...")
    paths.extend(plot_llm_metric_comparison(rows, out_dir, "kl"))
    print("Generating MSE-to-LLM plots...")
    paths.extend(plot_llm_metric_comparison(rows, out_dir, "mse"))
    print("Generating correlation-to-LLM plots...")
    paths.extend(plot_llm_correlation_comparison(rows, out_dir))
    print("Generating closest-baseline plots...")
    paths.extend(plot_closest_baseline(rows, out_dir))
    return paths
