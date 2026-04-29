"""Bayesian mixture-of-baselines post-processing analysis."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graphs import build_candidate_graphs
from .metrics import kl_divergence


BASELINES: tuple[tuple[str, str], ...] = (
    ("ideal_bayes", "bayes_distribution"),
    ("edge_learner", "edge_learner_distribution"),
    ("cache", "cache_distribution"),
    ("semantic_prior", "semantic_prior_distribution"),
)

BASELINE_COLORS = {
    "ideal_bayes": "#1976D2",
    "edge_learner": "#2E7D32",
    "cache": "#C62828",
    "semantic_prior": "#6A1B9A",
    "mixture": "#111111",
}


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list of rows in {path}")
    return rows


def _dist_from_row(row: Mapping, key: str, words: Sequence[str]) -> np.ndarray:
    raw = row[key]
    arr = np.array([raw[word] for word in words], dtype=float)
    total = arr.sum()
    if total <= 0:
        raise ValueError(f"Distribution {key} has nonpositive mass.")
    return arr / total


def usable_rows(rows: Sequence[dict]) -> tuple[list[dict], tuple[str, ...]]:
    kept = []
    words: tuple[str, ...] | None = None
    required = ["llm_distribution"] + [key for _, key in BASELINES]
    for row in rows:
        if not all(key in row for key in required):
            continue
        row_words = tuple(row["llm_distribution"].keys())
        if words is None:
            words = row_words
        elif row_words != words:
            raise ValueError("Rows use inconsistent vocabulary order.")
        kept.append(row)
    if words is None:
        raise ValueError(
            "No rows contain llm_distribution plus all four baseline distributions. "
            "Run LLM inference with semantic prior metrics first."
        )
    return kept, words


def softmax(theta: np.ndarray) -> np.ndarray:
    centered = theta - np.max(theta, axis=-1, keepdims=True)
    exps = np.exp(centered)
    return exps / exps.sum(axis=-1, keepdims=True)


def _objective_and_grad_theta(
    theta: np.ndarray,
    llm: np.ndarray,
    baselines: np.ndarray,
    alpha: np.ndarray,
    eps: float,
) -> tuple[float, np.ndarray]:
    """Mean KL(p_llm || p_mix) plus negative log Dirichlet prior."""

    lam = softmax(theta)
    mix = np.einsum("k,nkv->nv", lam, baselines)
    mix = np.clip(mix, eps, None)
    kl = np.sum(llm * (np.log(np.clip(llm, eps, None)) - np.log(mix)), axis=1)
    objective = float(np.mean(kl))

    grad_lam = -np.mean(np.einsum("nv,nkv->nk", llm / mix, baselines), axis=0)

    if alpha is not None:
        prior_coeff = alpha - 1.0
        if np.any(prior_coeff != 0.0):
            objective -= float(np.sum(prior_coeff * np.log(np.clip(lam, eps, None))))
            grad_lam -= prior_coeff / np.clip(lam, eps, None)

    grad_theta = lam * (grad_lam - np.dot(grad_lam, lam))
    return objective, grad_theta


def fit_simplex_weights(
    llm: np.ndarray,
    baselines: np.ndarray,
    alpha: Sequence[float] | float = 1.0,
    n_steps: int = 1200,
    lr: float = 0.05,
    eps: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Fit one simplex weight vector for a group of examples."""

    n_baselines = baselines.shape[1]
    alpha_arr = (
        np.full(n_baselines, float(alpha), dtype=float)
        if np.isscalar(alpha)
        else np.asarray(alpha, dtype=float)
    )
    theta = np.zeros(n_baselines, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    best_obj = float("inf")
    best_theta = theta.copy()

    for step in range(1, n_steps + 1):
        obj, grad = _objective_and_grad_theta(theta, llm, baselines, alpha_arr, eps)
        if obj < best_obj:
            best_obj = obj
            best_theta = theta.copy()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad ** 2)
        m_hat = m / (1.0 - 0.9 ** step)
        v_hat = v / (1.0 - 0.999 ** step)
        theta -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    lam = softmax(best_theta)
    mix = np.einsum("k,nkv->nv", lam, baselines)
    mean_kl = float(np.mean([kl_divergence(p, q, eps=eps) for p, q in zip(llm, mix)]))
    return lam, mean_kl


def _rows_to_arrays(rows: Sequence[dict], words: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    llm = []
    baseline_arrays = []
    for row in rows:
        llm.append(_dist_from_row(row, "llm_distribution", words))
        baseline_arrays.append([
            _dist_from_row(row, key, words)
            for _, key in BASELINES
        ])
    return np.stack(llm), np.asarray(baseline_arrays, dtype=float)


def fit_by_context_length(
    rows: Sequence[dict],
    words: Sequence[str],
    alpha: float,
    n_steps: int,
    lr: float,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["context_length"])].append(row)

    out = []
    for context_length in sorted(grouped):
        group_rows = grouped[context_length]
        llm, baselines = _rows_to_arrays(group_rows, words)
        weights, mix_kl = fit_simplex_weights(llm, baselines, alpha=alpha, n_steps=n_steps, lr=lr)
        individual = {}
        for idx, (name, _) in enumerate(BASELINES):
            individual[name] = float(
                np.mean([kl_divergence(p, q) for p, q in zip(llm, baselines[:, idx, :])])
            )
        out.append({
            "context_length": context_length,
            "n_examples": len(group_rows),
            "weights": {name: float(weights[i]) for i, (name, _) in enumerate(BASELINES)},
            "kl_llm_mixture": mix_kl,
            "kl_llm_individual": individual,
        })
    return out


def _smooth_objective_and_grad(
    params: np.ndarray,
    llm: np.ndarray,
    baselines: np.ndarray,
    log_t: np.ndarray,
    alpha: np.ndarray,
    eps: float,
) -> tuple[float, np.ndarray]:
    n_baselines = baselines.shape[1]
    a = params[:n_baselines]
    b = params[n_baselines:]
    theta = a[None, :] + log_t[:, None] * b[None, :]
    lam = softmax(theta)
    mix = np.einsum("nk,nkv->nv", lam, baselines)
    mix = np.clip(mix, eps, None)

    kl = np.sum(llm * (np.log(np.clip(llm, eps, None)) - np.log(mix)), axis=1)
    objective = float(np.mean(kl))
    grad_lam = -(np.einsum("nv,nkv->nk", llm / mix, baselines) / llm.shape[0])

    prior_coeff = alpha - 1.0
    if np.any(prior_coeff != 0.0):
        objective -= float(np.sum(prior_coeff[None, :] * np.log(np.clip(lam, eps, None))) / llm.shape[0])
        grad_lam -= prior_coeff[None, :] / np.clip(lam, eps, None) / llm.shape[0]

    dot = np.sum(grad_lam * lam, axis=1, keepdims=True)
    grad_theta = lam * (grad_lam - dot)
    grad_a = grad_theta.sum(axis=0)
    grad_b = (grad_theta * log_t[:, None]).sum(axis=0)
    return objective, np.concatenate([grad_a, grad_b])


def fit_smooth_context_model(
    rows: Sequence[dict],
    words: Sequence[str],
    alpha: float,
    n_steps: int,
    lr: float,
    eps: float = 1e-12,
) -> dict:
    llm, baselines = _rows_to_arrays(rows, words)
    lengths = np.array([int(row["context_length"]) for row in rows], dtype=float)
    log_t_raw = np.log(lengths)
    log_t = (log_t_raw - log_t_raw.mean()) / max(log_t_raw.std(), 1e-8)

    n_baselines = baselines.shape[1]
    alpha_arr = np.full(n_baselines, float(alpha), dtype=float)
    params = np.zeros(2 * n_baselines, dtype=float)
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    best_obj = float("inf")
    best_params = params.copy()

    for step in range(1, n_steps + 1):
        obj, grad = _smooth_objective_and_grad(params, llm, baselines, log_t, alpha_arr, eps)
        if obj < best_obj:
            best_obj = obj
            best_params = params.copy()
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad ** 2)
        params -= lr * (m / (1.0 - 0.9 ** step)) / (np.sqrt(v / (1.0 - 0.999 ** step)) + 1e-8)

    a = best_params[:n_baselines]
    b = best_params[n_baselines:]
    theta = a[None, :] + log_t[:, None] * b[None, :]
    lam_by_row = softmax(theta)
    mix = np.einsum("nk,nkv->nv", lam_by_row, baselines)
    mean_kl = float(np.mean([kl_divergence(p, q, eps=eps) for p, q in zip(llm, mix)]))

    unique_lengths = sorted(set(int(x) for x in lengths))
    weights_by_length = []
    for L in unique_lengths:
        z = (np.log(L) - log_t_raw.mean()) / max(log_t_raw.std(), 1e-8)
        weights = softmax(a + b * z)
        weights_by_length.append({
            "context_length": L,
            "weights": {name: float(weights[i]) for i, (name, _) in enumerate(BASELINES)},
        })

    return {
        "mean_kl_llm_mixture": mean_kl,
        "log_t_mean": float(log_t_raw.mean()),
        "log_t_std": float(max(log_t_raw.std(), 1e-8)),
        "a": {name: float(a[i]) for i, (name, _) in enumerate(BASELINES)},
        "b": {name: float(b[i]) for i, (name, _) in enumerate(BASELINES)},
        "weights_by_context_length": weights_by_length,
    }


def plot_weights(weight_rows: Sequence[dict], out_path: str | Path, title: str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lengths = [row["context_length"] for row in weight_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _ in BASELINES:
        values = [row["weights"][name] for row in weight_rows]
        ax.plot(lengths, values, "o-", lw=2, ms=4, color=BASELINE_COLORS[name], label=name)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Mixture weight")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_kl_comparison(fit_rows: Sequence[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lengths = [row["context_length"] for row in fit_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        lengths,
        [row["kl_llm_mixture"] for row in fit_rows],
        "o-",
        lw=2.5,
        ms=4,
        color=BASELINE_COLORS["mixture"],
        label="mixture",
    )
    for name, _ in BASELINES:
        ax.plot(
            lengths,
            [row["kl_llm_individual"][name] for row in fit_rows],
            "o-",
            lw=1.8,
            ms=3,
            color=BASELINE_COLORS[name],
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Context length")
    ax.set_ylabel("KL(LLM || model)")
    ax.set_title("Mixture KL vs individual baselines")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def _neighbor_mass(distribution: np.ndarray, words: Sequence[str], row: Mapping) -> float | None:
    source_graph = row.get("source_graph") or row.get("true_graph")
    graphs = build_candidate_graphs(words)
    if source_graph not in graphs:
        return None
    current_word = row.get("current_word")
    if current_word is None:
        return None
    word_to_idx = {word: i for i, word in enumerate(words)}
    return float(sum(distribution[word_to_idx[word]] for word in graphs[source_graph].get_valid_next_words(current_word)))


def plot_mixture_neighbor_fit(
    rows: Sequence[dict],
    words: Sequence[str],
    fit_rows: Sequence[dict],
    out_path: str | Path,
) -> Path:
    """Plot how the fitted mixture tracks the LLM neighbor-probability curve."""

    weights_by_length = {
        int(row["context_length"]): np.array(
            [row["weights"][name] for name, _ in BASELINES],
            dtype=float,
        )
        for row in fit_rows
    }
    grouped = defaultdict(lambda: {"llm": [], "mixture": []})

    for row in rows:
        L = int(row["context_length"])
        if L not in weights_by_length:
            continue
        llm = _dist_from_row(row, "llm_distribution", words)
        baselines = np.stack([_dist_from_row(row, key, words) for _, key in BASELINES])
        mixture = np.einsum("k,kv->v", weights_by_length[L], baselines)
        llm_mass = _neighbor_mass(llm, words, row)
        mixture_mass = _neighbor_mass(mixture, words, row)
        if llm_mass is None or mixture_mass is None:
            continue
        grouped[L]["llm"].append(llm_mass)
        grouped[L]["mixture"].append(mixture_mass)

    if not grouped:
        raise ValueError("Could not compute neighbor curve; rows need current_word and source_graph/true_graph.")

    lengths = sorted(grouped)
    llm_mean, llm_sem, mix_mean, mix_sem = [], [], [], []
    for L in lengths:
        m, s = _mean_sem(grouped[L]["llm"])
        llm_mean.append(m)
        llm_sem.append(s)
        m, s = _mean_sem(grouped[L]["mixture"])
        mix_mean.append(m)
        mix_sem.append(s)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths, llm_mean, "o-", lw=2.5, ms=4, color="#111111", label="LLM")
    if any(s > 0 for s in llm_sem):
        ax.fill_between(
            lengths,
            np.asarray(llm_mean) - np.asarray(llm_sem),
            np.asarray(llm_mean) + np.asarray(llm_sem),
            color="#111111",
            alpha=0.14,
            linewidth=0,
        )
    ax.plot(lengths, mix_mean, "o-", lw=2.5, ms=4, color="#E65100", label="Fitted mixture")
    if any(s > 0 for s in mix_sem):
        ax.fill_between(
            lengths,
            np.asarray(mix_mean) - np.asarray(mix_sem),
            np.asarray(mix_mean) + np.asarray(mix_sem),
            color="#E65100",
            alpha=0.18,
            linewidth=0,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Renormalized P(next in source-graph neighbors)")
    ax.set_title("LLM curve vs fitted mixture curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_mixture_analysis(
    input_path: str | Path,
    out_dir: str | Path | None = None,
    alpha: float = 1.2,
    n_steps: int = 1200,
    lr: float = 0.05,
    smooth: bool = True,
) -> list[Path]:
    rows, words = usable_rows(load_rows(input_path))
    output_dir = Path(out_dir) if out_dir is not None else Path(input_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    per_length = fit_by_context_length(rows, words, alpha=alpha, n_steps=n_steps, lr=lr)
    result = {
        "input_path": str(input_path),
        "dirichlet_alpha": alpha,
        "baselines": [name for name, _ in BASELINES],
        "per_context_length": per_length,
    }

    written = []
    if smooth:
        result["smooth_context_model"] = fit_smooth_context_model(
            rows,
            words,
            alpha=alpha,
            n_steps=n_steps,
            lr=lr,
        )

    json_path = output_dir / "mixture_analysis.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    written.append(json_path)
    written.append(plot_weights(per_length, output_dir / "mixture_weights_by_context.png", "Fitted mixture weights by context length"))
    written.append(plot_kl_comparison(per_length, output_dir / "mixture_kl_comparison.png"))
    try:
        written.append(
            plot_mixture_neighbor_fit(
                rows,
                words,
                per_length,
                output_dir / "mixture_neighbor_curve_fit.png",
            )
        )
    except ValueError as exc:
        print(f"Skipping mixture neighbor curve: {exc}")
    if smooth and "smooth_context_model" in result:
        written.append(
            plot_weights(
                result["smooth_context_model"]["weights_by_context_length"],
                output_dir / "mixture_weights_smooth_context.png",
                "Smooth context-length mixture weights",
            )
        )
    return written
