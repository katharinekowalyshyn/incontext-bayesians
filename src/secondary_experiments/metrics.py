"""Distribution-comparison metrics."""

from __future__ import annotations

import numpy as np


def _as_prob_vector(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D probability vector, got shape {arr.shape}.")
    total = arr.sum()
    if total <= 0:
        raise ValueError("Probability vector must have positive mass.")
    return arr / total


def kl_divergence(p, q, eps: float = 1e-12) -> float:
    """KL(p || q) with tiny clipping for numerical safety."""

    p_arr = np.clip(_as_prob_vector(p), eps, 1.0)
    q_arr = np.clip(_as_prob_vector(q), eps, 1.0)
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    return float(np.sum(p_arr * (np.log(p_arr) - np.log(q_arr))))


def mse(p, q) -> float:
    p_arr = _as_prob_vector(p)
    q_arr = _as_prob_vector(q)
    return float(np.mean((p_arr - q_arr) ** 2))


def pearson_corr(p, q) -> float:
    p_arr = _as_prob_vector(p)
    q_arr = _as_prob_vector(q)
    if np.std(p_arr) == 0.0 or np.std(q_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(p_arr, q_arr)[0, 1])
