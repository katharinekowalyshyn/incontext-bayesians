"""Milestones 3 & 4 — Upgrade fit with MST-complexity prior.

This script fits two parameterizations of the Upgrade model per *condition*
and reports AIC / BIC so we can let the data pick the better one.

(A) Per-graph Upgrade — ``--model per_graph`` (default)
------------------------------------------------------
Each in-context graph ``k ∈ {grid, ring}`` has its own sigmoid, driven by the
number of k-graph tokens seen so far ``ρ_k·N``::

    p̂_k(N, ρ) = p₀_k + (q_k − p₀_k) · σ(b_k + γ_k · (ρ_k · N)^(1 − α_k))
    b_k       = b₀ − λ · C_MST(G_k)

Free parameters θ = (b₀, λ, γ_grid, α_grid, q_grid, γ_ring, α_ring, q_ring).
``p₀_k`` is estimated empirically per (condition, ρ, graph) from early-L
observations, following Checkpoint-2 §3.1, and held fixed at the
*maximum-mixture* estimate across ρ-cells.

(B) Mixture-bias Upgrade — ``--model mixture_bias``
---------------------------------------------------
A shared sigmoid whose bias is a mixture of the two per-graph priors::

    p̂(N, ρ) = p₀(ρ) + (q − p₀(ρ)) · σ(b(ρ) + γ · N^(1−α))
    b(ρ)    = (1 − ρ)·(b₀ − λ C_MST(G_grid)) + ρ·(b₀ − λ C_MST(G_ring))
    p₀(ρ)  = (1 − ρ)·p₀_grid + ρ·p₀_ring

Free parameters θ = (b₀, λ, γ, α, q).

This is the "null" version of the Upgrade where the two graphs share
evidence machinery but differ only in prior mass.

Objective
---------
MSE over all (condition-kept walks) × (graph = grid or ring) × (ρ) × (L)
triples where an observation exists.  Walks split 0.5/0.25/0.25
(train/val/test) deterministically within each (ρ, graph) cell.

Model selection
---------------
Under an iid-Gaussian residual assumption the MLE of σ² is ``MSE``, so for
``n`` observations with ``k`` free parameters we report::

    AIC = n · (log(2π·MSE) + 1) + 2k
    BIC = n · (log(2π·MSE) + 1) + k · log n

Lower is better; ΔAIC = AIC_A − AIC_B.

Usage
-----
    python src/experiments/fit_upgrade.py --model both --plot
    python src/experiments/fit_upgrade.py --condition months_permuted --model per_graph
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_loading import (  # noqa: E402
    CONDITIONS, RESULTS_ROOT,
    apply_split, c_mst, ensure_dir, estimate_p0, to_cells,
)

FIT_DIR = os.path.join(RESULTS_ROOT, "upgrade_fits")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures", "upgrade")
RNG_SEED = 0xDAFA17
N_RESTARTS = 24   # Upgrade has 8 params — searches a wider space than baseline.


# ─── Common numerics ─────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _safe_log(x):
    return np.log(max(float(x), 1e-20))


def aic_bic(mse: float, n: int, k: int) -> tuple[float, float]:
    """Gaussian-residual AIC / BIC."""
    if n <= 0 or not np.isfinite(mse) or mse <= 0:
        return float("nan"), float("nan")
    gll = n * (_safe_log(2 * np.pi * mse) + 1.0)   # -2 · log-lik (Gaussian)
    return gll + 2 * k, gll + k * _safe_log(n)


# ─── Observation packing ────────────────────────────────────────────────────
#
# We collect every (walk, ρ, graph, L) observation into flat numpy arrays and
# fit by minimising MSE over the ``train`` mask.  This is both the simplest and
# the fastest option — scipy's L-BFGS-B handles thousands of observations
# comfortably in milliseconds.
#
# ``rho_k`` is defined per-row as the fraction of k-graph tokens:
#   * grid row at nominal ρ  →  ρ_grid = 1 − ρ
#   * ring row at nominal ρ  →  ρ_ring = ρ
# (ρ is the fraction of ring segments; see ``make_interleaved_sequence``.)

@dataclass
class ObsTable:
    condition: str
    N: np.ndarray
    rho_k: np.ndarray           # fraction of k-graph tokens for that row
    graph: np.ndarray           # 0 = grid, 1 = ring
    rho: np.ndarray             # nominal ρ of the sequence (for plotting)
    walk: np.ndarray            # walk index within (ρ, graph) cell
    split: np.ndarray           # 'train' / 'val' / 'test'
    y: np.ndarray
    p0_by_rho_graph: dict[tuple[float, str], float]  # for mixture_bias model

    @property
    def train(self) -> np.ndarray:
        return self.split == "train"

    @property
    def val(self) -> np.ndarray:
        return self.split == "val"

    @property
    def test(self) -> np.ndarray:
        return self.split == "test"


def build_observations(condition: str,
                       ratios: tuple[float, float, float] = (0.5, 0.25, 0.25)
                       ) -> ObsTable:
    cells = to_cells(condition)
    N_parts, rho_parts, rho_k_parts = [], [], []
    graph_parts, walk_parts, y_parts, split_parts = [], [], [], []
    p0_by_rho_graph: dict[tuple[float, str], float] = {}

    for cell in cells:
        apply_split(cell, ratios=ratios)
        cell.p0 = estimate_p0(cell)
        p0_by_rho_graph[(cell.rho, cell.graph)] = cell.p0

        splits = np.array(["train"] * cell.n_walks, dtype=object)
        splits[cell.val_idx] = "val"
        splits[cell.test_idx] = "test"

        # Flatten with NaN mask.
        acc = cell.acc
        L = cell.L
        for w in range(cell.n_walks):
            row = acc[w]
            mask = ~np.isnan(row)
            if not mask.any():
                continue
            Ns = L[mask]
            ys = row[mask]
            m = ys.size
            rho_k_val = (1.0 - cell.rho) if cell.graph == "grid" else cell.rho
            # At ρ=0 on a pure-grid run every token is on the grid → ρ_k=1;
            # the formula above already yields that.  At ρ=1 on pure-ring,
            # ρ_k=1 for the ring rows, 0 for any grid rows (which don't exist).
            N_parts.append(Ns)
            rho_parts.append(np.full(m, cell.rho))
            rho_k_parts.append(np.full(m, rho_k_val))
            graph_parts.append(np.full(m, 0 if cell.graph == "grid" else 1))
            walk_parts.append(np.full(m, w))
            y_parts.append(ys)
            split_parts.append(np.full(m, splits[w], dtype=object))

    return ObsTable(
        condition=condition,
        N=np.concatenate(N_parts),
        rho=np.concatenate(rho_parts),
        rho_k=np.concatenate(rho_k_parts),
        graph=np.concatenate(graph_parts).astype(int),
        walk=np.concatenate(walk_parts).astype(int),
        y=np.concatenate(y_parts).astype(float),
        split=np.concatenate(split_parts).astype(object),
        p0_by_rho_graph=p0_by_rho_graph,
    )


# ─── Model A: per-graph Upgrade ─────────────────────────────────────────────
#
# θ = (b0, lam, γ_g, α_g, q_g, γ_r, α_r, q_r), 8 params.
# p₀_k is fixed at the mean of ``p0_by_rho_graph[ρ, k]`` across the ρ values
# where graph k was observed.  This keeps λ and b₀ identifiable.

PER_GRAPH_PARAM_NAMES = ["b0", "lam", "gamma_grid", "alpha_grid", "q_grid",
                         "gamma_ring", "alpha_ring", "q_ring"]

C_GRID = c_mst("grid")
C_RING = c_mst("ring")


def _fixed_p0_per_graph(obs: ObsTable) -> dict[str, float]:
    p0s = {"grid": [], "ring": []}
    for (_, g), v in obs.p0_by_rho_graph.items():
        p0s[g].append(v)
    return {g: float(np.mean(vs)) if vs else float("nan") for g, vs in p0s.items()}


def per_graph_predict(theta: np.ndarray, obs: ObsTable,
                      p0s: dict[str, float]) -> np.ndarray:
    b0, lam, g_g, a_g, q_g, g_r, a_r, q_r = theta
    b_grid = b0 - lam * C_GRID
    b_ring = b0 - lam * C_RING

    is_grid = (obs.graph == 0)
    is_ring = ~is_grid

    # Effective context length for each graph-k row = ρ_k · N, clamped to ≥1.
    eff_grid = np.clip(obs.rho_k * obs.N, 1.0, None)
    eff_ring = np.clip(obs.rho_k * obs.N, 1.0, None)

    y_hat = np.empty_like(obs.y)
    if is_grid.any():
        z = b_grid + g_g * np.power(eff_grid[is_grid], 1.0 - a_g)
        y_hat[is_grid] = p0s["grid"] + (q_g - p0s["grid"]) * sigmoid(z)
    if is_ring.any():
        z = b_ring + g_r * np.power(eff_ring[is_ring], 1.0 - a_r)
        y_hat[is_ring] = p0s["ring"] + (q_r - p0s["ring"]) * sigmoid(z)
    return y_hat


def _per_graph_bounds(p0s: dict[str, float]):
    eps = 1e-3
    return [
        (-15.0, 15.0),                        # b0
        (-2.0, 2.0),                          # lam (can be negative → violates Bayes, flag)
        (1e-6, 50.0), (0.0, 0.99), (p0s["grid"] + eps, 1.0),
        (1e-6, 50.0), (0.0, 0.99), (p0s["ring"] + eps, 1.0),
    ]


def _per_graph_init(rng: np.random.Generator, p0s: dict[str, float]) -> np.ndarray:
    return np.array([
        rng.uniform(-4, 0),                                     # b0
        rng.uniform(-0.05, 0.3),                                # lam — small positive
        rng.uniform(0.05, 3.0), rng.uniform(0.2, 0.85),         # γ_g, α_g
        rng.uniform(max(p0s["grid"] + 0.05, 0.6), 0.98),        # q_g
        rng.uniform(0.05, 3.0), rng.uniform(0.2, 0.85),         # γ_r, α_r
        rng.uniform(max(p0s["ring"] + 0.05, 0.6), 0.98),        # q_r
    ])


# ─── Model B: mixture-bias Upgrade ──────────────────────────────────────────
#
# θ = (b0, lam, γ, α, q), 5 params.

MIXTURE_PARAM_NAMES = ["b0", "lam", "gamma", "alpha", "q"]


def mixture_bias_predict(theta: np.ndarray, obs: ObsTable,
                         p0s_by_rho_graph: dict[tuple[float, str], float]) -> np.ndarray:
    b0, lam, gamma, alpha, q = theta
    b_grid_prior = b0 - lam * C_GRID
    b_ring_prior = b0 - lam * C_RING
    # Mixed bias depends on nominal ρ of the sequence (shared across graphs
    # in that row — the row's own graph is bookkeeping only).
    b_eff = (1.0 - obs.rho) * b_grid_prior + obs.rho * b_ring_prior
    # Mixed p0 by nominal ρ.
    p0_grid = np.array([p0s_by_rho_graph.get((r, "grid"), np.nan) for r in obs.rho])
    p0_ring = np.array([p0s_by_rho_graph.get((r, "ring"), np.nan) for r in obs.rho])
    # If one side is missing (ρ=0 ring or ρ=1 grid), fall back to the other.
    p0_grid = np.where(np.isnan(p0_grid), p0_ring, p0_grid)
    p0_ring = np.where(np.isnan(p0_ring), p0_grid, p0_ring)
    p0_eff = (1.0 - obs.rho) * p0_grid + obs.rho * p0_ring

    N_safe = np.clip(obs.N, 1.0, None)
    z = b_eff + gamma * np.power(N_safe, 1.0 - alpha)
    return p0_eff + (q - p0_eff) * sigmoid(z)


def _mixture_bounds(p0_eff_min: float):
    eps = 1e-3
    return [
        (-15.0, 15.0),           # b0
        (-2.0, 2.0),             # lam
        (1e-6, 50.0),            # gamma
        (0.0, 0.99),             # alpha
        (p0_eff_min + eps, 1.0), # q
    ]


def _mixture_init(rng: np.random.Generator, p0_eff_min: float) -> np.ndarray:
    return np.array([
        rng.uniform(-4, 0),
        rng.uniform(-0.05, 0.3),
        rng.uniform(0.05, 3.0),
        rng.uniform(0.2, 0.85),
        rng.uniform(max(p0_eff_min + 0.05, 0.6), 0.98),
    ])


# ─── Unified fit driver ──────────────────────────────────────────────────────

@dataclass
class UpgradeFit:
    model: str
    condition: str
    params: dict[str, float]
    p0: dict[str, float]
    C_MST: dict[str, float] = field(default_factory=lambda: {"grid": C_GRID, "ring": C_RING})
    mse_train: float = float("nan")
    mse_val: float = float("nan")
    mse_test: float = float("nan")
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    k_params: int = 0
    aic: float = float("nan")
    bic: float = float("nan")
    converged: bool = False


def _mse(theta, obs, split_mask, predict_fn, *args):
    y_hat = predict_fn(theta, obs, *args)
    err = y_hat[split_mask] - obs.y[split_mask]
    return float(np.mean(err * err)) if err.size else float("nan")


def fit_model(obs: ObsTable, *, model: str,
              rng: np.random.Generator | None = None,
              n_restarts: int = N_RESTARTS) -> UpgradeFit:
    rng = rng or np.random.default_rng(RNG_SEED)

    if model == "per_graph":
        p0s = _fixed_p0_per_graph(obs)
        predict_fn = per_graph_predict
        bounds = _per_graph_bounds(p0s)
        init_fn = lambda: _per_graph_init(rng, p0s)
        param_names = PER_GRAPH_PARAM_NAMES
        predict_args: tuple = (p0s,)
    elif model == "mixture_bias":
        predict_fn = mixture_bias_predict
        p0_eff_min = min(v for v in obs.p0_by_rho_graph.values() if np.isfinite(v))
        bounds = _mixture_bounds(p0_eff_min)
        init_fn = lambda: _mixture_init(rng, p0_eff_min)
        param_names = MIXTURE_PARAM_NAMES
        predict_args = (obs.p0_by_rho_graph,)
        p0s = {f"rho={r:.2f},{g}": v for (r, g), v in obs.p0_by_rho_graph.items()}
    else:
        raise ValueError(f"unknown model: {model!r}")

    train_mask = obs.train
    def loss(theta):
        return _mse(theta, obs, train_mask, predict_fn, *predict_args)

    best = None
    for _ in range(n_restarts):
        theta0 = init_fn()
        try:
            res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-10})
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best[1]:
            best = (res.x.copy(), float(res.fun), bool(res.success))
    if best is None:
        theta = np.full(len(bounds), np.nan)
        train_mse = float("nan")
        converged = False
    else:
        theta, train_mse, converged = best

    val_mse = _mse(theta, obs, obs.val, predict_fn, *predict_args)
    test_mse = _mse(theta, obs, obs.test, predict_fn, *predict_args)
    k = len(theta)
    aic, bic = aic_bic(train_mse, int(train_mask.sum()), k)

    return UpgradeFit(
        model=model, condition=obs.condition,
        params={name: float(v) for name, v in zip(param_names, theta)},
        p0=p0s,
        mse_train=train_mse, mse_val=val_mse, mse_test=test_mse,
        n_train=int(train_mask.sum()), n_val=int(obs.val.sum()),
        n_test=int(obs.test.sum()), k_params=k,
        aic=aic, bic=bic, converged=converged,
    )


# ─── Plotting ────────────────────────────────────────────────────────────────

GRID_COLOR = "#1976D2"
RING_COLOR = "#C62828"


def plot_condition(obs: ObsTable, fits: dict[str, UpgradeFit]) -> str:
    """One figure per condition: 3-panel strip (ρ=0, 0.5, 1) with data and
    overlaid predictions from whichever models are available.
    """
    rhos = sorted(np.unique(obs.rho))
    fig, axes = plt.subplots(1, len(rhos), figsize=(4 + 3.2 * len(rhos), 4.4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    # Dense grid for plotting predictions per (graph, ρ).
    N_grid = np.geomspace(10, 2500, 200)

    for ax, rho in zip(axes, rhos):
        sel_rho = obs.rho == rho
        for g, color, g_id in [("grid", GRID_COLOR, 0), ("ring", RING_COLOR, 1)]:
            rows = sel_rho & (obs.graph == g_id)
            if not rows.any():
                continue
            # observed per-L mean ± SEM across walks
            Ls = np.unique(obs.N[rows])
            means, sems = [], []
            for L in Ls:
                vals = obs.y[rows & (obs.N == L)]
                if vals.size == 0:
                    continue
                means.append(vals.mean())
                sems.append(vals.std() / max(np.sqrt(vals.size), 1))
            ax.errorbar(Ls, means, yerr=sems, fmt="o", color=color,
                        alpha=0.85, label=g, ms=5, zorder=3)

        # Overlay each model's prediction on this panel.
        for i, (model_name, fit) in enumerate(fits.items()):
            ls = "-" if model_name == "per_graph" else "--"
            for g, color, g_id in [("grid", GRID_COLOR, 0), ("ring", RING_COLOR, 1)]:
                # Build a pseudo-obs for the curve.
                if not ((obs.rho == rho) & (obs.graph == g_id)).any():
                    continue
                rho_k = (1.0 - rho) if g == "grid" else rho
                pseudo = ObsTable(
                    condition=obs.condition,
                    N=N_grid,
                    rho=np.full_like(N_grid, rho),
                    rho_k=np.full_like(N_grid, rho_k),
                    graph=np.full(N_grid.size, g_id, dtype=int),
                    walk=np.zeros(N_grid.size, dtype=int),
                    y=np.zeros_like(N_grid),
                    split=np.full(N_grid.size, "train", dtype=object),
                    p0_by_rho_graph=obs.p0_by_rho_graph,
                )
                theta = np.array([fit.params[name] for name in fit.params])
                if model_name == "per_graph":
                    y_hat = per_graph_predict(theta, pseudo,
                                              {k: v for k, v in fit.p0.items()
                                               if k in ("grid", "ring")})
                else:
                    y_hat = mixture_bias_predict(theta, pseudo, obs.p0_by_rho_graph)
                ax.plot(N_grid, y_hat, ls=ls, color=color, lw=1.5, alpha=0.85,
                        label=None if i else f"{model_name}")

        ax.set_xscale("log")
        ax.set_xlim(40, 2500)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"ρ={rho:.2f}")
        ax.set_xlabel("Context length N")
        ax.grid(alpha=0.25)

    # Build a single combined legend.
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        axes[0].legend(handles, labels, fontsize=8, loc="lower right",
                       framealpha=0.85)
    axes[0].set_ylabel("P(next ∈ valid neighbors)")

    fit_summaries = " | ".join(
        f"{m}: λ={f.params.get('lam', float('nan')):+.3f}, AIC={f.aic:.1f}"
        for m, f in fits.items() if np.isfinite(f.aic)
    )
    fig.suptitle(
        f"Upgrade fit — {obs.condition}\n{fit_summaries}",
        fontsize=11,
    )
    fig.tight_layout()

    ensure_dir(FIG_DIR)
    path = os.path.join(FIG_DIR, f"upgrade_{obs.condition}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── I/O ─────────────────────────────────────────────────────────────────────

def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_condition_fits(condition: str, fits: dict[str, UpgradeFit]) -> str:
    ensure_dir(FIT_DIR)
    path = os.path.join(FIT_DIR, f"{condition}.json")
    payload = {
        "condition": condition,
        "C_MST": {"grid": C_GRID, "ring": C_RING},
        "models": {name: asdict(f) for name, f in fits.items()},
    }
    if len(fits) == 2:
        a = fits["per_graph"]; b = fits["mixture_bias"]
        payload["comparison"] = {
            "delta_aic_per_graph_minus_mixture": a.aic - b.aic,
            "delta_bic_per_graph_minus_mixture": a.bic - b.bic,
            "preferred_by_aic": "per_graph" if a.aic < b.aic else "mixture_bias",
            "preferred_by_bic": "per_graph" if a.bic < b.bic else "mixture_bias",
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument(
        "--model", choices=["per_graph", "mixture_bias", "both"],
        default="both",
        help="Which upgrade parameterisation to fit (default: both, with AIC/BIC compare).",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--split", nargs=3, type=float, default=(0.5, 0.25, 0.25),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    args = parser.parse_args()

    todo = CONDITIONS if args.condition == "all" else [args.condition]
    models = (["per_graph", "mixture_bias"] if args.model == "both"
              else [args.model])

    rng = np.random.default_rng(args.seed)
    for cond in todo:
        print(f"\n=== {cond} ===")
        obs = build_observations(cond, ratios=tuple(args.split))
        fits: dict[str, UpgradeFit] = {}
        for m in models:
            fit = fit_model(obs, model=m, rng=rng)
            fits[m] = fit
            ps = ", ".join(f"{k}={v:+.3f}" for k, v in fit.params.items())
            print(f"  [{m}] MSE tr/va/te = {fit.mse_train:.4f}/"
                  f"{fit.mse_val:.4f}/{fit.mse_test:.4f}   "
                  f"AIC={fit.aic:.1f}  BIC={fit.bic:.1f}")
            print(f"           {ps}")
        out = save_condition_fits(cond, fits)
        print(f"  saved fits → {out}")
        if args.plot:
            png = plot_condition(obs, fits)
            print(f"  saved plot → {png}")
        if len(fits) == 2:
            a, b = fits["per_graph"], fits["mixture_bias"]
            print(f"  ΔAIC (per_graph − mixture) = {a.aic - b.aic:+.2f}")
            print(f"  ΔBIC (per_graph − mixture) = {a.bic - b.bic:+.2f}")


if __name__ == "__main__":
    main()
