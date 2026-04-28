"""Tim's Milestone 2 — Bigelow-style Baseline sigmoid fit.

Model
-----
For each (condition, ρ, graph) cell we fit::

    p̂(N; θ) = p₀ + (q − p₀) · σ(b + γ · N^(1−α))

with θ = (b, γ, α, q).  ``p₀`` is held fixed at its empirical value across
the pre-transition context lengths (Checkpoint-2 §3.1).

Objective
---------
Mean-squared error against the observed LLM accuracy across all train
(walk, L) pairs in the cell.  Val/test MSE is reported for the held-out
walks.  ``N* = (−b / γ)^(1/(1−α))`` is computed whenever b < 0 and γ > 0 —
this is what Katie's M8 overlay needs.

Usage
-----
    # Fit every condition in the default data dir:
    python src/experiments/fit_baseline.py

    # Fit just one condition, with 200 bootstrap resamples of training walks:
    python src/experiments/fit_baseline.py --condition months_natural --bootstrap 200

    # Also write a diagnostic PNG per condition:
    python src/experiments/fit_baseline.py --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data_loading import (  # noqa: E402
    CONDITIONS, RESULTS_ROOT,
    Cell, apply_split, ensure_dir, estimate_p0, flatten, to_cells,
)

FIT_DIR = os.path.join(RESULTS_ROOT, "baseline_fits")
FIG_DIR = os.path.join(RESULTS_ROOT, "figures", "baseline")

# Optimisation search space.  Values outside these bounds indicate degenerate
# fits (e.g. q ≤ p₀) and we treat them as failures.
BOUNDS_B     = (-30.0, 30.0)
BOUNDS_GAMMA = (1e-6, 50.0)
BOUNDS_ALPHA = (0.0, 0.99)    # α = 1 makes N^0 = 1 and the sigmoid degenerate
N_RESTARTS = 16
RNG_SEED = 0xB1A5E


# ─── Model + loss ────────────────────────────────────────────────────────────

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def predict(N: np.ndarray, *, b: float, gamma: float, alpha: float,
            q: float, p0: float) -> np.ndarray:
    """Baseline sigmoid prediction at context length(s) ``N``."""
    N = np.asarray(N, dtype=float)
    # N=0 would make N^(1-α) = 0 regardless of α; clamp to avoid log(0) below.
    N_safe = np.clip(N, 1.0, None)
    z = b + gamma * np.power(N_safe, 1.0 - alpha)
    return p0 + (q - p0) * sigmoid(z)


def mse(theta: np.ndarray, N: np.ndarray, y: np.ndarray, p0: float) -> float:
    b, gamma, alpha, q = theta
    y_hat = predict(N, b=b, gamma=gamma, alpha=alpha, q=q, p0=p0)
    return float(np.mean((y_hat - y) ** 2))


def n_star(b: float, gamma: float, alpha: float) -> float | None:
    """Sigmoid inflection point ``N* = (−b/γ)^(1/(1−α))``.

    Defined only when the sigmoid argument can hit 0 for some N > 0: need
    b < 0 and γ > 0.  Returns ``None`` otherwise (fit saturates immediately).
    """
    if b >= 0 or gamma <= 0 or alpha >= 1.0:
        return None
    ratio = -b / gamma
    if ratio <= 0:
        return None
    return float(ratio ** (1.0 / (1.0 - alpha)))


# ─── Single-cell fit with multi-start ────────────────────────────────────────

@dataclass
class FitResult:
    b: float
    gamma: float
    alpha: float
    q: float
    p0: float
    N_star: float | None
    mse_train: float
    mse_val: float
    mse_test: float
    n_train: int
    n_val: int
    n_test: int
    converged: bool
    bootstrap: dict[str, list[float]] | None = None


def _fit_once(N_train: np.ndarray, y_train: np.ndarray, p0: float,
              rng: np.random.Generator) -> tuple[np.ndarray, float, bool]:
    best = None
    for _ in range(N_RESTARTS):
        # Sample starting point uniformly inside the feasible box.
        theta0 = np.array([
            rng.uniform(-6.0, 0.5),                    # b: typically negative
            rng.uniform(0.05, 2.0),                    # γ
            rng.uniform(0.1, 0.85),                    # α
            rng.uniform(max(p0 + 0.05, 0.6), 0.99),    # q
        ])
        bounds = [BOUNDS_B, BOUNDS_GAMMA, BOUNDS_ALPHA,
                  (p0 + 1e-3, 1.0)]
        try:
            res = minimize(
                mse, theta0, args=(N_train, y_train, p0),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )
        except Exception:
            continue
        if not np.isfinite(res.fun):
            continue
        if best is None or res.fun < best[1]:
            best = (res.x, float(res.fun), bool(res.success))
    if best is None:
        return np.full(4, np.nan), float("nan"), False
    return best


def fit_cell(cell: Cell, *, bootstrap: int = 0,
             rng: np.random.Generator | None = None) -> FitResult:
    rng = rng or np.random.default_rng(RNG_SEED)
    if cell.train_idx.size == 0:
        apply_split(cell)
    if cell.p0 is None:
        cell.p0 = estimate_p0(cell)
    p0 = float(cell.p0)

    N_tr, y_tr = flatten(cell, cell.train_idx)
    N_va, y_va = flatten(cell, cell.val_idx)
    N_te, y_te = flatten(cell, cell.test_idx)

    theta, train_mse, converged = _fit_once(N_tr, y_tr, p0, rng)
    b, gamma, alpha, q = theta

    def _mse(N, y):
        if N.size == 0 or not np.isfinite(theta).all():
            return float("nan")
        return mse(theta, N, y, p0)

    # Bootstrap: resample train walks with replacement, refit.
    boot: dict[str, list[float]] | None = None
    if bootstrap > 0 and cell.train_idx.size > 1:
        boot = {"b": [], "gamma": [], "alpha": [], "q": [], "N_star": []}
        for _ in range(bootstrap):
            pick = rng.choice(cell.train_idx, size=cell.train_idx.size, replace=True)
            Nb, yb = flatten(cell, pick)
            if Nb.size < 4:
                continue
            theta_b, _, _ = _fit_once(Nb, yb, p0, rng)
            if not np.isfinite(theta_b).all():
                continue
            bb, gg, aa, qq = theta_b
            boot["b"].append(float(bb))
            boot["gamma"].append(float(gg))
            boot["alpha"].append(float(aa))
            boot["q"].append(float(qq))
            ns = n_star(bb, gg, aa)
            boot["N_star"].append(float(ns) if ns is not None else float("nan"))

    return FitResult(
        b=float(b), gamma=float(gamma), alpha=float(alpha), q=float(q),
        p0=p0, N_star=n_star(b, gamma, alpha),
        mse_train=train_mse, mse_val=_mse(N_va, y_va), mse_test=_mse(N_te, y_te),
        n_train=cell.train_idx.size, n_val=cell.val_idx.size, n_test=cell.test_idx.size,
        converged=converged, bootstrap=boot,
    )


# ─── Condition-level driver ──────────────────────────────────────────────────

def fit_condition(condition: str, *, bootstrap: int = 0,
                  rng: np.random.Generator | None = None,
                  ratios: tuple[float, float, float] = (0.5, 0.25, 0.25)
                  ) -> tuple[list[Cell], dict]:
    cells = to_cells(condition)
    for cell in cells:
        apply_split(cell, ratios=ratios)
        cell.p0 = estimate_p0(cell)

    rng = rng or np.random.default_rng(RNG_SEED)
    results: dict[str, dict[str, dict]] = {}
    for cell in cells:
        fit = fit_cell(cell, bootstrap=bootstrap, rng=rng)
        results.setdefault(f"{cell.rho:.2f}", {})[cell.graph] = asdict(fit)
    return cells, {"condition": condition, "per_cell": results}


def save_fits(condition: str, payload: dict) -> str:
    ensure_dir(FIT_DIR)
    path = os.path.join(FIT_DIR, f"{condition}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o is None or isinstance(o, float) and np.isnan(o):
        return None
    raise TypeError(f"not JSON serializable: {type(o)}")


# ─── Plotting ────────────────────────────────────────────────────────────────

GRID_COLOR = "#1976D2"
RING_COLOR = "#C62828"


def plot_condition(cells: list[Cell], fits: dict, condition: str) -> str:
    """One figure per condition: 3-panel strip (ρ=0 / ρ=0.5 / ρ=1) with
    observed accuracy (mean ± SEM), fit, and N* marker.  Additional ρ values
    (from the upcoming ladder) get added to the right.
    """
    rhos_present = sorted({cell.rho for cell in cells})
    n_panels = max(len(rhos_present), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 + 3.2 * n_panels, 4.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    N_grid = np.geomspace(10, 2500, 200)

    for ax, rho in zip(axes, rhos_present):
        for cell in cells:
            if cell.rho != rho:
                continue
            color = GRID_COLOR if cell.graph == "grid" else RING_COLOR
            # Observed mean ± SEM across *all* walks (train+val+test).
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(cell.acc, axis=0)
                sem = np.nanstd(cell.acc, axis=0) / np.sqrt(
                    np.sum(~np.isnan(cell.acc), axis=0).clip(min=1))
            ax.plot(cell.L, mean, "o", color=color,
                    label=f"{cell.graph} (n={cell.n_walks})", zorder=3)
            ax.fill_between(cell.L, mean - sem, mean + sem,
                            color=color, alpha=0.15, zorder=2)

            fit = fits["per_cell"][f"{cell.rho:.2f}"][cell.graph]
            if np.isfinite(fit["b"]):
                y_hat = predict(
                    N_grid, b=fit["b"], gamma=fit["gamma"],
                    alpha=fit["alpha"], q=fit["q"], p0=fit["p0"],
                )
                ax.plot(N_grid, y_hat, "-", color=color, lw=1.8, alpha=0.9)
                if fit["N_star"] is not None and fit["N_star"] < N_grid.max():
                    ax.axvline(fit["N_star"], color=color, ls="--",
                               lw=1.0, alpha=0.6)
                    ax.text(fit["N_star"], 0.02,
                            f"N*≈{fit['N_star']:.0f}", rotation=90,
                            color=color, fontsize=8, ha="right", va="bottom")

        ax.set_xscale("log")
        ax.set_xlim(40, 2500)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Context length N")
        ax.set_title(f"ρ={rho:.2f}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.85, loc="lower right")

    axes[0].set_ylabel("P(next ∈ valid neighbors)")
    fig.suptitle(f"Baseline Bigelow fit — {condition}", fontsize=12)
    fig.tight_layout()

    ensure_dir(FIG_DIR)
    path = os.path.join(FIG_DIR, f"baseline_{condition}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Number of bootstrap resamples of training walks (0 = off).")
    parser.add_argument("--plot", action="store_true",
                        help="Also write a diagnostic PNG per condition.")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument(
        "--split", nargs=3, type=float, default=(0.5, 0.25, 0.25),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fractions over walks in each cell (default 0.5/0.25/0.25).",
    )
    args = parser.parse_args()

    todo = CONDITIONS if args.condition == "all" else [args.condition]
    rng = np.random.default_rng(args.seed)

    summary = []
    for cond in todo:
        print(f"\n=== {cond} ===")
        cells, payload = fit_condition(
            cond, bootstrap=args.bootstrap, rng=rng, ratios=tuple(args.split),
        )
        out_path = save_fits(cond, payload)
        print(f"  saved fits  → {out_path}")
        if args.plot:
            png = plot_condition(cells, payload, cond)
            print(f"  saved plot  → {png}")

        # Console summary.
        for rho_str, by_graph in payload["per_cell"].items():
            for graph, fit in by_graph.items():
                ns = fit["N_star"]
                ns_str = f"{ns:.0f}" if ns is not None else "  —"
                print(f"  ρ={rho_str}  {graph:>4}  "
                      f"b={fit['b']:+.3f} γ={fit['gamma']:.3f} "
                      f"α={fit['alpha']:.3f} q={fit['q']:.3f}  "
                      f"N*={ns_str:>4}  "
                      f"MSE tr/va/te = {fit['mse_train']:.4f}/"
                      f"{fit['mse_val']:.4f}/{fit['mse_test']:.4f}")
                summary.append((cond, rho_str, graph, fit["N_star"]))

    # Consolidated CSV for M8 overlay.
    ensure_dir(FIT_DIR)
    n_star_path = os.path.join(FIT_DIR, "N_star_summary.csv")
    with open(n_star_path, "w") as f:
        f.write("condition,rho,graph,N_star\n")
        for cond, rho, graph, ns in summary:
            f.write(f"{cond},{rho},{graph},{'' if ns is None else f'{ns:.2f}'}\n")
    print(f"\nN* summary → {n_star_path}")


if __name__ == "__main__":
    main()
