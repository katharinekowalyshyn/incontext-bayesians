"""Plot graph-steering JSONL outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def finite_mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def plot_layer_alpha_heatmap(rows: list[dict], out_path: Path, control: str = "real") -> Path:
    grouped: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("control") != control or row.get("eval_direction") != "target_plus_source_minus_target":
            continue
        grouped[(int(row["layer"]), float(row["alpha"]))].append(float(row["steered_metric"]))
    layers = sorted({layer for layer, _ in grouped})
    alphas = sorted({alpha for _, alpha in grouped})
    matrix = np.full((len(layers), len(alphas)), np.nan)
    for i, layer in enumerate(layers):
        for j, alpha in enumerate(alphas):
            matrix[i, j] = finite_mean(grouped.get((layer, alpha), []))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Layer")
    ax.set_title(f"Steering layer-alpha heatmap ({control})")
    fig.colorbar(im, ax=ax, label="Graph logit diff")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_control_comparison(rows: list[dict], out_path: Path) -> Path:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("eval_direction") != "target_plus_source_minus_target":
            continue
        value = row.get("normalized_effect")
        if value is None:
            continue
        grouped[row["control"]][float(row["alpha"])].append(float(value))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for control, by_alpha in sorted(grouped.items()):
        alphas = sorted(by_alpha)
        ax.plot(alphas, [finite_mean(by_alpha[a]) for a in alphas], marker="o", lw=2, label=control)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.axhline(1, color="black", lw=0.8, alpha=0.25, linestyle="--")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Normalized steering effect")
    ax.set_title("Real steering versus controls")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_seen_heldout(rows: list[dict], out_path: Path, control: str = "real") -> Path:
    grouped = {"seen": defaultdict(list), "heldout": defaultdict(list)}
    for row in rows:
        if row.get("control") != control or row.get("eval_direction") != "target_plus_source_minus_target":
            continue
        if float(row.get("alpha", 0.0)) <= 0:
            continue
        layer = int(row["layer"])
        seen = row.get("steered_clean_seen_logit_diff")
        heldout = row.get("steered_clean_heldout_logit_diff")
        if seen is not None:
            grouped["seen"][layer].append(float(seen))
        if heldout is not None:
            grouped["heldout"][layer].append(float(heldout))
    layers = sorted(set(grouped["seen"]) | set(grouped["heldout"]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, store in grouped.items():
        ax.plot(layers, [finite_mean(store.get(layer, [])) for layer in layers], marker="o", lw=2, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Positive-alpha clean-neighbor logit diff")
    ax.set_title("Seen versus held-out steering effect")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to steering rows.jsonl")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    rows = load_rows(args.input)
    out_dir = Path(args.out_dir)
    paths = [
        plot_layer_alpha_heatmap(rows, out_dir / "steering_layer_alpha_heatmap.png"),
        plot_control_comparison(rows, out_dir / "steering_control_comparison.png"),
        plot_seen_heldout(rows, out_dir / "seen_vs_heldout_steering.png"),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

