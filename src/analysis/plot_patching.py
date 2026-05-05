"""Plot activation-patching JSONL outputs."""

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


def mean(values: list[float]) -> float:
    arr = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    return float(arr.mean()) if len(arr) else float("nan")


def plot_final_curve(rows: list[dict], out_path: Path) -> Path:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("position_strategy") == "final" and row.get("normalized_effect") is not None:
            grouped[int(row["layer"])].append(float(row["normalized_effect"]))
    layers = sorted(grouped)
    values = [mean(grouped[layer]) for layer in layers]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(layers, values, marker="o", lw=2)
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.axhline(1, color="black", lw=0.8, alpha=0.25, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized patching effect")
    ax.set_title("Final-token residual patching")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_layer_position_heatmap(rows: list[dict], out_path: Path, strategy: str = "all") -> Path:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("position_strategy") != strategy or row.get("normalized_effect") is None:
            continue
        pos = int(row["positions"][0]) if row.get("positions") else -1
        grouped[(int(row["layer"]), pos)].append(float(row["normalized_effect"]))
    if not grouped:
        return out_path
    layers = sorted({layer for layer, _ in grouped})
    positions = sorted({pos for _, pos in grouped})
    matrix = np.full((len(layers), len(positions)), np.nan)
    for i, layer in enumerate(layers):
        for j, pos in enumerate(positions):
            matrix[i, j] = mean(grouped.get((layer, pos), []))
    fig, ax = plt.subplots(figsize=(max(7, len(positions) * 0.12), 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    if len(positions) <= 40:
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels(positions, rotation=90, fontsize=7)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(f"Activation patching heatmap ({strategy})")
    fig.colorbar(im, ax=ax, label="Normalized effect")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_seen_heldout(rows: list[dict], out_path: Path) -> Path:
    grouped = {
        "seen": defaultdict(list),
        "heldout": defaultdict(list),
    }
    for row in rows:
        if row.get("position_strategy") != "final":
            continue
        layer = int(row["layer"])
        seen = row.get("patched_clean_seen_logit_diff")
        heldout = row.get("patched_clean_heldout_logit_diff")
        if seen is not None:
            grouped["seen"][layer].append(float(seen))
        if heldout is not None:
            grouped["heldout"][layer].append(float(heldout))
    layers = sorted(set(grouped["seen"]) | set(grouped["heldout"]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, store in grouped.items():
        ax.plot(layers, [mean(store.get(layer, [])) for layer in layers], marker="o", lw=2, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Patched clean-neighbor logit diff")
    ax.set_title("Seen versus held-out intervention effect")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to activation-patching rows.jsonl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--heatmap_strategy", default="all")
    args = parser.parse_args()
    rows = load_rows(args.input)
    out_dir = Path(args.out_dir)
    paths = [
        plot_final_curve(rows, out_dir / "final_token_patching_curve.png"),
        plot_layer_position_heatmap(rows, out_dir / f"patching_heatmap_{args.heatmap_strategy}.png", args.heatmap_strategy),
        plot_seen_heldout(rows, out_dir / "seen_vs_heldout_patching.png"),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

