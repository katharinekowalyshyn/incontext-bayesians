"""
mixing_experiment.py

Measures Llama 3.1 8B (base) accuracy on competing graph structures via Ollama.

For rho in {0.0, 0.5, 1.0}:
  - rho=0.0: pure 4x4 grid walks
  - rho=0.5: interleaved grid + ring segments (across-sequence mixing)
  - rho=1.0: pure 12-node ring (months of year) walks

At each sampled context length, we pass the prefix to the model and check
whether its greedy next-token prediction is a valid graph neighbor.

Usage:
    python initial_experiments/mixing_experiment.py
    python initial_experiments/mixing_experiment.py --replot   # replot saved results
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphs import Ring, MONTHS
from sanity_check import Grid, WORDS, set_seed, make_interleaved_sequence

# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.1:8b-text-q4_0"

RHOS = [0.0, 0.5, 1.0]
N_SEQUENCES = 8          # sequences per rho value
SEQ_LEN = 1400           # total tokens per sequence (matches Park et al.)
SEGMENT_LEN = 100        # tokens per segment in mixed sequences
EVAL_LENGTHS = [50, 100, 200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1400]

DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "mixing_experiment")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Ollama interface ────────────────────────────────────────────────────────────

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Ollama running. Available models: {models}")
        if MODEL not in models:
            print(f"\nERROR: '{MODEL}' not found. Pull it with:\n  ollama pull {MODEL}\n")
            return False
        return True
    except Exception as e:
        print(f"Ollama not reachable at {OLLAMA_URL}: {e}")
        return False


def query_model(prefix_text, valid_neighbors, retries=2):
    """
    Feed prefix_text to the base LM, get one greedy token, check if valid.

    Returns:
        1.0  if generated token is a valid graph neighbor
        0.0  if not
        None if the query failed
    """
    payload = {
        "model": MODEL,
        "prompt": prefix_text,
        "stream": False,
        "raw": True,          # no chat template — pure next-token prediction
        "options": {
            "temperature": 0,
            "num_predict": 1,
            "num_ctx": 2048,
            "seed": 0,
        },
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            generated = r.json()["response"].strip().lower()
            return float(generated in valid_neighbors)
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2)
            else:
                return None
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"  [query error: {e}]", end="", flush=True)
                return None


# ── Experiment ─────────────────────────────────────────────────────────────────

def run_rho(grid, ring, rho, n_sequences, eval_lengths, seed_offset=0):
    """
    Run all sequences for one mixture ratio.

    Returns:
        grid_accs : {length: [0.0 or 1.0, ...]}  — positions labeled 'grid'
        ring_accs : {length: [0.0 or 1.0, ...]}  — positions labeled 'ring'
    """
    grid_accs = {L: [] for L in eval_lengths}
    ring_accs = {L: [] for L in eval_lengths}

    for seq_i in range(n_sequences):
        set_seed(42 + seq_i + seed_offset * 100)
        seq, labels = make_interleaved_sequence(grid, ring, SEQ_LEN, rho, SEGMENT_LEN)

        for L in tqdm(
            sorted(eval_lengths),
            desc=f"  ρ={rho} seq {seq_i+1}/{n_sequences}",
            leave=False,
        ):
            current_word = seq[L - 1]
            current_label = labels[L - 1]

            if current_label == "grid":
                valid = grid.get_valid_next_words(current_word)
            else:
                valid = ring.get_valid_next_words(current_word)

            prefix_text = " " + " ".join(seq[:L])
            acc = query_model(prefix_text, valid)

            if acc is not None:
                if current_label == "grid":
                    grid_accs[L].append(acc)
                else:
                    ring_accs[L].append(acc)

    return grid_accs, ring_accs


# ── Plots ───────────────────────────────────────────────────────────────────────

RHO_LABEL = {0.0: "ρ=0  (pure grid)", 0.5: "ρ=0.5  (mixed)", 1.0: "ρ=1  (pure ring)"}
GRID_COLOR = "#1976D2"
RING_COLOR = "#C62828"
RHO_COLORS = {0.0: "#1565C0", 0.5: "#6A1B9A", 1.0: "#B71C1C"}


def _plot_curve(ax, accs, color, label, eval_lengths):
    lengths = sorted(L for L in eval_lengths if accs.get(L))
    if not lengths:
        return
    means = [np.mean(accs[L]) for L in lengths]
    sems = [np.std(accs[L]) / max(np.sqrt(len(accs[L])), 1) for L in lengths]
    ax.plot(lengths, means, "o-", color=color, label=label, lw=2, ms=5, zorder=3)
    ax.fill_between(
        lengths,
        [m - s for m, s in zip(means, sems)],
        [m + s for m, s in zip(means, sems)],
        alpha=0.15, color=color, zorder=2,
    )


def plot_per_rho(all_results):
    """Three-panel plot: one subplot per rho, grid vs ring accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, rho in zip(axes, [0.0, 0.5, 1.0]):
        data = all_results[rho]
        _plot_curve(ax, data["grid"], GRID_COLOR, "Grid neighbors", EVAL_LENGTHS)
        _plot_curve(ax, data["ring"], RING_COLOR, "Ring neighbors", EVAL_LENGTHS)

        # Chance baselines: avg_degree / vocab (greedy, LM picks from full vocab,
        # but these reference lines show uniform-over-vocab-words baselines)
        ax.axhline(3 / 16, color=GRID_COLOR, lw=0.9, ls="--", alpha=0.45,
                   label="Grid chance (3/16)")
        ax.axhline(2 / 12, color=RING_COLOR, lw=0.9, ls="--", alpha=0.45,
                   label="Ring chance (2/12)")

        ax.set_xlabel("Context length (tokens)", fontsize=10)
        ax.set_title(RHO_LABEL[rho], fontsize=11)
        ax.set_xlim(0, SEQ_LEN + 50)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    axes[0].set_ylabel("P(next token ∈ valid neighbors)  [greedy]", fontsize=10)
    fig.suptitle(
        "Llama 3.1 8B (base) — accuracy on competing graph structures\n"
        f"({N_SEQUENCES} sequences per ρ, segment_len={SEGMENT_LEN})",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "mixing_per_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary(all_results):
    """Two-panel summary: grid accuracy and ring accuracy, all rhos overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, graph_type, color_base in zip(
        axes, ["grid", "ring"], [GRID_COLOR, RING_COLOR]
    ):
        for rho in [0.0, 0.5, 1.0]:
            accs = all_results[rho][graph_type]
            lengths = sorted(L for L in EVAL_LENGTHS if accs.get(L))
            if not lengths:
                continue
            means = [np.mean(accs[L]) for L in lengths]
            ax.plot(
                lengths, means, "o-",
                color=RHO_COLORS[rho],
                label=RHO_LABEL[rho],
                lw=2, ms=5,
            )

        chance = 3 / 16 if graph_type == "grid" else 2 / 12
        ax.axhline(chance, color="gray", lw=0.9, ls="--", alpha=0.5,
                   label=f"Chance ({chance:.2f})")
        ax.set_xlabel("Context length (tokens)", fontsize=10)
        ax.set_ylabel("Greedy accuracy", fontsize=10)
        ax.set_title(f"{graph_type.capitalize()} neighbor accuracy — effect of ρ", fontsize=11)
        ax.set_xlim(0, SEQ_LEN + 50)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Llama 3.1 8B (base) — does mixing suppress structure learning?",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "mixing_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_results(all_results, path):
    serializable = {
        str(rho): {
            graph: {str(L): vals for L, vals in accs.items()}
            for graph, accs in data.items()
        }
        for rho, data in all_results.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_results(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        float(rho_str): {
            graph: {int(L): vals for L, vals in accs.items()}
            for graph, accs in data.items()
        }
        for rho_str, data in raw.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip inference, replot from saved results")
    args = parser.parse_args()

    cache_path = os.path.join(DATA_DIR, "raw_results.json")

    grid = Grid()
    ring = Ring()

    if args.replot:
        if not os.path.exists(cache_path):
            print(f"No cached results found at {cache_path}. Run without --replot first.")
            sys.exit(1)
        print(f"Loading results from {cache_path}")
        all_results = load_results(cache_path)
    else:
        if not check_ollama():
            sys.exit(1)

        set_seed(42)

        all_results = {}
        for i, rho in enumerate(RHOS):
            print(f"\n[{i+1}/{len(RHOS)}] Running ρ={rho}  "
                  f"({N_SEQUENCES} sequences × {len(EVAL_LENGTHS)} eval points)")
            grid_accs, ring_accs = run_rho(grid, ring, rho, N_SEQUENCES, EVAL_LENGTHS,
                                           seed_offset=i)
            all_results[rho] = {"grid": grid_accs, "ring": ring_accs}

            # Save after each rho so we don't lose work if interrupted
            save_results(all_results, cache_path)
            print(f"  Checkpoint saved → {cache_path}")

    print("\nGenerating plots...")
    plot_per_rho(all_results)
    plot_summary(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
