"""
vocabulary_experiment.py

Two conditions testing how vocabulary choice affects competing-structure learning:

  Condition A — DISJOINT
    Grid (16 words) and Ring (12 neutral words: candle, brick, fern, ...) share
    no vocabulary.  Token identity alone disambiguates source graph.

  Condition B — OVERLAP
    Same topologies, but 3 ring words (rock, sand, box) also appear in the grid.
    The model must use structural context — not token identity — to determine
    which graph governs the current position.

For each condition we sweep rho in {0.0, 0.5, 1.0} and plot accuracy vs context
length.  The key comparisons are:
  - Does the neutral ring show a cleaner phase transition than months?
  - Does vocab overlap degrade accuracy (model gets confused at shared tokens)?
  - Does mixing suppress ring learning more when vocab is disjoint vs overlapping?

Usage:
    # run both conditions (default)
    python initial_experiments/vocabulary_experiment.py

    # run one condition only
    python initial_experiments/vocabulary_experiment.py --condition disjoint
    python initial_experiments/vocabulary_experiment.py --condition overlap

    # replot from saved results without re-running inference
    python initial_experiments/vocabulary_experiment.py --replot
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
from graphs import (
    Ring,
    RING_WORDS, RING_WORD_TO_COLOR,
    RING_WORDS_OVERLAP, OVERLAP_WORD_TO_COLOR,
    SHARED_WORDS,
)
from sanity_check import Grid, WORDS, WORD_TO_COLOR, set_seed, make_interleaved_sequence

# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.1:8b-text-q4_0"

RHOS = [0.0, 0.5, 1.0]
N_SEQUENCES = 8
SEQ_LEN = 1400
SEGMENT_LEN = 100
EVAL_LENGTHS = [50, 100, 200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1400]

DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "vocabulary_experiment")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(DATA_DIR, exist_ok=True)

CONDITIONS = {
    "disjoint": {
        "label": "Disjoint vocab",
        "ring_words": RING_WORDS,
        "description": "Ring uses neutral words with no grid overlap",
    },
    "overlap": {
        "label": "Overlapping vocab (rock, sand, box shared)",
        "ring_words": RING_WORDS_OVERLAP,
        "description": "3 ring words (rock, sand, box) also appear in grid",
    },
}

# ── Ollama ─────────────────────────────────────────────────────────────────────

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in models:
            print(f"ERROR: '{MODEL}' not found. Pull with: ollama pull {MODEL}")
            return False
        print(f"Ollama ready. Model: {MODEL}")
        return True
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        return False


def query_model(prefix_text, valid_neighbors, retries=2):
    """Greedy next-token prediction. Returns 1.0 if valid neighbor, else 0.0."""
    payload = {
        "model": MODEL,
        "prompt": prefix_text,
        "stream": False,
        "raw": True,
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
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"  [err: {e}]", end="", flush=True)
    return None


# ── Core experiment ────────────────────────────────────────────────────────────

def run_condition(condition_name, rho, n_sequences, eval_lengths, seed_offset=0):
    """
    Run one (condition, rho) cell.

    Returns:
        grid_accs : {length: [0.0|1.0, ...]}
        ring_accs : {length: [0.0|1.0, ...]}
        shared_accs : {length: [0.0|1.0, ...]}  — positions at shared-vocab words
                      (only populated for the overlap condition)
    """
    ring_words = CONDITIONS[condition_name]["ring_words"]
    grid = Grid()
    ring = Ring(words=ring_words)
    is_overlap = (condition_name == "overlap")

    grid_accs   = {L: [] for L in eval_lengths}
    ring_accs   = {L: [] for L in eval_lengths}
    shared_accs = {L: [] for L in eval_lengths}

    for seq_i in range(n_sequences):
        set_seed(42 + seq_i + seed_offset * 100)
        seq, labels = make_interleaved_sequence(grid, ring, SEQ_LEN, rho, SEGMENT_LEN)

        for L in tqdm(
            sorted(eval_lengths),
            desc=f"  {condition_name} ρ={rho} seq {seq_i+1}/{n_sequences}",
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
            if acc is None:
                continue

            is_shared = is_overlap and current_word in SHARED_WORDS

            if current_label == "grid":
                grid_accs[L].append(acc)
            else:
                ring_accs[L].append(acc)

            if is_shared:
                shared_accs[L].append(acc)

    return grid_accs, ring_accs, shared_accs


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_condition(results, condition_name):
    path = os.path.join(DATA_DIR, f"{condition_name}.json")
    serializable = {
        str(rho): {
            graph: {str(L): vals for L, vals in accs.items()}
            for graph, accs in data.items()
        }
        for rho, data in results.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_condition(condition_name):
    path = os.path.join(DATA_DIR, f"{condition_name}.json")
    with open(path) as f:
        raw = json.load(f)
    return {
        float(rho_str): {
            graph: {int(L): vals for L, vals in accs.items()}
            for graph, accs in data.items()
        }
        for rho_str, data in raw.items()
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

GRID_COLOR  = "#1976D2"
RING_COLOR  = "#C62828"
SHARE_COLOR = "#FF8F00"
RHO_LABEL   = {0.0: "ρ=0 (pure)", 0.5: "ρ=0.5 (mixed)", 1.0: "ρ=1 (pure)"}
RHO_COLORS  = {0.0: "#1565C0", 0.5: "#6A1B9A", 1.0: "#B71C1C"}


def _curve(ax, accs, color, label, eval_lengths, ls="-"):
    lengths = sorted(L for L in eval_lengths if accs.get(L))
    if not lengths:
        return
    means = [np.mean(accs[L]) for L in lengths]
    sems  = [np.std(accs[L]) / max(np.sqrt(len(accs[L])), 1) for L in lengths]
    ax.plot(lengths, means, f"o{ls}", color=color, label=label, lw=2, ms=5, zorder=3)
    ax.fill_between(
        lengths,
        [m - s for m, s in zip(means, sems)],
        [m + s for m, s in zip(means, sems)],
        alpha=0.15, color=color, zorder=2,
    )


def plot_condition(results, condition_name):
    """3-panel plot (one per rho) for a single condition."""
    cond_label = CONDITIONS[condition_name]["label"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, rho in zip(axes, [0.0, 0.5, 1.0]):
        data = results[rho]
        _curve(ax, data["grid"],   GRID_COLOR,  "Grid neighbors",   EVAL_LENGTHS)
        _curve(ax, data["ring"],   RING_COLOR,  "Ring neighbors",   EVAL_LENGTHS)
        if condition_name == "overlap" and any(data["shared"].get(L) for L in EVAL_LENGTHS):
            _curve(ax, data["shared"], SHARE_COLOR, "Shared-word positions", EVAL_LENGTHS, ls="--")

        ax.axhline(3 / 16, color=GRID_COLOR, lw=0.8, ls=":", alpha=0.5,
                   label="Grid chance (3/16)")
        ax.axhline(2 / 12, color=RING_COLOR, lw=0.8, ls=":", alpha=0.5,
                   label="Ring chance (2/12)")
        ax.set_xlabel("Context length (tokens)", fontsize=10)
        ax.set_title(RHO_LABEL[rho], fontsize=11)
        ax.set_xlim(0, SEQ_LEN + 50)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("P(next token ∈ valid neighbors)  [greedy]", fontsize=10)
    fig.suptitle(
        f"Llama 3.1 8B (base) — {cond_label}\n"
        f"({N_SEQUENCES} sequences per ρ, segment_len={SEGMENT_LEN})",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"vocab_{condition_name}_per_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison(all_results):
    """
    Head-to-head: disjoint vs overlap for the mixed condition (rho=0.5).
    One panel for grid accuracy, one for ring accuracy.
    Also includes a panel comparing pure-ring phase transitions across conditions.
    """
    if not all(c in all_results for c in ["disjoint", "overlap"]):
        print("  Skipping comparison plot (need both conditions)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"disjoint": "#1565C0", "overlap": "#E65100"}

    # Panel 1: pure ring (rho=1) — does neutral vocab fix the phase transition?
    ax = axes[0]
    for cond in ["disjoint", "overlap"]:
        accs = all_results[cond][1.0]["ring"]
        _curve(ax, accs, colors[cond], CONDITIONS[cond]["label"], EVAL_LENGTHS)
    ax.axhline(2 / 12, color="gray", lw=0.8, ls="--", alpha=0.5, label="Chance (2/12)")
    ax.set_title("Pure ring (ρ=1): disjoint vs overlap", fontsize=11)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Ring neighbor accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.05)

    # Panel 2: mixed (rho=0.5), grid accuracy — does overlap hurt grid learning?
    ax = axes[1]
    for cond in ["disjoint", "overlap"]:
        accs = all_results[cond][0.5]["grid"]
        _curve(ax, accs, colors[cond], CONDITIONS[cond]["label"], EVAL_LENGTHS)
    ax.axhline(3 / 16, color="gray", lw=0.8, ls="--", alpha=0.5, label="Chance (3/16)")
    ax.set_title("Mixed (ρ=0.5): grid accuracy", fontsize=11)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Grid neighbor accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: mixed (rho=0.5), ring accuracy — the key competition signal
    ax = axes[2]
    for cond in ["disjoint", "overlap"]:
        accs = all_results[cond][0.5]["ring"]
        _curve(ax, accs, colors[cond], CONDITIONS[cond]["label"], EVAL_LENGTHS)
    if all_results["overlap"][0.5].get("shared"):
        _curve(ax, all_results["overlap"][0.5]["shared"],
               SHARE_COLOR, "Overlap: shared-word positions", EVAL_LENGTHS, ls="--")
    ax.axhline(2 / 12, color="gray", lw=0.8, ls="--", alpha=0.5, label="Chance (2/12)")
    ax.set_title("Mixed (ρ=0.5): ring accuracy\n(key competition signal)", fontsize=11)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Ring neighbor accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Vocabulary experiment — disjoint vs overlapping ring words\n"
        "Llama 3.1 8B (base) via Ollama",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "vocab_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=["disjoint", "overlap", "both"], default="both",
        help="Which vocabulary condition to run (default: both)",
    )
    parser.add_argument(
        "--replot", action="store_true",
        help="Skip inference, regenerate plots from saved results",
    )
    args = parser.parse_args()

    conditions_to_run = (
        ["disjoint", "overlap"] if args.condition == "both"
        else [args.condition]
    )

    all_results = {}

    if args.replot:
        for cond in conditions_to_run:
            path = os.path.join(DATA_DIR, f"{cond}.json")
            if not os.path.exists(path):
                print(f"No saved results for '{cond}' at {path}. Run without --replot first.")
                sys.exit(1)
            print(f"Loading {cond} results from {path}")
            all_results[cond] = load_condition(cond)
    else:
        if not check_ollama():
            sys.exit(1)

        for cond in conditions_to_run:
            print(f"\n{'='*60}")
            print(f"Condition: {cond.upper()} — {CONDITIONS[cond]['description']}")
            print(f"Ring words: {CONDITIONS[cond]['ring_words']}")
            print(f"{'='*60}")

            cond_results = {}
            for i, rho in enumerate(RHOS):
                print(f"\n  [{i+1}/{len(RHOS)}] ρ={rho}")
                grid_accs, ring_accs, shared_accs = run_condition(
                    cond, rho, N_SEQUENCES, EVAL_LENGTHS, seed_offset=i
                )
                cond_results[rho] = {
                    "grid": grid_accs,
                    "ring": ring_accs,
                    "shared": shared_accs,
                }
                # checkpoint after each rho
                all_results[cond] = cond_results
                path = save_condition(cond_results, cond)
                print(f"  Checkpoint saved → {path}")

    print("\nGenerating plots...")
    for cond in conditions_to_run:
        plot_condition(all_results[cond], cond)
    plot_comparison(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
