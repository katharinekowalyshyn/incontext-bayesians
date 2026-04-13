"""
vocabulary_tl_experiment.py

TransformerLens version of the vocabulary experiment.
Uses a single forward pass per sequence to get real token probabilities
at every position — much faster and more precise than the Ollama greedy approach.

Two conditions:
  disjoint  — Ring uses fully neutral vocabulary (candle, brick, fern, ...)
  overlap   — 3 ring words (rock, sand, box) shared with grid vocabulary

For each condition, sweeps rho in {0.0, 0.5, 1.0} and measures
P(next token ∈ valid neighbors) at sampled context lengths up to 1400.

Usage (with conda env active):
    python src/initial_experiments/vocabulary_tl_experiment.py
    python src/initial_experiments/vocabulary_tl_experiment.py --condition disjoint
    python src/initial_experiments/vocabulary_tl_experiment.py --replot
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
ICLR_DIR  = os.path.join(REPO_ROOT, "iclr_induction-main")

sys.path.insert(0, ICLR_DIR)   # for utils.py (load_model, tokenize_sequence)
sys.path.insert(0, HERE)        # for graphs.py, sanity_check.py

from utils import load_model, MODEL_NAME
from graphs import (
    Ring,
    RING_WORDS, RING_WORD_TO_COLOR,
    RING_WORDS_OVERLAP, OVERLAP_WORD_TO_COLOR,
    SHARED_WORDS,
)
from sanity_check import Grid, WORDS as GRID_WORDS, WORD_TO_COLOR, set_seed, make_interleaved_sequence

# ── Config ─────────────────────────────────────────────────────────────────────

RHOS         = [0.0, 0.5, 1.0]
N_SEQUENCES  = 16       # more sequences → smoother curves (TL is fast, one pass each)
SEQ_LEN      = 1400
SEGMENT_LEN  = 100
EVAL_LENGTHS = [50, 100, 200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1400]

DATA_DIR = os.path.join(HERE, "results", "vocabulary_tl")
PLOT_DIR = os.path.join(HERE, "results")
os.makedirs(DATA_DIR, exist_ok=True)

CONDITIONS = {
    "disjoint": {
        "label": "Disjoint vocab",
        "ring_words": RING_WORDS,
        "description": "Ring: candle, brick, fern, lamp, dust, wool, reef, thorn, cask, flint, marsh, prism",
    },
    "overlap": {
        "label": "Overlapping vocab (rock, sand, box shared)",
        "ring_words": RING_WORDS_OVERLAP,
        "description": "3 ring words (rock, sand, box) also appear in grid",
    },
}

# ── Model helpers ──────────────────────────────────────────────────────────────

def tokenize_sequence(model, sequence):
    """Tokenize a space-separated word sequence with a leading space."""
    text = " " + " ".join(sequence)
    return model.tokenizer(text, return_tensors="pt").input_ids.to(model.cfg.device)


def get_word_token_id(model, word):
    """Return the single token ID for ' word' (with leading space).
    Raises if the word tokenizes to more than one token.
    """
    ids = model.tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"'{word}' tokenizes to {len(ids)} tokens: {ids}")
    return ids[0]


def build_vocab_token_map(model, words):
    """Return {word: token_id} for a list of words. Verifies single-token assumption."""
    mapping = {}
    multi_token = []
    for w in words:
        ids = model.tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            mapping[w] = ids[0]
        else:
            multi_token.append((w, ids))
    if multi_token:
        print(f"  WARNING: multi-token words (probabilities will be approximate): {multi_token}")
        for w, ids in multi_token:
            mapping[w] = ids[0]   # use first token as approximation
    return mapping


# ── Per-sequence accuracy ──────────────────────────────────────────────────────

@torch.no_grad()
def sequence_neighbor_probs(model, grid, ring, sequence, labels, eval_lengths,
                            grid_tok, ring_tok, is_overlap):
    """
    Single forward pass over a full sequence.
    Returns:
        grid_probs   : {L: float}  — P(next ∈ grid neighbors) at grid-labeled positions
        ring_probs   : {L: float}  — P(next ∈ ring neighbors) at ring-labeled positions
        shared_probs : {L: float}  — same but only at shared-vocab positions (overlap only)
    """
    tokens = tokenize_sequence(model, sequence)           # [1, seq_len+1]
    logits = model(tokens)                                 # [1, seq_len+1, vocab]
    probs  = torch.softmax(logits[0, 1:, :], dim=-1)      # [seq_len, vocab]  — remove BOS

    grid_probs, ring_probs, shared_probs = {}, {}, {}

    for L in eval_lengths:
        current_word  = sequence[L - 1]
        current_label = labels[L - 1]

        if current_label == "grid":
            valid  = grid.get_valid_next_words(current_word)
            tok_map = grid_tok
        else:
            valid  = ring.get_valid_next_words(current_word)
            tok_map = ring_tok

        # Sum probabilities of valid neighbor tokens
        p = sum(probs[L - 1, tok_map[nb]].item() for nb in valid if nb in tok_map)

        if current_label == "grid":
            grid_probs[L] = p
        else:
            ring_probs[L] = p

        if is_overlap and current_word in SHARED_WORDS:
            shared_probs[L] = p

    return grid_probs, ring_probs, shared_probs


# ── Run one (condition, rho) cell ──────────────────────────────────────────────

def run_condition_rho(model, condition_name, rho, n_sequences, eval_lengths,
                      seed_offset=0):
    ring_words  = CONDITIONS[condition_name]["ring_words"]
    grid        = Grid()
    ring        = Ring(words=ring_words)
    is_overlap  = (condition_name == "overlap")
    all_vocab   = list(set(GRID_WORDS) | set(ring_words))

    print(f"    Building token map for {len(all_vocab)} vocabulary words...")
    grid_tok = build_vocab_token_map(model, GRID_WORDS)
    ring_tok = build_vocab_token_map(model, ring_words)

    grid_accs   = {L: [] for L in eval_lengths}
    ring_accs   = {L: [] for L in eval_lengths}
    shared_accs = {L: [] for L in eval_lengths}

    for seq_i in tqdm(range(n_sequences),
                      desc=f"    {condition_name} ρ={rho}", leave=False):
        set_seed(42 + seq_i + seed_offset * 100)
        seq, labels = make_interleaved_sequence(grid, ring, SEQ_LEN, rho, SEGMENT_LEN)

        gp, rp, sp = sequence_neighbor_probs(
            model, grid, ring, seq, labels, eval_lengths,
            grid_tok, ring_tok, is_overlap,
        )
        for L, p in gp.items():
            grid_accs[L].append(p)
        for L, p in rp.items():
            ring_accs[L].append(p)
        for L, p in sp.items():
            shared_accs[L].append(p)

    return grid_accs, ring_accs, shared_accs


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_condition(results, condition_name):
    path = os.path.join(DATA_DIR, f"{condition_name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                str(rho): {
                    graph: {str(L): vals for L, vals in accs.items()}
                    for graph, accs in data.items()
                }
                for rho, data in results.items()
            },
            f, indent=2,
        )
    return path


def load_condition(condition_name):
    with open(os.path.join(DATA_DIR, f"{condition_name}.json")) as f:
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


def _curve(ax, accs, color, label, ls="-"):
    lengths = sorted(L for L in EVAL_LENGTHS if accs.get(L))
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


def _style_ax(ax, title):
    ax.set_xlabel("Context length (tokens)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, SEQ_LEN + 50)
    ax.set_ylim(-0.02, 0.55)   # probabilities are much lower than greedy accuracy
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)


def plot_condition(results, condition_name):
    """3-panel plot (one per rho) for a single condition."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, rho in zip(axes, [0.0, 0.5, 1.0]):
        data = results[rho]
        _curve(ax, data["grid"],   GRID_COLOR,  "Grid neighbors")
        _curve(ax, data["ring"],   RING_COLOR,  "Ring neighbors")
        if condition_name == "overlap":
            _curve(ax, data["shared"], SHARE_COLOR, "Shared-word positions", ls="--")

        # Chance: avg_degree / full_vocab_size (28 vocab words in model's distribution)
        ax.axhline(3 / len(model_vocab_size_proxy()),
                   color=GRID_COLOR, lw=0.8, ls=":", alpha=0.5, label="Grid chance")
        ax.axhline(2 / len(model_vocab_size_proxy()),
                   color=RING_COLOR,  lw=0.8, ls=":", alpha=0.5, label="Ring chance")
        _style_ax(ax, RHO_LABEL[rho])

    axes[0].set_ylabel("P(next token ∈ valid neighbors)", fontsize=10)
    fig.suptitle(
        f"Llama 3.1 8B (base, TransformerLens) — {CONDITIONS[condition_name]['label']}\n"
        f"({N_SEQUENCES} sequences per ρ)",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"tl_{condition_name}_per_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def model_vocab_size_proxy():
    """Placeholder — returns total vocabulary for chance-level computation."""
    # Llama-3.1-8B has 128256 tokens; we use this to set chance baseline
    return list(range(128256))


def plot_comparison(all_results):
    """Head-to-head comparison across conditions."""
    if not all(c in all_results for c in ["disjoint", "overlap"]):
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"disjoint": "#1565C0", "overlap": "#E65100"}

    # Panel 1: pure ring — does neutral vocab fix the phase transition?
    ax = axes[0]
    for cond in ["disjoint", "overlap"]:
        _curve_plain(ax, all_results[cond][1.0]["ring"],
                     colors[cond], CONDITIONS[cond]["label"])
    ax.set_title("Pure ring (ρ=1): disjoint vs overlap", fontsize=11)
    ax.set_ylabel("P(next ∈ ring neighbors)")

    # Panel 2: mixed, grid accuracy
    ax = axes[1]
    for cond in ["disjoint", "overlap"]:
        _curve_plain(ax, all_results[cond][0.5]["grid"],
                     colors[cond], CONDITIONS[cond]["label"])
    ax.set_title("Mixed (ρ=0.5): grid accuracy", fontsize=11)
    ax.set_ylabel("P(next ∈ grid neighbors)")

    # Panel 3: mixed, ring accuracy — the key competition signal
    ax = axes[2]
    for cond in ["disjoint", "overlap"]:
        _curve_plain(ax, all_results[cond][0.5]["ring"],
                     colors[cond], CONDITIONS[cond]["label"])
    if any(all_results["overlap"][0.5]["shared"].get(L) for L in EVAL_LENGTHS):
        _curve_plain(ax, all_results["overlap"][0.5]["shared"],
                     SHARE_COLOR, "Overlap: shared-word positions", ls="--")
    ax.set_title("Mixed (ρ=0.5): ring accuracy\n(competition signal)", fontsize=11)
    ax.set_ylabel("P(next ∈ ring neighbors)")

    for ax in axes:
        ax.set_xlabel("Context length (tokens)", fontsize=10)
        ax.set_xlim(0, SEQ_LEN + 50)
        ax.set_ylim(-0.02, 0.55)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Vocabulary experiment — disjoint vs overlapping ring words\n"
        "Llama 3.1 8B (base), TransformerLens",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "tl_vocab_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def _curve_plain(ax, accs, color, label, ls="-"):
    lengths = sorted(L for L in EVAL_LENGTHS if accs.get(L))
    if not lengths:
        return
    means = [np.mean(accs[L]) for L in lengths]
    ax.plot(lengths, means, f"o{ls}", color=color, label=label, lw=2, ms=5)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=["disjoint", "overlap", "both"], default="both",
    )
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    conditions_to_run = (
        ["disjoint", "overlap"] if args.condition == "both" else [args.condition]
    )

    all_results = {}

    if args.replot:
        for cond in conditions_to_run:
            print(f"Loading {cond}...")
            all_results[cond] = load_condition(cond)
    else:
        print(f"Loading {MODEL_NAME} via TransformerLens...")
        model = load_model()
        model.eval()
        print(f"Model loaded on {model.cfg.device}.")

        for cond in conditions_to_run:
            print(f"\n{'='*60}")
            print(f"Condition: {cond.upper()}")
            print(f"  {CONDITIONS[cond]['description']}")
            print(f"{'='*60}")

            cond_results = {}
            for i, rho in enumerate(RHOS):
                print(f"\n  [{i+1}/{len(RHOS)}] ρ={rho}  "
                      f"({N_SEQUENCES} sequences × {len(EVAL_LENGTHS)} eval points)")
                grid_accs, ring_accs, shared_accs = run_condition_rho(
                    model, cond, rho, N_SEQUENCES, EVAL_LENGTHS, seed_offset=i,
                )
                cond_results[rho] = {
                    "grid": grid_accs,
                    "ring": ring_accs,
                    "shared": shared_accs,
                }
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
