"""
vocabulary_tl_experiment.py

TransformerLens version of the vocabulary/ring experiment.
Uses a single forward pass per sequence — fast and precise.

Four conditions:

  months_natural  — Ring uses months in natural sequential order (Jan→Feb→...→Dec).
                    The model's semantic prior matches the in-context ring structure.
                    Expect fast, clean learning.

  months_permuted — Ring uses a permuted month order where no naturally-adjacent
                    months are in-context neighbors (Jan→Aug→Mar→Oct→May→Dec→Jul→
                    Feb→Sep→Apr→Nov→Jun). The semantic prior directly conflicts with
                    the in-context structure. Replicates the days-of-week experiment
                    from Park et al. (ICLR 2025) with months.

  neutral_disjoint — Ring uses semantically neutral words (candle, brick, fern, ...)
                     fully disjoint from the grid vocabulary. No semantic prior;
                     serves as a clean control for in-context learning speed.

  neutral_overlap  — 3 ring words (rock, sand, box) are shared with the grid
                     vocabulary. Tests ambiguity at shared tokens.

For each condition, sweeps rho in {0.0, 0.5, 1.0} and measures
P(next token ∈ valid neighbors) — summed softmax probability over ground-truth
neighbor tokens — at sampled context lengths up to 2000.

Our contribution: the interleaved mixing approach (rho > 0), where grid and ring
random-walk segments are concatenated into a single sequence. This tests whether
two competing in-context structures interfere with each other's learning.

Usage:
    python src/initial_experiments/vocabulary_tl_experiment.py
    python src/initial_experiments/vocabulary_tl_experiment.py --condition months_natural
    python src/initial_experiments/vocabulary_tl_experiment.py --replot
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformer_lens import HookedTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from graphs import (
    Ring,
    MONTHS, MONTHS_PERMUTED,
    RING_WORDS, RING_WORDS_OVERLAP, SHARED_WORDS,
)
from sanity_check import Grid, WORDS as GRID_WORDS, set_seed, make_interleaved_sequence

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME       = "meta-llama/Llama-3.1-8B"
RHOS             = [0.0, 0.5, 1.0]
SEQ_LEN          = 2000
SEGMENT_LEN      = 100
EVAL_LENGTHS     = [50, 100, 200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1400, 1600, 1800, 2000]
MIN_PLOT_SAMPLES = 8   # eval points with fewer sequences are dropped from plots
Y_AXIS_LIMITS    = (0.0, 1.0)

DATA_DIR = os.path.join(HERE, "results", "vocabulary_tl")
PLOT_DIR = os.path.join(HERE, "results")
os.makedirs(DATA_DIR, exist_ok=True)

CONDITIONS = {
    "months_natural": {
        "label": "Months — natural order",
        "ring_words": MONTHS,
        "description": (
            "Jan→Feb→...→Dec ring; semantic prior matches in-context structure"
        ),
        "has_overlap": False,
    },
    "months_permuted": {
        "label": "Months — permuted (prior conflicts with in-context)",
        "ring_words": MONTHS_PERMUTED,
        "description": (
            "Months permuted so no naturally-adjacent months are in-context neighbors "
            "(min natural distance between any in-context-adjacent pair = 5)"
        ),
        "has_overlap": False,
    },
    "neutral_disjoint": {
        "label": "Neutral words — disjoint",
        "ring_words": RING_WORDS,
        "description": (
            "Semantically neutral words, fully disjoint from grid; no semantic prior"
        ),
        "has_overlap": False,
    },
    "neutral_overlap": {
        "label": "Neutral words — overlapping vocab",
        "ring_words": RING_WORDS_OVERLAP,
        "description": "3 ring words (rock, sand, box) shared with grid",
        "has_overlap": True,
    },
}

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device,
    )
    model.eval()
    return model


# ── Token map ─────────────────────────────────────────────────────────────────

def build_token_map(model, words):
    """Return {word: token_id} for a list of words. Warns on multi-token words."""
    mapping = {}
    multi_token = []
    for w in words:
        ids = model.tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            mapping[w] = ids[0]
        else:
            multi_token.append((w, ids))
            mapping[w] = ids[0]  # use first token as approximation
    if multi_token:
        print(f"  WARNING: multi-token words (using first token): {multi_token}")
    return mapping


# ── Per-sequence accuracy ──────────────────────────────────────────────────────

@torch.no_grad()
def sequence_neighbor_probs(model, grid, ring, sequence, labels, eval_lengths,
                            grid_tok, ring_tok, is_overlap):
    """
    Single forward pass over the full sequence.

    At each eval position L, computes:
        P(next token ∈ valid neighbors) = sum of softmax probabilities over the
        ground-truth neighbor tokens for the current graph.

    Returns:
        grid_probs   : {L: float}
        ring_probs   : {L: float}
        shared_probs : {L: float}  (overlap condition only)
    """
    # Build the input sequence directly from token IDs.  All vocabulary words
    # are verified single-token, so the token count equals the word count (plus
    # BOS) and we can index probs with L-1.
    tok_map_all = {**grid_tok, **ring_tok}
    bos         = model.tokenizer.bos_token_id
    input_ids   = [bos] + [tok_map_all[w] for w in sequence]
    n_ctx       = model.cfg.n_ctx
    if len(input_ids) > n_ctx:
        input_ids = input_ids[:n_ctx]
    tokens = torch.tensor([input_ids], dtype=torch.long).to(model.cfg.device)
    logits = model(tokens)                             # [1, T, vocab]
    probs  = torch.softmax(logits[0, 1:, :], dim=-1)  # [T-1, vocab] — skip BOS

    grid_probs, ring_probs, shared_probs = {}, {}, {}

    for L in eval_lengths:
        if L - 1 >= probs.shape[0]:
            continue  # sequence was truncated before this eval point; skip

        current_word  = sequence[L - 1]
        current_label = labels[L - 1]

        if current_label == "grid":
            valid   = grid.get_valid_next_words(current_word)
            tok_map = grid_tok
        else:
            valid   = ring.get_valid_next_words(current_word)
            tok_map = ring_tok

        # Sum softmax probability over ground-truth valid neighbors
        p = sum(probs[L - 1, tok_map[nb]].item() for nb in valid if nb in tok_map)

        if current_label == "grid":
            grid_probs[L] = p
        else:
            ring_probs[L] = p

        if is_overlap and current_word in SHARED_WORDS:
            shared_probs[L] = p

    return grid_probs, ring_probs, shared_probs


# ── Run one (condition, rho) cell ──────────────────────────────────────────────

def run_condition_rho(model, condition_name, rho, eval_lengths, seed_offset=0):
    ring_words = CONDITIONS[condition_name]["ring_words"]
    is_overlap = CONDITIONS[condition_name]["has_overlap"]
    grid       = Grid()
    ring       = Ring(words=ring_words)

    # Use one sequence per graph node: 16 for pure-grid runs, 12 (ring.n) otherwise.
    n_sequences = len(grid.words) if rho == 0.0 else ring.n

    grid_tok = build_token_map(model, GRID_WORDS)
    ring_tok = build_token_map(model, ring_words)

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
RHO_LABEL   = {0.0: "ρ=0 (pure grid)", 0.5: "ρ=0.5 (mixed)", 1.0: "ρ=1 (pure ring)"}

COND_COLORS = {
    "months_natural":   "#1565C0",
    "months_permuted":  "#AD1457",
    "neutral_disjoint": "#2E7D32",
    "neutral_overlap":  "#E65100",
}


def _curve(ax, accs, color, label, ls="-"):
    lengths = sorted(L for L in EVAL_LENGTHS if len(accs.get(L, [])) >= MIN_PLOT_SAMPLES)
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
    ax.set_ylim(*Y_AXIS_LIMITS)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)


def plot_condition(results, condition_name):
    """3-panel plot (one per rho) for a single condition."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, rho in zip(axes, [0.0, 0.5, 1.0]):
        data = results[rho]
        _curve(ax, data["grid"], GRID_COLOR, "Grid neighbors")
        _curve(ax, data["ring"], RING_COLOR, "Ring neighbors")
        if CONDITIONS[condition_name]["has_overlap"]:
            _curve(ax, data["shared"], SHARE_COLOR, "Shared-word positions", ls="--")
        _style_ax(ax, RHO_LABEL[rho])

    axes[0].set_ylabel("P(next token ∈ valid neighbors)", fontsize=10)
    fig.suptitle(
        f"Llama 3.1 8B (base, TransformerLens) — {CONDITIONS[condition_name]['label']}\n"
        f"(16 seq ρ=0 / 12 seq ρ>0, segment_len={SEGMENT_LEN})",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, f"tl_{condition_name}_per_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_comparison(all_results):
    """
    3-panel head-to-head across all available conditions.

    Panel 1 — pure ring (ρ=1): baseline learnability per condition.
               Key question: does semantic prior help (natural) or hurt (permuted)?
    Panel 2 — mixed (ρ=0.5), grid accuracy: does the ring condition affect
               how well grid structure is learned under competition?
    Panel 3 — mixed (ρ=0.5), ring accuracy: the competition signal.
               Does mixing suppress ring learning, and does semantic prior modulate this?
    """
    if len(all_results) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: pure ring baseline
    ax = axes[0]
    for cond, data in all_results.items():
        if 1.0 in data:
            _curve(ax, data[1.0]["ring"], COND_COLORS[cond], CONDITIONS[cond]["label"])
    _style_ax(ax, "Pure ring (ρ=1): ring accuracy by condition")
    ax.set_ylabel("P(next ∈ ring neighbors)", fontsize=10)

    # Panel 2: mixed, grid accuracy
    ax = axes[1]
    for cond, data in all_results.items():
        if 0.5 in data:
            _curve(ax, data[0.5]["grid"], COND_COLORS[cond], CONDITIONS[cond]["label"])
    _style_ax(ax, "Mixed (ρ=0.5): grid accuracy")
    ax.set_ylabel("P(next ∈ grid neighbors)", fontsize=10)

    # Panel 3: mixed, ring accuracy (competition signal)
    ax = axes[2]
    for cond, data in all_results.items():
        if 0.5 in data:
            _curve(ax, data[0.5]["ring"], COND_COLORS[cond], CONDITIONS[cond]["label"])
    _style_ax(ax, "Mixed (ρ=0.5): ring accuracy\n(competition signal)")
    ax.set_ylabel("P(next ∈ ring neighbors)", fontsize=10)

    fig.suptitle(
        "Vocabulary & semantic prior experiment — all conditions\n"
        "Llama 3.1 8B (base), TransformerLens",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "tl_all_conditions_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=list(CONDITIONS.keys()) + ["all"],
        default="all",
        help="Which condition(s) to run (default: all)",
    )
    parser.add_argument(
        "--replot", action="store_true",
        help="Skip inference, regenerate plots from saved results",
    )
    args = parser.parse_args()

    conditions_to_run = (
        list(CONDITIONS.keys()) if args.condition == "all" else [args.condition]
    )

    all_results = {}

    if args.replot:
        for cond in conditions_to_run:
            print(f"Loading {cond}...")
            all_results[cond] = load_condition(cond)
    else:
        print(f"Loading {MODEL_NAME} via TransformerLens...")
        model = load_model()
        print(f"Model loaded on {model.cfg.device}.\n")

        for cond in conditions_to_run:
            print(f"{'='*60}")
            print(f"Condition: {cond.upper()}")
            print(f"  {CONDITIONS[cond]['description']}")
            print(f"{'='*60}")

            cond_results = {}
            for i, rho in enumerate(RHOS):
                n_seq = 16 if rho == 0.0 else 12
                print(f"\n  [{i+1}/{len(RHOS)}] ρ={rho}  "
                      f"({n_seq} sequences × {len(EVAL_LENGTHS)} eval points)")
                grid_accs, ring_accs, shared_accs = run_condition_rho(
                    model, cond, rho, EVAL_LENGTHS, seed_offset=i,
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
