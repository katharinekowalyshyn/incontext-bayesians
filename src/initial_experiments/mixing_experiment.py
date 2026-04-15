"""
mixing_experiment.py

Measures Llama 3.1 8B (base) performance on competing graph structures via
TransformerLens (single forward pass per sequence).

For rho in {0.0, 0.5, 1.0}:
  - rho=0.0: pure 4x4 grid walks
  - rho=0.5: interleaved grid + ring segments (across-sequence mixing)
  - rho=1.0: pure 12-node ring (months of year) walks

At each sampled context length, we measure:
    P(next token in valid neighbors)
as the summed softmax probability over ground-truth neighbor tokens.

Usage:
    python initial_experiments/mixing_experiment.py
    python initial_experiments/mixing_experiment.py --replot   # replot saved results
"""

import os
import sys
import json
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from transformer_lens import HookedTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphs import Ring, MONTHS
from sanity_check import Grid, WORDS, set_seed, make_interleaved_sequence

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B"

RHOS = [0.0, 0.5, 1.0]
N_SEQUENCES = 8          # sequences per rho value
SEQ_LEN = 1400           # total tokens per sequence (matches Park et al.)
SEGMENT_LEN = 100        # tokens per segment in mixed sequences
EVAL_LENGTHS = [50, 100, 200, 300, 400, 500, 600, 700, 850, 1000, 1200, 1400]
Y_AXIS_LIMITS = (0.0, 1.0)

DATA_DIR = os.path.join(os.path.dirname(__file__), "results", "mixing_experiment")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── TransformerLens helpers ────────────────────────────────────────────────────

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        device=device,
    )
    model.eval()
    return model


def build_token_map(model, words):
    """Return {word: token_id}; warns if any word splits into multiple tokens."""
    mapping = {}
    multi_token = []
    for w in words:
        ids = model.tokenizer.encode(" " + w, add_special_tokens=False)
        if len(ids) == 1:
            mapping[w] = ids[0]
        else:
            multi_token.append((w, ids))
            mapping[w] = ids[0]
    if multi_token:
        print(f"  WARNING: multi-token words (using first token): {multi_token}")
    return mapping


@torch.no_grad()
def sequence_neighbor_probs(model, grid, ring, sequence, labels, eval_lengths, grid_tok, ring_tok):
    """
    Single forward pass on one sequence.
    Returns:
        grid_probs: {L: float}
        ring_probs: {L: float}
    """
    tok_map_all = {**grid_tok, **ring_tok}
    bos = model.tokenizer.bos_token_id
    input_ids = [bos] + [tok_map_all[w] for w in sequence]
    n_ctx = model.cfg.n_ctx
    if len(input_ids) > n_ctx:
        input_ids = input_ids[:n_ctx]

    tokens = torch.tensor([input_ids], dtype=torch.long).to(model.cfg.device)
    logits = model(tokens)
    probs = torch.softmax(logits[0, 1:, :], dim=-1)  # skip BOS

    grid_probs, ring_probs = {}, {}
    for L in eval_lengths:
        if L - 1 >= probs.shape[0]:
            continue

        current_word = sequence[L - 1]
        current_label = labels[L - 1]
        if current_label == "grid":
            valid = grid.get_valid_next_words(current_word)
            tok_map = grid_tok
        else:
            valid = ring.get_valid_next_words(current_word)
            tok_map = ring_tok

        p = sum(probs[L - 1, tok_map[nb]].item() for nb in valid if nb in tok_map)
        if current_label == "grid":
            grid_probs[L] = p
        else:
            ring_probs[L] = p

    return grid_probs, ring_probs


# ── Experiment ─────────────────────────────────────────────────────────────────

def run_rho(model, grid, ring, rho, n_sequences, eval_lengths, seed_offset=0):
    """
    Run all sequences for one mixture ratio.

    Returns:
        grid_accs : {length: [float, ...]}  — P(next token in valid grid neighbors)
        ring_accs : {length: [float, ...]}  — P(next token in valid ring neighbors)
    """
    grid_accs = {L: [] for L in eval_lengths}
    ring_accs = {L: [] for L in eval_lengths}
    grid_tok = build_token_map(model, grid.words)
    ring_tok = build_token_map(model, ring.words)

    for seq_i in range(n_sequences):
        set_seed(42 + seq_i + seed_offset * 100)
        seq, labels = make_interleaved_sequence(grid, ring, SEQ_LEN, rho, SEGMENT_LEN)
        gp, rp = sequence_neighbor_probs(
            model, grid, ring, seq, labels, eval_lengths, grid_tok, ring_tok
        )
        for L, p in gp.items():
            grid_accs[L].append(p)
        for L, p in rp.items():
            ring_accs[L].append(p)

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
    """Three-panel plot: one subplot per rho, grid vs ring neighbor probability."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, rho in zip(axes, [0.0, 0.5, 1.0]):
        data = all_results[rho]
        _plot_curve(ax, data["grid"], GRID_COLOR, "Grid neighbors", EVAL_LENGTHS)
        _plot_curve(ax, data["ring"], RING_COLOR, "Ring neighbors", EVAL_LENGTHS)

        # Reference baselines if mass were uniform over each graph vocabulary.
        ax.axhline(3 / 16, color=GRID_COLOR, lw=0.9, ls="--", alpha=0.45,
                   label="Grid chance (3/16)")
        ax.axhline(2 / 12, color=RING_COLOR, lw=0.9, ls="--", alpha=0.45,
                   label="Ring chance (2/12)")

        ax.set_xlabel("Context length (tokens)", fontsize=10)
        ax.set_title(RHO_LABEL[rho], fontsize=11)
        ax.set_xlim(0, SEQ_LEN + 50)
        ax.set_ylim(*Y_AXIS_LIMITS)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    axes[0].set_ylabel("P(next token in valid neighbors)", fontsize=10)
    fig.suptitle(
        "Llama 3.1 8B (base, TransformerLens) — competing graph structures\n"
        f"({N_SEQUENCES} sequences per ρ, segment_len={SEGMENT_LEN})",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "mixing_per_rho.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary(all_results):
    """Two-panel summary: grid and ring neighbor probability, all rhos overlaid."""
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
        ax.set_ylabel("P(next token in valid neighbors)", fontsize=10)
        ax.set_title(f"{graph_type.capitalize()} neighbor probability — effect of ρ", fontsize=11)
        ax.set_xlim(0, SEQ_LEN + 50)
        ax.set_ylim(*Y_AXIS_LIMITS)
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Llama 3.1 8B (base, TransformerLens) — does mixing suppress structure learning?",
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
    ring = Ring(words=MONTHS)

    if args.replot:
        if not os.path.exists(cache_path):
            print(f"No cached results found at {cache_path}. Run without --replot first.")
            sys.exit(1)
        print(f"Loading results from {cache_path}")
        all_results = load_results(cache_path)
    else:
        print(f"Loading {MODEL_NAME} via TransformerLens...")
        model = load_model()
        print(f"Model loaded on {model.cfg.device}.")

        set_seed(42)

        all_results = {}
        for i, rho in enumerate(RHOS):
            print(f"\n[{i+1}/{len(RHOS)}] Running ρ={rho}  "
                  f"({N_SEQUENCES} sequences × {len(EVAL_LENGTHS)} eval points)")
            grid_accs, ring_accs = run_rho(
                model, grid, ring, rho, N_SEQUENCES, EVAL_LENGTHS, seed_offset=i
            )
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
