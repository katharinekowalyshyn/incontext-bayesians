"""Figures 3 and 4: ablation experiments on induction and previous-token heads."""

import os
import json
import random as _random

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import einops
import tqdm

from utils import (
    WORDS, LAYER, SEQ_LEN, WORD_TO_COLOR,
    Grid, set_seed, load_model,
    get_model_accuracies, get_activations, compute_class_means,
    compute_pca_directions, make_ablation_hooks, setup_plotting, save_figure,
    smooth, plotly_pca_layout, plotly_line_layout, plotly_pca_traces, save_plotly,
)

DATA_DIR = "results/ablation/data"
PLOTS_DIR = "results/ablation/plots"
N_LOOKBACK = 200
K_VALUES = [1, 2, 4, 8, 16, 32]
N_TEST_SEQS = 32       # repeated-token sequences for head identification
REPEAT_LEN = 32        # tokens per half of the repeated sequence
N_RANDOM_SETS = 4      # random control: number of random head sets
N_RANDOM_HEADS = 32    # random control: heads per set


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Identify induction and previous-token heads (Appendix A)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_head_scores(model):
    """Score every head for induction and previous-token behaviour.

    Returns (induction_scores, prev_token_scores), each [n_layers, n_heads].
    """
    set_seed(42)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    induction_scores = np.zeros((n_layers, n_heads))
    prev_token_scores = np.zeros((n_layers, n_heads))

    name_filters = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]

    for seq_idx in tqdm.tqdm(range(N_TEST_SEQS), desc="Identifying heads"):
        # Random tokens from lower half of vocab, repeated once → 64 tokens
        random_tokens = torch.randint(0, model.cfg.d_vocab // 2, (REPEAT_LEN,))
        repeated = einops.repeat(random_tokens, "s -> (two s)", two=2)

        _, cache = model.run_with_cache(repeated, names_filter=name_filters)

        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [n_heads, seq, seq]
            for head in range(n_heads):
                p = pattern[head]  # [seq, seq]

                # Induction score: mean of offset-(REPEAT_LEN-1) diagonal
                # position i attending to i - (REPEAT_LEN - 1)
                offset = REPEAT_LEN - 1
                diag = p.diagonal(offset=-offset)
                induction_scores[layer, head] += diag[offset:].mean().item()

                # Previous-token score: mean of offset-1 diagonal
                diag_prev = p.diagonal(offset=-1)
                prev_token_scores[layer, head] += diag_prev[1:].mean().item()

        del cache

    # Average across test sequences
    induction_scores /= N_TEST_SEQS
    prev_token_scores /= N_TEST_SEQS

    return induction_scores, prev_token_scores


def scores_to_ranked(scores):
    """Convert [n_layers, n_heads] score array to sorted list of (score, layer, head)."""
    n_layers, n_heads = scores.shape
    return sorted(
        [(scores[l, h], l, h) for l in range(n_layers) for h in range(n_heads)],
        reverse=True,
    )


def print_top_heads(induction_ranked, prev_token_ranked, k=32):
    print(f"\nTop-{k} induction heads:")
    for score, layer, head in induction_ranked[:k]:
        print(f"  L{layer:>2d}.H{head:>2d}  score={score:.4f}")
    print(f"\nTop-{k} previous-token heads:")
    for score, layer, head in prev_token_ranked[:k]:
        print(f"  L{layer:>2d}.H{head:>2d}  score={score:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Fig 3 — Ablation accuracy curves
# ═══════════════════════════════════════════════════════════════════════════════

def run_accuracy_sweep(model, grid, ranked_heads, k_values, label, induction_ranked, prev_token_ranked):
    """Run ablation accuracy experiments for a single head type.

    Returns dict mapping k → mean_accuracy array (plus k=0 baseline
    and 'random' → random control).
    """
    results = {}

    for k in [0] + k_values:
        heads = [(l, h) for _, l, h in ranked_heads[:k]]
        hooks = make_ablation_hooks(heads)

        set_seed(42)
        sequences = grid.generate_batch(SEQ_LEN)
        all_accs = []
        for seq in tqdm.tqdm(sequences, desc=f"{label} k={k}"):
            all_accs.append(get_model_accuracies(model, grid, seq, fwd_hooks=hooks))
        results[k] = np.array(all_accs).mean(axis=0)

    # Random control: 4 random sets of 32 heads (excluding top-32 of each type)
    excluded = set()
    for _, l, h in induction_ranked[:32]:
        excluded.add((l, h))
    for _, l, h in prev_token_ranked[:32]:
        excluded.add((l, h))
    all_heads = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    eligible = [lh for lh in all_heads if lh not in excluded]

    random_accs = []
    for r in range(N_RANDOM_SETS):
        set_seed(r + 100)
        chosen = _random.sample(eligible, N_RANDOM_HEADS)
        hooks = make_ablation_hooks(chosen)

        set_seed(42)
        sequences = grid.generate_batch(SEQ_LEN)
        set_accs = []
        for seq in tqdm.tqdm(sequences, desc=f"{label} random set {r}"):
            set_accs.append(get_model_accuracies(model, grid, seq, fwd_hooks=hooks))
        random_accs.append(np.array(set_accs).mean(axis=0))
    results["random"] = np.mean(random_accs, axis=0)

    return results


def save_accuracy_data(path, results):
    save_dict = {f"k{k}": results[k] for k in [0] + K_VALUES}
    save_dict["random"] = results["random"]
    np.savez(path, **save_dict)
    print(f"Cached {path}")


def load_accuracy_data(path):
    data = np.load(path)
    results = {k: data[f"k{k}"] for k in [0] + K_VALUES}
    results["random"] = data["random"]
    return results


def plot_ablation_accuracy(results, k_values, title, filename,
                           cmap_name="Reds", head_type="heads"):
    """Plot one ablation accuracy figure."""
    fig, ax = plt.subplots(figsize=(5, 2.8))

    # Baseline (k=0)
    ax.plot(smooth(results[0]), color="black", linewidth=1.0, label="No ablation", zorder=10)

    # Random control (plotted second so it appears second in legend)
    ax.plot(smooth(results["random"]), color="gray", linewidth=1.0,
            label=f"Ablate {N_RANDOM_HEADS} random heads", zorder=9)

    # Ablation curves: dark → light as more heads are ablated
    cmap = plt.colormaps[cmap_name]
    colors = [cmap(v) for v in np.linspace(0.85, 0.35, len(k_values))]
    for i, k in enumerate(k_values):
        ax.plot(smooth(results[k]), color=colors[i], linewidth=1.0,
                label=f"Ablate top-{k} {head_type}")

    ax.set_xscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="gray", fontsize=7)

    save_figure(fig, PLOTS_DIR, filename)
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    import matplotlib.colors as mcolors
    pfig = go.Figure()
    x = list(range(len(smooth(results[0]))))

    pfig.add_trace(go.Scatter(
        x=x, y=smooth(results[0]).tolist(), mode="lines",
        line=dict(color="black", width=2), name="No ablation",
        hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
    ))
    pfig.add_trace(go.Scatter(
        x=x, y=smooth(results["random"]).tolist(), mode="lines",
        line=dict(color="gray", width=2), name=f"Ablate {N_RANDOM_HEADS} random heads",
        hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
    ))
    for i, k in enumerate(k_values):
        pfig.add_trace(go.Scatter(
            x=x, y=smooth(results[k]).tolist(), mode="lines",
            line=dict(color=mcolors.to_hex(colors[i]), width=2),
            name=f"Ablate top-{k} {head_type}",
            hovertemplate="pos: %{x}<br>accuracy: %{y:.3f}<extra></extra>",
        ))

    html_stem = os.path.splitext(filename)[0]
    pfig.update_layout(**plotly_line_layout(title, "Sequence length", "Accuracy"))
    save_plotly(pfig, PLOTS_DIR, f"{html_stem}.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Fig 4 — Ablation PCA
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pca_projected(model, grid, sequence, heads_to_ablate):
    """Compute PCA-projected class means for one ablation condition. Returns [16, 2] numpy."""
    hooks = make_ablation_hooks(heads_to_ablate) if heads_to_ablate else []
    activations = get_activations(model, sequence, LAYER, N_LOOKBACK, fwd_hooks=hooks)
    class_means = compute_class_means(activations, sequence, WORDS, N_LOOKBACK)
    pca_dirs = compute_pca_directions(class_means, top_n=2)
    projected = (class_means @ pca_dirs.T).cpu().numpy()
    return projected


def plot_ablation_pca(grid, projected, condition_name, filename):
    """PCA scatter for one ablation condition."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Grid edges
    A = grid.build_adjacency_matrix()
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if A[i, j]:
                ax.plot(
                    [projected[i, 0].item(), projected[j, 0].item()],
                    [projected[i, 1].item(), projected[j, 1].item()],
                    color="gray", alpha=0.3, linestyle="--", linewidth=0.5,
                )

    for i, word in enumerate(WORDS):
        ax.scatter(projected[i, 0].item(), projected[i, 1].item(),
                   color=WORD_TO_COLOR[word], s=120, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(
            word, (projected[i, 0].item(), projected[i, 1].item()),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(condition_name, fontsize=10)
    ax.set_aspect("equal")
    save_figure(fig, PLOTS_DIR, filename)
    print(f"Saved {filename}")

    # ── Plotly interactive ───────────────────────────────────────────────────
    pfig = go.Figure(data=plotly_pca_traces(projected, grid))
    pfig.update_layout(**plotly_pca_layout(condition_name))
    html_stem = os.path.splitext(filename)[0]
    save_plotly(pfig, PLOTS_DIR, f"{html_stem}.html")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    setup_plotting()
    grid = Grid()
    os.makedirs(DATA_DIR, exist_ok=True)
    model = None  # lazy-loaded only when data is missing

    # ── Step 1: head scores ───────────────────────────────────────────────────
    head_path = os.path.join(DATA_DIR, "head_scores.npz")
    if os.path.exists(head_path):
        print("Loading cached head scores...")
        data = np.load(head_path)
        induction_scores, prev_token_scores = data["induction_scores"], data["prev_token_scores"]
    else:
        if model is None:
            model = load_model()
        induction_scores, prev_token_scores = compute_head_scores(model)
        np.savez(head_path, induction_scores=induction_scores, prev_token_scores=prev_token_scores)
        print(f"Cached {head_path}")

    induction_ranked = scores_to_ranked(induction_scores)
    prev_token_ranked = scores_to_ranked(prev_token_scores)
    print_top_heads(induction_ranked, prev_token_ranked)

    # ── Step 2: ablation accuracy curves ──────────────────────────────────────
    ind_acc_path = os.path.join(DATA_DIR, "accuracy_induction.npz")
    if os.path.exists(ind_acc_path):
        print("Loading cached induction accuracy data...")
        ind_results = load_accuracy_data(ind_acc_path)
    else:
        if model is None:
            model = load_model()
        print("\n=== Induction head ablation ===")
        ind_results = run_accuracy_sweep(model, grid, induction_ranked, K_VALUES,
                                         "Induction", induction_ranked, prev_token_ranked)
        save_accuracy_data(ind_acc_path, ind_results)

    pt_acc_path = os.path.join(DATA_DIR, "accuracy_prev_token.npz")
    if os.path.exists(pt_acc_path):
        print("Loading cached prev-token accuracy data...")
        pt_results = load_accuracy_data(pt_acc_path)
    else:
        if model is None:
            model = load_model()
        print("\n=== Previous-token head ablation ===")
        pt_results = run_accuracy_sweep(model, grid, prev_token_ranked, K_VALUES,
                                        "Prev-token", induction_ranked, prev_token_ranked)
        save_accuracy_data(pt_acc_path, pt_results)

    plot_ablation_accuracy(ind_results, K_VALUES,
                           "Ablating induction heads", "ablation_induction.pdf",
                           cmap_name="Reds", head_type="induction heads")
    plot_ablation_accuracy(pt_results, K_VALUES,
                           "Ablating previous-token heads", "ablation_prev_token.pdf",
                           cmap_name="Blues", head_type="prev-token heads")

    # ── Step 3: ablation PCA ──────────────────────────────────────────────────
    seq_path = os.path.join(DATA_DIR, "sequence.json")
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            sequence = json.load(f)
    else:
        set_seed(42)
        sequence = grid.generate_sequence(SEQ_LEN)
        with open(seq_path, "w") as f:
            json.dump(sequence, f)

    top32_induction = [(l, h) for _, l, h in induction_ranked[:32]]
    top32_prev_token = [(l, h) for _, l, h in prev_token_ranked[:32]]

    pca_conditions = [
        ("baseline", "No ablation",
         [], "pca_baseline.pdf"),
        ("induction_ablated", "Top 32 induction heads ablated",
         top32_induction, "pca_induction_ablated.pdf"),
        ("prev_token_ablated", "Top 32 previous-token heads ablated",
         top32_prev_token, "pca_prev_token_ablated.pdf"),
    ]

    for cache_name, title, heads_to_ablate, plot_filename in pca_conditions:
        pca_path = os.path.join(DATA_DIR, f"pca_{cache_name}.npz")
        if os.path.exists(pca_path):
            print(f"Loading cached PCA data ({cache_name})...")
            projected = np.load(pca_path)["projected"]
        else:
            if model is None:
                model = load_model()
            projected = compute_pca_projected(model, grid, sequence, heads_to_ablate)
            np.savez(pca_path, projected=projected)
            print(f"Cached {pca_path}")

        plot_ablation_pca(grid, projected, title, plot_filename)


if __name__ == "__main__":
    main()
