"""Shared utilities for reproducing 'In-context learning of representations
can be explained by induction circuits.'"""

import os
import random
import functools

import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import einops

from torch import Tensor
from jaxtyping import Float
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

torch.set_grad_enabled(False)

# ── Constants ──────────────────────────────────────────────────────────────────

WORDS = [
    "apple", "bird", "car", "egg",
    "house", "milk", "plane", "opera",
    "box", "sand", "sun", "mango",
    "rock", "math", "code", "phone",
]

GRID_ROWS = 4
GRID_COLS = 4
MODEL_NAME = "meta-llama/Llama-3.1-8B"
LAYER = 26
SEQ_LEN = 1400
N_SEQUENCES = 16
SMOOTHING_WINDOW = 30

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#ffff33", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]

WORD_TO_COLOR = {word: color for word, color in zip(WORDS, COLORS)}


# ── Grid ───────────────────────────────────────────────────────────────────────

class Grid:
    def __init__(self, words=WORDS, rows=GRID_ROWS, cols=GRID_COLS):
        if rows * cols != len(words):
            raise ValueError(
                f"Grid dimensions ({rows}x{cols}={rows * cols}) "
                f"do not match number of words ({len(words)})"
            )
        self.words = words
        self.rows = rows
        self.cols = cols
        self.grid = np.array(words).reshape(rows, cols).tolist()
        self.word_to_row = {w: i // cols for i, w in enumerate(words)}
        self.word_to_col = {w: i % cols for i, w in enumerate(words)}

    # ── sequence generation ────────────────────────────────────────────────

    def generate_sequence(self, seq_len, start_word=None):
        """Random walk on the grid.  Optionally fix the starting word."""
        if start_word is not None:
            row, col = self.word_to_row[start_word], self.word_to_col[start_word]
        else:
            row, col = np.random.randint(0, self.rows), np.random.randint(0, self.cols)

        sequence = [self.grid[row][col]]
        while len(sequence) < seq_len:
            moves = self._valid_moves(row, col)
            direction = np.random.choice(moves)
            if direction == "up":    row -= 1
            elif direction == "down":  row += 1
            elif direction == "left":  col -= 1
            elif direction == "right": col += 1
            sequence.append(self.grid[row][col])
        return sequence

    def generate_batch(self, seq_len):
        """16 sequences, each starting at a different grid word."""
        return [self.generate_sequence(seq_len, start_word=w) for w in self.words]

    # ── adjacency ──────────────────────────────────────────────────────────

    def get_valid_next_words(self, word):
        row, col = self.word_to_row[word], self.word_to_col[word]
        next_words = []
        for move in self._valid_moves(row, col):
            if move == "up":    next_words.append(self.grid[row - 1][col])
            elif move == "down":  next_words.append(self.grid[row + 1][col])
            elif move == "left":  next_words.append(self.grid[row][col - 1])
            elif move == "right": next_words.append(self.grid[row][col + 1])
        return next_words

    def build_adjacency_matrix(self):
        """Return a 16x16 binary adjacency matrix (symmetric)."""
        n = len(self.words)
        A = np.zeros((n, n))
        for i, word in enumerate(self.words):
            for neighbor in self.get_valid_next_words(word):
                j = self.words.index(neighbor)
                A[i, j] = 1
        return A

    # ── internals ──────────────────────────────────────────────────────────

    def _valid_moves(self, row, col):
        moves = []
        if row > 0:              moves.append("up")
        if row < self.rows - 1:  moves.append("down")
        if col > 0:              moves.append("left")
        if col < self.cols - 1:  moves.append("right")
        return moves


# ── Seeding ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(cache_dir=None, device=None):
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device=device, cache_dir=cache_dir,
    )


# ── Tokenization ──────────────────────────────────────────────────────────────

def tokenize_sequence(model, sequence):
    """Tokenize a word sequence. Returns input_ids tensor."""
    text = " " + " ".join(sequence)
    return model.tokenizer(text, return_tensors="pt").input_ids


# ── Accuracy ───────────────────────────────────────────────────────────────────

def get_model_accuracies(model, grid, sequence, fwd_hooks=[]):
    """Per-position probability assigned to valid next tokens."""
    tokens = tokenize_sequence(model, sequence)
    logits = model.run_with_hooks(tokens.to(model.cfg.device), fwd_hooks=fwd_hooks)
    probs = torch.softmax(logits, dim=-1)
    probs = probs[0, 1:, :]  # remove BOS position

    accuracies = []
    for i in range(len(sequence)):
        valid_next = grid.get_valid_next_words(sequence[i])
        token_ids = torch.tensor(
            [model.tokenizer.encode(" " + w, add_special_tokens=False) for w in valid_next]
        ).squeeze()
        accuracies.append(probs[i, token_ids].sum().item())
    return accuracies


# ── Activations ────────────────────────────────────────────────────────────────

def get_activations(model, sequence, layer, n_lookback, fwd_hooks=[]):
    """Return last n_lookback residual-stream activations at `layer`.

    Shape: [n_lookback, d_model]
    """
    tokens = tokenize_sequence(model, sequence)
    names_filter = [f"blocks.{layer}.hook_resid_pre"]

    with model.hooks(fwd_hooks=fwd_hooks):
        _, cache = model.run_with_cache(
            tokens.to(model.cfg.device), names_filter=names_filter,
        )

    acts = cache[f"blocks.{layer}.hook_resid_pre"]  # [1, seq+1, d_model]
    # Remove BOS then take last n_lookback
    acts = acts[0, 1:, :]  # [seq_len, d_model]
    return acts[-n_lookback:, :]  # [n_lookback, d_model]


# ── PCA helpers ────────────────────────────────────────────────────────────────

def compute_class_means(
    activations: Float[Tensor, "n_lookback d_model"],
    sequence: list[str],
    words: list[str],
    n_lookback: int,
) -> Float[Tensor, "n_words d_model"]:
    """Mean activation per word over the last n_lookback positions."""
    tail = sequence[-n_lookback:]
    means = []
    for word in words:
        idxs = [i for i, w in enumerate(tail) if w == word]
        if idxs:
            means.append(activations[idxs].mean(dim=0))
        else:
            means.append(torch.zeros(activations.shape[-1], device=activations.device))
    return torch.stack(means)  # [16, d_model]


def compute_pca_directions(
    class_means: Float[Tensor, "n_words d_model"],
    top_n: int = 2,
) -> Float[Tensor, "top_n d_model"]:
    """PCA on the 16 class-mean vectors. Returns top_n right singular vectors."""
    centered = class_means - class_means.mean(dim=0, keepdim=True)
    _, _, V = torch.svd(centered)
    return einops.rearrange(V, "d_model n -> n d_model")[:top_n, :]


# ── Ablation hooks ─────────────────────────────────────────────────────────────

def head_ablation_hook(
    activation: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head: int,
) -> Float[Tensor, "batch seq n_heads d_head"]:
    activation[:, :, head, :] = 0.0
    return activation


def make_ablation_hooks(heads_to_ablate):
    """Return a list of (name, hook_fn) pairs that zero-ablate the given heads.

    heads_to_ablate: list of (layer, head) tuples.
    """
    return [
        (
            utils.get_act_name("z", layer),
            functools.partial(head_ablation_hook, head=head),
        )
        for layer, head in heads_to_ablate
    ]


# ── Plotting ───────────────────────────────────────────────────────────────────

def setup_plotting():
    plt.style.use("science")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.top": False,
        "ytick.right": False,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.color": "gray",
        "grid.linewidth": 0.4,
        "savefig.bbox": "standard",  # override scienceplots' tight default
    })


def save_figure(fig, directory, filename):
    os.makedirs(directory, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    fig.tight_layout()
    fig.savefig(os.path.join(directory, stem + ".pdf"))
    fig.savefig(os.path.join(directory, stem + ".png"), dpi=300)
    plt.close(fig)


def smooth(arr, window=SMOOTHING_WINDOW):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


# ── Plotly helpers ────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    autosize=False,
    margin=dict(l=60, r=40, t=60, b=60),
)

PLOTLY_AXIS = dict(
    showgrid=True, gridcolor="rgba(128,128,128,0.12)", gridwidth=0.4,
    zeroline=False, showline=True, linecolor="black", linewidth=1,
    ticks="outside", tickcolor="black",
)


def plotly_pca_layout(title):
    """Standard layout for PCA scatter plots in plotly."""
    return dict(
        title=title,
        width=600, height=600,
        xaxis=dict(title="PC1", scaleanchor="y", scaleratio=1, **PLOTLY_AXIS),
        yaxis=dict(title="PC2", **PLOTLY_AXIS),
        **PLOTLY_LAYOUT,
    )


def plotly_line_layout(title, xlabel, ylabel):
    """Standard layout for line plots in plotly."""
    return dict(
        title=title,
        width=700, height=400,
        xaxis=dict(title=xlabel, type="log", **PLOTLY_AXIS),
        yaxis=dict(title=ylabel, **PLOTLY_AXIS),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray", borderwidth=1),
        **PLOTLY_LAYOUT,
    )


def save_plotly(fig, directory, filename):
    """Save a plotly figure as HTML with fixed dimensions."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    fig.write_html(path, config={"responsive": False}, include_plotlyjs="cdn")
    print(f"Saved {path}")


def plotly_pca_traces(projected, grid, words=WORDS, word_to_color=WORD_TO_COLOR):
    """Return list of plotly traces for a PCA scatter: edges + star centroids."""
    import plotly.graph_objects as go
    traces = []
    A = grid.build_adjacency_matrix()
    # Collect all edges into a single trace with None breaks
    edge_x, edge_y = [], []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if A[i, j]:
                edge_x += [projected[i, 0].item(), projected[j, 0].item(), None]
                edge_y += [projected[i, 1].item(), projected[j, 1].item(), None]
    traces.append(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="gray", width=0.8, dash="dash"),
        opacity=0.4, showlegend=False, hoverinfo="skip",
    ))
    # Centroids
    for i, word in enumerate(words):
        traces.append(go.Scatter(
            x=[projected[i, 0].item()], y=[projected[i, 1].item()],
            mode="markers+text",
            marker=dict(size=16, symbol="star", color=word_to_color[word],
                        line=dict(width=1, color="black")),
            text=word, textposition="top right", textfont=dict(size=10),
            hovertemplate=f"centroid: <b>{word}</b><extra></extra>",
            hoverlabel=dict(bgcolor=word_to_color[word], font=dict(color="black")),
            showlegend=False,
        ))
    return traces
