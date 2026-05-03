"""PCA/representation analysis for secondary graph experiments.

This is a scoped copy/adaptation of the PCA utilities in
``src/initial_experiments/pca_analysis.py`` and ``iclr_induction-main/utils.py``.
It keeps the same residual-stream hook convention and paper-style sliding
window class means, but uses the stand-alone secondary graph definitions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT_CONFIG, ExperimentConfig
from .graphs import UndirectedGraph, build_candidate_graphs
from .sequence_generation import generate_sequence
from .vocabulary import WORDS, WORD_TO_COLOR, build_token_map, validate_vocabulary


DEFAULT_LAYER = 26
DEFAULT_WINDOW = 50
DEFAULT_SNAPSHOT_T = (200, 400, 1400)
DEFAULT_ENERGY_T = (60, 80, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1400)


@dataclass
class PCAResult:
    true_graph: str
    layer: int
    seq_len: int
    window: int
    snapshot_Ts: tuple[int, ...]
    energy_Ts: tuple[int, ...]
    class_means_by_T: dict[int, np.ndarray]
    present_by_T: dict[int, np.ndarray]
    energy_by_graph: dict[str, tuple[np.ndarray, np.ndarray]]


def compute_top_k_pca(class_means: np.ndarray, k: int = 4) -> np.ndarray:
    """Top-k PCA directions of ``class_means``. Copied from initial PCA code."""

    centered = class_means - class_means.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:k]


def compute_class_means_np(
    activations: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
) -> np.ndarray:
    """Per-word mean activation. Shape: ``[len(words), d_model]``."""

    if len(activations) != len(tokens):
        raise ValueError(
            f"activations ({len(activations)}) and tokens ({len(tokens)}) must align"
        )
    tokens = list(tokens)
    means = np.zeros((len(words), activations.shape[1]), dtype=activations.dtype)
    for j, word in enumerate(words):
        idxs = [i for i, token in enumerate(tokens) if token == word]
        if idxs:
            means[j] = activations[idxs].mean(axis=0)
    return means


def class_means_sliding(
    activations: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
    T: int,
    window: int = DEFAULT_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Class means over positions ``[T - window, T)``."""

    if T > len(activations):
        raise ValueError(f"T={T} exceeds sequence length {len(activations)}.")
    if T - window < 0:
        raise ValueError(f"T={T} is shorter than window={window}.")
    sl_tokens = list(tokens[T - window:T])
    means = compute_class_means_np(activations[T - window:T], sl_tokens, words)
    present = np.array([word in sl_tokens for word in words], dtype=bool)
    return means, present


def dirichlet_energy(H: np.ndarray, A: np.ndarray, normalize: bool = True) -> float:
    """Normalized graph Dirichlet energy, copied from initial PCA analysis."""

    H = np.asarray(H, dtype=float)
    A = np.asarray(A, dtype=float)
    n = H.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A shape {A.shape} != ({n}, {n}).")
    D = np.diag(A.sum(axis=1))
    L = D - A
    energy = float(np.trace(H.T @ L @ H))
    if not normalize:
        return energy
    H_c = H - H.mean(axis=0, keepdims=True)
    denom = float(np.trace(H_c.T @ D @ H_c))
    return energy / denom if denom > 0 else float("nan")


def dirichlet_energy_curve(
    activations: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
    A: np.ndarray,
    Ts: Sequence[int] = DEFAULT_ENERGY_T,
    window: int = DEFAULT_WINDOW,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Dirichlet energy at each requested context length."""

    kept_Ts, energies = [], []
    for T in Ts:
        if T - window < 0 or T > len(activations):
            continue
        H, present = class_means_sliding(activations, tokens, words, T, window=window)
        if present.sum() < 2:
            continue
        H_p = H[present]
        A_p = np.asarray(A)[np.ix_(present, present)]
        kept_Ts.append(T)
        energies.append(dirichlet_energy(H_p, A_p, normalize=normalize))
    return np.asarray(kept_Ts), np.asarray(energies)


def _edges_from_adjacency(A: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i in range(A.shape[0]) for j in range(i + 1, A.shape[1]) if A[i, j]]


def _draw_scatter(
    ax: plt.Axes,
    projected: np.ndarray,
    words: Sequence[str],
    A: np.ndarray,
    pc_x: int,
    pc_y: int,
    title: str,
    show_labels: bool,
) -> None:
    for i, j in _edges_from_adjacency(A):
        ax.plot(
            [projected[i, 0], projected[j, 0]],
            [projected[i, 1], projected[j, 1]],
            color="gray",
            alpha=0.35,
            linestyle="--",
            linewidth=0.6,
        )
    for i, word in enumerate(words):
        ax.scatter(
            projected[i, 0],
            projected[i, 1],
            color=WORD_TO_COLOR.get(word, "gray"),
            s=130,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        if show_labels:
            ax.annotate(
                word,
                (projected[i, 0], projected[i, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.5),
            )
    ax.set_xlabel(f"PC{pc_x}")
    ax.set_ylabel(f"PC{pc_y}")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)


def _fill_pca_snapshot_column(
    ax_pc12: plt.Axes,
    ax_pc34: plt.Axes,
    T: int,
    result: PCAResult,
    graph: UndirectedGraph,
    words: Sequence[str],
    overlay_graphs: Mapping[str, UndirectedGraph] | None,
    show_labels: bool,
) -> None:
    """Draw PC1/2 (top) and PC3/4 (bottom) for one snapshot context length ``T``."""

    H_full = result.class_means_by_T[T]
    present = result.present_by_T[T]
    n_present = int(present.sum())
    if n_present < 3:
        for ax in (ax_pc12, ax_pc34):
            ax.text(
                0.5,
                0.5,
                f"T={T}: only {n_present} words seen",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        return

    A = graph.build_adjacency_matrix()
    H = H_full[present]
    words_p = [word for word, ok in zip(words, present) if ok]
    A_p = A[np.ix_(present, present)]

    A_draw = A_p.copy()
    if overlay_graphs:
        for gname, og in overlay_graphs.items():
            Ao = og.build_adjacency_matrix()
            Ao_p = Ao[np.ix_(present, present)]
            A_draw = np.maximum(A_draw, Ao_p)

    pca_dirs = compute_top_k_pca(H, k=4)
    projected = H @ pca_dirs.T
    n_missing = len(words) - n_present
    tag = "" if n_missing == 0 else f" ({n_missing} unseen)"
    for ax, (lo, hi) in zip(
        (ax_pc12, ax_pc34),
        [(0, 2), (2, 4)],
        strict=True,
    ):
        _draw_scatter(
            ax,
            projected[:, lo:hi],
            words_p,
            A_draw,
            pc_x=lo + 1,
            pc_y=lo + 2,
            title=f"T = {T}{tag}",
            show_labels=show_labels,
        )


def plot_pca_snapshots(
    result: PCAResult,
    graph: UndirectedGraph,
    out_path: str | Path,
    words: Sequence[str] = WORDS,
    overlay_graphs: Mapping[str, UndirectedGraph] | None = None,
) -> Path:
    """Paper-style PCA snapshots: PC1/2 and PC3/4 for each T."""

    Ts = sorted(result.class_means_by_T)
    fig, axes = plt.subplots(2, len(Ts), figsize=(4.2 * len(Ts), 8.5), squeeze=False)

    for col, T in enumerate(Ts):
        _fill_pca_snapshot_column(
            axes[0, col],
            axes[1, col],
            T,
            result,
            graph,
            words,
            overlay_graphs,
            show_labels=(col == 0),
        )

    fig.suptitle(
        f"Llama class-mean PCA: true {result.true_graph}, layer {result.layer}, "
        f"Nw={result.window}",
        fontsize=11,
        y=1.00,
    )
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dirichlet_energy_overlay(result: PCAResult, out_path: str | Path) -> Path:
    """Plot normalized Dirichlet energy under all candidate graph adjacencies."""

    colors = {
        "grid": "#1976D2",
        "ring": "#C62828",
        "chain": "#2E7D32",
        "star": "#6A1B9A",
        "uniform": "#616161",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for graph_name, (Ts, energies) in result.energy_by_graph.items():
        if len(Ts) == 0:
            continue
        ax.plot(
            Ts,
            energies,
            "o-",
            lw=2,
            ms=4,
            color=colors.get(graph_name, "gray"),
            label=f"E_{graph_name}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Context length T")
    ax.set_ylabel("Normalized Dirichlet energy")
    ax.set_title(f"Representation energy by graph: true {result.true_graph}")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def sequence_activations(
    model,
    sequence: Sequence[str],
    layer: int = DEFAULT_LAYER,
    words: Sequence[str] = WORDS,
    token_map: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Run one forward pass and return residual-stream activations."""

    import torch

    vocab = validate_vocabulary(words)
    tok_map = dict(build_token_map(model, vocab) if token_map is None else token_map)
    bos = model.tokenizer.bos_token_id
    input_ids = [bos] + [tok_map[word] for word in sequence]
    if len(input_ids) > model.cfg.n_ctx:
        input_ids = input_ids[: model.cfg.n_ctx]

    tokens = torch.tensor([input_ids], dtype=torch.long).to(model.cfg.device)
    hook_name = f"blocks.{layer}.hook_resid_pre"
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    acts = cache[hook_name][0, 1:, :].detach().cpu().float().numpy()
    if len(acts) != min(len(sequence), model.cfg.n_ctx - 1):
        raise RuntimeError(
            f"Activation length {len(acts)} does not match tokenized sequence length."
        )
    return acts


def pca_result_for_sequence(
    sequence: Sequence[str],
    activations: np.ndarray,
    true_graph: str,
    graphs: Mapping[str, UndirectedGraph],
    layer: int,
    seq_len: int,
    window: int,
    snapshot_Ts: Sequence[int],
    energy_Ts: Sequence[int],
    words: Sequence[str] = WORDS,
) -> PCAResult:
    """Compute class-mean snapshots and graph-energy curves for one sequence."""

    vocab = validate_vocabulary(words)
    class_means_by_T: dict[int, np.ndarray] = {}
    present_by_T: dict[int, np.ndarray] = {}
    for T in snapshot_Ts:
        if T - window < 0 or T > len(activations):
            continue
        H, present = class_means_sliding(activations, sequence, vocab, T, window=window)
        class_means_by_T[int(T)] = H
        present_by_T[int(T)] = present

    energy_by_graph = {}
    for graph_name, graph in graphs.items():
        Ts, energies = dirichlet_energy_curve(
            activations,
            sequence,
            vocab,
            graph.build_adjacency_matrix(),
            Ts=energy_Ts,
            window=window,
        )
        energy_by_graph[graph_name] = (Ts, energies)

    return PCAResult(
        true_graph=true_graph,
        layer=layer,
        seq_len=seq_len,
        window=window,
        snapshot_Ts=tuple(int(T) for T in snapshot_Ts),
        energy_Ts=tuple(int(T) for T in energy_Ts),
        class_means_by_T=class_means_by_T,
        present_by_T=present_by_T,
        energy_by_graph=energy_by_graph,
    )


def average_pca_results(results: Sequence[PCAResult]) -> PCAResult:
    """Average snapshot means and energy curves across seeds for one true graph."""

    if not results:
        raise ValueError("Need at least one PCAResult to average.")

    first = results[0]
    class_sums: dict[int, np.ndarray] = {}
    present_counts: dict[int, np.ndarray] = {}
    for result in results:
        for T, H in result.class_means_by_T.items():
            present = result.present_by_T[T]
            if T not in class_sums:
                class_sums[T] = np.zeros_like(H)
                present_counts[T] = np.zeros(H.shape[0], dtype=int)
            class_sums[T][present] += H[present]
            present_counts[T][present] += 1

    class_means_by_T = {}
    present_by_T = {}
    for T, H_sum in class_sums.items():
        counts = present_counts[T]
        present = counts > 0
        H = np.zeros_like(H_sum)
        H[present] = H_sum[present] / counts[present, None]
        class_means_by_T[T] = H
        present_by_T[T] = present

    grouped_energy = defaultdict(list)
    for result in results:
        for graph_name, (Ts, energies) in result.energy_by_graph.items():
            grouped_energy[graph_name].append((Ts, energies))

    energy_by_graph = {}
    for graph_name, curves in grouped_energy.items():
        Ts0 = curves[0][0]
        if not all(np.array_equal(Ts, Ts0) for Ts, _ in curves):
            raise ValueError(f"Energy T grids differ for {graph_name}.")
        stacked = np.stack([energies for _, energies in curves], axis=0)
        energy_by_graph[graph_name] = (Ts0, stacked.mean(axis=0))

    return PCAResult(
        true_graph=first.true_graph,
        layer=first.layer,
        seq_len=first.seq_len,
        window=first.window,
        snapshot_Ts=first.snapshot_Ts,
        energy_Ts=first.energy_Ts,
        class_means_by_T=class_means_by_T,
        present_by_T=present_by_T,
        energy_by_graph=energy_by_graph,
    )


def save_pca_npz(result: PCAResult, out_path: str | Path) -> Path:
    """Persist PCA arrays for later inspection/replotting."""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "true_graph": np.array(result.true_graph),
        "layer": np.array(result.layer),
        "seq_len": np.array(result.seq_len),
        "window": np.array(result.window),
        "snapshot_Ts": np.array(result.snapshot_Ts),
        "energy_Ts": np.array(result.energy_Ts),
    }
    for T, H in result.class_means_by_T.items():
        payload[f"class_means_T{T}"] = H
        payload[f"present_T{T}"] = result.present_by_T[T]
    for graph_name, (Ts, energies) in result.energy_by_graph.items():
        payload[f"energy_Ts_{graph_name}"] = Ts
        payload[f"energies_{graph_name}"] = energies
    np.savez(out, **payload)
    return out


def load_pca_npz(path: str | Path) -> PCAResult:
    """Load ``PCAResult`` written by :func:`save_pca_npz`."""

    path = Path(path)
    z = np.load(path, allow_pickle=True)
    true_graph = str(z["true_graph"].item())
    layer = int(z["layer"].item())
    seq_len = int(z["seq_len"].item())
    window = int(z["window"].item())
    snapshot_Ts = tuple(int(x) for x in z["snapshot_Ts"])
    energy_Ts = tuple(int(x) for x in z["energy_Ts"])

    class_means_by_T: dict[int, np.ndarray] = {}
    present_by_T: dict[int, np.ndarray] = {}
    prefix_means = "class_means_T"
    prefix_present = "present_T"
    for key in z.files:
        if key.startswith(prefix_means):
            t_str = key[len(prefix_means) :]
            class_means_by_T[int(t_str)] = np.asarray(z[key], dtype=float)
        elif key.startswith(prefix_present):
            t_str = key[len(prefix_present) :]
            present_by_T[int(t_str)] = np.asarray(z[key], dtype=bool)

    energy_by_graph: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in z.files:
        if key.startswith("energies_"):
            gname = key[len("energies_") :]
            ts_key = f"energy_Ts_{gname}"
            if ts_key in z.files:
                energy_by_graph[gname] = (
                    np.asarray(z[ts_key]),
                    np.asarray(z[key]),
                )

    return PCAResult(
        true_graph=true_graph,
        layer=layer,
        seq_len=seq_len,
        window=window,
        snapshot_Ts=snapshot_Ts,
        energy_Ts=energy_Ts,
        class_means_by_T=class_means_by_T,
        present_by_T=present_by_T,
        energy_by_graph=energy_by_graph,
    )


def write_pca_evolution_gif(
    result: PCAResult,
    graph: UndirectedGraph,
    out_path: str | Path,
    words: Sequence[str] = WORDS,
    overlay_graphs: Mapping[str, UndirectedGraph] | None = None,
    duration_ms: int = 900,
    figsize: tuple[float, float] = (10.8, 4.85),
    dpi: int = 125,
) -> Path:
    """Animate snapshot PCA panels (one frame per stored context length ``T``) as a GIF.

    Each frame is laid out **horizontally**: PC1/2 (left), PC3/4 (right).  Requires Pillow.
    Intended for README / slides; regenerate with ``scripts/make_pca_gif.py``.
    """

    import io

    from PIL import Image

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Ts = sorted(result.class_means_by_T)
    frames: list[Image.Image] = []
    for T in Ts:
        fig, (ax_pc12, ax_pc34) = plt.subplots(1, 2, figsize=figsize)
        _fill_pca_snapshot_column(
            ax_pc12,
            ax_pc34,
            T,
            result,
            graph,
            words,
            overlay_graphs,
            show_labels=True,
        )
        fig.suptitle(
            f"Class-mean PCA · true {result.true_graph} · layer {result.layer} · "
            f"Nw={result.window}",
            fontsize=11,
        )
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        with Image.open(buf) as im:
            frames.append(im.convert("RGB").copy())

    if not frames:
        raise ValueError("No snapshot times in PCAResult; cannot build GIF.")

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for f in frames:
        f.close()
    return out


def run_pca_analysis(
    config: ExperimentConfig = DEFAULT_CONFIG,
    true_graphs: Sequence[str] | None = None,
    layer: int = DEFAULT_LAYER,
    snapshot_Ts: Sequence[int] = DEFAULT_SNAPSHOT_T,
    energy_Ts: Sequence[int] = DEFAULT_ENERGY_T,
    window: int = DEFAULT_WINDOW,
) -> list[Path]:
    """Run PCA analysis for pure-graph conditions and write plots/NPZ files."""

    from .llm_inference import load_model

    graph_map = build_candidate_graphs()
    selected_true_graphs = tuple(config.true_graphs if true_graphs is None else true_graphs)
    model = load_model(config.model_name, device=config.device, dtype=config.dtype)
    token_map = build_token_map(model)

    written: list[Path] = []
    for true_graph in selected_true_graphs:
        print(f"[pca] true_graph={true_graph}")
        per_seed = []
        graph = graph_map[true_graph]
        for idx, seed in enumerate(config.seeds, start=1):
            print(f"  seed={seed} ({idx}/{len(config.seeds)}): generating sequence...")
            sequence = generate_sequence(graph, seq_len=config.seq_len, seed=seed)
            print(f"  seed={seed}: extracting layer {layer} activations...")
            acts = sequence_activations(
                model,
                sequence,
                layer=layer,
                token_map=token_map,
            )
            print(f"  seed={seed}: computing PCA snapshots and energy curves...")
            per_seed.append(
                pca_result_for_sequence(
                    sequence=sequence[: len(acts)],
                    activations=acts,
                    true_graph=true_graph,
                    graphs=graph_map,
                    layer=layer,
                    seq_len=config.seq_len,
                    window=window,
                    snapshot_Ts=snapshot_Ts,
                    energy_Ts=energy_Ts,
                )
            )

        print(f"[pca] true_graph={true_graph}: averaging {len(per_seed)} seed results...")
        result = average_pca_results(per_seed)
        print(f"[pca] true_graph={true_graph}: writing plots and NPZ...")
        written.append(
            plot_pca_snapshots(
                result,
                graph=graph,
                out_path=Path(config.output_dir) / f"pca_snapshots_{true_graph}.png",
            )
        )
        written.append(
            plot_dirichlet_energy_overlay(
                result,
                out_path=Path(config.output_dir) / f"dirichlet_energy_{true_graph}.png",
            )
        )
        written.append(save_pca_npz(result, Path(config.output_dir) / f"pca_{true_graph}.npz"))
    return written


def run_pca_analysis_mixed(
    config: ExperimentConfig = DEFAULT_CONFIG,
    mix_ratios: Mapping[str, float] | None = None,
    mix_name: str = "mix",
    layer: int = DEFAULT_LAYER,
    snapshot_Ts: Sequence[int] = DEFAULT_SNAPSHOT_T,
    energy_Ts: Sequence[int] = DEFAULT_ENERGY_T,
    window: int = DEFAULT_WINDOW,
) -> list[Path]:
    """Run PCA analysis for a mixed-graph condition and write plots/NPZ files.

    ``mix_ratios`` maps graph name → weight, e.g. ``{"grid": 70.0, "ring": 30.0}``.
    The weights are normalised internally; they can be percentages or fractions.

    The PCA snapshot plot draws edges for all mixed graphs as separate overlays
    so both structures are visible in the projected class-mean space.
    """
    from .llm_inference import load_model
    from .sequence_generation import generate_mixed_sequence

    if mix_ratios is None:
        raise ValueError("mix_ratios is required for run_pca_analysis_mixed.")

    graph_map = build_candidate_graphs()
    # Normalise weights to fractions so generate_mixed_sequence is happy.
    total_w = sum(mix_ratios.values())
    norm_ratios = {k: v / total_w for k, v in mix_ratios.items()}

    model = load_model(config.model_name, device=config.device, dtype=config.dtype)
    token_map = build_token_map(model)

    # Dominant graph = highest-weight graph in the mix (used as primary for plot title).
    dominant = max(norm_ratios, key=lambda k: norm_ratios[k])
    dominant_graph = graph_map[dominant]
    # All mixed graphs get their edges overlaid (skip dominant since it's the primary).
    overlay_graphs = {
        k: graph_map[k] for k in norm_ratios if k != dominant
    }

    per_seed: list[PCAResult] = []
    for idx, seed in enumerate(config.seeds, start=1):
        rng = np.random.default_rng(seed)
        print(f"  seed={seed} ({idx}/{len(config.seeds)}): generating mixed sequence {dict(mix_ratios)}...")
        sequence, _ = generate_mixed_sequence(
            graph_map,
            norm_ratios,
            seq_len=config.seq_len,
            seed=seed,
        )
        print(f"  seed={seed}: extracting layer {layer} activations...")
        acts = sequence_activations(model, sequence, layer=layer, token_map=token_map)
        del rng
        print(f"  seed={seed}: computing PCA snapshots and energy curves...")
        per_seed.append(
            pca_result_for_sequence(
                sequence=sequence[: len(acts)],
                activations=acts,
                true_graph=mix_name,
                graphs=graph_map,
                layer=layer,
                seq_len=config.seq_len,
                window=window,
                snapshot_Ts=snapshot_Ts,
                energy_Ts=energy_Ts,
            )
        )

    print(f"[pca] {mix_name}: averaging {len(per_seed)} seed results...")
    result = average_pca_results(per_seed)
    print(f"[pca] {mix_name}: writing plots and NPZ...")
    out_dir = Path(config.output_dir)
    written: list[Path] = []
    written.append(
        plot_pca_snapshots(
            result,
            graph=dominant_graph,
            out_path=out_dir / f"pca_snapshots_{mix_name}.png",
            overlay_graphs=overlay_graphs,
        )
    )
    written.append(
        plot_dirichlet_energy_overlay(
            result,
            out_path=out_dir / f"dirichlet_energy_{mix_name}.png",
        )
    )
    written.append(save_pca_npz(result, out_dir / f"pca_{mix_name}.npz"))
    return written
