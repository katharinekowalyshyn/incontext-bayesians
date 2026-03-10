# In-context learning of representations can be explained by induction circuits

This repository contains minimal reproduction scripts for the blog post:

> **In-context learning of representations can be explained by induction circuits**
> Andy Arditi (Northeastern University)

The blog post responds to [Park et al., 2025](https://arxiv.org/abs/2501.00070), who find that when LLMs process random walks on a graph in-context, their token representations come to mirror the graph's connectivity structure. We offer a simpler mechanistic explanation: the task can be solved by **induction circuits**, and the geometric structure of representations is a byproduct of **previous-token mixing** within those circuits.

## Setup

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/). Scripts `01` and `02` require a CUDA-capable GPU with at least 48 GB of memory (for Llama-3.1-8B). `03_neighbor_mixing.py` runs on CPU only and has no GPU requirement.

```bash
# 1. Create and activate a virtual environment
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r requirements.txt

# 3. Log in to Hugging Face (required for Llama-3.1-8B access)
huggingface-cli login
```

Llama-3.1-8B is a gated model — you'll need to have accepted the license on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B) before running scripts `01` and `02`.

## Scripts

### `01_reproduce.py` — Figures 2 and 6

Reproduces the core results from Park et al.: a language model performing in-context learning on a grid random walk task.

```bash
python 01_reproduce.py
```

**Outputs** (in `results/reproduce/plots/`):

| File | Description |
|------|-------------|
| `accuracy_curve.{pdf,png,html}` | **Fig 2 left.** Accuracy (probability on valid next tokens) as a function of context length. |
| `pca_class_means.{pdf,png,html}` | **Fig 2 right.** PCA of the 16 class-mean activations at layer 26 (last 200 positions). |
| `bigram_pca.{pdf,png,html}` | **Fig 6.** Individual activations projected onto the same PCA directions. Fill color = current token, border color = previous token. |

### `02_ablation.py` — Figures 3 and 4

Tests the induction circuit hypothesis by ablating attention heads.

```bash
python 02_ablation.py
```

This script first identifies induction heads and previous-token heads using repeated random token sequences (Appendix A), then ablates the top-k heads of each type and measures the effect on accuracy and representations.

**Outputs** (in `results/ablation/plots/`):

| File | Description |
|------|-------------|
| `ablation_induction.{pdf,png,html}` | **Fig 3 left.** Accuracy curves when ablating top-k induction heads (k = 1, 2, 4, 8, 16, 32). |
| `ablation_prev_token.{pdf,png,html}` | **Fig 3 right.** Accuracy curves when ablating top-k previous-token heads. |
| `pca_baseline.{pdf,png,html}` | **Fig 4 left.** Class-mean PCA with no ablation (baseline). |
| `pca_induction_ablated.{pdf,png,html}` | **Fig 4 center.** Class-mean PCA with top-32 induction heads ablated. |
| `pca_prev_token_ablated.{pdf,png,html}` | **Fig 4 right.** Class-mean PCA with top-32 previous-token heads ablated. |

### `03_neighbor_mixing.py` — Figure 5

Demonstrates that a single round of previous-token (neighbor) mixing can explain the emergent grid structure in representations. **No model or GPU required.**

```bash
python 03_neighbor_mixing.py
```

**Outputs** (in `results/neighbor_mixing/plots/`):

| File | Description |
|------|-------------|
| `before_mixing.{pdf,png,html}` | **Fig 5 left.** PCA of 16 random Gaussian vectors in R^4096. |
| `after_mixing.{pdf,png,html}` | **Fig 5 right.** PCA after applying one round of neighbor mixing: each embedding is updated by adding the mean of its grid neighbors' embeddings. |

## The Grid Task

The task uses a 4×4 grid of common English words:

```
 apple  bird   car    egg
 house  milk   plane  opera
 box    sand   sun    mango
 rock   math   code   phone
```

A random walk produces a sequence of words by moving to adjacent cells (up/down/left/right). The model is given this sequence and must predict valid next words at each position. A "correct" prediction is one that places probability on tokens corresponding to grid neighbors of the current word.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── utils.py                 # Shared utilities (Grid, model loading, PCA, plotting)
├── 01_reproduce.py          # Fig 2 (accuracy + PCA) and Fig 6 (bigram PCA)
├── 02_ablation.py           # Fig 3 (ablation accuracy) and Fig 4 (ablation PCA)
├── 03_neighbor_mixing.py    # Fig 5 (toy model of neighbor mixing)
└── results/
    ├── reproduce/
    │   ├── data/            # Cached activations, accuracies, sequences
    │   └── plots/           # PDF, PNG, and interactive HTML figures
    ├── ablation/
    │   ├── data/            # Cached head scores, ablation accuracies, PCA data
    │   └── plots/
    └── neighbor_mixing/
        ├── data/            # Cached mixing projections
        └── plots/
```

## Reproducibility

All scripts use `set_seed(42)` for deterministic results. The 16 accuracy curves use uniform initialization (one sequence starting at each grid position) to ensure all positions are represented.

- **Model:** `meta-llama/Llama-3.1-8B` via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- **Analysis layer:** 26 (residual stream pre-attention)
- **Sequence length:** 1,400 tokens
- **PCA lookback:** last 200 positions

## Citation

If you use this code, please cite:

```bibtex
TODO
```

## License

MIT
