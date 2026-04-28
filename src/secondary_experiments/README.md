# Secondary Graph Baseline Experiments

All baselines predict over the same controlled 16-word vocabulary.

## Ideal Bayesian Graph Observer

The ideal Bayesian graph observer knows the candidate graph family:
`grid`, `ring`, `chain`, `star`, and `uniform`.  It infers a posterior over
which candidate graph generated the observed context, then uses that posterior
to produce a next-token distribution.  This tests whether the LLM behaves like
it is performing global latent-structure inference over a known hypothesis set.

## Bayesian Edge Learner

The Bayesian edge learner does not know the candidate graph shapes.  It learns
an unknown undirected graph edge-by-edge.  Seeing either `u -> v` or `v -> u`
increases the inferred edge strength for the unordered pair `{u, v}`.  This
tests whether the LLM behaves like it is learning graph adjacency locally but
symmetrically, rather than identifying one global graph hypothesis.

## Cache / Bigram Baseline

The cache baseline is a directed local transition-count model.  It predicts
what followed the current token earlier in the same context.  Unlike the edge
learner, observing `u -> v` affects future predictions after `u`, but does not
symmetrically affect predictions after `v`.  This tests whether the LLM can be
explained by local copying or directed bigram induction.

## Semantic Prior

The semantic-prior baseline queries the LLM with no graph/random-walk context:
`[BOS] current_word`.  It measures the model's pretrained preferences among
the same 16 graph words before any in-context graph learning.  This tests how
much of the LLM's behavior is explained by word associations rather than
context-induced graph structure.

## Main Question

As context length increases, does the LLM's next-token distribution become
closer to global graph inference, learned undirected edges, local directed
transition copying, or pretrained semantic priors?

## Running Experiments

Use `--device cpu` to force CPU.  Use `--seeds 0` for a quick one-seed run;
the default is 16 seeds.

### One Graph, Full LLM Analysis

```bash
python -m src.secondary_experiments.run_experiment \
  --true-graphs grid \
  --device cpu \
  --seeds 0 \
  --out-dir src/secondary_experiments/results/grid_full
```

Outputs include:

- `llm_results.json`
- `semantic_prior.json`
- `posterior_grid.png`
- `llm_vocab_mass_grid.png`
- `neighbor_probability_grid.png`
- `kl_to_llm_grid.png`
- `mse_to_llm_grid.png`
- `corr_to_llm_grid.png`
- `closest_baseline_kl_grid.png`

### Baseline-Only Check

```bash
python -m src.secondary_experiments.run_experiment \
  --skip-llm \
  --true-graphs grid \
  --seeds 0 \
  --out-dir src/secondary_experiments/results/grid_baseline_check
```

If `llm_results.json` already exists in the output folder, `--skip-llm` uses
that cached LLM data for plotting instead of recomputing inference.

### Mixed Transition Experiment

```bash
python -m src.secondary_experiments.run_experiment \
  --mix grid:80,ring:20 \
  --mix-name mix_grid80_ring20 \
  --device cpu \
  --seeds 0 \
  --out-dir src/secondary_experiments/results/mix_grid80_ring20
```

The mix schedule is deterministic and balanced across transitions, not one
contiguous block.  For `grid:80,ring:20`, about every fifth transition is from
the ring.

### Sequential Runs

```bash
python -m src.secondary_experiments.run_all \
  --graphs grid ring chain uniform \
  --device cpu \
  --seeds 0 \
  --out-root src/secondary_experiments/results/sequential
```

Run only mixes:

```bash
python -m src.secondary_experiments.run_all \
  --no-pure \
  --mixes grid:80,ring:20 \
  --device cpu \
  --seeds 0 \
  --out-root src/secondary_experiments/results/mixes
```

Run a rho ladder from grid to ring:

```bash
python -m src.secondary_experiments.run_all \
  --rho-ladder \
  --ladder-graphs grid ring \
  --device cpu \
  --seeds 0 \
  --out-root src/secondary_experiments/results/rho_ladder
```

`run_all.py` launches each experiment in a subprocess, so it reloads the model
for each condition.  This is memory-safe but not model-load-time efficient.

### PCA Analysis

PCA is separate from the behavioral analysis.

```bash
python -m src.secondary_experiments.run_pca \
  --true-graphs grid \
  --device cpu \
  --seeds 0 \
  --out-dir src/secondary_experiments/results/grid_full
```

Default PCA snapshots are `T=200,400,1400`; default energy curve points are
`T=60,80,100,150,200,300,400,600,800,1000,1200,1400`.

### Mixture-of-Baselines Analysis

After a full LLM run, fit a Bayesian mixture over the four baselines:

```bash
python -m src.secondary_experiments.run_mixture \
  --input src/secondary_experiments/results/grid_full/llm_results.json \
  --out-dir src/secondary_experiments/results/grid_full
```

This writes:

- `mixture_analysis.json`
- `mixture_weights_by_context.png`
- `mixture_kl_comparison.png`
- `mixture_weights_smooth_context.png`

The per-context model fits one simplex weight vector at each context length.
The smooth model fits `lambda(t) = softmax(a + b * log(t))` across all rows.
