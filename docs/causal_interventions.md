# Causal Graph-Representation Interventions

This pipeline extends the secondary graph experiments with activation patching,
graph-difference steering, and attention-head ablation.  It reuses the existing
16-word vocabulary, deterministic graph definitions, random-walk generator, BOS
plus word-token prompt format, TransformerLens model loader, and token map from
`src/secondary_experiments/`.

## Clean/corrupt pair generation

Clean and corrupt prompts are generated from different graph families but end at
the same current node.  Because the graph hypotheses are undirected, the helper
generates a random walk starting at the desired final node with the existing
`UndirectedGraph.generate_sequence(...)` method and reverses it.  The reversed
walk is still valid under the same graph support, and the final token is aligned
across clean and corrupt contexts.

The prompt format is unchanged from the behavioral experiments:

```text
[BOS] apple bird ... current_word
```

The model is always evaluated at the final position, predicting the next graph
word.

## Primary metric

For a clean graph `G_clean`, corrupt graph `G_corrupt`, and current node `X`,
each output logit vector is scored as:

```text
graph_logit_diff =
  mean_logit(neighbors_G_clean(X)) - mean_logit(neighbors_G_corrupt(X))
```

For activation patching, the normalized effect is:

```text
(patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)
```

For steering target contexts, it is:

```text
(steered_metric - target_metric) / (source_metric - target_metric)
```

Rows with small denominators are retained but marked with
`normalization_usable=false` and `normalized_effect=null`.

## Seen versus held-out edges

The critical cache-control split is computed from observed transitions involving
the final token in the evaluation context:

- `seen_neighbors`: true graph neighbors of the final token whose edge appeared
  in either direction in the context.
- `heldout_neighbors`: true graph neighbors of the final token whose edge did
  not appear in the context.
- `non_neighbors`: controlled-vocabulary tokens that are not graph neighbors.

For clean/corrupt patching, the split is aligned to the corrupt evaluation
context.  This asks whether an intervention increases logits for graph neighbors
that the corrupt prompt itself did not locally observe.

## Activation patching

Example:

```bash
python experiments/activation_patching.py \
  --model meta-llama/Llama-3.1-8B \
  --clean_graph grid \
  --corrupt_graph ring \
  --num_pairs 500 \
  --context_length 128 \
  --seed 0 \
  --positions final \
  --output_dir results/patching/grid_vs_ring
```

By default, all layers are patched at `blocks.{layer}.hook_resid_post`.  You can
choose layers and position strategies:

```bash
python experiments/activation_patching.py \
  --clean_graph grid --corrupt_graph ring \
  --layers 16 17 18 19 20 21 22 23 24 25 26 27 28 \
  --positions final same_token_occurrences edge_observation_positions \
  --num_pairs 500 --context_length 128 --seed 0 \
  --output_dir results/patching/grid_vs_ring_positions
```

Supported activations are `resid_pre`, `resid_post`, `attn_out`, and `mlp_out`.
Residual post-layer patching is the recommended first pass.

Outputs:

- `config.json`
- `rows.jsonl`, one row per pair/layer/position intervention
- `summary.json`

Plotting:

```bash
python -m src.analysis.plot_patching \
  --input results/patching/grid_vs_ring/rows.jsonl \
  --out_dir results/patching/grid_vs_ring
```

## Steering

The steering vector is trained on contexts disjoint from evaluation contexts:

```text
v_source_minus_target[layer] =
  mean_resid(source contexts, layer, position)
  - mean_resid(target contexts, layer, position)
```

Example:

```bash
python experiments/steering.py \
  --model meta-llama/Llama-3.1-8B \
  --source_graph grid \
  --target_graph ring \
  --num_train_contexts 1000 \
  --num_eval_contexts 500 \
  --context_length 128 \
  --layers 20 21 22 23 24 25 26 27 28 \
  --alphas -5 -2 -1 -0.5 0 0.5 1 2 5 \
  --seed 0 \
  --output_dir results/steering/grid_minus_ring
```

Each evaluation pair includes:

- `target_plus_source_minus_target`: ring context plus `alpha * v_grid_minus_ring`.
- `source_minus_source_minus_target`: grid context minus `alpha * v_grid_minus_ring`.
- controls: real vector, norm-matched random vector, shuffled-label vector.

Optional no-context stress test:

```bash
python experiments/steering.py \
  --source_graph grid --target_graph ring \
  --num_train_contexts 1000 --num_eval_contexts 500 \
  --context_length 128 \
  --layers 20 21 22 23 24 25 26 27 28 \
  --no_context_eval \
  --output_dir results/steering/no_context_grid_minus_ring
```

Plotting:

```bash
python -m src.analysis.plot_steering \
  --input results/steering/grid_minus_ring/rows.jsonl \
  --out_dir results/steering/grid_minus_ring
```

## Attention-head ablation

After residual patching and steering identify candidate layers, the head script
scores attention heads by how much the final token attends to:

- previous occurrences of the same token;
- positions participating in observed transitions involving the final token.

It then ablates top candidate heads and measures graph-logit effects:

```bash
python experiments/head_ablation.py \
  --model meta-llama/Llama-3.1-8B \
  --clean_graph grid \
  --corrupt_graph ring \
  --num_pairs 200 \
  --context_length 128 \
  --layers 20 21 22 23 24 25 26 27 28 \
  --top_candidate_heads 32 \
  --seed 0 \
  --output_dir results/head_ablation/grid_vs_ring
```

## Recommended aggregate reporting

Run at least three seeds and multiple context lengths, then aggregate by
`layer`, `position_strategy`, `alpha`, `control`, and edge split.  Main tables
should report mean effect, standard error or bootstrap intervals, number of
prompt pairs, number of seeds, and denominator-exclusion counts.  The JSONL rows
include full prompt metadata so any aggregate can be regenerated.

