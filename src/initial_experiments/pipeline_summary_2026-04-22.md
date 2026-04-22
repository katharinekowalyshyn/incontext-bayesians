# Representation-level pipeline results — 2026-04-22

**Context.** Milestones 7 and 8 of the Checkpoint-2 plan call for
computing residual-stream Dirichlet energy and PCA snapshots of
Llama-3.1-8B class-mean activations for every vocabulary condition.
This document summarises the run is done as of 2026-04-22 at ~10am
and the follow-up analysis that the Baseline (Tim, M2) and
Upgrade (Dan, M3–M4) fits now need to plug into.

## What I ran

`scripts/run_pca_pipeline.sh` (Llama-3.1-8B, layer 26, 8 walks × 1400 tokens
per condition, paper `Nw=50` sliding window, snapshots at T ∈ {200, 400, 1400}).
Total wall-clock: **~4 minutes** on one GPU. Pipeline processed:

1. `grid`              (4×4 grid, 16 words — replication of Park Fig 2/4)
2. `months_natural`    (12-month ring, natural Jan→Dec order)
3. `months_permuted`   (12-month ring with the Park et al. conflicting order)
4. `neutral_disjoint`  (16-word semantically neutral ring, disjoint from grid)
5. `neutral_overlap`   (12-word ring, 3 tokens shared with grid vocab)

…followed by the cross-condition overlay.

## Key numbers (mean ± 1σ over 8 walks, T = 1400)

| condition          | E_G @ 1400  | Llama acc @ 1400 | r(E_G, acc) across walks | ΔE_G (60 → 1400) |
|---------------------|:-----------:|:----------------:|:------------------------:|:-----------------:|
| `grid`              | 0.915 ± 0.059 | 0.889 ± 0.023   |   +0.30                  |  **+0.030**  (smallest) |
| `months_natural`    | **0.702 ± 0.016** | 0.817 ± 0.030 | +0.60                  |  **+0.109** (largest) |
| `months_permuted`   | 0.997 ± 0.032 | **0.977 ± 0.005** | +0.54                  |  +0.107           |
| `neutral_disjoint`  | 0.828 ± 0.076 | 0.949 ± 0.029   |   +0.72                  |  +0.043           |
| `neutral_overlap`   | 0.880 ± 0.035 | 0.957 ± 0.013   |   +0.96                  |  +0.062           |

Raw per-walk arrays are cached in
`pca_pipeline_{condition}.npz` and include per-walk Dirichlet energies,
per-walk `P(next ∈ valid nbrs)` curves, class means at T ∈ {200, 400, 1400},
and per-word "present in window" masks.

## What the data say (quick read)

1. **Representations do reorganise over context** in every condition
   (ΔE_G > 0 everywhere), consistent with the belief account of §1.
2. **The biggest E_G drops are in the ring conditions** (≈ 0.11 for both
   months-natural and months-permuted) and the **smallest in `grid`** (0.03).
   The 12-node rings appear easier for Llama to re-represent as
   in-context graphs than the 16-node 4×4 grid, which is the opposite
   of what one would expect from Kolmogorov complexity alone (C_MST
   favours the ring by 60 vs 44 bits, §4.3). **We should flag this to Dan
   before he locks in the complexity proxy.**
3. **E_G @ 1400 ordering is informative.** `months_natural` is the
   lowest-energy representation (prior aligned with structure); `months_permuted`
   is the highest (prior directly conflicts). This is a clean
   representation-level analogue of the behavioural finding in §5.1 Fig 1.
4. **Cross-walk E_G / accuracy correlations are positive in every
   condition.** The belief account predicts *negative* (more aligned reps
   → better accuracy), so this is either (a) a sample-size artefact with
   only 8 walks, (b) an accuracy-saturation artefact (acc > 0.95 in three
   conditions leaves no variance for E_G to explain), or (c) a real effect.
   **Needs re-checking with more walks.**

## Artefacts generated (per condition)

- `pca_snapshots_{cond}_paper.png` — Fig-2 style per-snapshot PCA + spectral overlay.
- `pca_before_during_after_{cond}.png` — shared-basis 3-phase trajectory.
- `dirichlet_energy_{cond}.png` — E_G vs T with ±1σ band (grid also overlays E on the Hamiltonian-ring adjacency).
- `energy_vs_accuracy_{cond}.png` — per-walk scatter + per-T correlation bars.
- `pca_pipeline_{cond}.npz` — all raw arrays for replotting.

Overlays:
- `dirichlet_energy_overlay.png`
- `energy_vs_accuracy_overlay.png`
- `pca_months_permuted_fig3.png` — Park et al. Fig 3 reproduction.

Earlier (cached) artefacts that remain usable:
- `pca_after_pc1234.png`, `spectral_embedding_reference.png`.

## Code changes that went in alongside this run

`src/initial_experiments/pca_analysis.py`:

- `CONDITIONS` registry mirroring `vocabulary_tl_experiment.py`.
- `run_with_model(..., n_walks=N)` — multi-walk averaging with ±1σ bands.
- Per-position LLM `P(next ∈ valid nbrs)` extracted in the same forward pass.
- `plot_energy_vs_accuracy` — per-walk scatter + per-T Pearson/Spearman.
- `run_overlay(sigmoid_json=…)` — cross-condition E_G overlay, optionally annotated with the Baseline sigmoid inflection N*.
- `run_layer_sweep(condition, layers, …)` — per-layer E_G curves (M7 "each layer").
- `n_star_from_sigmoid(b, γ, α)` helper.

`src/initial_experiments/bayesian_model.py`:

- **`mst_log_prior(adjacencies, lam, b0)`** — the Checkpoint-2 §4.3 Upgrade prior (`C_MST(G) = |E_MST| · ⌈log₂|V|⌉` bits). Verified: grid & 16-ring = 60 bits, 12-ring = 44 bits; λ=0 reduces to flat prior as the §4.2 unit-test contract requires.
- `mst_complexity_bits(A)` and `_mst_edge_count(A)` helpers.
- `dirichlet_energy_log_prior` is now clearly labelled as **APPENDIX-ONLY** (§4.3 "alternative proxies"), not the primary Upgrade.

`scripts/run_pca_pipeline.sh` — nohup-friendly driver; logs per condition.

---

## What we need from Tim (Milestone 2, due Apr 26)

The Baseline sigmoid fit is the only thing that keeps Katie's
representation-level analysis from directly speaking to the
"belief vs. circuitry" hypothesis in §1.

1. **Baseline sigmoid-fit script** — reparameterised Adam, 20 restarts,
   bootstrap CIs, as per §3.4.
2. **Output format for the fitted parameters.** Please emit the per-cell
   fit as a JSON keyed by condition (optionally with a `rho` subkey):

   ```json
   {
     "grid":             {"b": -3.5, "gamma": 0.42, "alpha": 0.62, "q": 0.90},
     "months_natural":   {"b": -1.1, "gamma": 0.51, "alpha": 0.71, "q": 0.84},
     "months_permuted":  {"b": -4.0, "gamma": 0.35, "alpha": 0.55, "q": 0.98},
     "neutral_disjoint": {"b": -2.2, "gamma": 0.48, "alpha": 0.66, "q": 0.96},
     "neutral_overlap":  {"b": -2.0, "gamma": 0.47, "alpha": 0.65, "q": 0.96}
   }
   ```

   `(b, γ, α)` is all we need to compute `N* = (−b/γ)^(1/(1−α))`.
   Drop the JSON into the repo and run:

   ```bash
   python src/initial_experiments/pca_analysis.py \
          --overlay --sigmoid-json path/to/fit.json
   ```

   …and the `dirichlet_energy_overlay.png` will be re-drawn with a
   dotted `axvline` at N* per condition — this is the §5.4 /
   Milestone-8 "overlay Dirichlet-energy drop with sigmoid inflection"
   deliverable.
3. **Per-walk fit (optional but ideal).** The NPZ files include
   per-walk P(next ∈ nbrs) curves (`acc_per_walk`, shape [8, len(Ts)]),
   one row per walk. If your fit script can consume those directly, we
   can report per-walk `N*` distributions in the bootstrap CI rather
   than fitting to the per-T mean.
4. **Which accuracy series to fit.** Our `acc_per_walk` uses a
   trailing-window mean of `P(next ∈ valid nbrs)`, which matches Nw=50.
   `src/initial_experiments/results/vocabulary_tl/*.json` from the
   full vocabulary experiment uses point-samples at
   `{50, 100, …, 2000}`. You should pick one of these (I'd suggest
   the `vocabulary_tl` JSON since it's the series the §5 figures are
   already drawn from) and document the choice.

**Open question for Tim:** the Llama accuracy curves *saturate*
differently across conditions (0.82 for months_natural vs 0.98 for
months_permuted at T=1400). Your reparameterisation of `q` via
`q = p₀ + (1 − p₀)σ(q̃)` should handle that, but the `p₀` prior term
needs to be estimated per-condition since the pre-transition baseline
is vocabulary-dependent. Please estimate `p₀` from the first ~50
tokens of each condition rather than using a single shared value.

## What we need from Dan (Milestones 3 & 4, due Apr 27–28)

1. **Lock the C_MST proxy.** The helper is in the repo — use
   `bayesian_model.mst_log_prior` (not `dirichlet_energy_log_prior`,
   that's the appendix alternative). Current values:

   | graph | \|E_MST\| | ⌈log₂\|V\|⌉ | C_MST (bits) |
   |---|:---:|:---:|:---:|
   | 4×4 grid       | 15 | 4 | 60 |
   | 16-node ring   | 15 | 4 | 60 |
   | 12-node ring   | 11 | 4 | 44 |

   **Note** the 4×4 grid and the 16-node ring are *tied* under this
   proxy (both trees have n−1=15 edges on 16 nodes). If the Upgrade
   is supposed to prefer one over the other, C_MST alone can't do it —
   consider that when you design the ablation.

2. **Per-graph evidence parameters (γ_k, α_k).** The changelog entry
   in CP2 §A item 3 commits to these. The NPZ from this run contains
   all the data you need; no further LLM inference required.

3. **Upgrade fit script** (M4). Reuse Tim's optimiser scaffold from
   M2; the only diff is the joint `(b₀, λ, γ_grid, γ_ring, α_grid, α_ring)`
   parameter vector and the per-graph accuracy relabelling.
4. **Unit tests per §4.2.** Please add:
   - `test_lambda_zero_matches_baseline`: with `lam=0`, `mst_log_prior` is flat, so the joint MLE should match the Baseline fit on each pure-graph cell.
   - `test_matched_per_graph_params_matches_baseline`: with `γ_grid = γ_ring` and `α_grid = α_ring`, the Upgrade collapses to §3 on pure-graph cells.

## What Katie still owes (Milestones 7–9)

- **M7 per-layer sweep.** The sweep entry point is in place
  (`pca_analysis.py --layer-sweep --layers 8,16,20,24,26,28`).
  Will run after this writeup — ~1 min per condition × 5 conditions ≈ 5 min GPU.
- **M8 N\* overlay.** Blocked on Tim's M2 JSON (see above). Rendering
  is already wired into `run_overlay(sigmoid_json=…)`; pure NPZ → PNG
  once the fit lands.
- **M9 bootstrap CIs.** NPZ now stores per-walk arrays; any
  percentile-based CI procedure can consume them directly.

## Open decisions before the Final Report

1. The **grid's ΔE_G drop is small (0.03)** while the rings see 0.1.
   Either (a) this is a real finding and belongs in §5, (b) it's a
   layer-choice artefact — we hardcoded layer 26; other layers may
   show a cleaner drop. The layer sweep will tell us.
2. **Positive cross-walk correlation r(E_G, acc)** in all conditions.
   If this persists at n_walks=16, it contradicts the belief account's
   naive prediction and needs an explanation in §5.4.
3. **12-node vs 16-node graphs make the P(next ∈ nbrs) accuracy scales
   incomparable** (chance=0.17 vs 0.25). When Tim reports MSE gap
   numbers between Baseline and Upgrade, we should report them
   *per-condition* and not average across conditions with different
   graph sizes.

## Reference: to reproduce / extend

```bash
# full pipeline (Katie's side; writes everything above)
nohup bash scripts/run_pca_pipeline.sh \
    > logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 & disown

# bigger bootstrap (after freezing the final writeup)
N_WALKS=16 SEQ_LEN=2000 nohup bash scripts/run_pca_pipeline.sh \
    > logs/pipeline_n16.log 2>&1 & disown

# per-layer E_G for one condition
python src/initial_experiments/pca_analysis.py \
    --layer-sweep --condition grid --n-walks 8 \
    --layers 8,16,20,24,26,28

# re-render overlay with Tim's sigmoid fit (once it exists)
python src/initial_experiments/pca_analysis.py \
    --overlay --sigmoid-json path/to/fit.json
```
