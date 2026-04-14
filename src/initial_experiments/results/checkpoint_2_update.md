# Checkpoint 2 Update

## Overview

Three initial experiments were run on Llama 3.1 8B (base) to test whether the model can learn competing graph structures in-context. The experiments probe two graph topologies — a 4x4 grid and a 12-node ring — under varying mixture ratios (ρ) and vocabulary conditions.

---

## Experiment 1: Mixing Experiment

**Setup:** Sequences interleave a 4x4 grid (arbitrary word vocabulary) and a 12-node ring (months of year) at ρ ∈ {0.0, 0.5, 1.0}. Accuracy is measured greedily via Ollama: the model's single next-token prediction is checked against valid graph neighbors.

**Results:**
- ρ=0 (pure grid): Accuracy rises to ~1.0 by ~500 tokens and holds. Clean, fast learning.
- ρ=0.5 (mixed): Grid accuracy remains near 1.0. Ring accuracy is near chance and highly noisy.
- ρ=1 (pure ring, months): Ring accuracy is erratic (0.25–0.75) with no convergence even in isolation.

**Takeaway:** Grid structure is learned robustly. The months-of-year ring fails to show a clean phase transition even without competition, likely due to strong pre-existing associations with month names in the model's weights.

---

## Experiment 2: Vocabulary Experiment — Disjoint Vocab (TransformerLens)

**Setup:** Replaces months with fully neutral ring words (candle, brick, fern, lamp, dust, wool, reef, thorn, cask, flint, marsh, prism). Uses TransformerLens for a single forward pass per sequence, measuring summed probability over valid neighbors rather than greedy accuracy. 16 sequences per condition.

**Results:**
- ρ=0 (pure grid): Grid neighbor probability rises cleanly from ~0.4 at 50 tokens to ~0.5 by 200–300 tokens.
- ρ=0.5 (mixed): Grid probability stays high. Ring probability collapses near zero and stays suppressed across all context lengths.
- ρ=1 (pure ring): Ring probability is noisy (~0.1–0.5) with no clean convergence — weaker and more variable than grid learning.

**Takeaway:** Neutral vocabulary does not fix ring learning. The ring topology itself is harder to learn in-context than the grid. Under mixing, ring learning is completely suppressed.

---

## Experiment 3: Vocabulary Experiment — Overlapping Vocab (TransformerLens)

**Setup:** 3 ring words (rock, sand, box) are shared with the grid vocabulary. Shared-word positions are tracked separately to probe ambiguity.

**Results:**
- Very similar to disjoint condition: grid learns well, ring is suppressed under mixing.
- Shared-word positions show elevated variance and occasionally elevated probability, indicating the model is uncertain at ambiguous tokens.
- Vocabulary overlap does not substantially degrade grid accuracy.

**Takeaway:** Overlap introduces local ambiguity at shared tokens but does not qualitatively change the competition outcome. Grid still dominates.

---

## Summary Assessment

| Condition | Grid learning | Ring learning | Competition signal |
|---|---|---|---|
| ρ=0, grid only | Strong | N/A | N/A |
| ρ=1, ring only | N/A | Weak / noisy | N/A |
| ρ=0.5, mixed | Strong (survives) | Suppressed to ~0 | Asymmetric |

**What is working:**
- Grid structure is robustly learned in-context and survives interleaving with a competing structure.
- The setup successfully produces a measurable asymmetry: one structure wins, the other is suppressed.
- TransformerLens gives cleaner signal than greedy Ollama probing.

**Key concerns:**
1. The ring does not cleanly converge even at ρ=1 (pure ring sequences). This means the "competition" result may reflect the ring simply being harder to learn, not genuine interference from the grid.
2. Before interpreting the ρ=0.5 results as evidence of competition, a cleaner ring baseline is needed — one that shows reliable convergence in isolation.
3. Ring neighbor probability under ρ=1 is noisy across both vocabulary conditions, suggesting the issue is structural rather than vocabulary-driven.

**Next steps:**
- Investigate why ring learning is weaker and noisier than grid learning (topology difference, degree, sequence statistics).
- Consider testing alternative ring structures or sequence lengths to establish a clean ring baseline.
- Once both structures learn reliably in isolation, re-run the mixing experiment to isolate true interference effects.
