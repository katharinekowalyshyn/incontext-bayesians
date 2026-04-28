"""Bayesian model fits for the ICL-as-Bayes project (Checkpoint 2 §§3–4).

Entry points
------------
- fit_baseline.py  — Tim's M2: Bigelow sigmoid per (condition, ρ, graph).
- fit_upgrade.py   — Dan's M3/M4: Upgrade with MST complexity prior λ.
- generate_rho_ladder.py — wrapper around vocabulary_tl_experiment to fill in
  the ρ ladder {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}.

Shared helpers live in `data_loading.py`.
"""
