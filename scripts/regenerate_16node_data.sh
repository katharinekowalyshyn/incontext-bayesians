#!/usr/bin/env bash
# Regenerate the vocabulary_tl accuracy data for the two-condition, 16-node
# refactor: conditions ∈ {disjoint, overlap}, ρ ladder ∈ {0.0, 0.2, 0.3, 0.4,
# 0.5, 0.6, 0.7, 0.8, 1.0}, 16 walks × 15 eval points per cell.
#
# Launch in the background:
#
#   nohup bash scripts/regenerate_16node_data.sh \
#       > logs/regen_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Override the condition set with:
#
#   CONDITIONS="disjoint" nohup bash scripts/regenerate_16node_data.sh \
#       > logs/regen_disjoint.log 2>&1 & disown
#
# Runtime is dominated by Llama-3.1-8B forward passes: roughly
# (9 ρ × 16 walks × 1–2s/walk)  ≈  2–5 min per condition on an RTX-class GPU,
# so the full 2-condition run takes ~5–10 min.  After it completes, both
# fit scripts pick up the new data automatically.

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/home/katie/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate incontext-bayesians

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs
LOG_DIR="${REPO_ROOT}/logs"

CONDITIONS="${CONDITIONS:-disjoint overlap}"
PYTHON="python"
DATA_SCRIPT="${REPO_ROOT}/src/initial_experiments/vocabulary_tl_experiment.py"
BASELINE_SCRIPT="${REPO_ROOT}/src/experiments/fit_baseline.py"
UPGRADE_SCRIPT="${REPO_ROOT}/src/experiments/fit_upgrade.py"

echo "───────────────────────────────────────────────────────────────"
echo "vocabulary_tl + Bayesian-fit pipeline (16-node refactor)"
echo "  start      = $(date -u +'%Y-%m-%d %H:%M:%SZ')"
echo "  REPO_ROOT  = ${REPO_ROOT}"
echo "  CONDITIONS = ${CONDITIONS}"
echo "  PYTHON     = $(${PYTHON} --version 2>&1)"
echo "  CUDA       = $(${PYTHON} -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)' 2>&1 || echo 'torch missing')"
echo "───────────────────────────────────────────────────────────────"

# ── Phase 1: regenerate LLM accuracy data per condition ──────────────────────
for COND in ${CONDITIONS}; do
    echo ""
    echo "=== [$(date +%H:%M:%S)] regenerating data: ${COND} ==="
    COND_LOG="${LOG_DIR}/regen_${COND}.log"
    ${PYTHON} "${DATA_SCRIPT}" --condition "${COND}" 2>&1 | tee "${COND_LOG}"
    echo "    log  → ${COND_LOG}"
done

# ── Phase 2: baseline Bigelow fit (Tim's M2) ─────────────────────────────────
echo ""
echo "=== [$(date +%H:%M:%S)] Tim: Baseline sigmoid fit ==="
BASELINE_LOG="${LOG_DIR}/fit_baseline.log"
${PYTHON} "${BASELINE_SCRIPT}" --plot --bootstrap 100 2>&1 | tee "${BASELINE_LOG}"
echo "    log  → ${BASELINE_LOG}"

# ── Phase 3: upgrade fits (Dan's M3/M4) ──────────────────────────────────────
echo ""
echo "=== [$(date +%H:%M:%S)] Dan: Upgrade fit (both parameterisations) ==="
UPGRADE_LOG="${LOG_DIR}/fit_upgrade.log"
${PYTHON} "${UPGRADE_SCRIPT}" --model both --plot 2>&1 | tee "${UPGRADE_LOG}"
echo "    log  → ${UPGRADE_LOG}"

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "Pipeline complete  —  $(date -u +'%Y-%m-%d %H:%M:%SZ')"
echo "Data:            ${REPO_ROOT}/src/initial_experiments/results/vocabulary_tl/"
echo "Baseline fits:   ${REPO_ROOT}/src/experiments/results/baseline_fits/"
echo "Upgrade  fits:   ${REPO_ROOT}/src/experiments/results/upgrade_fits/"
echo "Figures:         ${REPO_ROOT}/src/experiments/results/figures/"
echo "───────────────────────────────────────────────────────────────"
