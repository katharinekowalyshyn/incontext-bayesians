#!/usr/bin/env bash
# Run the full PCA + Dirichlet-energy + accuracy pipeline for every
# experimental condition, then produce the cross-condition overlay
# plots.  Designed to be run in the background via:
#
#   nohup bash scripts/run_pca_pipeline.sh > logs/pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
#
# Or with explicit arg overrides:
#
#   N_WALKS=16 SEQ_LEN=2000 nohup bash scripts/run_pca_pipeline.sh \
#       > logs/pipeline.log 2>&1 & disown
#
# The script is idempotent: rerunning it will overwrite the
# `results/pca_pipeline_*.npz` artefacts and regenerate every plot.

set -euo pipefail

# ── Activate the conda env ────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/home/katie/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate incontext-bayesians

# ── Locate repo root regardless of CWD ────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs
LOG_DIR="${REPO_ROOT}/logs"

# ── Config (override via env vars on the command line) ────────────────────────
N_WALKS="${N_WALKS:-8}"
SEQ_LEN="${SEQ_LEN:-1400}"
LAYER="${LAYER:-26}"
SEED="${SEED:-42}"
# Space-separated list of conditions; override with e.g.
#   CONDITIONS="grid months_permuted" nohup bash scripts/run_pca_pipeline.sh ...
CONDITIONS="${CONDITIONS:-grid months_natural months_permuted neutral_disjoint neutral_overlap}"

PYTHON="python"
SCRIPT="${REPO_ROOT}/src/initial_experiments/pca_analysis.py"

echo "───────────────────────────────────────────────────────────────"
echo "PCA pipeline  —  $(date -u +'%Y-%m-%d %H:%M:%SZ')"
echo "  REPO_ROOT  = ${REPO_ROOT}"
echo "  N_WALKS    = ${N_WALKS}"
echo "  SEQ_LEN    = ${SEQ_LEN}"
echo "  LAYER      = ${LAYER}"
echo "  SEED       = ${SEED}"
echo "  CONDITIONS = ${CONDITIONS}"
echo "  PYTHON     = $(${PYTHON} --version 2>&1)"
echo "  CUDA       = $(${PYTHON} -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)' 2>&1 || echo 'torch missing')"
echo "───────────────────────────────────────────────────────────────"

# ── Per-condition runs ────────────────────────────────────────────────────────
for COND in ${CONDITIONS}; do
    echo ""
    echo "=== [$(date +%H:%M:%S)] condition: ${COND} ==="
    COND_LOG="${LOG_DIR}/pca_pipeline_${COND}.log"
    # `tee` so the combined log still reflects progress of each run.
    ${PYTHON} "${SCRIPT}" \
        --with-model \
        --condition "${COND}" \
        --n-walks "${N_WALKS}" \
        --seq-len "${SEQ_LEN}" \
        --layer   "${LAYER}" \
        --seed    "${SEED}" \
        2>&1 | tee "${COND_LOG}"
    echo "    log  → ${COND_LOG}"
done

# ── Cross-condition overlay ───────────────────────────────────────────────────
echo ""
echo "=== [$(date +%H:%M:%S)] cross-condition overlay ==="
OVERLAY_LOG="${LOG_DIR}/pca_pipeline_overlay.log"
${PYTHON} "${SCRIPT}" --overlay 2>&1 | tee "${OVERLAY_LOG}"
echo "    log  → ${OVERLAY_LOG}"

echo ""
echo "───────────────────────────────────────────────────────────────"
echo "PCA pipeline complete  —  $(date -u +'%Y-%m-%d %H:%M:%SZ')"
echo "Artifacts under: ${REPO_ROOT}/src/initial_experiments/results/"
echo "Per-condition logs: ${LOG_DIR}/pca_pipeline_*.log"
echo "───────────────────────────────────────────────────────────────"
