#!/bin/bash
# Run the TransformerLens mixing experiment in the background.
# Requires the incontext-bayesians conda environment.
#
# Usage:
#   ./run_mixing_tl.sh           # full run (inference + plots)
#   ./run_mixing_tl.sh --replot  # regenerate plots from cached JSON only

cd "$(dirname "$0")"

LOG="mixing_experiment.log"
ARGS="$*"

source /home/katie/miniconda3/etc/profile.d/conda.sh
conda activate incontext-bayesians

nohup python src/initial_experiments/mixing_experiment.py $ARGS \
    > "$LOG" 2>&1 &

echo "Started PID $!"
echo "Tail logs with: tail -f $LOG"
