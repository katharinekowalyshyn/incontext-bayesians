#!/bin/bash
# Run the TransformerLens vocabulary experiment in the background.
# Requires the incontext-bayesians conda environment.
#
# Usage:
#   ./run_vocabulary_tl.sh                  # both conditions
#   ./run_vocabulary_tl.sh disjoint         # disjoint only
#   ./run_vocabulary_tl.sh overlap          # overlap only

cd "$(dirname "$0")"

CONDITION=${1:-both}
LOG="vocabulary_tl_experiment.log"

source /home/katie/miniconda3/etc/profile.d/conda.sh
conda activate incontext-bayesians

nohup python src/initial_experiments/vocabulary_tl_experiment.py --condition all\
    > "$LOG" 2>&1 &

echo "Started PID $! (condition: $CONDITION)"
echo "Tail logs with: tail -f $LOG"
