#!/bin/bash
cd "$(dirname "$0")"
nohup python iclr_induction-main/initial_experiments/vocabulary_experiment.py \
    > vocabulary_experiment.log 2>&1 &
echo "Started PID $!"
echo "Tail logs with: tail -f vocabulary_experiment.log"
