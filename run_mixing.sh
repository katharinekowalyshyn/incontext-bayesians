#!/bin/bash
cd "$(dirname "$0")"
nohup python iclr_induction-main/initial_experiments/mixing_experiment.py \
    > mixing_experiment.log 2>&1 &
echo "Started PID $!"
echo "Tail logs with: tail -f mixing_experiment.log"
