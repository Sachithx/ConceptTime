#!/usr/bin/env bash
# run.sh — optional wrapper that preloads the active conda env's libstdc++.
#
# Only needed if you hit a "GLIBCXX_3.4.xx not found" error when importing
# torch (a mismatch between a recent PyTorch build and an older system
# libstdc++). If plain `python ...` works for you, you do not need this.
#
# Usage:
#   ./run.sh python src/evaluate.py --dataset Epilepsy \
#       --artifact checkpoints/Epilepsy/pipeline_best.pt \
#       --density  checkpoints/Epilepsy/gaussian_entropy_best.pt

CONDA_LIB="${CONDA_PREFIX:-/usr}/lib"
if [ -f "${CONDA_LIB}/libstdc++.so.6" ]; then
    export LD_PRELOAD="${CONDA_LIB}/libstdc++.so.6:${LD_PRELOAD}"
fi

exec "$@"
