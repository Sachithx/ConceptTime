#!/usr/bin/env bash
# run.sh — wrapper that preloads the conda env's libstdc++ to fix the
# GLIBCXX_3.4.31 mismatch between PyTorch 2.10 / optree and the system library.
#
# Usage:
#   ./run.sh python train_gaussian_entropy_model.py --dataset HAR
#   ./run.sh python entropy_diagnostics.py --dataset HAR
#   ./run.sh python entropy_diagnostics.py --dataset HAR --ckpt2 output/HAR/seed1337.pt

CONDA_LIB=<>
export LD_PRELOAD="${CONDA_LIB}/libstdc++.so.6:${LD_PRELOAD}"

exec "$@"
