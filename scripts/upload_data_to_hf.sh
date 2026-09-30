#!/usr/bin/env bash
# upload_data_to_hf.sh — publish the large HAR / Sleep-EDF test splits to a
# HuggingFace *dataset* repo, so the GitHub repo can stay small.
#
# One-time setup:
#   pip install huggingface_hub
#   huggingface-cli login              # paste a WRITE token from hf.co/settings/tokens
#
# Usage (run from the repo root):
#   REPO=sachithabey/ConceptTime bash scripts/upload_data_to_hf.sh
set -euo pipefail

REPO="${REPO:-sachithabey/ConceptTime}"

# Create the dataset repo if it does not exist (no-op if it already does).
huggingface-cli repo create "$REPO" --repo-type dataset -y || true

# Upload each large file to the same relative path used in this repo.
huggingface-cli upload "$REPO" dataset/HAR/test.pt        dataset/HAR/test.pt        --repo-type dataset
huggingface-cli upload "$REPO" dataset/SLeep-EDF/test.pt  dataset/SLeep-EDF/test.pt  --repo-type dataset

echo "Uploaded HAR + Sleep-EDF test splits to https://huggingface.co/datasets/$REPO"
echo "Now set HF_REPO in download_data.py (or pass --repo) to: $REPO"
