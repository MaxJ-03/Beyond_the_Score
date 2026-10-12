#!/bin/bash
# Prepare everything the evaluation needs, in one command:
#   1. all MIR datasets (Spanish 2022, 2024, 2026 and English 2022, 2024)
#   2. all model weights, followed by the parameter counts for the FLOPs metric
#
# Run on a Snellius login node after scripts/setup_env.sh and `hf auth login`.
# The model download takes a while, so run it in the background:
#   nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf_cache}
mkdir -p logs results

echo "=== Step 1/2: datasets ==="
python scripts/prepare_data.py

echo "=== Step 2/2: models and parameter counts ==="
python scripts/download_models.py

echo "=== Preparation complete ==="
