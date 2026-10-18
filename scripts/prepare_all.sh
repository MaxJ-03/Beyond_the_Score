#!/bin/bash
# Prepare everything the evaluation needs, in one command:
#   1. all MIR datasets (Spanish and English, 2020 to 2026) and the translation checks
#   2. the real-world datasets (CUPCase, LiveMedBench, K-QA)
#   3. model weights (including the judge) and parameter counts: only missing models are
#      downloaded and only models missing from results/param_counts.json are counted, so
#      this step does nothing when everything is already in place
#
# Run on a Snellius login node after scripts/setup_env.sh and `hf auth login`.
# A first download takes a while, so run it in the background:
#   nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &
# SKIP_MODELS=1 runs only the dataset steps.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf_cache}
mkdir -p logs results

echo "=== Step 1/3: MIR datasets ==="
python src/prepare_data.py

echo "=== Step 2/3: real-world datasets ==="
python src/prepare_real_world.py

echo "=== Step 3/3: models and parameter counts ==="
if [[ "${SKIP_MODELS:-0}" == "1" ]]; then
    echo "Skipped (SKIP_MODELS=1)."
else
    python src/models.py
fi

echo "=== Preparation complete ==="
