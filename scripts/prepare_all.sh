#!/bin/bash
# Prepare MIR and real-world datasets, model weights and parameter counts (skips what exists).
# Run in the background: nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &  (SKIP_MODELS=1: data only)
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
