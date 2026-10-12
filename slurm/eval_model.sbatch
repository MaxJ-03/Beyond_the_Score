#!/bin/bash
# Evaluate one model on all default datasets (or those given in $DATASETS).
# Submitted by scripts/submit_all.sh, which sets MODEL_KEY and the time limit.
#SBATCH --job-name=medeval
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY must be set}"
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}

module purge
module load 2024
module load "$PYTHON_MODULE"
source "$HOME/venvs/medeval/bin/activate"

export HF_HOME="/scratch-shared/$USER/hf_cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

DATASET_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a DATASET_ARGS <<< "$DATASETS"
    DATASET_ARGS=(--datasets "${DATASET_ARGS[@]}")
fi

python -m medeval.run_eval --model "$MODEL_KEY" --out results --latency-sample "${LATENCY_SAMPLE:-20}" "${DATASET_ARGS[@]}"
