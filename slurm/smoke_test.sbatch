#!/bin/bash
# Short end-to-end test on a real GPU: one small model, 20 questions per dataset.
# Optional: MODEL_KEY (default qwen3-4b) and DATASETS (default: default_datasets).
#SBATCH --job-name=medeval_smoke
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}

module purge
module load 2024
module load "$PYTHON_MODULE"
source "$HOME/venvs/medeval/bin/activate"

export HF_HOME="/scratch-shared/$USER/hf_cache"
export HF_HUB_OFFLINE=1

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
DATASET_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a DATASET_ARGS <<< "$DATASETS"
    DATASET_ARGS=(--datasets "${DATASET_ARGS[@]}")
fi

python -m medeval.run_eval --model "${MODEL_KEY:-qwen3-4b}" --limit 20 --out results_smoke "${DATASET_ARGS[@]}"
