#!/bin/bash
# One GPU job that runs the command in $TASK_CMD (set by contamination/run.sh).
#SBATCH --job-name=contamination
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
: "${TASK_CMD:?TASK_CMD must be set}"
module purge
module load 2024
module load "${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
source "$HOME/venvs/medeval/bin/activate"
export HF_HOME="/scratch-shared/$USER/hf_cache" HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "Running: $TASK_CMD"
eval "$TASK_CMD"
