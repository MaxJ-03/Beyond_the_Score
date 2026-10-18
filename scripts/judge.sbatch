#!/bin/bash
# Grade all open-ended answers that have no grades yet with the judge (gpt-oss-120b).
# The judge is loaded once for all models and datasets; rerunning only grades new answers.
#   sbatch scripts/judge.sbatch                       grade everything that is missing
#   VALIDATE=1 sbatch scripts/judge.sbatch            agreement with the K-QA physician labels
#   JUDGE_ARGS="--models qwen3-4b --force" sbatch scripts/judge.sbatch
#SBATCH --job-name=medeval_judge
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}

module purge
module load 2024
module load "$PYTHON_MODULE"
source "$HOME/venvs/medeval/bin/activate"

export HF_HOME="/scratch-shared/$USER/hf_cache"
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if [[ "${VALIDATE:-0}" == "1" ]]; then
    python src/judge.py --validate
else
    read -r -a EXTRA_ARGS <<< "${JUDGE_ARGS:-}"
    python src/judge.py "${EXTRA_ARGS[@]}"
fi
