#!/bin/bash
# Contamination tests (src/sanitize.py) for $MODEL_KEY on all MIR datasets; optional LORA, LORA_NAME, DATASETS.
# Smoke test: SMOKE=1 sbatch scripts/sanitize.sbatch (unit tests + smallest model on 5 questions).
#SBATCH --job-name=medeval_sanitize
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --export=ALL

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

if [[ "${SMOKE:-0}" == "1" ]]; then
    python -m pytest -v tests/test_sanitize_metrics.py tests/test_sanitize_smoke.py
    exit 0
fi
: "${MODEL_KEY:?MODEL_KEY must be set}"

EXTRA_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a DATASET_KEYS <<< "$DATASETS"
    EXTRA_ARGS+=(--datasets "${DATASET_KEYS[@]}")
fi
if [[ -n "${TESTS:-}" ]]; then
    read -r -a TEST_NAMES <<< "$TESTS"
    EXTRA_ARGS+=(--tests "${TEST_NAMES[@]}")
fi
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${LORA:-}" ]]; then
    EXTRA_ARGS+=(--lora "$LORA")
    if [[ -n "${LORA_NAME:-}" ]]; then
        EXTRA_ARGS+=(--lora-name "$LORA_NAME")
    fi
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

python src/sanitize.py --models "$MODEL_KEY" --out "${OUT:-results}" "${EXTRA_ARGS[@]}"
