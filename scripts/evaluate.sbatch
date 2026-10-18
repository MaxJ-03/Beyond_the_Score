#!/bin/bash
# Evaluate one model on all default datasets (or those given in $DATASETS).
#
# Full run: submitted by scripts/submit_all.sh, which sets MODEL_KEY and the time limit.
# Smoke test: one model, 20 questions per dataset, written to results_smoke/:
#   SMOKE=1 sbatch scripts/evaluate.sbatch                  (default model qwen3-4b)
#   SMOKE=1 MODEL_KEY=gemma3-4b sbatch scripts/evaluate.sbatch
# Optional variables: DATASETS (space-separated keys), LATENCY_SAMPLE (default 20, 0 in the
# smoke test), LIMIT and OUT (set by the smoke test).
#SBATCH --job-name=medeval
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --export=ALL

set -euo pipefail

if [[ "${SMOKE:-0}" == "1" ]]; then
    MODEL_KEY=${MODEL_KEY:-qwen3-4b}
    LIMIT=${LIMIT:-20}
    OUT=${OUT:-results_smoke}
    LATENCY_SAMPLE=${LATENCY_SAMPLE:-0}
fi
: "${MODEL_KEY:?MODEL_KEY must be set}"
PYTHON_MODULE=${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}

module purge
module load 2024
module load "$PYTHON_MODULE"
source "$HOME/venvs/medeval/bin/activate"

export HF_HOME="/scratch-shared/$USER/hf_cache"
export HF_HUB_OFFLINE=1
# Use vLLM's built-in sampler instead of FlashInfer's, which compiles GPU code at startup
# and needs the CUDA compiler. Greedy decoding gives identical results with both.
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

EXTRA_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a DATASET_KEYS <<< "$DATASETS"
    EXTRA_ARGS+=(--datasets "${DATASET_KEYS[@]}")
fi
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

python src/run_eval.py --model "$MODEL_KEY" --out "${OUT:-results}" \
    --latency-sample "${LATENCY_SAMPLE:-20}" "${EXTRA_ARGS[@]}"
