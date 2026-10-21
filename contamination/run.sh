#!/bin/bash
# Snellius driver of the contamination experiments. Run from the project folder on a login node:
#   bash contamination/run.sh planted [gemma3-4b|medgemma-4b]  planted contamination (train, evaluate, Min-K%)
#   bash contamination/run.sh medqa                            MedQA, nine models (stage3 settings)
#   bash contamination/run.sh tests                            memorization tests, nine models + planted models
#   bash contamination/run.sh analyse                          all tables (CPU, seconds)
#   bash contamination/run.sh status                           today's jobs
# Every GPU stage only submits jobs; run each stage once.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
export HF_HOME="/scratch-shared/$USER/hf_cache"
mkdir -p logs

MODELS="gemma3-1b gemma3-4b gemma3-27b medgemma-4b medgemma-27b qwen3-1.7b qwen3-4b llama3-8b med42-8b"
PLANTED_DATA="mir_2020_es mir_2021_es mir_2022_es"
TEST_DATA="medqa mir_2026 $PLANTED_DATA mir_2023_es mir_2024_es mir_2026_es"

submit() {  # submit NAME TIME DEPENDENCY COMMAND...   prints the job id
    local name=$1 time=$2 dep=$3; shift 3
    local args=(--parsable --job-name="$name" --time="$time" --export=ALL,TASK_CMD="$*")
    [[ -n $dep ]] && args+=(--dependency="afterok:$dep")
    local id; id=$(sbatch "${args[@]}" contamination/job.sbatch 2>/dev/null)
    echo "  $name -> job ${id:-FAILED}" >&2
    echo "$id"
}
time_for() { [[ $1 == *27b* ]] && echo 01:00:00 || echo 00:30:00; }

case "${1:-}" in
planted)
    base=${2:-gemma3-4b}
    [[ -f contamination/planted_split.csv ]] || python contamination/planted.py split
    mkdir -p "/scratch-shared/$USER/planted"
    [[ -e checkpoints/planted ]] || { mkdir -p checkpoints; ln -s "/scratch-shared/$USER/planted" checkpoints/planted; }
    train=$(submit "train_$base" 00:15:00 "" python contamination/planted.py train --model $base)
    submit "mink_$base" 00:15:00 "" python contamination/planted.py mink --model $base >/dev/null
    for e in 1 3 10; do
        m=planted-$base-ep$e
        submit "eval_$m" 00:15:00 "$train" python contamination/common.py eval --model $m --datasets $PLANTED_DATA mir_2026_es >/dev/null
        submit "mink_$m" 00:15:00 "$train" python contamination/planted.py mink --model $m >/dev/null
        submit "tests_$m" 00:30:00 "$train" python contamination/memorization_tests.py run --model $m --datasets $PLANTED_DATA --n 0 >/dev/null
    done ;;
medqa)
    [[ -f data/medqa_test.jsonl ]] || python scripts/prepare_medqa.py --hf-name GBaker/MedQA-USMLE-4-options --split test --out data/medqa_test.jsonl
    for m in $MODELS; do submit "medqa_$m" "$(time_for $m)" "" python contamination/common.py eval --model $m --datasets medqa >/dev/null; done ;;
tests)
    for m in $MODELS; do submit "tests_$m" "$(time_for $m)" "" python contamination/memorization_tests.py run --model $m --datasets $TEST_DATA >/dev/null; done ;;
analyse)
    python contamination/common.py reparse
    python contamination/planted.py analyse --base gemma3-4b
    python contamination/planted.py analyse --base medgemma-4b
    python contamination/memorization_tests.py summary
    echo "Paired comparison: python contamination/paired_shift.py --project <stage3 checkout>" ;;
status)
    sacct -u "$USER" -S "$(date +%Y-%m-%d)T00:00" -X -o JobID,JobName%40,State%15,Elapsed ;;
*)
    sed -n 2,8p "$0" ;;
esac
