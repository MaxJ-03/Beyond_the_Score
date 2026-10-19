#!/bin/bash
# Submit contamination-test jobs (scripts/sanitize.sbatch) for every model that still has missing
# outputs in results/sanitize/. Each job runs one model on all MIR datasets and skips the outputs
# that already exist, so rerunning this script after an interrupted or timed-out job only does
# the remaining work. The time limit of each job is the model's slurm_time in configs/models.yaml.
#
# The jobs run one after another and are appended to the end of the existing chain of this
# project's jobs (also those of scripts/submit_all.sh), so only one job at a time is checked
# against the compute budget. Models that already have a queued job are skipped.
#
# Usage:
#   bash scripts/submit_sanitize.sh                                  all models, missing outputs only
#   MODELS="gemma3-27b qwen3-4b" bash scripts/submit_sanitize.sh     only these models
#   DATASETS="mir_2022_es mir_2026_es" bash scripts/submit_sanitize.sh
#   OVERWRITE=1 bash scripts/submit_sanitize.sh                      rerun everything, overwriting outputs
#   PARALLEL=1 bash scripts/submit_sanitize.sh                       submit all jobs at once instead of in a chain
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
mkdir -p logs

python - "${OVERWRITE:-0}" "${DATASETS:-}" "${MODELS:-}" <<'PY' > logs/sanitize_list.txt
import sys
sys.path.insert(0, "src")
from sanitize import pending_work

overwrite, datasets, models = sys.argv[1] == "1", sys.argv[2].split(), sys.argv[3].split()
for key, time in pending_work(models or None, datasets or None, overwrite):
    print(key, time)
PY

if [[ ! -s logs/sanitize_list.txt ]]; then
    echo "Nothing to submit: all models already have contamination-test outputs for the selected datasets."
    exit 0
fi

queued=$(squeue -u "$USER" -h -o "%i %j" | awk '$2 ~ /^medeval/' | sort -n)
previous=$(awk 'END {print $1}' <<< "$queued")

submit() {
    local name=$1 job_id
    shift
    if [[ -n "$previous" && "${PARALLEL:-0}" != "1" ]]; then
        set -- --dependency="afterany:$previous" "$@"
    fi
    if ! job_id=$(sbatch --parsable --job-name="$name" "$@"); then
        echo "Submission of $name failed. Jobs submitted so far stay queued; rerun this script" \
            "later to submit the rest." >&2
        exit 1
    fi
    echo "$job_id"
}

while read -r key time; do
    if grep -qx "[0-9]* medeval_sanitize_${key}" <<< "$queued"; then
        echo "Skipped $key: a job for this model is already queued"
        continue
    fi
    job_id=$(submit "medeval_sanitize_${key}" --time="$time" \
        --export=ALL,MODEL_KEY="$key",DATASETS="${DATASETS:-}",OVERWRITE="${OVERWRITE:-0}" scripts/sanitize.sbatch)
    echo "Submitted $key (time limit $time) as job $job_id"
    previous=$job_id
done < logs/sanitize_list.txt
