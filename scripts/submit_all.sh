#!/bin/bash
# Submit evaluation jobs for every model, but only for the datasets that do not have
# results yet. Rerunning this script after adding a dataset therefore only evaluates
# the new dataset. Results with answers that were cut off at a lower token limit than the
# current one in configs/datasets.yaml are outdated and are submitted again as well.
#
# The jobs run one after another: each job starts when the previous one has ended, so only
# one job at a time is checked against the compute budget. A grading job (scripts/judge.sbatch)
# is queued after the last evaluation job. A job that is cancelled or hits its time limit does
# not stop the chain; rerunning this script submits whatever is still missing.
#
# Usage:
#   bash scripts/submit_all.sh                      all available datasets, missing results only
#   DATASETS="medqa" bash scripts/submit_all.sh     specific datasets
#   INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh   include optional models
#   FORCE=1 bash scripts/submit_all.sh              rerun everything, overwriting results
#   MODELS="gemma3-27b" bash scripts/submit_all.sh  only these models
#   JUDGE=0 bash scripts/submit_all.sh              do not queue the grading job
#   PARALLEL=1 bash scripts/submit_all.sh           submit all jobs at once instead of in a chain
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
mkdir -p logs

python - "${INCLUDE_OPTIONAL:-0}" "${FORCE:-0}" "${DATASETS:-}" "${MODELS:-}" <<'PY' > logs/submit_list.txt
import json
import sys
sys.path.insert(0, "src")
from config import PROJECT_ROOT, default_datasets, load_datasets_config, load_models

include_optional, force = sys.argv[1] == "1", sys.argv[2] == "1"
requested, models = sys.argv[3].split(), sys.argv[4].split()
ds_cfg = load_datasets_config()
eval_cfg = ds_cfg["evaluation"]
datasets = requested or default_datasets(ds_cfg)


def needs_run(model, dataset, thinking):
    """True if the result is missing, or was cut off at a lower token limit than the current one."""
    results = PROJECT_ROOT / "results" / "raw" / model / f"{dataset}.jsonl"
    meta = results.with_suffix(".meta.json")
    if force or not results.exists() or not meta.exists():
        return True
    if ds_cfg["datasets"][dataset].get("task", "multiple_choice") != "multiple_choice":
        limit = eval_cfg["max_tokens_open"]
    else:
        limit = eval_cfg["max_tokens_thinking"] if thinking else eval_cfg["max_tokens_direct"]
    if json.loads(meta.read_text(encoding="utf-8")).get("max_tokens", limit) >= limit:
        return False
    with open(results, encoding="utf-8") as handle:
        return any(json.loads(line).get("finish_reason") == "length" for line in handle)


for key, cfg in load_models().items():
    if cfg.get("role") == "judge" or (cfg.get("optional") and not include_optional):
        continue
    if models and key not in models:
        continue
    rerun = [d for d in datasets if needs_run(key, d, cfg.get("thinking", False))]
    if rerun:
        print(key, cfg["slurm_time"], ",".join(rerun))
PY

if [[ ! -s logs/submit_list.txt ]]; then
    echo "Nothing to submit: all models already have results for the selected datasets."
    exit 0
fi

# Jobs of this project that are still pending or running. Models that already have a queued
# job are skipped, and new jobs are appended to the end of the existing chain, so the script can
# be rerun safely while earlier jobs are still in the queue.
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

while read -r key time datasets; do
    if grep -qx "[0-9]* medeval_${key}" <<< "$queued"; then
        echo "Skipped $key: a job for this model is already queued"
        continue
    fi
    job_id=$(submit "medeval_${key}" --time="$time" \
        --export=ALL,MODEL_KEY="$key",DATASETS="${datasets//,/ }" scripts/evaluate.sbatch)
    echo "Submitted $key (time limit $time) for [${datasets//,/ }] as job $job_id"
    previous=$job_id
done < logs/submit_list.txt

if [[ "${JUDGE:-1}" == "1" ]]; then
    if grep -qx "[0-9]* medeval_judge" <<< "$queued"; then
        echo "Skipped the grading job: it is already queued"
    else
        job_id=$(submit medeval_judge scripts/judge.sbatch)
        echo "Submitted the grading job as job $job_id (starts after the evaluation jobs)"
    fi
fi
