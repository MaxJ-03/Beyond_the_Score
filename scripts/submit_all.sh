#!/bin/bash
# Submit evaluation jobs for every model, but only for the datasets that do not have
# results yet. Rerunning this script after adding a dataset therefore only evaluates
# the new dataset.
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
#   JUDGE=0 bash scripts/submit_all.sh              do not queue the grading job
#   PARALLEL=1 bash scripts/submit_all.sh           submit all jobs at once instead of in a chain
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
mkdir -p logs

python - "${INCLUDE_OPTIONAL:-0}" "${FORCE:-0}" "${DATASETS:-}" <<'PY' > logs/submit_list.txt
import sys
sys.path.insert(0, "src")
from config import PROJECT_ROOT, default_datasets, load_datasets_config, load_models

include_optional, force, requested = sys.argv[1] == "1", sys.argv[2] == "1", sys.argv[3].split()
datasets = requested or default_datasets(load_datasets_config())
for key, cfg in load_models().items():
    if cfg.get("role") == "judge" or (cfg.get("optional") and not include_optional):
        continue
    missing = [d for d in datasets
               if force or not (PROJECT_ROOT / "results" / "raw" / key / f"{d}.jsonl").exists()]
    if missing:
        print(key, cfg["slurm_time"], ",".join(missing))
PY

if [[ ! -s logs/submit_list.txt ]]; then
    echo "Nothing to submit: all models already have results for the selected datasets."
    exit 0
fi

previous=""
while read -r key time datasets; do
    dependency=()
    if [[ -n "$previous" && "${PARALLEL:-0}" != "1" ]]; then
        dependency=(--dependency="afterany:$previous")
    fi
    job_id=$(sbatch --parsable --job-name="medeval_${key}" --time="$time" ${dependency[@]+"${dependency[@]}"} \
        --export=ALL,MODEL_KEY="$key",DATASETS="${datasets//,/ }" scripts/evaluate.sbatch)
    echo "Submitted $key (time limit $time) for [${datasets//,/ }] as job $job_id"
    previous=$job_id
done < logs/submit_list.txt

if [[ "${JUDGE:-1}" == "1" ]]; then
    job_id=$(sbatch --parsable --dependency="afterany:$previous" scripts/judge.sbatch)
    echo "Submitted the grading job as job $job_id (starts after job $previous)"
fi
