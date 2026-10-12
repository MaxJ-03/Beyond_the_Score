#!/bin/bash
# Submit evaluation jobs for every model, but only for the datasets that do not have
# results yet. Rerunning this script after adding a dataset therefore only evaluates
# the new dataset.
#
# Usage:
#   bash scripts/submit_all.sh                      all available datasets, missing results only
#   DATASETS="medqa" bash scripts/submit_all.sh     specific datasets
#   INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh   include optional models
#   FORCE=1 bash scripts/submit_all.sh              rerun everything, overwriting results
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
mkdir -p logs

python - "${INCLUDE_OPTIONAL:-0}" "${FORCE:-0}" "${DATASETS:-}" <<'PY' > logs/submit_list.txt
import sys
from medeval.config import PROJECT_ROOT, default_datasets, load_datasets_config, load_models

include_optional, force, requested = sys.argv[1] == "1", sys.argv[2] == "1", sys.argv[3].split()
datasets = requested or default_datasets(load_datasets_config())
for key, cfg in load_models().items():
    if cfg.get("optional") and not include_optional:
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

while read -r key time datasets; do
    job_id=$(sbatch --parsable --job-name="medeval_${key}" --time="$time" \
        --export=ALL,MODEL_KEY="$key",DATASETS="${datasets//,/ }" slurm/eval_model.sbatch)
    echo "Submitted $key for [${datasets//,/ }] as job $job_id"
done < logs/submit_list.txt
