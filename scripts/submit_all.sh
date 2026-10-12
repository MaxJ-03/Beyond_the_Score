#!/bin/bash
# Submit one evaluation job per model listed in configs/models.yaml.
# Usage:
#   bash scripts/submit_all.sh                      all non-optional models
#   INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh   include optional variants
#   DATASETS="mir_2024 mir_2026" bash scripts/submit_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/venvs/medeval/bin/activate"
mkdir -p logs

python - "${INCLUDE_OPTIONAL:-0}" <<'PY' > logs/submit_list.txt
import sys
from medeval.config import load_models
include_optional = sys.argv[1] == "1"
for key, cfg in load_models().items():
    if cfg.get("optional") and not include_optional:
        continue
    print(key, cfg["slurm_time"])
PY

while read -r key time; do
    job_id=$(sbatch --parsable --job-name="medeval_${key}" --time="$time" \
        --export=ALL,MODEL_KEY="$key",DATASETS="${DATASETS:-}" slurm/eval_model.sbatch)
    echo "Submitted $key (time limit $time) as job $job_id"
done < logs/submit_list.txt
