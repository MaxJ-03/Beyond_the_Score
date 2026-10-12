#!/bin/bash
# Translate all MIR exam years with one fixed configuration, then validate the results.
# Run this once, before the evaluation pipeline.
#
# Input:  data/raw/mir_<year>_es.csv  (Spanish, one row per question)
# Output: data/mir_<year>_en.csv      (Spanish and English side by side, used by the pipeline)
#         data/mir_<year>_en.meta.json (translation settings and date)
#         data/translation_checks/     (validation reports and spot-check sample)
#
# Configuration through environment variables:
#   YEARS        exam years to translate (default: "2022 2024 2026")
#   BACKEND      google-cloud | deep-translator | mock (default: google-cloud)
#   PROJECT_ID   Google Cloud project ID (required for google-cloud)
#   MODEL        Cloud Translation model (default: general/nmt)
#   COLUMN_MAP   optional JSON column mapping passed to translate_mir.py
#   PYTHON       Python interpreter (default: python)
#
# Example:
#   BACKEND=google-cloud PROJECT_ID=my-project bash scripts/prepare_mir.sh
set -euo pipefail
cd "$(dirname "$0")/.."

YEARS=${YEARS:-"2022 2024 2026"}
BACKEND=${BACKEND:-google-cloud}
MODEL=${MODEL:-general/nmt}
PYTHON=${PYTHON:-python}

EXTRA_ARGS=()
if [[ "$BACKEND" == "google-cloud" ]]; then
    : "${PROJECT_ID:?PROJECT_ID must be set for the google-cloud backend}"
    EXTRA_ARGS+=(--project-id "$PROJECT_ID" --model "$MODEL")
fi
if [[ -n "${COLUMN_MAP:-}" ]]; then
    EXTRA_ARGS+=(--column-map "$COLUMN_MAP")
fi

OUTPUTS=()
for year in $YEARS; do
    input="data/raw/mir_${year}_es.csv"
    output="data/mir_${year}_en.csv"
    if [[ ! -f "$input" ]]; then
        echo "Missing input file $input" >&2
        exit 1
    fi
    echo "Translating MIR $year with backend $BACKEND ..."
    "$PYTHON" scripts/translate_mir.py --input "$input" --output "$output" --backend "$BACKEND" "${EXTRA_ARGS[@]}"
    OUTPUTS+=("$output")
done

echo "Validating translations ..."
"$PYTHON" scripts/validate_translations.py "${OUTPUTS[@]}"
echo "Done. Review data/translation_checks/ and complete the spot check before running the pipeline."
