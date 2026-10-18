# medeval

Evaluation pipeline for the accuracy and inference cost of open LLMs on the Spanish MIR
medical licensing exam: older editions (2020 to 2024) against the newest editions (2025 and
2026), in the original Spanish and in English, with checks for benchmark memorization, and
later on real-world clinical questions.

The full user manual is in [`docs/MANUAL.md`](docs/MANUAL.md). This README is the short version.

## The whole workflow

On a Snellius login node, inside the project folder:

```bash
# Once
bash scripts/setup_env.sh
source ~/venvs/medeval/bin/activate
hf auth login

# Prepare datasets, model weights and parameter counts (background; models already
# downloaded are skipped)
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &

# Test, then run everything
SMOKE=1 sbatch scripts/evaluate.sbatch
bash scripts/submit_all.sh

# Analyse
python src/analyze.py
python src/contamination.py
```

In every new session, activate the environment first with `source ~/venvs/medeval/bin/activate`.
On your own computer, `pip install -r requirements.txt` installs everything needed for data
preparation and analysis (all dependencies except vLLM, which only the GPU jobs need).

## Project layout

```
configs/models.yaml          models, settings and the first unseen exam year per model
configs/datasets.yaml        datasets, prompts, generation and analysis settings
scripts/                     everything run on Snellius
  setup_env.sh                 environment (once)
  prepare_all.sh               datasets, model weights, parameter counts
  submit_all.sh                one evaluation job per model, missing results only
  evaluate.sbatch              the evaluation job (also the smoke test, SMOKE=1)
src/                         all Python code, one file per pipeline stage
  config.py                    reading the configuration files
  prepare_data.py              builds every dataset file and checks the translations
  mir_sources.py               selection rules, official PDFs, MIR 2025 extraction
  mir_corrections.py           audited corrections to source text
  prepare_medqa.py             optional MedQA reference set
  models.py                    model download and parameter counting
  run_eval.py                  data loading, prompts, model run, answer parsing
  analyze.py                   accuracy, cost, rankings, Pareto frontier, exam shift, plots
  contamination.py             year-difficulty adjustment, unparseable check, positive control
docs/MANUAL.md               full user manual
data/external/               files from outside the pipeline (tracked by Git)
data/raw/, data/             prepared datasets (tracked by Git)
results/raw/                 model answers (tracked by Git)
results/analysis/            analysis outputs (tracked by Git)
```

Each stage file (`prepare_data`, `prepare_medqa`, `models`, `run_eval`, `analyze`,
`contamination`) runs with `python src/<file>.py` from the project folder; `--help` shows
its options.

## Datasets

| Dataset key | Content | Source |
|---|---|---|
| `mir_2020_es` to `mir_2024_es` | Original Spanish | CareQA (Hugging Face, from the official exam booklets) |
| `mir_2020` to `mir_2024` | English | CareQA's official English version (GPT-4 translation, validated by human evaluators) |
| `mir_2025_es` | Original Spanish | Official Version 0 booklet and definitive answer key in `data/external/` |
| `mir_2025` | English | 2025 translation in `data/external/` (Google machine translation, same procedure as 2026) |
| `mir_2026_es` | Original Spanish | Scraped 2026 file in `data/external/` |
| `mir_2026` | English | Scraped 2026 translation in `data/external/` (Google machine translation) |
| `medqa` | Optional reference | `python src/prepare_medqa.py`; run only on request |

All years use the same selection: the main questions (175 for 2020 and 2021, 200 from 2022
on), without reserve, image-based or annulled questions. The source files that do not come
from the internet live in `data/external/` (tracked by Git):

| File | Content |
|---|---|
| `mir2025_questions_v0.pdf`, `mir2025_answers_definitive_v0.pdf` | official 2025 booklet and answer key |
| `mir_2025_en_mlt.csv` | English translation of the selected 2025 questions |
| `mir2026_text_primary_es.csv` | original Spanish 2026 questions and answers |
| `mir_2026_en_mlt.csv` | the same 2026 questions with the English translation |
| `mir_human_difficulty_2020_2026.csv` | human reference accuracy per exam year, with sources |

Every English file has the same format and contains exactly the questions of its Spanish
counterpart, with the Spanish text in parallel columns.

## How runs are organised

* Every evaluation job runs one model on one H100 and answers all questions of all
  available datasets. Datasets whose input file exists are picked up automatically.
* `submit_all.sh` only submits work that has no results yet. After adding a dataset, run it
  again and only the new dataset is evaluated. `FORCE=1 bash scripts/submit_all.sh` reruns
  everything.
* `prepare_all.sh` can be rerun at any time: it only downloads missing models and only
  counts parameters of models that are not counted yet.

## Seen and unseen exam years

Two splits of the exam years are reported:

* **Common split** (`analysis.common_split` in `configs/datasets.yaml`): the same years for
  every model, 2020 to 2024 as seen and 2025 to 2026 as unseen, so results are directly
  comparable. It is not exact for Llama 3 and Med42 (2024 is already unseen) or Qwen3 (2025
  may have been seen).
* **Per-model split** (`unseen_from` in `configs/models.yaml`): unseen from the first exam
  after each model's training cutoff. Because the cutoffs differ between models, this split
  allows `src/contamination.py` to separate the effect of being unseen from the difficulty
  of each exam year.

## Translations

The English versions come from different translators: GPT-4 for 2020 to 2024 (CareQA) and
Google machine translation for 2025 and 2026. The Spanish comparison is therefore the
cleanest test of old versus new questions, and comparing Spanish and English results shows
whether translation affects accuracy. Report both translation sources in the paper.
`python src/prepare_data.py` checks all English files and writes a spot-check sample to
`data/translation_checks/`.

## Outputs

| File | Content |
|---|---|
| `results/analysis/summary.csv` | accuracy (95% CI), chance-corrected accuracy, invalid and truncated rates, tokens, FLOPs, GPU-seconds and median latency per question, weights memory, cost per above-chance correct answer, Pareto frontier membership |
| `results/analysis/rankings.csv` | rank of each model per dataset, by accuracy and by cost-efficiency |
| `results/analysis/rank_correlations.csv` | agreement of cost-efficiency rankings between datasets |
| `results/analysis/exam_shift.csv` | accuracy on unseen minus seen exam years, per model, language and split |
| `results/analysis/accuracy_vs_flops_<dataset>.png` | accuracy against FLOPs, with the Pareto frontier |
| `results/analysis/contamination/` | year-difficulty adjustment (main contamination result), unparseable-answer check, MedExpQA positive control |
