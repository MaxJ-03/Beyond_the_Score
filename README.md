# medeval

Evaluation pipeline for the accuracy and inference cost of open LLMs on the Spanish MIR
medical licensing exam: older editions (2022 and 2024) against the newest edition (2026),
in the original Spanish and in English, and later on real-world clinical questions.

The full user manual is in [`docs/MANUAL.md`](docs/MANUAL.md). This README is the short version.

## The whole workflow

On a Snellius login node, inside the project folder:

```bash
# Once
bash scripts/setup_env.sh
source ~/venvs/medeval/bin/activate
hf auth login

# Prepare datasets, models and parameter counts (runs in the background, about an hour)
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &

# Test, then run everything
sbatch slurm/smoke_test.sbatch
bash scripts/submit_all.sh

# Analyse
python -m medeval.analyze
```

In every new session, activate the environment first with `source ~/venvs/medeval/bin/activate`.

## Datasets

| Dataset key | Content | Source |
|---|---|---|
| `mir_2022_es`, `mir_2024_es` | Original Spanish | CareQA (Hugging Face, from the official exam booklets) |
| `mir_2022`, `mir_2024` | English | CareQA's official English version (GPT-4 translation, validated by human evaluators) |
| `mir_2026_es` | Original Spanish | Scraped 2026 file in `data/external/` |
| `mir_2026` | English | Scraped 2026 translation in `data/external/` (Google machine translation) |
| `medqa` | Optional reference | `scripts/prepare_medqa.py`; run only on request |

All years use the same selection: the 200 main questions, without reserve, image-based or
annulled questions. The 2026 source files live in `data/external/` (tracked by Git):

| File | Content |
|---|---|
| `mir2026_text_primary_es.csv` (and `.jsonl`) | original Spanish questions and answers (included) |
| `mir_2026_en_mlt.csv` | the same questions with the English translation (add this file) |

Every English file has the same format and contains exactly the questions of its Spanish
counterpart, with the Spanish text in parallel columns.

## How runs are organised

* Every evaluation job runs one model on one H100 and answers all questions of all
  available datasets. Datasets whose input file exists are picked up automatically.
* `submit_all.sh` only submits work that has no results yet. After adding a dataset (for
  example MIR 2026 in English), run it again and only the new dataset is evaluated.
  `FORCE=1 bash scripts/submit_all.sh` reruns everything.
* `results/analysis/` is tracked by Git, so results can be pushed from Snellius and
  pulled on any computer.

## Translations

The English versions come from different translators: GPT-4 for 2022 and 2024 (CareQA) and
Google machine translation for 2026 (scraped file). The Spanish comparison is therefore the
cleanest test of old versus new questions, and comparing Spanish and English results shows
whether translation affects accuracy. Report both translation sources in the paper.
`scripts/prepare_data.py` validates all English files and writes a spot-check sample to
`data/translation_checks/`.

## Outputs

| File | Content |
|---|---|
| `results/analysis/summary.csv` | accuracy (95% CI), chance-corrected accuracy, invalid and truncated rates, tokens, FLOPs, GPU-seconds and median latency per question, weights memory, cost per above-chance correct answer, Pareto frontier membership |
| `results/analysis/rankings.csv` | rank of each model per dataset, by accuracy and by cost-efficiency |
| `results/analysis/rank_correlations.csv` | agreement of cost-efficiency rankings between datasets |
| `results/analysis/exam_shift.csv` | accuracy on 2026 minus accuracy on 2022 and 2024, per model and language |
| `results/analysis/accuracy_vs_flops_<dataset>.png` | accuracy against FLOPs, with the Pareto frontier |

## Project layout

```
configs/models.yaml       models and settings
configs/datasets.yaml     datasets, prompts, generation and analysis settings
src/medeval/              pipeline code
scripts/                  setup, data preparation, model download, job submission
slurm/                    Slurm job scripts
docs/MANUAL.md            full user manual
data/external/            files from outside the pipeline (tracked by Git)
data/raw/, data/          prepared datasets (generated)
results/                  raw outputs (generated) and analysis (tracked by Git)
```
