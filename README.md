# medeval

Evaluation pipeline for the accuracy and inference cost of open LLMs on medical
multiple-choice questions: older editions of the Spanish MIR licensing exam (2022 and 2024),
the newest edition (2026), and, later, real-world clinical questions. Optional runs cover
the original Spanish text (translation effect) and MedQA (reproduction of reported scores).

Each model is loaded once with vLLM and evaluated on all datasets. For every question the
pipeline stores the response, the parsed answer, correctness and token counts; for every
dataset it stores the measured generation time. The analysis step turns these into
accuracy, chance-corrected accuracy, FLOPs and GPU time per question, cost per
above-chance correct answer, model rankings and the accuracy shift between exam editions.

The complete user manual, from logging in to Snellius to interpreting the results, is in
[`docs/MANUAL.md`](docs/MANUAL.md). This README is a short overview.

## Project layout

```
configs/models.yaml      models, vLLM settings, Slurm time limits
configs/datasets.yaml    datasets, prompts, generation and analysis settings
src/medeval/             pipeline code (data loading, prompts, parsing, evaluation, analysis)
scripts/                 setup, data preparation (including MIR translation), model download,
                         parameter counting, job submission
docs/MANUAL.md           full user manual
slurm/                   Slurm job scripts
data/                    input datasets (not version-controlled)
results/                 raw outputs and analysis (not version-controlled)
```

## Running on Snellius

All commands below are run from the project directory on a Snellius login node.

### 1. Copy the project and set up the environment (once)

```bash
scp -r medeval snellius:~/          # from your own computer
ssh snellius
cd ~/medeval
module avail Python                  # check that the module in scripts/setup_env.sh exists
bash scripts/setup_env.sh
```

If the Python module name differs, run `PYTHON_MODULE=<name> bash scripts/setup_env.sh`
and set the same `PYTHON_MODULE` when submitting jobs. The installed package versions are
written to `requirements.lock.txt`; report the vLLM version in the paper.

### 2. Log in to Hugging Face (once)

Accept the licenses for Gemma 3 and MedGemma on their Hugging Face pages, then:

```bash
source ~/venvs/medeval/bin/activate
huggingface-cli login
```

### 3. Prepare the datasets (own computer or login node, no GPU)

```bash
pip install -r requirements-data.txt
python scripts/download_mir.py careqa --years 2022 2024
python scripts/download_mir.py pdf --year 2026 --questions-pdf <booklet.pdf> --answers-pdf <key.pdf>
BACKEND=google-cloud PROJECT_ID=<your-project> bash scripts/prepare_mir.sh
```

`download_mir.py` collects the Spanish questions per year into `data/raw/` with the same
selection rules for every year (an existing CSV can be imported with the `csv` source).
`prepare_mir.sh` translates all years with one fixed configuration and writes
`data/mir_<year>_en.csv`, a `.meta.json` file with the translation settings, and validation
reports plus a spot-check sample in `data/translation_checks/`. Complete the spot check before
running the evaluation. Optionally, prepare MedQA with `scripts/prepare_medqa.py`.
See section 5 of the manual for details.

### 4. Download the models and count their parameters

```bash
python scripts/download_models.py
python scripts/count_params.py --out results/param_counts.json
```

The models take roughly 140 GB and are stored in `/scratch-shared/$USER/hf_cache`, where
files unused for 14 days are deleted. If that happens, run the download script again.

### 5. Run a smoke test (about 20 minutes on one H100)

```bash
sbatch slurm/smoke_test.sbatch
squeue -u $USER                      # job status
cat logs/medeval_smoke_*.out         # output when finished
```

Check in `results_smoke/raw/qwen3-4b/*.jsonl` that the responses look sensible and that
`predicted` is filled in for most questions.

### 6. Submit all models

```bash
bash scripts/submit_all.sh                        # one job per model
INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh     # also the Qwen3 thinking variant
```

### 7. Analyse

```bash
python -m medeval.analyze --results results
```

Outputs in `results/analysis/`:

| File | Content |
|---|---|
| `summary.csv` | accuracy (with 95% CI), chance-corrected accuracy, invalid and truncated answer rates, tokens, FLOPs, GPU-seconds and median latency per question, weights memory, cost per above-chance correct answer, Pareto frontier membership |
| `rankings.csv` | rank of each model per dataset, by accuracy and by cost-efficiency |
| `rank_correlations.csv` | Spearman correlation of cost-efficiency rankings between datasets |
| `exam_shift.csv` | accuracy on the new exam minus accuracy on older editions, per language, with bootstrap CI |
| `accuracy_vs_flops_<dataset>.png` | accuracy against FLOPs per question, with the Pareto frontier |

## Testing without a GPU

The mock backend returns random answers, so the full pipeline can be tested locally:

```bash
pip install -e . pandas pyyaml matplotlib
python scripts/make_test_data.py                  # synthetic data; never overwrites real files
python -m medeval.run_eval --model qwen3-4b --backend mock
python -m medeval.analyze --results results
```

Delete the synthetic files in `data/` and the `results/` directory before using real data.

## Design choices

These defaults are set in the configuration files and can be changed there:

- **Hardware:** every model runs on one H100 (94 GB), so measured GPU time is comparable
  across models. The largest models (27B in bfloat16) fit on a single H100.
- **Decoding:** greedy (temperature 0) with a fixed seed. The direct-answer prompt asks for
  the letter only, with at most 64 output tokens.
- **Qwen3 thinking:** disabled by default for comparability with Gemma; the optional
  `qwen3-4b-think` variant measures the cost of reasoning.
- **Answer parsing:** an explicit "Answer: X" statement, then a leading letter, then a single
  standalone letter. Responses without an unambiguous answer count as incorrect and are
  reported as the invalid rate.
- **FLOPs:** 2 × N × (input tokens + output tokens), with N the language-model parameters
  counted from the weights (vision components excluded). Switch to non-embedding parameters
  with `flops_param_count` in `configs/datasets.yaml`.
- **Chance correction:** (accuracy − c) / (1 − c), where c is the mean of 1/k over the
  questions (k = number of options). Cost per above-chance correct answer is undefined
  when the model is at or below chance.

## Adding the real-world dataset

Convert it to the jsonl format (`id, year, question, options, answer`), place it in `data/`,
uncomment the `real_world` entry in `configs/datasets.yaml` and add it to `default_datasets`.
No code changes are needed.

## Estimated cost

With about 600 MIR questions (about 1,800 including the Spanish versions and MedQA) and short direct answers, each model job should take
roughly 10 to 30 minutes on one H100 (192 SBU per GPU-hour), so the default models
should need on the order of a few hundred SBU. The thinking variant generates far more
tokens and can cost several times more. Check actual usage with `accinfo` after the smoke
test and adjust before submitting everything.
