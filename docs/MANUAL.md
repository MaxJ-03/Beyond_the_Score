# medeval: User Manual

This manual explains how to run the complete study, from logging in to Snellius to
reading the final results. Follow the parts in order the first time; afterwards, use the
quick reference at the end.

**Contents**

1. What the pipeline does
2. Before you start
3. Connecting to Snellius
4. Installing the pipeline
5. Preparing data and models
6. Checking the models
7. Running the evaluation
8. Analysing the results
9. Configuration reference
10. Extending the study
11. Troubleshooting
12. Reproducibility checklist for the paper
13. Quick reference

---

## 1. What the pipeline does

The study measures how accurate open LLMs are on medical multiple-choice questions, and
how much compute each correct answer costs, on three kinds of questions:

| Question type | Dataset | Role in the study |
|---|---|---|
| Standard benchmark | MIR 2022 and 2024 | Older editions of the exam, possibly seen in training |
| Same-type benchmark | MIR 2026 | Same exam, released after all models |
| Real-world questions | to be decided | Transfer to clinical practice |
| Reference (optional) | MedQA (USMLE) | Reproduce widely reported scores |

Every MIR edition is evaluated in the original Spanish and in English. English for 2022 and
2024 comes from CareQA's official translation (GPT-4); English for 2026 comes from the
scraped translation (Google machine translation).

Models: Gemma 3 1B, 4B and 27B, MedGemma 4B and 27B (text), Qwen3 1.7B and 4B, plus an
optional Qwen3 4B variant with thinking enabled.

```
Stage 1: preparation (once)                 Stage 2: evaluation              Stage 3: analysis
  scripts/prepare_all.sh                      scripts/submit_all.sh            python -m medeval.analyze
  datasets + models + parameter counts  ──>   one Slurm job per model    ──>   accuracy, cost metrics,
                                              (only missing results)           rankings, exam shift, plots
```

* **Stage 1** runs on a Snellius login node (internet access, no GPU).
* **Stage 2** runs as Slurm jobs on Snellius GPU nodes. Each model is loaded once and answers
  all questions of all available datasets.
* **Stage 3** only reads result files and takes seconds.

## 2. Before you start

You need:

* A Snellius account (for example `scur0455`) with an accepted usage agreement and a GPU budget.
* A Hugging Face account. Gemma 3 and MedGemma are gated: open each model page on
  huggingface.co while logged in and accept the license. MedGemma additionally requires
  accepting the Health AI Developer Foundations terms. Approval is usually immediate.
* A Hugging Face access token (huggingface.co, Settings, Access Tokens, type "Read").
* The scraped MIR 2026 files in `data/external/` (MIR 2026 is not on Hugging Face; see
  section 5): the original Spanish `mir2026_text_primary_es.csv` (included in the project)
  and the translated `mir_2026_en_mlt.csv` (add it to the repository).

## 3. Connecting to Snellius

### 3.1 First login

1. Go to https://portal.cua.surf.nl, log in, change your password and accept the Usage Agreement.
2. In a terminal (PowerShell on Windows):

   ```bash
   ssh scur0455@snellius.surf.nl
   ```

   Answer "yes" to the fingerprint question and enter your password.

Five wrong passwords in a row lead to a 24-hour ban on that login node.

### 3.2 Logging in from outside the university

Snellius only accepts connections from approved networks. From home, either connect to the
TU/e VPN first, or use SURF's door node:

```bash
ssh scur0455@doornode.hpcv.surf.nl
```

Choose Snellius from the menu and enter your password again.

### 3.3 Key-based login (recommended)

On your own computer:

```bash
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub
```

Copy the printed line, then in the SURF portal open "Public ssh keys", click "add key",
paste it and confirm with your password.

Add a shortcut to `~/.ssh/config` on your computer (Windows: `C:\Users\<you>\.ssh\config`):

```
Host snellius
    HostName snellius.surf.nl
    User scur0455
    IdentityFile ~/.ssh/id_ed25519
```

From now on, `ssh snellius` logs you in.

### 3.4 Copying files

```bash
scp -r medeval snellius:~/                 # project folder to Snellius
scp snellius:~/medeval/results/analysis/summary.csv .   # a result back to your computer
```

### 3.5 Getting the code onto Snellius with Git

The project lives in a private GitHub repository. On Snellius:

```bash
git config --global credential.helper store
git clone https://github.com/<owner>/<repository>.git
```

GitHub asks for a username and a password; the password must be a **personal access token**,
not your GitHub password (github.com, Settings, Developer settings, Personal access tokens,
Fine-grained, access to the repository with "Contents: Read and write"). After changing code
on your computer and pushing it, update Snellius with `git pull`.

### 3.6 Rules on Snellius

* The login node is only for editing files, installing software and submitting jobs.
  Never run models there; use Slurm jobs.
* Your home directory (`~`) has a 200 GB quota. Model weights go to
  `/scratch-shared/scur0455/`, where files unused for 14 days are deleted automatically.
* Every GPU-hour is charged to your budget: 192 SBU per H100 GPU-hour. Check your balance
  with `accinfo`.

## 4. Installing the pipeline

Get the project onto Snellius (section 3.5), log in, go to the project folder (the folder
containing `configs`, `scripts` and `src`) and run:

```bash
module load 2024
module avail Python               # press q to leave the list
```

Check that `Python/3.12.3-GCCcore-13.3.0` appears in the list. If the name differs, use the
listed name in the commands below by prefixing them with `PYTHON_MODULE=<name>`.

```bash
bash scripts/setup_env.sh
```

This creates a Python environment in `~/venvs/medeval`, installs vLLM and the other
dependencies, installs the pipeline itself, creates the cache folder on scratch and records
all installed versions in `requirements.lock.txt`. It takes about 10 to 15 minutes.

Then log in to Hugging Face once:

```bash
source ~/venvs/medeval/bin/activate
hf auth login                    # paste your access token (older versions: huggingface-cli login)
```

In every new terminal session, activate the environment again with
`source ~/venvs/medeval/bin/activate` before running any script.

## 5. Preparing data and models

### 5.1 Files in `data/external/`

`data/external/` holds the files that do not come from the internet automatically. The folder
is tracked by Git, so files added on your computer reach Snellius with `git pull`.

| File | Content | Status |
|---|---|---|
| `mir2026_text_primary_es.csv` | MIR 2026, original Spanish questions and answers (170 questions) | included |
| `mir2026_text_primary_es.jsonl` | the same in JSON Lines format (not used by the pipeline; kept for reference) | included |
| `mir_2026_en_mlt.csv` | MIR 2026, the same questions with the English translation | **add this file** |

MIR 2022 and 2024 need no files here: they are downloaded from CareQA on Hugging Face
(Spanish and English) by the preparation step.

### 5.2 One command

```bash
source ~/venvs/medeval/bin/activate
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &
tail -f logs/prepare.log          # Ctrl+C stops watching, not the preparation
```

This runs two steps:

1. **Datasets** (`scripts/prepare_data.py`, about a minute):
   * MIR 2022 and 2024 in Spanish, from CareQA on Hugging Face, written to
     `data/raw/mir_<year>_es.csv`
   * MIR 2022 and 2024 in English, from CareQA's official English version (translated with
     GPT-4 and validated by human evaluators), written to `data/mir_<year>_en.csv` with the
     Spanish text in parallel columns
   * MIR 2026 in Spanish, from `data/external/mir2026_text_primary_es.csv`, written to
     `data/raw/mir_2026_es.csv`
   * MIR 2026 in English, from `data/external/mir_2026_en_mlt.csv`, written to
     `data/mir_2026_en.csv` (`scripts/prepare_mir2026_en.py`)
   * a validation of all English files, with a spot-check sample, in `data/translation_checks/`
2. **Models** (`scripts/download_models.py`, about an hour): downloads all weights (about
   140 GB) to `/scratch-shared/$USER/hf_cache` and then writes the parameter counts for the
   FLOPs metric to `results/param_counts.json`.

At the end of the dataset step, a summary lists every dataset and its number of questions.
Expect roughly 170 to 180 questions per edition, with the same number in Spanish and English
for each edition.

### 5.3 Selection rules

All editions use the same rules, applied by `scripts/download_mir.py`:

* only the 200 main questions (the 10 reserve questions, 201 to 210, are removed, because
  the 2026 source has none)
* questions referring to an attached image ("Pregunta vinculada a la imagen") are removed;
  general mentions of imaging ("prueba de imagen") are kept
* annulled questions (no valid answer) are removed

Each edition gets a report, `data/raw/mir_<year>_es.report.json`, with the number of removed
questions per rule. The English files of 2022 and 2024 have a `.meta.json` file reporting how
many questions were matched between the Spanish and English versions, and whether any answers
differ between them (unmatched questions and answer mismatches should both be 0).

Points to settle before reporting results:

* The 2026 answers in the scraped file come from an exam-preparation website. Compare them
  with the official definitive answer key of the Ministry of Health. The answer distribution
  of 2026 is less balanced than in 2022 and 2024, which makes this check important.
* The original Spanish text of 2026 comes from the scraper; check a few questions for
  formatting artifacts (words split by hyphens, leftover numbering).

### 5.4 Translations

The English versions come from two translators: GPT-4 for 2022 and 2024 (CareQA, validated by
human evaluators) and Google machine translation for 2026 (scraped file). Report both in the
paper. Because the Spanish questions are the original exam text for every edition, the
Spanish comparison of old and new editions is free of translation effects; comparing
Spanish and English results per edition shows whether translation affects accuracy.

`data/translation_checks/` contains three files: `validation_summary.csv` (one row per edition;
the editions should look similar), `validation_flags.csv` (fields with possible problems, such
as numbers missing from the English text) and `spot_check_sample.csv` (15 questions per edition
for a Spanish-speaking group member to compare, filling in `reviewer_ok` and `reviewer_notes`).

### 5.5 Optional: MedQA

```bash
python scripts/prepare_medqa.py --hf-name awinml/medqa --split test --out data/medqa_test.jsonl
DATASETS="medqa" bash scripts/submit_all.sh
```

MedQA is marked as optional, so it is only evaluated when requested.

## 6. Checking the models

`prepare_all.sh` prints the parameter count of each model at the end of `logs/prepare.log`.
Check that they roughly match the model names (about 1B, 4B and 27B for Gemma, 1.7B and 4B for
Qwen). A "401" or "gated repo" error during the download means the license of that model has
not been accepted yet (section 2). Because scratch files are deleted after 14 days without
use, rerun `python scripts/download_models.py` if a job later reports a missing model.

## 7. Running the evaluation

### 7.1 Smoke test

Run the smoke test first, and again after changing prompts or settings. It evaluates one small
model on 20 questions of every available dataset (about 20 minutes, at most 64 SBU):

```bash
sbatch slurm/smoke_test.sbatch
squeue -u $USER                    # wait until the job disappears
cat logs/medeval_smoke_*.out
head -n 3 results_smoke/raw/qwen3-4b/mir_2026_es.jsonl
```

Each line in the `.jsonl` files shows the model's `response`, the `predicted` letter and
whether it was `correct`. Check that:

* `predicted` is filled in for nearly all questions (many `null` values mean the model answers
  in a format the parser does not recognise)
* `finish_reason` is `stop` (the value `length` means the answer was cut off)
* accuracy is clearly above chance (25% with four options)

### 7.2 Full run

```bash
bash scripts/submit_all.sh
```

This submits one job per model for every available dataset that has no results yet, so it can
safely be run again at any time: after adding a dataset, only the new dataset is evaluated.

| Variation | Command |
|---|---|
| Only some datasets | `DATASETS="mir_2026 mir_2026_es" bash scripts/submit_all.sh` |
| Include the optional Qwen3 thinking variant | `INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh` |
| Rerun everything, overwriting results | `FORCE=1 bash scripts/submit_all.sh` |
| Change the latency sample (default 20) | `LATENCY_SAMPLE=50 bash scripts/submit_all.sh` |

### 7.3 Monitoring jobs

| Task | Command |
|---|---|
| List your jobs | `squeue -u $USER` |
| Follow a job's output | `tail -f logs/medeval_<model>_<jobid>.out` |
| Runtime and status after completion | `sacct -j <jobid> --format=JobID,JobName,Elapsed,State` |
| Cancel a job | `scancel <jobid>` |
| Remaining budget | `accinfo` |

A job in state `PD` (pending) is waiting for a free GPU.

## 8. Analysing the results

When all jobs have finished:

```bash
python -m medeval.analyze
git add results/analysis && git commit -m "Update analysis" && git push
```

The analysis folder is tracked by Git, so the results can be pulled on any computer. Pushing
from Snellius requires a GitHub token with write access (section 3.5).

### 8.1 Output files

All files are written to `results/analysis/`.

**`summary.csv`**: one row per model and dataset.

| Column | Meaning |
|---|---|
| `accuracy`, `accuracy_ci_low`, `accuracy_ci_high` | share correct, with 95% bootstrap confidence interval |
| `chance_level` | expected accuracy of random guessing (mean of 1/k) |
| `chance_corrected_acc` | (accuracy − chance) / (1 − chance); 0 means no better than guessing |
| `invalid_rate` | share of responses without a recognisable answer (counted as wrong) |
| `truncated_rate` | share of responses cut off at the token limit |
| `input_tokens_per_question`, `output_tokens_per_question` | mean tokens processed and generated |
| `flops_per_question` | 2 × N × (input + output tokens), N from `param_counts.json` |
| `gpu_seconds_per_question` | measured generation time × number of GPUs, per question |
| `flops_per_above_chance_correct` | FLOPs per question / chance-corrected accuracy; empty if at or below chance |
| `gpu_seconds_per_above_chance_correct` | the same with measured GPU time |
| `median_latency_seconds` | median time to answer one question on its own (from the latency sample) |
| `weights_memory_gb` | memory needed for the model weights in bfloat16 |
| `on_pareto_frontier` | true if no other model is both cheaper (FLOPs) and more accurate on this dataset |
| `language` | `en` for translated questions, `es` for the original Spanish |
| `gpu_name` | GPU used, to confirm all models ran on the same hardware |

Peak GPU memory is reported as the memory needed for the weights rather than measured,
because vLLM reserves a fixed share of the GPU (90%) regardless of the model's actual needs.

**`rankings.csv`**: each model's rank per dataset, by accuracy and by cost-efficiency
(lowest FLOPs per above-chance correct answer ranks first).

**`rank_correlations.csv`**: Spearman correlation between datasets of the cost-efficiency
rankings. A value near 1 means the same models are most cost-efficient on both datasets; low
or negative values mean the ranking on one dataset is a poor guide to the other.

**`exam_shift.csv`**: per model and language, accuracy on the new exam edition minus
accuracy on the older editions, with a 95% bootstrap confidence interval. A negative difference whose
interval excludes zero indicates lower accuracy on the unseen edition.

**`accuracy_vs_flops_<dataset>.png`**: accuracy against FLOPs per question (log scale),
with the Pareto frontier as a solid line and the chance level as a dashed line.

**Translation effect**: compare `accuracy` for `mir_<year>` and `mir_<year>_es` per model. A
large difference means translation affects the results, which should be reported.

### 8.2 Interpreting with care

* With about 180 questions per MIR year, confidence intervals are roughly ±7 percentage
  points. Treat smaller differences as inconclusive.
* A drop from older to new MIR editions can come from exposure to the older questions or from
  the new exam being harder. Compare the drop across models: a similar drop for all models
  points to difficulty, a drop concentrated in some models points to exposure.
* Check `invalid_rate` before comparing models; a model that answers in an unexpected
  format loses accuracy for reasons unrelated to medical knowledge.

## 9. Configuration reference

### 9.1 `configs/models.yaml`

Settings under `defaults` apply to all models unless a model overrides them.

| Field | Meaning |
|---|---|
| `hf_id` | Hugging Face model identifier |
| `family`, `size_b`, `medical`, `release` | metadata used in the analysis and the paper |
| `dtype` | numeric precision (bfloat16) |
| `max_model_len` | maximum tokens (prompt plus answer) per question |
| `gpu_memory_utilization` | share of GPU memory vLLM may use |
| `tensor_parallel_size` | number of GPUs per model (1 for all current models) |
| `seed` | random seed |
| `thinking` | whether reasoning before the answer is allowed |
| `chat_template_kwargs` | options passed to the chat template, e.g. Qwen3's `enable_thinking` |
| `llm_kwargs` | extra vLLM options, e.g. disabling image inputs for Gemma 3 |
| `slurm_time` | time limit of the evaluation job |
| `optional` | if true, only submitted with `INCLUDE_OPTIONAL=1` |

### 9.2 `configs/datasets.yaml`

| Block | Field | Meaning |
|---|---|---|
| `datasets.<key>` | `format`, `path` | file type (`jsonl` or `csv`) and location |
| | `group` | `older_exam`, `new_exam`, `real_world` or `reference`; drives the analysis |
| | `language` | `en` or `es`; exam shifts are computed per language |
| | `columns` | for CSV files: which column holds the id, year, question, options and answer |
| | `optional` | if true, only evaluated when requested with `DATASETS=...` |
| `default_datasets` | | `auto`: every non-optional dataset whose input file exists |
| `evaluation` | `system_prompt`, `instruction_direct`, `instruction_thinking` | prompt texts |
| | `temperature` | 0 means always the most likely answer |
| | `max_tokens_direct`, `max_tokens_thinking` | answer length limits |
| `analysis` | `flops_param_count` | `lm_total` or `lm_non_embedding` |
| | `bootstrap_samples`, `bootstrap_seed` | settings for confidence intervals |

After changing prompts or generation settings, rerun the smoke test and then all models, so
every model is evaluated with identical settings.

## 10. Extending the study

### 10.1 Adding the real-world dataset

1. Convert it to jsonl, one question per line:

   ```json
   {"id": "rw-001", "year": "2026", "question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "answer": "C"}
   ```

2. Save it as `data/real_world.jsonl`.
3. In `configs/datasets.yaml`, uncomment the `real_world` entry.
4. Run `bash scripts/submit_all.sh`; the new dataset is detected automatically and only it is
   evaluated.

The pipeline handles multiple-choice questions only. Datasets with free-text answers would
need an additional grading step, which is not part of the pipeline.

### 10.2 Adding a model

Add an entry to `configs/models.yaml` (copy an existing one and change `hf_id` and the
metadata), then run `python scripts/download_models.py --models <key>` (which also updates the
parameter counts), the smoke test with `sbatch --export=ALL,MODEL_KEY=<key> slurm/smoke_test.sbatch`,
and finally `bash scripts/submit_all.sh`, which only runs the new model.

## 11. Troubleshooting

| Problem | Likely cause | Solution |
|---|---|---|
| `ssh: connect ... timed out` | Off-campus network not allowed | Use the TU/e VPN or `doornode.hpcv.surf.nl` |
| `Permission denied` after several attempts | Ban after 5 failed logins | Wait 24 hours or use another login node; set up key login |
| `module: Python/... not found` | Different module name | `module avail Python` and set `PYTHON_MODULE` |
| `401` or `gated repo` while downloading | License not accepted or not logged in | Accept the license on the model page; `hf auth login` |
| Job fails with "cannot find model" or offline error | Weights missing or deleted from scratch | Run `download_models.py` again on a login node |
| `Disk quota exceeded` | Weights stored in home | Check that `HF_HOME` points to `/scratch-shared/$USER/hf_cache` |
| CUDA out of memory | Context length too large for the GPU | Lower `max_model_len` for that model |
| Job stays pending for a long time | GPUs busy | Wait; check with `squeue -u $USER` |
| Job hits its time limit | Time limit too short | Increase `slurm_time` in `models.yaml` and resubmit |
| High `invalid_rate` | Model answers in an unexpected format | Inspect `response` fields; adjust the prompt or the parser and rerun all models |
| High `truncated_rate` | Token limit too low (mainly with thinking) | Increase `max_tokens_thinking` or `max_model_len` |
| `download_mir.py pdf` finds far fewer than 210 questions | Different PDF layout | Check the warnings; use the `csv` source with an existing scrape instead |

## 12. Reproducibility checklist for the paper

Record and report:

* vLLM and PyTorch versions (`requirements.lock.txt`, also stored in every `.meta.json`)
* GPU type (`gpu_name` in `summary.csv`), dtype, temperature, seed and token limits
* exact model identifiers and whether Qwen3 thinking was enabled
* MedQA version (dataset name, split, number of options, number of questions)
* MIR sources (CareQA or official PDFs), selection rules (200 main questions; image-based,
  annulled and reserve questions removed) and the number of questions per year
  (`data/raw/mir_<year>_es.report.json`)
* translation sources (CareQA's GPT-4 translation for 2022 and 2024, the scraped Google
  machine translation for 2026; see `data/mir_<year>_en.meta.json`) and the spot-check result
* the parameter counts used for FLOPs (`results/param_counts.json`) and the choice of
  `lm_total` or `lm_non_embedding`
* the prompts (`configs/datasets.yaml`)

Publish the code, the configuration files, the item identifiers and the raw outputs in
`results/raw/`.

## 13. Quick reference

```bash
# Every session
ssh snellius
cd ~/<project folder> && source ~/venvs/medeval/bin/activate

# Once
bash scripts/setup_env.sh
hf auth login

# Preparation: datasets, models, parameter counts
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &

# Evaluation
sbatch slurm/smoke_test.sbatch
bash scripts/submit_all.sh          # safe to rerun; only missing results are submitted
squeue -u $USER

# Analysis
python -m medeval.analyze
git add results/analysis && git commit -m "Update analysis" && git push
```
