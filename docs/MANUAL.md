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

The study measures how accurate open LLMs are on medical multiple-choice questions, how much
compute each correct answer costs, and whether accuracy on older exam editions is inflated by
memorization:

| Question type | Dataset | Role in the study |
|---|---|---|
| Standard benchmark | MIR 2020 to 2024 | Older editions of the exam, possibly seen in training |
| Same-type benchmark | MIR 2025 and 2026 | Newest editions, held after most (2025) or all (2026) model cutoffs |
| Real-world questions | to be decided | Transfer to clinical practice |
| Reference (optional) | MedQA (USMLE) | Reproduce widely reported scores |

Every MIR edition is evaluated in the original Spanish and in English. English for 2020 to
2024 comes from CareQA's official translation (GPT-4); English for 2025 and 2026 comes from
Google machine translation.

Models: Gemma 3 1B, 4B and 27B, MedGemma 4B and 27B (text), Llama 3 8B and Med42-v2 8B,
Qwen3 1.7B and 4B.

```
Stage 1: preparation                    Stage 2: evaluation              Stage 3: analysis
  scripts/prepare_all.sh                  scripts/submit_all.sh            python src/analyze.py
  datasets + models + parameter    ──>    one Slurm job per model    ──>   python src/contamination.py
  counts (only what is missing)           (only missing results)           accuracy, cost, exam shift,
                                                                           memorization checks, plots
```

* **Stage 1** runs on a Snellius login node (internet access, no GPU).
* **Stage 2** runs as Slurm jobs on Snellius GPU nodes. Each model is loaded once and answers
  all questions of all available datasets.
* **Stage 3** only reads result files and takes seconds to a minute; it also runs on your own
  computer.

The code follows these stages: every file in `src/` is one step, and every file in
`scripts/` is a command run on Snellius.

| Stage | File (`python src/<file>.py`) | What it does |
|---|---|---|
| 1 | `prepare_data` | builds all dataset files and checks the translations (uses `mir_sources`, `mir_corrections`) |
| 1 | `prepare_medqa` | optional MedQA reference set |
| 1 | `models` | downloads missing model weights and counts parameters |
| 2 | `run_eval` | loads the datasets, builds the prompts, runs the model and parses the answers |
| 3 | `analyze` | accuracy, cost metrics, rankings, Pareto frontier, exam shift, plots |
| 3 | `contamination` | year-difficulty adjustment, unparseable-answer check, MedExpQA positive control |

## 2. Before you start

You need:

* A Snellius account (for example `scur0455`) with an accepted usage agreement and a GPU budget.
* A Hugging Face account. Gemma 3, MedGemma and Llama 3 are gated: open each model page on
  huggingface.co while logged in and accept the license. MedGemma additionally requires
  accepting the Health AI Developer Foundations terms. Approval is usually immediate.
* A Hugging Face access token (huggingface.co, Settings, Access Tokens, type "Read").

All files that do not come from the internet (the 2025 and 2026 exam sources, the
translations and the human reference scores) are included in `data/external/`; see section 5.

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

This creates a Python environment in `~/venvs/medeval` (or reuses it; `FRESH=1` rebuilds it),
installs vLLM and all other dependencies listed in `requirements.txt`,
creates the cache folder on scratch and records all installed versions in
`requirements.lock.txt`. It takes about 10 to 15 minutes.

Then log in to Hugging Face once:

```bash
source ~/venvs/medeval/bin/activate
hf auth login                    # paste your access token (older versions: huggingface-cli login)
```

In every new terminal session, activate the environment again with
`source ~/venvs/medeval/bin/activate` before running any command.

**On your own computer** (data preparation and analysis only, no GPU): create a virtual
environment and run `pip install -r requirements.txt` in the project folder. This installs
everything except vLLM.

## 5. Preparing data and models

### 5.1 Files in `data/external/`

`data/external/` holds the files that do not come from the internet automatically. The folder
is tracked by Git, so files added on your computer reach Snellius with `git pull`.

| File | Content |
|---|---|
| `mir2025_questions_v0.pdf`, `mir2025_answers_definitive_v0.pdf` | MIR 2025, official Version 0 booklet and definitive answer key |
| `mir_2025_en_mlt.csv` | MIR 2025, English translation of the selected questions (same procedure as 2026) |
| `mir2026_text_primary_es.csv` | MIR 2026, original Spanish questions and answers (170 questions) |
| `mir_2026_en_mlt.csv` | MIR 2026, the same questions with the English translation |
| `mir_human_difficulty_2020_2026.csv` | human reference accuracy per exam year (top 10% of candidates), with sources |

MIR 2020 to 2024 need no files here: they are downloaded from CareQA on Hugging Face (Spanish
and English) by the preparation step.

### 5.2 One command

```bash
source ~/venvs/medeval/bin/activate
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &
tail -f logs/prepare.log          # Ctrl+C stops watching, not the preparation
```

This runs two steps:

1. **Datasets** (`python src/prepare_data.py`, about a minute):
   * MIR 2020 to 2024 in Spanish, from CareQA on Hugging Face, written to
     `data/raw/mir_<year>_es.csv`
   * MIR 2020 to 2024 in English, from CareQA's official English version (translated with
     GPT-4 and validated by human evaluators), written to `data/mir_<year>_en.csv` with the
     Spanish text in parallel columns
   * MIR 2025 in Spanish, from the official PDFs, written to `data/raw/mir_2025_es.csv`; the
     extraction stops if the PDFs or the result differ from the audited version
   * MIR 2025 and 2026 in English, from the translations in `data/external/`, written to
     `data/mir_<year>_en.csv`
   * MIR 2026 in Spanish, from `data/external/mir2026_text_primary_es.csv`, written to
     `data/raw/mir_2026_es.csv`
   * a check of all English files, with a spot-check sample, in `data/translation_checks/`
2. **Models** (`python src/models.py`): downloads the weights of every model that is not
   yet complete in `/scratch-shared/$USER/hf_cache` (about 140 GB for a first download, which
   takes about an hour), then counts the parameters of every model missing from
   `results/param_counts.json`. When everything is in place, this step only reports that and
   finishes immediately. `SKIP_MODELS=1 bash scripts/prepare_all.sh` runs only the dataset step.

At the end of the dataset step, a summary lists every dataset and its number of questions.
Expect 145 to 172 questions per edition, with the same number in Spanish and English for
each edition. To rerun only the translation checks: `python src/prepare_data.py --checks-only`.

### 5.3 Selection rules

All editions use the same rules, applied in `src/mir_sources.py`:

* only the main questions (175 for 2020 and 2021, 200 from 2022 on); reserve questions are
  removed
* questions referring to an attached image ("Pregunta vinculada a la imagen") are removed;
  general mentions of imaging ("prueba de imagen") are kept
* annulled questions (no valid answer) are removed

Audited corrections of transcription errors in the sources (`src/mir_corrections.py`)
are applied before selection. Each edition gets a report,
`data/raw/mir_<year>_es.report.json`, with the number of removed questions per rule. Each
English file has a `.meta.json` file reporting how it was matched to the Spanish version.

Points to settle before reporting results:

* The 2026 answers in the scraped file come from an exam-preparation website. Compare them
  with the official definitive answer key of the Ministry of Health. The answer distribution
  of 2026 is less balanced than in other years, which makes this check important.
* The 2025 human reference value in `mir_human_difficulty_2020_2026.csv` is marked "VERIFY"
  in its notes; confirm it on the Ministry portal.

### 5.4 Translations

The English versions come from two translators: GPT-4 for 2020 to 2024 (CareQA, validated by
human evaluators) and Google machine translation for 2025 and 2026. Report both in the paper.
Because the Spanish questions are the original exam text for every edition, the Spanish
comparison of old and new editions is free of translation effects; comparing Spanish and
English results per edition shows whether translation affects accuracy. Zero-width spaces
left by the machine translation are removed during preparation. In 2025, question 38, the
translation corrects a typographical error in the Spanish dose ("0,3-05 mL" rendered as
"0.3-0.5 mL").

`data/translation_checks/` contains three files: `validation_summary.csv` (one row per edition;
the editions should look similar), `validation_flags.csv` (fields with possible problems, such
as numbers missing from the English text; numbers written out in words are also flagged) and
`spot_check_sample.csv` (15 questions per edition for a Spanish-speaking group member to
compare, filling in `reviewer_ok` and `reviewer_notes`).

### 5.5 Optional: MedQA

```bash
python src/prepare_medqa.py --hf-name awinml/medqa --split test --out data/medqa_test.jsonl
DATASETS="medqa" bash scripts/submit_all.sh
```

MedQA is marked as optional, so it is only evaluated when requested.

## 6. Checking the models

`prepare_all.sh` prints the parameter count of each newly counted model at the end of
`logs/prepare.log`; all counts are in `results/param_counts.json`. Check that they roughly
match the model names (about 1B, 4B and 27B for Gemma, 1.7B and 4B for Qwen, 8B for Llama).
A "401" or "gated repo" error during the download means the license of that model has not been
accepted yet (section 2). Because scratch files are deleted after 14 days without use, rerun
`python src/models.py` (or `prepare_all.sh`) if a job later reports a missing model; only
the missing models are downloaded again.

## 7. Running the evaluation

### 7.1 Smoke test

Run the smoke test first, and again after changing prompts or settings. It evaluates one small
model on 20 questions of every available dataset (about 20 minutes, at most 64 SBU):

```bash
SMOKE=1 sbatch scripts/evaluate.sbatch                  # default model qwen3-4b
SMOKE=1 MODEL_KEY=gemma3-4b sbatch scripts/evaluate.sbatch
squeue -u $USER                    # wait until the job disappears
cat logs/medeval_*.out
head -n 3 results_smoke/raw/qwen3-4b/mir_2026_es.jsonl
```

Each line in the `.jsonl` files shows the model's `response`, the `predicted` letter and
whether it was `correct`. Check that:

* `predicted` is filled in for nearly all questions (many `null` values mean the model answers
  in a format the parser does not recognise)
* `finish_reason` is `stop` (the value `length` means the answer was cut off)
* accuracy is clearly above chance (25% with four options)

Without a GPU, `python src/run_eval.py --model qwen3-4b --backend mock --limit 5` tests
data loading, prompting and the output files with random answers.

### 7.2 Full run

```bash
bash scripts/submit_all.sh
```

This submits one job per model (`scripts/evaluate.sbatch`) for every available dataset that
has no results yet, so it can safely be run again at any time: after adding a dataset, only
the new dataset is evaluated.

| Variation | Command |
|---|---|
| Only some datasets | `DATASETS="mir_2025 mir_2026" bash scripts/submit_all.sh` |
| Include optional models | `INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh` |
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
python src/analyze.py
python src/contamination.py
git add results && git commit -m "Update results" && git push
```

`results/raw/` and `results/analysis/` are tracked by Git, so the results can be pulled on any
computer. Pushing from Snellius requires a GitHub token with write access (section 3.5). The
FLOPs columns need `results/param_counts.json`, which is not tracked: run `analyze` where
`prepare_all.sh` was run, or copy the file.

### 8.1 Output files of `src/analyze.py`

All files are written to `results/analysis/`.

**`summary.csv`**: one row per model and dataset.

| Column | Meaning |
|---|---|
| `accuracy`, `accuracy_ci_low`, `accuracy_ci_high` | share correct, with 95% bootstrap confidence interval |
| `chance_level` | expected accuracy of random guessing (mean of 1/k) |
| `chance_corrected_acc` | (accuracy - chance) / (1 - chance); 0 means no better than guessing |
| `invalid_rate` | share of responses without a recognisable answer (counted as wrong) |
| `truncated_rate` | share of responses cut off at the token limit |
| `input_tokens_per_question`, `output_tokens_per_question` | mean tokens processed and generated |
| `flops_per_question` | 2 x N x (input + output tokens), N from `param_counts.json` |
| `gpu_seconds_per_question` | measured generation time x number of GPUs, per question |
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

**`exam_shift.csv`**: per model and language, accuracy on unseen minus seen exam years, with
a 95% bootstrap confidence interval, for two splits of the years (column `split`):

* `common`: the same years for every model, set in `analysis.common_split` (seen 2020 to 2024,
  unseen 2025 and 2026), so the rows are directly comparable; it is not exact for Llama 3 and
  Med42 (2024 is already unseen) or Qwen3 (2025 may have been seen)
* `per_model`: unseen from each model's `unseen_from` year in `configs/models.yaml`

**`accuracy_vs_flops_<dataset>.png`**: accuracy against FLOPs per question (log scale),
with the Pareto frontier as a solid line and the chance level as a dashed line.

**Translation effect**: compare `accuracy` for `mir_<year>` and `mir_<year>_es` per model. A
large difference means translation affects the results, which should be reported.

### 8.2 Output files of `src/contamination.py`

The three checks use the Spanish answers in `results/raw/` and write to
`results/analysis/contamination/`. Shifts are in percentage points; negative means lower
accuracy on unseen exams, which is what memorization predicts.

| File | Content |
|---|---|
| `model_based.csv` | **main result**: the effect of an exam being unseen, after removing the difficulty of each exam year, pooled and per model (regression `correct ~ model + year + unseen`, per-model split, standard errors clustered by question) |
| `year_difficulty.csv` | difficulty of each year: the model-based estimate next to the human reference |
| `human_based.csv` | check: shift after subtracting the human reference accuracy per year (years with a reference only) |
| `accuracy_by_year.csv` | accuracy per model and year, with seen or unseen status |
| `unparseable_by_year.csv`, `unparseable_reasons.csv`, `unparseable_answers.csv` | how often answers had no option letter, per year and why |
| `shift_with_without.csv` | the year-adjusted shift with all answers and with parseable answers only |
| `mir_overlap.csv`, `matched_questions.csv` | positive control: MedGemma's lead over Gemma 3 on MIR questions in its MedExpQA training data versus other MIR questions of the same years |

The difficulty adjustment works because the training cutoffs differ between models: within
the same exam year, models that could have seen the exam are compared with models that could
not. With the common split, "unseen" would be the same year for every model and the two
effects could not be separated.

Options:

| Task | Command |
|---|---|
| Only some checks | `python src/contamination.py --checks year unparseable` |
| Sensitivity check with a different cutoff | `python src/contamination.py --unseen-from qwen3-1.7b=2025 qwen3-4b=2025 --out results/analysis/contamination_qwen2025` |
| MedExpQA from local files (no download) | `python src/contamination.py --medexpqa-dir <folder>` |

### 8.3 Interpreting with care

* With about 170 questions per MIR year, confidence intervals are roughly ±7 percentage
  points. Treat smaller differences as inconclusive.
* A drop from older to new MIR editions can come from exposure to the older questions or from
  the new exam being easier or harder. Use the year-adjusted result in `model_based.csv`, and
  compare it with the human-based check.
* Check `invalid_rate` before comparing models; a model that answers in an unexpected
  format loses accuracy for reasons unrelated to medical knowledge.

## 9. Configuration reference

### 9.1 `configs/models.yaml`

Settings under `defaults` apply to all models unless a model overrides them.

| Field | Meaning |
|---|---|
| `hf_id` | Hugging Face model identifier |
| `family`, `size_b`, `medical`, `release` | metadata used in the analysis and the paper |
| `unseen_from` | first MIR exam year after the training cutoff (per-model split) |
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
| | `group` | `older_exam`, `new_exam`, `real_world` or `reference`; reported in `summary.csv` |
| | `language` | `en` or `es`; exam shifts are computed per language |
| | `columns` | for CSV files: which column holds the id, year, question, options and answer |
| | `optional` | if true, only evaluated when requested with `DATASETS=...` |
| `default_datasets` | | `auto`: every non-optional dataset whose input file exists |
| `evaluation` | `system_prompt`, `instruction_direct`, `instruction_thinking` | prompt texts |
| | `temperature` | 0 means always the most likely answer |
| | `max_tokens_direct`, `max_tokens_thinking` | answer length limits |
| `analysis` | `flops_param_count` | `lm_total` or `lm_non_embedding` |
| | `bootstrap_samples`, `bootstrap_seed` | settings for confidence intervals |
| | `common_split` | seen and unseen exam years shared by all models |
| | `exclude_from_regression` | models left out of the year-difficulty regression (near chance) |
| | `human_difficulty` | file with the human reference accuracy per year |

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

Add an entry to `configs/models.yaml` (copy an existing one and change `hf_id`, the metadata
and `unseen_from`), then run `python src/models.py --models <key>` (downloads the weights
and adds the parameter count), the smoke test with
`SMOKE=1 MODEL_KEY=<key> sbatch scripts/evaluate.sbatch`, and finally
`bash scripts/submit_all.sh`, which only runs the new model.

## 11. Troubleshooting

| Problem | Likely cause | Solution |
|---|---|---|
| `ssh: connect ... timed out` | Off-campus network not allowed | Use the TU/e VPN or `doornode.hpcv.surf.nl` |
| `Permission denied` after several attempts | Ban after 5 failed logins | Wait 24 hours or use another login node; set up key login |
| `module: Python/... not found` | Different module name | `module avail Python` and set `PYTHON_MODULE` |
| `401` or `gated repo` while downloading | License not accepted or not logged in | Accept the license on the model page; `hf auth login` |
| Job fails with "cannot find model" or offline error | Weights missing or deleted from scratch | Run `python src/models.py` again on a login node |
| `Disk quota exceeded` | Weights stored in home | Check that `HF_HOME` points to `/scratch-shared/$USER/hf_cache` |
| CUDA out of memory | Context length too large for the GPU | Lower `max_model_len` for that model |
| Job stays pending for a long time | GPUs busy | Wait; check with `squeue -u $USER` |
| Job hits its time limit | Time limit too short | Increase `slurm_time` in `models.yaml` and resubmit |
| High `invalid_rate` | Model answers in an unexpected format | Inspect `response` fields; adjust the prompt or the parser and rerun all models |
| High `truncated_rate` | Token limit too low (mainly with thinking) | Increase `max_tokens_direct`, `max_tokens_thinking` or `max_model_len` |
| MIR 2025 preparation stops with "STOP" | The PDFs or the extracted text differ from the audited version | Check that the Version 0 PDFs in `data/external/` are unchanged |
| FLOPs columns empty in `summary.csv` | `results/param_counts.json` missing | Run `python src/models.py` or copy the file from Snellius |

## 12. Reproducibility checklist for the paper

Record and report:

* vLLM and PyTorch versions (`requirements.lock.txt`, also stored in every `.meta.json`)
* GPU type (`gpu_name` in `summary.csv`), dtype, temperature, seed and token limits
* exact model identifiers, that Qwen3 thinking was disabled, and the `unseen_from` year of
  every model with its source
* MedQA version (dataset name, split, number of options, number of questions), if used
* MIR sources (CareQA, official PDFs, scraped 2026 file), selection rules and the number of
  questions per year (`data/raw/mir_<year>_es.report.json`)
* translation sources (CareQA's GPT-4 translation for 2020 to 2024, Google machine
  translation for 2025 and 2026; see `data/mir_<year>_en.meta.json`) and the spot-check result
* the human reference accuracies and their sources (`data/external/mir_human_difficulty_2020_2026.csv`)
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

# Preparation: datasets, missing models, missing parameter counts
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &

# Evaluation
SMOKE=1 sbatch scripts/evaluate.sbatch
bash scripts/submit_all.sh          # safe to rerun; only missing results are submitted
squeue -u $USER

# Analysis
python src/analyze.py
python src/contamination.py
git add results && git commit -m "Update results" && git push
```
