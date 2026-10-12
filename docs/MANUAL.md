# medeval: User Manual

This manual explains how to run the complete study, from logging in to Snellius to
reading the final results. Follow the parts in order the first time; afterwards, use the
quick reference at the end.

**Contents**

1. What the pipeline does
2. Before you start
3. Connecting to Snellius
4. Installing the pipeline
5. Preparing the data
6. Downloading the models
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
| Translation check (optional) | MIR in the original Spanish | Does translation affect accuracy? |
| Reference (optional) | MedQA (USMLE) | Reproduce widely reported scores |

Models: Gemma 3 1B, 4B and 27B, MedGemma 4B and 27B (text), Qwen3 1.7B and 4B, plus an
optional Qwen3 4B variant with thinking enabled.

The work is split into three stages, each run separately:

```
Stage 1: data preparation (once)               Stage 2: evaluation (once per model)   Stage 3: analysis
  download MIR (Spanish) ─ translate ─ validate ─> data/mir_<year>_en.csv ─> results/raw/ ─> accuracy, cost metrics,
  (optional) MedQA from Hugging Face ──────────> data/medqa_test.jsonl        (one job per    rankings, exam shift,
                                                                               model, 1 H100)  Pareto plots
```

* **Stage 1** runs on your own computer or a Snellius login node. It needs internet access
  but no GPU.
* **Stage 2** runs as Slurm jobs on Snellius GPU nodes. Each model is loaded once and answers
  all questions of all datasets.
* **Stage 3** runs anywhere (it only reads result files) and takes seconds.

## 2. Before you start

You need:

* A Snellius account (for example `scur0455`) with an accepted usage agreement and a GPU budget.
* A Hugging Face account. Gemma 3 and MedGemma are gated: open each model page on
  huggingface.co while logged in and accept the license. MedGemma additionally requires
  accepting the Health AI Developer Foundations terms. Approval is usually immediate.
* A Hugging Face access token (huggingface.co, Settings, Access Tokens, type "Read").
* For the MIR translation: either a Google Cloud project with the Cloud Translation API
  enabled, or the free `deep-translator` package (see section 5.3 for the trade-off).
* For MIR 2026 (not on Hugging Face): the official booklet and answer key as PDF, or the
  CSV from Asen's scraper (see section 5.2).

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

### 3.5 Rules on Snellius

* The login node is only for editing files, installing software and submitting jobs.
  Never run models there; use Slurm jobs.
* Your home directory (`~`) has a 200 GB quota. Model weights go to
  `/scratch-shared/scur0455/`, where files unused for 14 days are deleted automatically.
* Every GPU-hour is charged to your budget: 192 SBU per H100 GPU-hour. Check your balance
  with `accinfo`.

## 4. Installing the pipeline

Copy the project to Snellius (section 3.4), log in and run:

```bash
cd ~/medeval
module avail Python
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
huggingface-cli login            # paste your access token
```

In every new terminal session, activate the environment again with
`source ~/venvs/medeval/bin/activate` before running any script.

## 5. Preparing the data

All data preparation runs on your own computer or a Snellius login node (internet, no GPU).
Install its dependencies once:

```bash
pip install -r requirements-data.txt
```

All input data lives in `data/`. The pipeline reads these files:

```
data/mir_2022_en.csv      MIR 2022 (Spanish and English side by side)
data/mir_2024_en.csv      MIR 2024
data/mir_2026_en.csv      MIR 2026
data/medqa_test.jsonl     optional, only for the MedQA reference run
```

The Spanish versions (`mir_<year>_es` datasets) are read from the same files, from the
Spanish columns.

### 5.1 MedQA (optional reference)

```bash
python scripts/prepare_medqa.py --hf-name awinml/medqa --split test --out data/medqa_test.jsonl
```

The script detects the layout of the dataset and prints the columns it used and the first
converted item. Check that item: the options should be labelled A, B, C, D (and E if
present) and the answer letter should match the correct option. If detection fails, pass
the column names explicitly with `--question-col`, `--options-col` and `--answer-col`.

Note the number of options and the split in the paper, since reported MedQA scores refer to
specific versions (most often the 4-option US test set with 1,273 questions).

### 5.2 MIR: collecting the Spanish questions

`scripts/download_mir.py` collects the Spanish questions of each year into
`data/raw/mir_<year>_es.csv`, in one shared format, from one of three sources:

**MIR 2020 to 2024: download from CareQA** (Hugging Face, built from the official
ministry booklets, with image-based and annulled questions already removed):

```bash
python scripts/download_mir.py careqa --years 2022 2024
```

**MIR 2026: from the official PDFs.** Download the exam booklet (use the official version,
"V0") and the definitive answer key from the Ministry of Health website, then:

```bash
python scripts/download_mir.py pdf --year 2026 \
    --questions-pdf cuaderno_mir_2026_v0.pdf --answers-pdf respuestas_mir_2026.pdf
```

URLs can be given instead of file paths. PDF layouts change between years, so check the
printed report: about 200 main and 10 reserve questions should be found, and the script
warns about question numbers it could not parse completely. Make sure the answer key
belongs to the same exam version as the booklet.

**Alternatively, from an existing CSV** (for example Asen's scraped 2026 file):

```bash
python scripts/download_mir.py csv --year 2026 --input mir2026_text_primary_es.csv \
    --column-map '{"pregunta": "question", "a": "option_a", "b": "option_b", "c": "option_c", "d": "option_d", "respuesta": "answer"}'
```

Adjust the column mapping to the actual column names. Answers may be letters (A to D) or
numbers (1 to 4).

All three sources apply the same rules: questions that mention an image are removed,
annulled questions (no valid answer) are removed, and whitespace is normalized. Each year
also gets a report, `data/raw/mir_<year>_es.report.json`, listing how many questions were
removed and why. Compare the reports across years; the numbers of remaining questions should
be similar. The output columns are:

| Column | Content |
|---|---|
| `id` | unique identifier, e.g. `mir2026-017` |
| `year` | exam year |
| `question_number` | number in the original exam |
| `question` | question text in Spanish |
| `option_a` ... `option_d` | the four answer options in Spanish |
| `answer` | correct letter, A to D |

Decide as a group whether to keep the 10 reserve questions, and apply the same choice to
every year (for the PDF source, `--max-question 200` drops them).

### 5.3 MIR: translation

All years must be translated with the same method, in the same period. Translate the
Spanish source files yourself, rather than combining your own translation of one year with
an existing translation of another (for example CareQA's English version), because a
difference in translation quality could otherwise explain differences in accuracy.

Choose one backend and use it for all years:

| Backend | Requirements | Reproducibility |
|---|---|---|
| `google-cloud` (recommended) | Google Cloud project with the Cloud Translation API enabled; `gcloud auth application-default login` | Model is selectable and recorded |
| `deep-translator` | none (free Google Translate web service) | Model cannot be pinned; translate all years in one session |

Run the translation and validation for all years in one go:

```bash
BACKEND=google-cloud PROJECT_ID=<your-project> bash scripts/prepare_mir.sh
# or
BACKEND=deep-translator bash scripts/prepare_mir.sh
```

Optional settings: `YEARS="2022 2024 2025 2026"` to change the years,
`MODEL=general/translation-llm` to use Google's LLM-based translation model, and
`COLUMN_MAP='{"pregunta": "question"}'` if your source columns are named differently.

Each question is translated together with its options, so that a term appearing in both
is translated the same way. If the translation damages the option markers, that question
is retranslated field by field; these cases are listed in the metadata file.

Outputs per year: `data/mir_<year>_en.csv` (Spanish and English side by side) and
`data/mir_<year>_en.meta.json` (backend, model and date; needed for the paper).

### 5.4 MIR: checking the translation

`prepare_mir.sh` ends by running the validation script, which writes three files to
`data/translation_checks/`:

* `validation_summary.csv`: one row per year. The years should look similar: comparable
  English-to-Spanish length ratios, a low fallback rate and few flagged fields.
* `validation_flags.csv`: every suspicious field, for example a number in the Spanish text
  that is missing from the English text (a lab value of 3,5 should appear as 3.5), an empty
  translation, or an unusually short or long translation. Check each flagged question and
  fix the English text by hand where needed.
* `spot_check_sample.csv`: 15 random questions per year. A group member who reads Spanish
  should compare Spanish and English, focusing on drug names, abbreviations, lab values and
  negations, and fill in the `reviewer_ok` and `reviewer_notes` columns. Report the share of
  acceptable translations per year in the paper.

## 6. Downloading the models

Compute nodes should not download anything, so download all weights first, from a login node:

```bash
source ~/venvs/medeval/bin/activate
python scripts/download_models.py
```

This stores about 140 GB in `/scratch-shared/scur0455/hf_cache`. A "401" or "gated repo"
error means the license for that model has not been accepted yet (section 2).

Then count the language-model parameters, which the FLOPs metric needs:

```bash
python scripts/count_params.py --out results/param_counts.json
```

The script reads only the file headers, so it runs in seconds. It excludes the image
components of the multimodal Gemma checkpoints and prints the counts per model.

Because scratch files are deleted after 14 days without use, run `download_models.py` again
if a job reports that a model cannot be found.

## 7. Running the evaluation

### 7.1 Smoke test

Always run the smoke test first, after installing and after changing prompts or settings.
It evaluates one small model on 20 questions per dataset (about 20 minutes, at most 64 SBU):

```bash
sbatch slurm/smoke_test.sbatch
```

When it has finished, check:

```bash
cat logs/medeval_smoke_*.out
head -n 3 results_smoke/raw/qwen3-4b/mir_2026.jsonl
```

Each line in the `.jsonl` files shows the model's `response`, the `predicted` letter and
whether it was `correct`. Look for:

* `predicted` filled in for nearly all questions. Many `null` values mean the model answers
  in a format the parser does not recognise; inspect the responses.
* `finish_reason` equal to `stop`. The value `length` means the answer was cut off.
* Accuracy clearly above chance (25% with four options).

To test another model the same way: `sbatch --export=ALL,MODEL_KEY=gemma3-4b slurm/smoke_test.sbatch`.

### 7.2 Full run

```bash
bash scripts/submit_all.sh
```

This submits one job per model, each with the time limit set in `configs/models.yaml`. To
include the optional Qwen3 thinking variant: `INCLUDE_OPTIONAL=1 bash scripts/submit_all.sh`.
To run only some datasets: `DATASETS="mir_2024 mir_2026" bash scripts/submit_all.sh`.
To include the Spanish versions and MedQA:
`DATASETS="mir_2022 mir_2024 mir_2026 mir_2022_es mir_2024_es mir_2026_es medqa" bash scripts/submit_all.sh`.

Each job also measures latency by answering the first 20 questions of every dataset one at a
time. Change the number with `LATENCY_SAMPLE=<n>` or disable it with `LATENCY_SAMPLE=0`.

### 7.3 Monitoring jobs

| Task | Command |
|---|---|
| List your jobs | `squeue -u $USER` |
| Follow a job's output | `tail -f logs/medeval_<model>_<jobid>.out` |
| Runtime and status after completion | `sacct -j <jobid> --format=JobID,JobName,Elapsed,State` |
| Cancel a job | `scancel <jobid>` |
| Remaining budget | `accinfo` |

A job in state `PD` (pending) is waiting for a free GPU; this can take from minutes to hours.

### 7.4 Rerunning

Rerunning a model overwrites its files in `results/raw/<model>/`. To keep an earlier run,
rename that folder first. A single model can be submitted on its own with:

```bash
sbatch --job-name=medeval_qwen3-4b --time=01:00:00 --export=ALL,MODEL_KEY=qwen3-4b slurm/eval_model.sbatch
```

## 8. Analysing the results

When all jobs have finished:

```bash
python -m medeval.analyze --results results
```

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
| `default_datasets` | | datasets evaluated when none are specified |
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
3. In `configs/datasets.yaml`, uncomment the `real_world` entry and add `real_world` to
   `default_datasets`.
4. Run the smoke test, then submit the models with `DATASETS="real_world" bash scripts/submit_all.sh`.

The pipeline handles multiple-choice questions only. Datasets with free-text answers would
need an additional grading step, which is not part of the pipeline.

### 10.2 Adding a model

Add an entry to `configs/models.yaml` (copy an existing one and change `hf_id` and the
metadata), then run `download_models.py --models <key>`, `count_params.py`, the smoke test
with `MODEL_KEY=<key>`, and finally submit the job.

## 11. Troubleshooting

| Problem | Likely cause | Solution |
|---|---|---|
| `ssh: connect ... timed out` | Off-campus network not allowed | Use the TU/e VPN or `doornode.hpcv.surf.nl` |
| `Permission denied` after several attempts | Ban after 5 failed logins | Wait 24 hours or use another login node; set up key login |
| `module: Python/... not found` | Different module name | `module avail Python` and set `PYTHON_MODULE` |
| `401` or `gated repo` while downloading | License not accepted or not logged in | Accept the license on the model page; `huggingface-cli login` |
| Job fails with "cannot find model" or offline error | Weights missing or deleted from scratch | Run `download_models.py` again on a login node |
| `Disk quota exceeded` | Weights stored in home | Check that `HF_HOME` points to `/scratch-shared/$USER/hf_cache` |
| CUDA out of memory | Context length too large for the GPU | Lower `max_model_len` for that model |
| Job stays pending for a long time | GPUs busy | Wait; check with `squeue -u $USER` |
| Job hits its time limit | Time limit too short | Increase `slurm_time` in `models.yaml` and resubmit |
| High `invalid_rate` | Model answers in an unexpected format | Inspect `response` fields; adjust the prompt or the parser and rerun all models |
| High `truncated_rate` | Token limit too low (mainly with thinking) | Increase `max_tokens_thinking` or `max_model_len` |
| Translation script: missing columns | Source CSV uses other names | Rename columns or pass `COLUMN_MAP` |
| `download_mir.py pdf` finds far fewer than 210 questions | Different PDF layout | Check the warnings; use the `csv` source with an existing scrape instead |
| Many translation fallbacks | Translator alters the option markers | Usually harmless; check those questions in the spot check |

## 12. Reproducibility checklist for the paper

Record and report:

* vLLM and PyTorch versions (`requirements.lock.txt`, also stored in every `.meta.json`)
* GPU type (`gpu_name` in `summary.csv`), dtype, temperature, seed and token limits
* exact model identifiers and whether Qwen3 thinking was enabled
* MedQA version (dataset name, split, number of options, number of questions)
* MIR sources (CareQA or official PDFs), selection rules (image and annulled questions
  removed, reserve questions) and the number of questions per year
  (`data/raw/mir_<year>_es.report.json`)
* translation backend, model and date (`data/mir_<year>_en.meta.json`) and the spot-check result
* the parameter counts used for FLOPs (`results/param_counts.json`) and the choice of
  `lm_total` or `lm_non_embedding`
* the prompts (`configs/datasets.yaml`)

Publish the code, the configuration files, the item identifiers and the raw outputs in
`results/raw/`.

## 13. Quick reference

```bash
# Every session
ssh snellius
cd ~/medeval && source ~/venvs/medeval/bin/activate

# Once: installation
bash scripts/setup_env.sh && huggingface-cli login

# Once: data (own computer or login node)
pip install -r requirements-data.txt
python scripts/download_mir.py careqa --years 2022 2024
python scripts/download_mir.py pdf --year 2026 --questions-pdf <booklet.pdf> --answers-pdf <key.pdf>
BACKEND=google-cloud PROJECT_ID=<project> bash scripts/prepare_mir.sh
python scripts/prepare_medqa.py --hf-name awinml/medqa --split test --out data/medqa_test.jsonl   # optional

# Once: models
python scripts/download_models.py
python scripts/count_params.py --out results/param_counts.json

# Evaluation
sbatch slurm/smoke_test.sbatch
bash scripts/submit_all.sh
squeue -u $USER

# Analysis
python -m medeval.analyze --results results
```
