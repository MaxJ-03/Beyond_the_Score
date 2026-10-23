# Beyond the Score: What Medical Benchmarks Miss

Code, data and results for our study of open LLMs on the Spanish MIR medical licensing exam:

- **Accuracy and inference cost** on older (2020–2024) and newer (2025–2026) MIR editions, in
  the original Spanish and in English, and on real-world clinical questions (CUPCase,
  LiveMedBench, K-QA).
- **Memorization checks:** whether accuracy drops on editions released after each model's
  training cutoff, plus likelihood (Min-K% Prob) and completion tests on the raw question text.

All results in `results/` were produced on one NVIDIA H100 (Snellius) with **vLLM 0.31.0** and
**PyTorch 2.13.0 (CUDA 13.0)**, greedy decoding, seed 0. The full manual is
[`docs/MANUAL.md`](docs/MANUAL.md); the contamination experiments are described in
[`contamination/README.md`](contamination/README.md).

## Repository layout

```
configs/         models.yaml (models, cutoffs), datasets.yaml (datasets, prompts, analysis),
                 judge_prompts.yaml (official grading prompts), sanitize.yaml (contamination tests)
src/             one Python file per pipeline stage (python src/<file>.py --help)
scripts/         Snellius job scripts (setup, preparation, evaluation, grading, contamination tests)
contamination/   planted contamination, LLMSanitize memorization tests, paired comparisons
tests/           python -m pytest tests/
data/            prepared datasets; data/external/ holds the source files not downloadable online
results/raw/     every model answer          results/judged/    judge grades of open-ended answers
results/sanitize/ contamination-test scores  results/analysis/  all tables and figures
```

## Data

| Dataset keys | Questions | Source |
|---|---|---|
| `mir_2020_es` … `mir_2024_es` | 145, 149, 172, 172, 171 | CareQA (Hugging Face `HPAI-BSC/CareQA`), from the official booklets |
| `mir_2025_es` | 170 | official Version 0 booklet and definitive answer key (`data/external/`) |
| `mir_2026_es` | 170 | 2026 file extracted from the official booklet (`data/external/`) |
| `mir_2020` … `mir_2026` (English) | same as Spanish | 2020–2024: CareQA's GPT-4 translation; 2025–2026: Google machine translation |
| `cupcase` | 3,508 | CUPCase (`ofir408/CupCase`), case reports with four candidate diagnoses |
| `livemedbench` | 2,664 (analysis: cases posted from 2025-06-01) | LiveMedBench (`JuelieYann/LiveMedBench`), English cases graded against rubrics |
| `kqa` | 201 | K-QA (GitHub `Itaymanes/K-QA`), patient questions graded against physician statements |
| `medqa` (optional) | 1,273 | MedQA 4-option test set (`GBaker/MedQA-USMLE-4-options`), contamination experiments only |

MIR selection: main questions only (175 in 2020–2021, 200 from 2022), without reserve,
image-linked or annulled questions. Each English file contains exactly the questions of its
Spanish counterpart. Because the translators differ, Spanish is the primary comparison.

## Models

| Key | Hugging Face id | Training cutoff | Release | First unseen MIR |
|---|---|---|---|---|
| `gemma3-1b`, `gemma3-4b`, `gemma3-27b` | `google/gemma-3-{1b,4b,27b}-it` | 2024-08 | 2025-03-12 | 2025 |
| `medgemma-4b`, `medgemma-27b` | `google/medgemma-4b-it`, `google/medgemma-27b-text-it` | 2024-08 | 2025-05-20 | 2025 |
| `qwen3-1.7b`, `qwen3-4b` | `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (thinking off) | release used | 2025-04-29 | 2026 |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | 2023-03 | 2024-04-18 | 2024 |
| `med42-8b` | `m42-health/Llama3-Med42-8B` | 2023-03 | 2024-08 | 2024 |

Open-ended answers are graded by `openai/gpt-oss-120b` (judge only, never evaluated). Seen/unseen
splits: the **common split** (2020–2024 seen, 2025–2026 unseen for every model) and the
**per-model split** (unseen from the first MIR after the training cutoff, last column).

## Reproducing the results

### 0. Requirements

- A Slurm cluster with an NVIDIA H100 (94 GB); the scripts target Snellius (`gpu_h100`
  partition, module `Python/3.12.3-GCCcore-13.3.0`). Adjust the module lines in `scripts/` and
  `contamination/job.sbatch` for another cluster.
- A Hugging Face account with the licenses of Gemma 3, MedGemma (Health AI Developer
  Foundations) and Llama 3 accepted.
- About 300 GB of disk for model weights (`HF_HOME=/scratch-shared/$USER/hf_cache`).

### 1. Install (once, on a login node)

```bash
git clone https://github.com/MaxJ-03/RTDM_Group_4.git && cd RTDM_Group_4
bash scripts/setup_env.sh                  # venv in ~/venvs/medeval, vLLM, requirements.lock.txt
source ~/venvs/medeval/bin/activate        # in every new session
pip install vllm==0.31.0                   # pin the version used for the paper
pip install peft rouge-score nltk scikit-learn   # only for contamination/
hf auth login
```

### 2. Prepare data and models

```bash
nohup bash scripts/prepare_all.sh > logs/prepare.log 2>&1 &
```

This rebuilds the MIR and real-world datasets, downloads the model weights and counts their
parameters (`results/param_counts.json`). The prepared datasets are committed, so
`git diff data/` should show no changes. `SKIP_MODELS=1` runs only the dataset steps.

### 3. Smoke tests

```bash
SMOKE=1 sbatch scripts/evaluate.sbatch     # one model, 20 questions per dataset -> results_smoke/
SMOKE=1 sbatch scripts/sanitize.sbatch     # unit tests + contamination tests on gemma3-1b
VALIDATE=1 sbatch scripts/judge.sbatch     # judge agreement with physicians (K-QA)
```

### 4. Accuracy evaluation (results/raw/, results/judged/)

```bash
bash scripts/submit_all.sh                 # one chained job per model, grading job at the end
squeue -u $USER                            # monitor; rerun the script to submit anything missing
python src/check_results.py                # completeness check once all jobs have finished
```

`submit_all.sh` only submits missing or outdated results; `FORCE=1` reruns everything.

### 5. Contamination tests on the raw question text (results/sanitize/)

```bash
bash scripts/submit_sanitize.sh            # completion and Min-K% Prob, all models, all MIR datasets
```

### 6. Contamination experiments (contamination/)

```bash
python src/prepare_medqa.py --hf-name GBaker/MedQA-USMLE-4-options --split test --out data/medqa_test.jsonl
bash contamination/run.sh planted gemma3-4b      # planted contamination (LoRA), then:
bash contamination/run.sh planted medgemma-4b
bash contamination/run.sh medqa                  # MedQA, nine models
bash contamination/run.sh tests                  # LLMSanitize memorization tests
bash contamination/run.sh status                 # wait until all jobs have finished
```

Each stage submits its jobs once; do not run a stage twice. Checkpoints are stored in
`/scratch-shared/$USER/planted` (not in Git).

### 7. Analysis (CPU, seconds to minutes)

```bash
python src/analyze.py                      # summary.csv, rankings, exam_shift.csv, plots
python src/compare_layers.py               # standard exam vs new exam vs real world
python src/contamination.py                # year-difficulty adjustment (main contamination result)
python src/sanitize_summary.py             # Min-K% and completion: seen vs unseen editions
python src/conda_check.py                  # reports in the CONDA database (needs internet)
bash contamination/run.sh analyse          # planted contamination and memorization tests
python contamination/paired_shift.py       # medical vs base model lead on unseen editions
python "contamination/paired_tests (2).py" # paired medical vs base tests per dataset
```

**Analysis only:** all model outputs are committed, so step 7 runs on any computer without a GPU
after `pip install -r requirements.txt` (and the extra `contamination/` packages).

## Main outputs (results/analysis/)

| File | Content |
|---|---|
| `summary.csv` | quality with 95% CI, chance-corrected quality, invalid/truncated rates, tokens, FLOPs, GPU-seconds, points per TFLOP, Pareto frontier |
| `rankings.csv`, `rank_correlations.csv` | rankings by quality and cost-efficiency, and their agreement across datasets |
| `exam_shift.csv` | accuracy on unseen minus seen editions, per model, language and split |
| `layer_summary.csv`, `layer_agreement.csv`, `generalization.csv` | comparison of the three layers |
| `judge_validation.csv` | judge agreement with physicians on K-QA |
| `contamination/` | year-difficulty adjustment (`model_based.csv`), unparseable-answer check, MedExpQA overlap, LiveMedBench dates, CONDA reports |
| `sanitize/` | Min-K% and completion test: per edition, seen vs unseen (ROC-AUC), edition-adjusted effect, plot |
| `memorization_tests/`, `paired_shift/`, `paired_tests/`, `medqa/` | contamination experiments (see `contamination/README.md`) |
| `accuracy_vs_flops_<dataset>.png` | accuracy against FLOPs with the Pareto frontier |

## Reproducibility notes

- Software and hardware versions, settings and timing are stored in every `.meta.json` next to
  the results; `requirements.lock.txt` (written by `setup_env.sh`) records all package versions.
- Decoding is greedy (temperature 0); small differences between reruns can still come from GPU
  batching in bf16.
- Prompts are in `configs/datasets.yaml`, judge prompts in `configs/judge_prompts.yaml`, all
  contamination-test settings in `configs/sanitize.yaml`.
- MIR sources and selection: `src/mir_sources.py`, `src/mir_corrections.py`; source files and
  human reference accuracies: `data/external/`.
