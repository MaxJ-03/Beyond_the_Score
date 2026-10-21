# Contamination experiments

Experiments that test **how much test-question exposure our methods can detect** and add
memorization tests that the team pipeline (branch `stage3`) does not run. Everything is in this
folder; no shared file is changed and no `stage3` result is repeated. Where the same models are
evaluated, the `stage3` settings are used (1,024-token answer limit, MedGemma 27B answering
directly, `stage3` answer parser), so the numbers can be reported next to the team's results.

| File | What it does |
|---|---|
| `common.py` | shared helpers (models incl. planted checkpoints, `stage3` settings and parser) and the multiple-choice evaluation (`eval`) |
| `planted.py` | **Planted contamination**: split, LoRA training, Min-K% scoring, analysis |
| `memorization_tests.py` | **Memorization tests** ported from LLMSanitize: guided completion, masked-option guessing, Min-K% likelihood |
| `paired_shift.py` | **Paired base-medical comparison** on the `stage3` MIR results |
| `run.sh`, `job.sbatch` | job submission on Snellius |
| `planted_split.csv` | the exposed / held-out split used in the paper |

Extra packages: `pip install peft rouge-score nltk scipy scikit-learn`.

## The experiments and the research question each serves

**1. Planted contamination (RQ2).** The edition comparison finds no large drop on unseen MIR
editions. That can mean "no contamination" or "the test cannot see it". We contaminate Gemma 3 4B
and MedGemma 4B on purpose and measure how much exposure is needed before accuracy, and before
Min-K% Prob, reveal it. MIR 2020-2022 is split at random within each edition into 232 exposed and
234 held-out questions; LoRA (rank 16, alpha 32, dropout 0.05, AdamW, learning rate 2e-4,
batch 8) on the exposed questions written with their answers; checkpoints after 1, 3, 10 epochs.
Planted effect = accuracy gain on exposed minus gain on held-out questions (bootstrap CI).

| Model | Epochs | Planted effect (pp, 95% CI) | Min-K% AUC |
|---|---|---|---|
| Gemma 3 4B | 1 / 3 / 10 | +9.0 [-0.9, +19.2] / +20.7 [+11.7, +29.7] / +42.2 [+32.3, +51.7] | 0.71 / 1.00 / 1.00 |
| MedGemma 4B | 1 / 3 / 10 | +2.5 [-6.5, +11.1] / +22.0 [+13.0, +31.0] / +37.5 [+28.4, +46.5] | 0.72 / 1.00 / 1.00 |

Accuracy reveals contamination from about three exposures. Min-K% detects a single one only on the
text as it was trained; in the LLMSanitize layout its AUC is 0.54 / 0.55 after one epoch, 0.76 / 0.81
after three and 0.94 / 0.95 after ten, and guided completion and masked-option guessing show no
consistent difference between exposed and held-out questions even after ten epochs.

**2. Paired base-medical comparison (RQ2, general vs medical models).** Each medical model is
compared with its own base model on the same questions, so edition difficulty cancels without the
regression's assumption that a harder edition is equally harder for every model. On the `stage3`
results, MedGemma 27B's lead shrinks by 4.2 points on MIR 2025-2026 (95% CI [-8.3, -0.1],
permutation p = 0.095); MedGemma 4B -3.7 [-10.0, +2.5]; Med42 +1.4 [-4.1, +7.0].

**3. Memorization tests (RQ2, completion test).** Three closed-data tests of LLMSanitize, ported
line by line because the library needs vLLM 0.3.3:
* *Guided completion*: given the question and first option, the model writes the rest, with and
  without the dataset name; a better match with the name (ROUGE-L, bootstrap p) means the model
  recognises the dataset.
* *Masked-option guessing*: one wrong option is hidden; reproducing it exactly cannot come from
  medical knowledge (exact match with the hidden option, as in the paper; the library's comparison
  with the correct option is a bug and is stored separately).
* *Min-K% likelihood*: MedQA vs MIR 2026 (English). Min-K% on older vs newer MIR editions is done
  by `stage3` (`src/sanitize.py`) and is not repeated.
100 questions per dataset; the planted checkpoints are the known positive. Results: guided
completion gives a similar positive difference on the unseen MIR 2026 as on older editions (except
Llama 3, negative, and MedGemma 4B, about zero); masked-option guessing reproduces the hidden option
for at most 4% of questions; Min-K% separates MedQA from MIR 2026 with AUC 0.876-0.912 for every
model.

**4. MedQA reference (supports RQ2; context for RQ1).** The 1,273 four-option MedQA test
questions for all nine models: the benchmark most likely seen in training, and the "seen" side of
the Min-K% comparison. Accuracy 32.8% (Gemma 3 1B) to 69.0% (MedGemma 27B); 9.3% of MedGemma
27B's answers reach the 1,024-token limit.

## Result files

| File | Content |
|---|---|
| `contamination/planted_out/accuracy_<model>.csv` | planted effect per epoch (exposed, held-out, MIR 2026 accuracy) |
| `contamination/planted_out/mink_auc_<model>.csv` | Min-K% AUC, exposed vs held-out, per epoch |
| `results/analysis/memorization_tests/` | `guided_completion.csv`, `masked_option.csv`, `mink_medqa_vs_mir2026.csv`, `planted_known_positive.csv` |
| `results/analysis/paired_shift/` | `lead_by_year.csv`, `lead_shift.csv`, `lead_shift_loo.csv` |
| `results/analysis/medqa/medqa_accuracy.csv` | MedQA accuracy, unparseable and cut-off answers per model |

## Commands (Snellius login node, project folder)

```bash
bash contamination/run.sh planted gemma3-4b      # planted contamination (and medgemma-4b)
bash contamination/run.sh medqa                  # MedQA, nine models
bash contamination/run.sh tests                  # memorization tests, nine models
bash contamination/run.sh analyse                # all tables, once the jobs have finished
python contamination/paired_shift.py --project <stage3 checkout>
```

Each stage submits its jobs once; do not run a stage twice. Checkpoints (about 9 GB each) are kept
in `/scratch-shared/$USER/planted` and linked as `checkpoints/planted`; do not add them to Git.
