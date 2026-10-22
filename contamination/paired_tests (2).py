"""Paired comparison of each medical model with its base model on the same questions.

Multiple choice (MIR Spanish, MIR English, CUPCase): accuracy difference, paired bootstrap 95% CI
and exact McNemar test. K-QA and LiveMedBench (cases from 2025-06-01): mean difference and paired
bootstrap 95% CI. Also writes the mean LiveMedBench answer length per model.
Seed 0 and 4,000 resamples for every bootstrap.

Usage (from the repository root): python contamination/paired_tests.py [--root .] [--out folder]
Output: results/analysis/paired_tests/paired_tests.csv and livemedbench_length.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

PAIRS = [("medgemma-4b", "gemma3-4b"), ("medgemma-27b", "gemma3-27b"), ("med42-8b", "llama3-8b")]
MODELS = ["gemma3-1b", "qwen3-1.7b", "gemma3-4b", "medgemma-4b", "qwen3-4b",
          "llama3-8b", "med42-8b", "gemma3-27b", "medgemma-27b"]
YEARS = range(2020, 2027)

# Multiple-choice sets: name in the output -> dataset files in results/raw/<model>/ and expected size
MC_SETS = {
    "MIR ES": ([f"mir_{y}_es" for y in YEARS], 1149),
    "MIR EN": ([f"mir_{y}" for y in YEARS], 1149),
    "CUPCase": (["cupcase"], 3508),
}
# Judged sets: (dataset, field in results/judged/<model>/<dataset>.jsonl, scale, unit, expected size)
JUDGED_SETS = [
    ("kqa", "score", 100, "comprehensiveness x100", 201),
    ("kqa", "hallucinations", 1, "contradicted statements per answer", 201),
    ("livemedbench", "score", 100, "rubric score x100", 1926),
]
LIVEMEDBENCH_MIN_DATE = "2025-06-01"  # cases posted after the release of every evaluated model
N_RESAMPLES = 4000
SEED = 0


def read_jsonl(path):
    """All records of a JSON-lines file."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def paired_bootstrap_ci(diff):
    """95% percentile CI of the mean of per-question differences (questions resampled together)."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(N_RESAMPLES, len(diff)))
    return np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])


def load_values(root, folder, model, datasets, field, keep=None):
    """{(dataset, item_id): value} for one model; keep = optional set of item ids to retain."""
    values = {}
    for ds in datasets:
        for r in read_jsonl(root / "results" / folder / model / f"{ds}.jsonl"):
            if keep is None or r["item_id"] in keep:
                key = (ds, r["item_id"])
                assert key not in values, f"duplicate question {key} for {model}"
                values[key] = r[field]
    return values


def paired_arrays(med_values, base_values, expected_n, label):
    """Values of both models on exactly the same questions, in the same order."""
    assert set(med_values) == set(base_values), f"{label}: the two models answered different questions"
    assert len(med_values) == expected_n, f"{label}: expected {expected_n} questions, found {len(med_values)}"
    keys = sorted(med_values)
    med = np.array([float(med_values[k]) for k in keys])
    base = np.array([float(base_values[k]) for k in keys])
    assert not np.isnan(med).any() and not np.isnan(base).any(), f"{label}: missing values"
    return med, base


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".", help="repository root (default: current folder)")
    parser.add_argument("--out", default=None, help="output folder (default: results/analysis/paired_tests)")
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "results" / "analysis" / "paired_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    # LiveMedBench cases used in the paper: posted on or after the minimum date
    dates = {r["id"]: (r.get("date") or "") for r in read_jsonl(root / "data" / "livemedbench_en.jsonl")}
    livemedbench_ids = {case for case, date in dates.items() if date >= LIVEMEDBENCH_MIN_DATE}

    rows = []

    # 1. Multiple choice: accuracy difference, paired bootstrap CI and exact McNemar test
    for set_name, (datasets, expected_n) in MC_SETS.items():
        for med_model, base_model in PAIRS:
            label = f"{set_name} {med_model} vs {base_model}"
            med, base = paired_arrays(load_values(root, "raw", med_model, datasets, "correct"),
                                      load_values(root, "raw", base_model, datasets, "correct"),
                                      expected_n, label)
            diff = med - base                              # +1, 0 or -1 per question
            only_med = int(((med == 1) & (base == 0)).sum())   # discordant: only the medical model correct
            only_base = int(((med == 0) & (base == 1)).sum())  # discordant: only the base model correct
            p_value = binomtest(only_med, only_med + only_base, 0.5).pvalue  # exact McNemar, two-sided
            low, high = paired_bootstrap_ci(diff)
            rows.append([set_name, "accuracy (%)", med_model, base_model, len(diff),
                         100 * med.mean(), 100 * base.mean(), 100 * diff.mean(), 100 * low, 100 * high,
                         only_med, only_base, f"{p_value:.2g}"])

    # 2. Judged open-ended sets: mean difference and paired bootstrap CI
    for dataset, field, scale, unit, expected_n in JUDGED_SETS:
        keep = livemedbench_ids if dataset == "livemedbench" else None
        for med_model, base_model in PAIRS:
            label = f"{dataset} {field} {med_model} vs {base_model}"
            med, base = paired_arrays(load_values(root, "judged", med_model, [dataset], field, keep),
                                      load_values(root, "judged", base_model, [dataset], field, keep),
                                      expected_n, label)
            diff = med - base
            low, high = paired_bootstrap_ci(diff)
            rows.append([{"kqa": "K-QA", "livemedbench": "LiveMedBench"}[dataset], unit, med_model, base_model,
                         len(diff), scale * med.mean(), scale * base.mean(), scale * diff.mean(),
                         scale * low, scale * high, "", "", ""])

    with open(out_dir / "paired_tests.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["set", "metric", "medical_model", "base_model", "n_questions",
                         "medical_mean", "base_mean", "difference", "ci95_low", "ci95_high",
                         "only_medical_correct", "only_base_correct", "mcnemar_p"])
        for row in rows:
            writer.writerow([round(v, 4) if isinstance(v, float) else v for v in row])

    # 3. LiveMedBench answer length per model (generated tokens), on the same 1,926 cases
    with open(out_dir / "livemedbench_length.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n_cases", "mean_output_tokens"])
        for model in MODELS:
            tokens = [r["output_tokens"] for r in read_jsonl(root / "results" / "raw" / model / "livemedbench.jsonl")
                      if r["item_id"] in livemedbench_ids]
            assert len(tokens) == 1926, f"{model}: expected 1926 LiveMedBench cases, found {len(tokens)}"
            writer.writerow([model, len(tokens), round(float(np.mean(tokens)), 1)])

    print(f"Wrote {out_dir / 'paired_tests.csv'} and {out_dir / 'livemedbench_length.csv'}")


if __name__ == "__main__":
    main()
