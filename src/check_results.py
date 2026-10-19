"""Check that every model has complete and usable results for every dataset.

For each model and each default dataset the script reports the number of items (compared with
the dataset size), the share of answers cut off at the token limit, the share of answers with a
reasoning section (written although the prompt asks for the answer only), and for multiple-choice
datasets the accuracy and the share of answers without a recognizable letter. For open-ended
datasets it reports the share of empty answers, the mean answer length, and the grading status
with the mean score and the share of judge verdicts that could not be read.

Usage:
    python src/check_results.py                      table and list of problems
    python src/check_results.py --out logs/check.csv also save the table
"""

import argparse
import json

import pandas as pd

from config import PROJECT_ROOT, default_datasets, load_datasets_config, load_models
from items import load_dataset
from judge import answers_newer_than_grades
from run_eval import has_reasoning

LIMIT_PERCENT = 5.0  # a share of cut-off or unparsed answers above this is listed as a problem


def check_pair(model, dataset, spec, ids, problems):
    raw = PROJECT_ROOT / "results" / "raw" / model / f"{dataset}.jsonl"
    if not raw.exists():
        problems.append(f"{model} {dataset}: no results")
        return None
    answers = pd.read_json(raw, lines=True, dtype={"item_id": str})
    expected = len(ids) if ids is not None else None
    if ids is not None:  # answers to items outside the dataset (e.g. before min_date) are ignored
        answers = answers[answers["item_id"].isin(ids)]
    meta_path = raw.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    row = {"model": model, "dataset": dataset, "n": len(answers), "expected": expected,
           "max_tokens": meta.get("max_tokens"),
           "truncated_%": round(100 * (answers["finish_reason"] == "length").mean(), 1),
           "reasoning_%": round(100 * answers["response"].map(has_reasoning).mean(), 1)}
    if expected and len(answers) != expected:
        problems.append(f"{model} {dataset}: {len(answers)} of {expected} items")
    if row["truncated_%"] > LIMIT_PERCENT:
        problems.append(f"{model} {dataset}: {row['truncated_%']}% answers cut off")

    if spec.get("task", "multiple_choice") == "multiple_choice":
        row["accuracy_%"] = round(100 * answers["correct"].mean(), 1)
        row["invalid_%"] = round(100 * answers["predicted"].isna().mean(), 1)
        if row["invalid_%"] > LIMIT_PERCENT:
            problems.append(f"{model} {dataset}: {row['invalid_%']}% answers not parsed")
        return row

    row["empty_%"] = round(100 * (answers["response"].fillna("").str.strip() == "").mean(), 1)
    row["out_tokens"] = round(answers["output_tokens"].mean())
    judged = PROJECT_ROOT / "results" / "judged" / model / f"{dataset}.jsonl"
    if not judged.exists():
        problems.append(f"{model} {dataset}: not graded")
        return row
    grades = pd.read_json(judged, lines=True, dtype={"item_id": str})
    if ids is not None:
        grades = grades[grades["item_id"].isin(ids)]
    row["graded"] = len(grades)
    row["score"] = round(grades["score"].mean(), 3)
    row["unparsed_%"] = round(100 * grades["n_unparsed"].sum() / max(grades["n_checks"].sum(), 1), 1)
    if len(grades) != len(answers):
        problems.append(f"{model} {dataset}: {len(grades)} of {len(answers)} answers graded")
    if answers_newer_than_grades(raw, judged):
        problems.append(f"{model} {dataset}: answers are newer than the grades (grade again)")
    if row["unparsed_%"] > LIMIT_PERCENT:
        problems.append(f"{model} {dataset}: {row['unparsed_%']}% judge verdicts not parsed")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="Optional CSV file for the table.")
    args = parser.parse_args()

    ds_cfg = load_datasets_config()
    datasets = default_datasets(ds_cfg)
    models = [k for k, m in load_models().items() if m.get("role") != "judge" and not m.get("optional")]
    expected = {}
    for dataset in datasets:
        try:
            expected[dataset] = {item.item_id for item in load_dataset(dataset, ds_cfg["datasets"][dataset])}
        except (OSError, ValueError, KeyError) as exc:
            expected[dataset] = None
            print(f"Warning: dataset {dataset} cannot be loaded ({exc})")

    problems, rows = [], []
    for model in models:
        for dataset in datasets:
            row = check_pair(model, dataset, ds_cfg["datasets"][dataset], expected[dataset], problems)
            if row:
                rows.append(row)

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(PROJECT_ROOT / args.out, index=False)
    print("\nProblems:" if problems else "\nAll models have complete results for all datasets.")
    for problem in problems:
        print("  " + problem)


if __name__ == "__main__":
    main()
