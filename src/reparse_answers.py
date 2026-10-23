"""Parse the stored multiple-choice answers again with the current answer parser.
Usage: python src/reparse_answers.py [--models ...]"""

import argparse
import json

from config import PROJECT_ROOT, load_datasets_config
from run_eval import extract_answer, has_reasoning


def reparse(path):
    """Recompute predicted and correct in one result file; returns counts before and after."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    before = {"invalid": sum(r["predicted"] is None for r in records),
              "correct": sum(bool(r["correct"]) for r in records)}
    for record in records:
        letters = "ABCDEFGHIJ"[: int(record["n_options"])]
        record["predicted"] = extract_answer(record["response"], letters)
        record["correct"] = record["predicted"] == record["answer"]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    after = {"invalid": sum(r["predicted"] is None for r in records),
             "correct": sum(r["correct"] for r in records),
             "reasoning": sum(has_reasoning(r["response"]) for r in records)}
    return len(records), before, after


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="Models to reparse (default: all).")
    args = parser.parse_args()

    datasets = load_datasets_config()["datasets"]
    n_changed = 0
    for path in sorted((PROJECT_ROOT / "results" / "raw").glob("*/*.jsonl")):
        model, dataset = path.parent.name, path.stem
        if args.models and model not in args.models:
            continue
        if datasets.get(dataset, {}).get("task", "multiple_choice") != "multiple_choice":
            continue
        n, before, after = reparse(path)
        if before["invalid"] != after["invalid"] or before["correct"] != after["correct"]:
            n_changed += 1
            print(f"[{model}] {dataset}: unparsed {before['invalid']} -> {after['invalid']}, "
                  f"correct {before['correct']} -> {after['correct']} of {n} "
                  f"({after['reasoning']} answers with a reasoning section)")
    print(f"{n_changed} result files changed.")


if __name__ == "__main__":
    main()
