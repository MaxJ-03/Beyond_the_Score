"""Convert a MedQA dataset from Hugging Face into the pipeline's jsonl format.

MedQA is distributed in several layouts on Hugging Face. This script detects the most
common ones automatically:
    options as a dict ({"A": "...", "B": "..."}), as a list, or as separate columns
    the answer as a letter, a zero-based index, or the full answer text
Column names can be overridden if detection fails. The detected layout and the first
converted item are printed so the conversion can be checked.

Usage:
    python scripts/prepare_medqa.py --hf-name awinml/medqa --split test --out data/medqa_test.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-name", required=True)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--question-col", default=None)
    parser.add_argument("--options-col", default=None)
    parser.add_argument("--answer-col", default=None)
    return parser.parse_args()


def first_present(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def get_options(row, options_col):
    if options_col:
        value = row[options_col]
        if isinstance(value, dict):
            return {str(k).strip().upper(): str(v).strip() for k, v in sorted(value.items())}
        if isinstance(value, list):
            if value and isinstance(value[0], dict):  # list of {"key": "A", "value": "..."}
                return {str(o["key"]).strip().upper(): str(o["value"]).strip() for o in value}
            return {LETTERS[i]: str(v).strip() for i, v in enumerate(value)}
    for prefix in ("op", "option_", "opt"):
        cols = [f"{prefix}{c}" for c in "abcde"]
        present = [c for c in cols if c in row and row[c] not in (None, "")]
        if present:
            return {LETTERS[i]: str(row[c]).strip() for i, c in enumerate(present)}
    raise ValueError("Could not find answer options; pass --options-col.")


def get_answer(row, answer_col, options):
    value = row[answer_col]
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        return LETTERS[int(value)]
    text = str(value).strip()
    if text.upper() in options:
        return text.upper()
    matches = [letter for letter, option in options.items() if option.lower() == text.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not map answer '{text}' to an option letter.")


def main():
    args = parse_args()
    ds = load_dataset(args.hf_name, args.hf_config, split=args.split)
    columns = ds.column_names
    print(f"Columns: {columns}")

    question_col = args.question_col or first_present(columns, ["question", "sent1", "Question"])
    options_col = args.options_col or first_present(columns, ["options", "choices", "Options"])
    answer_col = args.answer_col or first_present(
        columns, ["answer_idx", "cop", "label", "answer", "correct_answer", "Answer"])
    print(f"Using question='{question_col}', options='{options_col}', answer='{answer_col}'")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for index, row in enumerate(ds):
            options = get_options(row, options_col)
            record = {
                "id": str(row.get("id", index)),
                "year": "",
                "question": str(row[question_col]).strip(),
                "options": options,
                "answer": get_answer(row, answer_col, options),
            }
            if n_written == 0:
                print("First converted item:\n" + json.dumps(record, indent=2, ensure_ascii=False))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} items to {out_path}")


if __name__ == "__main__":
    main()
