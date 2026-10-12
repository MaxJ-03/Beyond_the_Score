"""Create small synthetic datasets for testing the pipeline with the mock backend.

Files are written to data/ only if they do not exist yet, so real data is never
overwritten. Remove the synthetic files before placing the real datasets.

Usage:
    python scripts/make_test_data.py
"""

import json
import random
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    rng = random.Random(0)
    DATA.mkdir(exist_ok=True)

    medqa = DATA / "medqa_test.jsonl"
    if not medqa.exists():
        with open(medqa, "w", encoding="utf-8") as handle:
            for i in range(50):
                options = {letter: f"Option {letter} for question {i}" for letter in "ABCD"}
                handle.write(json.dumps({"id": f"medqa-{i}", "year": "", "options": options,
                                         "question": f"Synthetic question {i}?",
                                         "answer": rng.choice("ABCD")}) + "\n")
        print(f"Created {medqa}")

    for year in (2022, 2024, 2026):
        path = DATA / f"mir_{year}_en.csv"
        if path.exists():
            print(f"{path} exists, left unchanged.")
            continue
        rows = []
        for i in range(40):
            row = {"id": f"mir{year}-{i}", "year": year, "answer": rng.choice("ABCD"),
                   "question_en": f"Synthetic MIR {year} question {i}?"}
            for letter in "abcd":
                row[f"option_{letter}_en"] = f"Option {letter.upper()}"
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Created {path}")


if __name__ == "__main__":
    main()
