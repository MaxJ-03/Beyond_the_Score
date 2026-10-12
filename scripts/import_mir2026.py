"""Import the MIR 2026 file into the shared source format.

Selects the questions that are neither image-based nor annulled, renames the Spanish
columns to the names expected by download_mir.py, and writes
data/external/mir_2026_es_selected.csv. Afterwards run:

    python scripts/download_mir.py csv --year 2026 --input data/external/mir_2026_es_selected.csv

Usage:
    python scripts/import_mir2026.py --input data/external/mir_2026_en_mlt.csv
"""

import argparse
from pathlib import Path

import pandas as pd

COLUMN_MAP = {
    "question_id": "question_number",
    "question_es": "question",
    "option_1_es": "option_a",
    "option_2_es": "option_b",
    "option_3_es": "option_c",
    "option_4_es": "option_d",
    "correct_answer": "answer",
}
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "external" / "mir_2026_es_selected.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Asen's MIR 2026 CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in [*COLUMN_MAP, "has_image", "annulled"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found {list(df.columns)}.")

    selected = df[~df["has_image"].astype(bool) & ~df["annulled"].astype(bool)]
    selected = selected[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT, index=False)
    print(f"{len(df)} rows read, {len(selected)} kept (not image-based, not annulled). Wrote {OUTPUT}")


if __name__ == "__main__":
    main()