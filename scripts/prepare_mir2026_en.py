"""Build data/mir_2026_en.csv from the scraped MIR 2026 translation.

The output has exactly the same format as the English CareQA files of 2022 and 2024:
    id, year, answer, translation_mode, question_es, question_en,
    option_a_es, option_a_en, ..., option_d_es, option_d_en

It contains exactly the questions selected in data/raw/mir_2026_es.csv (same ids, same
answers), with the English text taken from the translated file. Run it after the Spanish
2026 file has been prepared; scripts/prepare_data.py does this automatically.

Usage:
    python scripts/prepare_mir2026_en.py
    python scripts/prepare_mir2026_en.py --translation data/external/<other file>.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPANISH = ROOT / "data" / "raw" / "mir_2026_es.csv"
OUTPUT = ROOT / "data" / "mir_2026_en.csv"
DEFAULT_TRANSLATION = "data/external/mir_2026_en_mlt.csv"
LETTERS = ["a", "b", "c", "d"]
ENGLISH_COLUMNS = ["question_id", "question_en", "option_1_en", "option_2_en", "option_3_en", "option_4_en",
                   "correct_answer"]


def answer_letter(value):
    text = str(value).strip().upper()
    if text in "ABCD" and len(text) == 1:
        return text
    return "ABCD"[int(text) - 1] if text.isdigit() and 1 <= int(text) <= 4 else None


def build(translation_path):
    translation_path = Path(translation_path)
    if not translation_path.is_absolute():
        translation_path = ROOT / translation_path
    if not SPANISH.exists():
        raise SystemExit(f"{SPANISH} not found. Prepare the Spanish 2026 file first (scripts/prepare_data.py).")
    if not translation_path.exists():
        print(f"English 2026 translation not found at {translation_path}; skipped.")
        return None

    spanish = pd.read_csv(SPANISH)
    english = pd.read_csv(translation_path)
    missing = [c for c in ENGLISH_COLUMNS if c not in english.columns]
    if missing:
        raise SystemExit(f"Translation file: missing columns {missing}. Found {list(english.columns)}.")
    english = english.set_index("question_id")
    if not english.index.is_unique:
        raise SystemExit("Translation file: question_id is not unique.")

    rows, unmatched, answer_mismatches, empty_fields = [], [], 0, 0
    for _, row in spanish.iterrows():
        number = int(row["question_number"])
        if number not in english.index:
            unmatched.append(number)
            continue
        en = english.loc[number]
        answer_mismatches += answer_letter(en["correct_answer"]) != row["answer"]
        record = {"id": row["id"], "year": 2026, "answer": row["answer"],
                  "translation_mode": "scraped_translation",
                  "question_es": row["question"], "question_en": str(en["question_en"]).strip()}
        for letter, number_option in zip(LETTERS, range(1, 5)):
            record[f"option_{letter}_es"] = row[f"option_{letter}"]
            record[f"option_{letter}_en"] = str(en[f"option_{number_option}_en"]).strip()
        empty_fields += sum(not str(record[k]).strip() or str(record[k]) == "nan"
                            for k in record if k.endswith("_en"))
        rows.append(record)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    meta = {
        "output": str(OUTPUT),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source_spanish": str(SPANISH),
        "source_translation": str(translation_path),
        "translation": "scraped MIR 2026 translation (Google machine translation), see the translation file",
        "rows": len(out),
        "spanish_questions_without_translation": unmatched,
        "answer_mismatches_between_files": int(answer_mismatches),
        "empty_english_fields": int(empty_fields),
    }
    OUTPUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"English 2026: {len(out)} questions written to {OUTPUT} ({len(unmatched)} without translation, "
          f"{answer_mismatches} answer mismatches, {empty_fields} empty English fields)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translation", default=DEFAULT_TRANSLATION, help="Translated MIR 2026 file.")
    args = parser.parse_args()
    build(args.translation)


if __name__ == "__main__":
    main()
