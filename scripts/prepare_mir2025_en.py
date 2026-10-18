"""Build data/mir_2025_en.csv from the MIR 2025 translation.

The output has exactly the same format as the other English MIR files:
    id, year, answer, translation_mode, question_es, question_en,
    option_a_es, option_a_en, ..., option_d_es, option_d_en

It contains exactly the questions selected in data/raw/mir_2025_es.csv (same ids, same
answers). The Spanish text is taken from that file, so the parallel Spanish columns are
identical to the text evaluated in the Spanish run; the English text is taken from the
translated file. The translation was produced with the same procedure as the MIR 2026
translation (Google machine translation), so both new editions share one translation mode.

The translated file must contain the columns id, question_number, answer, question_en and
option_a_en to option_d_en. The script fails if any selected question is missing from the
translation, if ids or question numbers disagree, or if an answer letter differs.

Known deviation from a literal translation: in question 38, option C, the Spanish source
contains the typographical error "0,3-05 mL", which the translation renders as "0.3-0.5 mL".
The translation is used as provided; scripts/validate_translations.py reports the case.
Zero-width spaces introduced by the machine translation are removed.

Run it after the Spanish 2025 file has been prepared; scripts/prepare_data.py does this
automatically.

Usage:
    python scripts/prepare_mir2025_en.py
    python scripts/prepare_mir2025_en.py --translation data/external/<other file>.csv
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPANISH = ROOT / "data" / "raw" / "mir_2025_es.csv"
OUTPUT = ROOT / "data" / "mir_2025_en.csv"
DEFAULT_TRANSLATION = "data/external/mir_2025_en_mlt.csv"
LETTERS = ["a", "b", "c", "d"]
TRANSLATION_MODE = "scraped_translation"
REQUIRED_COLUMNS = ["id", "question_number", "answer", "question_en",
                    *[f"option_{letter}_en" for letter in LETTERS]]


def clean_text(value):
    """Strip the text and remove zero-width spaces left by the machine translation."""
    return str(value).replace("\u200b", "").strip()


def build(translation_path):
    translation_path = Path(translation_path)
    if not translation_path.is_absolute():
        translation_path = ROOT / translation_path
    if not SPANISH.exists():
        raise SystemExit(f"{SPANISH} not found. Prepare the Spanish 2025 file first (scripts/prepare_data.py).")
    if not translation_path.exists():
        print(f"English 2025 translation not found at {translation_path}; skipped.")
        return None

    spanish = pd.read_csv(SPANISH)
    english = pd.read_csv(translation_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in english.columns]
    if missing:
        raise SystemExit(f"Translation file: missing columns {missing}. Found {list(english.columns)}.")
    english = english.set_index("id")
    if not english.index.is_unique:
        raise SystemExit("Translation file: id is not unique.")

    unmatched = [i for i in spanish["id"] if i not in english.index]
    if unmatched:
        raise SystemExit(f"Translation file: {len(unmatched)} selected questions missing, e.g. {unmatched[:5]}.")

    rows, problems, empty_fields = [], [], 0
    for _, row in spanish.iterrows():
        en = english.loc[row["id"]]
        if int(en["question_number"]) != int(row["question_number"]):
            problems.append(f"{row['id']}: question number {en['question_number']} != {row['question_number']}")
        if str(en["answer"]).strip().upper() != row["answer"]:
            problems.append(f"{row['id']}: answer {en['answer']} != {row['answer']}")
        record = {"id": row["id"], "year": 2025, "answer": row["answer"],
                  "translation_mode": TRANSLATION_MODE,
                  "question_es": row["question"], "question_en": clean_text(en["question_en"])}
        for letter in LETTERS:
            record[f"option_{letter}_es"] = row[f"option_{letter}"]
            record[f"option_{letter}_en"] = clean_text(en[f"option_{letter}_en"])
        empty_fields += sum(not str(record[k]).strip() or str(record[k]) == "nan"
                            for k in record if k.endswith("_en"))
        rows.append(record)
    if problems:
        raise SystemExit("Translation file disagrees with the Spanish file:\n  " + "\n  ".join(problems))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    meta = {
        "output": str(OUTPUT),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source_spanish": str(SPANISH),
        "source_translation": str(translation_path),
        "translation": "MIR 2025 translation, same procedure as MIR 2026 (Google machine translation)",
        "rows": len(out),
        "translated_questions_not_selected": int(len(english) - len(out)),
        "empty_english_fields": int(empty_fields),
    }
    OUTPUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"English 2025: {len(out)} questions written to {OUTPUT} ({empty_fields} empty English fields)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--translation", default=DEFAULT_TRANSLATION, help="Translated MIR 2025 file.")
    args = parser.parse_args()
    build(args.translation)


if __name__ == "__main__":
    main()
