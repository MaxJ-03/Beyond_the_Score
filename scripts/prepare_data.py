"""Prepare all MIR datasets with one command.

What it produces:
    data/raw/mir_2020_es.csv through mir_2024_es.csv    Spanish, from CareQA
    data/mir_2020_en.csv through mir_2024_en.csv        CareQA English translations
    data/raw/mir_2025_es.csv                             Spanish, from audited Version 0 PDFs
    data/raw/mir_2026_es.csv                             original Spanish, from the scraped
                                                         file data/external/mir2026_text_primary_es.csv
    data/mir_2026_en.csv                                 English, from the scraped translation
                                                         data/external/mir_2026_en_mlt.csv
    data/translation_checks/                             validation of all English files

All English files have Spanish text in parallel columns. The main-question cutoff is
175 for MIR 2020-2021 and 200 for MIR 2022-2026; image-linked, annulled, and reserve
questions are excluded. Audited transcription corrections run during preparation.

Usage:
    python scripts/prepare_data.py
"""

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LETTERS = ["a", "b", "c", "d"]

def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_mir = _load_script("download_mir")
prepare_mir2026_en = _load_script("prepare_mir2026_en")
audited_fixes = _load_script("audited_mir_corrections")

ASEN_COLUMNS = {
    "question_id": "question_number",
    "question_es": "question",
    "option_1_es": "option_a",
    "option_2_es": "option_b",
    "option_3_es": "option_c",
    "option_4_es": "option_d",
    "correct_answer": "answer",
}




def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024], help="CareQA years.")
    parser.add_argument("--mir2026-es", default="data/external/mir2026_text_primary_es.csv",
                        help="Scraped MIR 2026 file with the original Spanish questions.")
    parser.add_argument("--mir2026-en", default="data/external/mir_2026_en_mlt.csv",
                        help="Scraped MIR 2026 file with the English translation.")
    parser.add_argument("--max-question", type=int, default=200)
    return parser.parse_args()


def load_careqa(config):
    from datasets import load_dataset

    df = load_dataset("HPAI-BSC/CareQA", config, split="test").to_pandas()
    required = {"question", "op1", "op2", "op3", "op4", "cop", "year", "category", "exam_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CareQA {config}: missing columns {sorted(missing)}. Found {list(df.columns)}.")
    df["category"] = df["category"].astype(str).str.lower()
    df["year"] = df["year"].astype(int)
    return df


def prepare_careqa(years, max_question):
    spanish = load_careqa("CareQA_es")
    english = load_careqa("CareQA_en")
    use_unique_id = (
        "unique_id" in spanish.columns and "unique_id" in english.columns
        and spanish["unique_id"].isin(english["unique_id"]).mean() > 0.9
    )
    key = "unique_id" if use_unique_id else "exam_id"
    print(f"Matching Spanish and English CareQA questions on '{key}'.")

    for year in years:
        es_year = spanish[(spanish["year"] == year) & (spanish["category"] == "medicine")]
        en_year = english[(english["year"] == year) & (english["category"] == "medicine")]
        if es_year.empty:
            print(f"No CareQA questions for {year}; skipped.")
            continue

        es_frame = pd.DataFrame({
            "question_number": es_year["exam_id"].values,
            "question": es_year["question"].values,
            "option_a": es_year["op1"].values, "option_b": es_year["op2"].values,
            "option_c": es_year["op3"].values, "option_d": es_year["op4"].values,
            "answer": es_year["cop"].values,
            "match_key": es_year[key].values,
        })
        print(f"\n--- MIR {year} (Spanish, CareQA) ---")
        if year == 2020:
            es_frame = audited_fixes.fix_2020_spanish(es_frame)
        main_limit = min(max_question, 175) if year in (2020, 2021) else max_question
        selected = download_mir.finalize(
            es_frame, year, keep_images=False, max_question=main_limit
        )

        en_index = en_year.set_index(key)
        if not en_index.index.is_unique:
            raise SystemExit(f"CareQA_en {year}: '{key}' is not unique; cannot match translations.")
        matched = selected[selected["match_key"].isin(en_index.index)]
        unmatched = len(selected) - len(matched)

        rows, answer_mismatches = [], 0
        for _, row in matched.iterrows():
            en = en_index.loc[row["match_key"]]
            en_answer = download_mir.answer_to_letter(en["cop"])
            answer_mismatches += en_answer != row["answer"]
            record = {"id": row["id"], "year": year, "answer": row["answer"],
                      "translation_mode": "careqa_gpt4",
                      "question_es": row["question"], "question_en": str(en["question"]).strip()}
            for letter, number in zip(LETTERS, range(1, 5)):
                record[f"option_{letter}_es"] = row[f"option_{letter}"]
                en_option = str(en[f"op{number}"]).strip()
                if year == 2020 and int(row["question_number"]) == 104:
                    en_option = audited_fixes.fix_2020_english_option(letter, en_option)
                record[f"option_{letter}_en"] = en_option
            rows.append(record)

        out = DATA / f"mir_{year}_en.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        meta = {
            "output": str(out),
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "source": "HPAI-BSC/CareQA, configurations CareQA_es and CareQA_en",
            "translation": "official CareQA English version, translated with GPT-4 and "
                           "validated by human evaluators (Arias-Duart et al., 2025)",
            "match_key": key,
            "rows": len(rows),
            "spanish_questions_without_english_match": unmatched,
            "answer_mismatches_between_languages": int(answer_mismatches),
        }
        out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(f"English: {len(rows)} questions written to {out} "
              f"({unmatched} without match, {answer_mismatches} answer mismatches)")


def prepare_2026(path, max_question):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not path.exists():
        print(f"\nMIR 2026 Spanish file not found at {path}; skipped. Place it there and rerun.")
        return
    df = pd.read_csv(path)
    missing = [c for c in [*ASEN_COLUMNS, "has_image", "annulled"] if c not in df.columns]
    if missing:
        raise SystemExit(f"MIR 2026 file: missing columns {missing}. Found {list(df.columns)}.")
    df = audited_fixes.fix_2026_source(df)
    selected = df[~df["has_image"].astype(bool) & ~df["annulled"].astype(bool)]
    frame = selected[list(ASEN_COLUMNS)].rename(columns=ASEN_COLUMNS)
    print(f"\n--- MIR 2026 (Spanish, scraped file: {len(df)} rows, "
          f"{len(df) - len(selected)} image-based or annulled removed) ---")
    download_mir.finalize(frame, 2026, keep_images=False, max_question=max_question)


def validate_translations():
    files = sorted(DATA.glob("mir_*_en.csv"))
    if not files:
        return
    validator = _load_script("validate_translations")
    import sys

    saved = sys.argv
    sys.argv = ["validate_translations.py", *map(str, files), "--out-dir", str(DATA / "translation_checks")]
    try:
        print("\n=== Translation checks ===")
        validator.main()
    finally:
        sys.argv = saved


def summary():
    print("\n=== Datasets ready for the pipeline ===")
    for year in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
        for language, path in (("es", DATA / "raw" / f"mir_{year}_es.csv"), ("en", DATA / f"mir_{year}_en.csv")):
            status = f"{len(pd.read_csv(path))} questions" if path.exists() else "missing"
            print(f"  MIR {year} {language}: {status}")


def main():
    args = parse_args()
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    prepare_careqa(args.years, args.max_question)
    _load_script("prepare_mir2025").main()
    prepare_2026(args.mir2026_es, args.max_question)
    if (DATA / "raw" / "mir_2026_es.csv").exists():
        print("\n--- MIR 2026 (English, scraped translation) ---")
        prepare_mir2026_en.build(args.mir2026_en)
    validate_translations()
    summary()


if __name__ == "__main__":
    main()
