"""Prepare all MIR datasets with one command.

What it produces:
    data/raw/mir_2020_es.csv to mir_2024_es.csv    Spanish, from CareQA on Hugging Face
    data/mir_2020_en.csv to mir_2024_en.csv        English, CareQA's GPT-4 translation
    data/raw/mir_2025_es.csv                       Spanish, from the audited Version 0 PDFs
    data/mir_2025_en.csv                           English, from data/external/mir_2025_en_mlt.csv
    data/raw/mir_2026_es.csv                       Spanish, from the scraped file
                                                   data/external/mir2026_text_primary_es.csv
    data/mir_2026_en.csv                           English, from data/external/mir_2026_en_mlt.csv
    data/translation_checks/                       checks of all English files

All English files have the Spanish text in parallel columns and contain exactly the
questions of their Spanish counterpart. The main-question cutoff is 175 for MIR 2020-2021
and 200 for MIR 2022-2026; image-linked, annulled and reserve questions are excluded (see
src/mir_sources.py). Audited transcription corrections (src/mir_corrections.py) are
applied during preparation.

Sections:
    1. CareQA, MIR 2020 to 2024 (Spanish and English)
    2. MIR 2026 (Spanish)
    3. English files from the 2025 and 2026 machine translations
    4. translation checks
    5. main

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --checks-only     only rerun the translation checks
"""

import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd

import mir_corrections
from config import PROJECT_ROOT, resolve_path
from mir_sources import answer_to_letter, finalize, prepare_mir2025

DATA = PROJECT_ROOT / "data"
LETTERS = ["a", "b", "c", "d"]


# ---------------------------------------------------------------- 1. CareQA, MIR 2020 to 2024

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
            es_frame = mir_corrections.fix_2020_spanish(es_frame)
        main_limit = min(max_question, 175) if year in (2020, 2021) else max_question
        selected = finalize(
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
            en_answer = answer_to_letter(en["cop"])
            answer_mismatches += en_answer != row["answer"]
            record = {"id": row["id"], "year": year, "answer": row["answer"],
                      "translation_mode": "careqa_gpt4",
                      "question_es": row["question"], "question_en": str(en["question"]).strip()}
            for letter, number in zip(LETTERS, range(1, 5)):
                record[f"option_{letter}_es"] = row[f"option_{letter}"]
                en_option = str(en[f"op{number}"]).strip()
                if year == 2020 and int(row["question_number"]) == 104:
                    en_option = mir_corrections.fix_2020_english_option(letter, en_option)
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


# ---------------------------------------------------------------- 2. MIR 2026 (Spanish)

# Columns of the scraped MIR 2026 file and their names in the common format.
ASEN_COLUMNS = {
    "question_id": "question_number",
    "question_es": "question",
    "option_1_es": "option_a",
    "option_2_es": "option_b",
    "option_3_es": "option_c",
    "option_4_es": "option_d",
    "correct_answer": "answer",
}


def prepare_2026(path, max_question):
    path = resolve_path(path)
    if not path.exists():
        print(f"\nMIR 2026 Spanish file not found at {path}; skipped. Place it there and rerun.")
        return
    df = pd.read_csv(path)
    missing = [c for c in [*ASEN_COLUMNS, "has_image", "annulled"] if c not in df.columns]
    if missing:
        raise SystemExit(f"MIR 2026 file: missing columns {missing}. Found {list(df.columns)}.")
    df = mir_corrections.fix_2026_source(df)
    selected = df[~df["has_image"].astype(bool) & ~df["annulled"].astype(bool)]
    frame = selected[list(ASEN_COLUMNS)].rename(columns=ASEN_COLUMNS)
    print(f"\n--- MIR 2026 (Spanish, scraped file: {len(df)} rows, "
          f"{len(df) - len(selected)} image-based or annulled removed) ---")
    finalize(frame, 2026, keep_images=False, max_question=max_question)


# ---------------------------------------------------------------- 3. English files from machine translations

# The 2025 and 2026 translations were produced with the same procedure (Google machine
# translation) and share one translation mode. Each entry gives the columns of the
# translated file: the key that identifies a question, the matching column in the Spanish
# file, the answer column and the English option columns. Known deviation in 2025: in
# question 38, option C, the Spanish source contains the typographical error "0,3-05 mL",
# which the translation renders as "0.3-0.5 mL"; the translation checks report the case.
TRANSLATIONS = {
    2025: {"key": "id", "spanish_key": "id", "answer": "answer",
           "options": {"a": "option_a_en", "b": "option_b_en", "c": "option_c_en", "d": "option_d_en"}},
    2026: {"key": "question_id", "spanish_key": "question_number", "answer": "correct_answer",
           "options": {"a": "option_1_en", "b": "option_2_en", "c": "option_3_en", "d": "option_4_en"}},
}
TRANSLATION_MODE = "scraped_translation"


def clean_text(value):
    """Strip the text and remove zero-width spaces left by the machine translation."""
    return str(value).replace("\u200b", "").strip()


def build_english(year, translation_path):
    """Write data/mir_<year>_en.csv from the translation of the selected Spanish questions.

    The output has the same format as the CareQA English files:
        id, year, answer, translation_mode, question_es, question_en,
        option_a_es, option_a_en, ..., option_d_es, option_d_en
    The Spanish text is taken from data/raw/mir_<year>_es.csv, so it is identical to the text
    evaluated in the Spanish run. Stops if a selected question is missing from the
    translation or if an answer letter differs between the two files.
    """
    spec = TRANSLATIONS[year]
    spanish_path = DATA / "raw" / f"mir_{year}_es.csv"
    translation_path = resolve_path(translation_path)
    if not spanish_path.exists():
        raise SystemExit(f"{spanish_path} not found. Prepare the Spanish {year} file first.")
    if not translation_path.exists():
        print(f"English {year} translation not found at {translation_path}; skipped.")
        return None

    spanish = pd.read_csv(spanish_path)
    english = pd.read_csv(translation_path)
    if year == 2026:
        english = mir_corrections.fix_2026_translation(english)
    required = [spec["key"], spec["answer"], "question_en", *spec["options"].values()]
    missing = [c for c in required if c not in english.columns]
    if missing:
        raise SystemExit(f"Translation file: missing columns {missing}. Found {list(english.columns)}.")
    english = english.set_index(spec["key"])
    if not english.index.is_unique:
        raise SystemExit(f"Translation file: {spec['key']} is not unique.")

    rows, problems, empty_fields = [], [], 0
    for _, row in spanish.iterrows():
        key = int(row["question_number"]) if spec["spanish_key"] == "question_number" else row[spec["spanish_key"]]
        if key not in english.index:
            problems.append(f"{row['id']}: missing from the translation")
            continue
        en = english.loc[key]
        if answer_to_letter(en[spec["answer"]]) != row["answer"]:
            problems.append(f"{row['id']}: answer {en[spec['answer']]} != {row['answer']}")
        record = {"id": row["id"], "year": year, "answer": row["answer"],
                  "translation_mode": TRANSLATION_MODE,
                  "question_es": row["question"], "question_en": clean_text(en["question_en"])}
        for letter in LETTERS:
            record[f"option_{letter}_es"] = row[f"option_{letter}"]
            record[f"option_{letter}_en"] = clean_text(en[spec["options"][letter]])
        empty_fields += sum(not str(record[k]).strip() or str(record[k]) == "nan"
                            for k in record if k.endswith("_en"))
        rows.append(record)
    if problems:
        raise SystemExit("Translation file disagrees with the Spanish file:\n  " + "\n  ".join(problems))

    out_path = DATA / f"mir_{year}_en.csv"
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    meta = {
        "output": str(out_path),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "source_spanish": str(spanish_path),
        "source_translation": str(translation_path),
        "translation": f"MIR {year} translation (Google machine translation), see the translation file",
        "rows": len(out),
        "translated_questions_not_selected": int(len(english) - len(out)),
        "empty_english_fields": int(empty_fields),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"English {year}: {len(out)} questions written to {out_path} ({empty_fields} empty English fields)")
    return out


# ---------------------------------------------------------------- 4. translation checks
#
# Checks per English file: empty translated fields and leftover option markers, answer
# letters restricted to A to D, numbers in the Spanish source that are missing from the
# English text (decimal commas are normalized, so 3,5 matches 3.5), and English to Spanish
# length ratios far outside the typical range for the file. Written to
# data/translation_checks/: validation_summary.csv (one row per year), validation_flags.csv
# (every flagged field) and spot_check_sample.csv (a random sample per year for manual review).

FIELDS = ["question", "option_a", "option_b", "option_c", "option_d"]
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
MARKER_PATTERN = re.compile(r"\[\[\s*[A-D]?\s*\]\]")


def numbers_in(text):
    """Return the set of numbers in a text, with decimal commas written as points."""
    return {n.replace(",", ".") for n in NUMBER_PATTERN.findall(str(text))}


def check_file(path):
    """Check one English file; return its data, a summary row and the flagged fields."""
    df = pd.read_csv(path)
    flags = []
    ratios = []

    for _, row in df.iterrows():
        for field in FIELDS:
            es, en = str(row[f"{field}_es"]), str(row[f"{field}_en"])
            if not en.strip() or en == "nan":
                flags.append((row["id"], field, "empty translation"))
                continue
            if MARKER_PATTERN.search(en):
                flags.append((row["id"], field, "leftover option marker"))
            missing = numbers_in(es) - numbers_in(en)
            if missing:
                flags.append((row["id"], field, f"numbers missing: {sorted(missing)}"))
            if len(es) > 20:
                ratios.append((row["id"], field, len(en) / len(es)))
        if str(row["answer"]).strip().upper() not in {"A", "B", "C", "D"}:
            flags.append((row["id"], "answer", "invalid answer letter"))

    ratio_series = pd.Series([r[2] for r in ratios])
    low, high = ratio_series.quantile(0.01), ratio_series.quantile(0.99)
    for row_id, field, ratio in ratios:
        if ratio < low or ratio > high:
            flags.append((row_id, field, f"unusual length ratio {ratio:.2f}"))

    summary = {
        "file": path,
        "years": ", ".join(str(y) for y in sorted(df["year"].unique())),
        "questions": len(df),
        "answer_distribution": df["answer"].value_counts().sort_index().to_dict(),
        "mean_question_chars_en": round(df["question_en"].astype(str).str.len().mean(), 1),
        "median_length_ratio_en_es": round(ratio_series.median(), 3),
        "flagged_fields": len(flags),
    }
    flag_df = pd.DataFrame(flags, columns=["id", "field", "issue"])
    flag_df.insert(0, "file", path)
    return df, summary, flag_df


def check_translations(out_dir=DATA / "translation_checks", sample_size=15, seed=42):
    """Run the translation checks on every data/mir_<year>_en.csv file."""
    files = [str(path) for path in sorted(DATA.glob("mir_*_en.csv"))]
    if not files:
        return
    print("\n=== Translation checks ===")
    summaries, all_flags, samples = [], [], []
    for path in files:
        df, summary, flags = check_file(path)
        summaries.append(summary)
        all_flags.append(flags)
        sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
        samples.append(sample.assign(source_file=path, reviewer_ok="", reviewer_notes=""))

    summary_df = pd.DataFrame(summaries)
    pd.set_option("display.width", 200)
    print(summary_df.to_string(index=False))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "validation_summary.csv", index=False)
    pd.concat(all_flags, ignore_index=True).to_csv(out_dir / "validation_flags.csv", index=False)
    pd.concat(samples, ignore_index=True).to_csv(out_dir / "spot_check_sample.csv", index=False)
    print(f"\nWrote validation_summary.csv, validation_flags.csv and spot_check_sample.csv to {out_dir}")


# ---------------------------------------------------------------- 5. main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024], help="CareQA years.")
    parser.add_argument("--mir2025-en", default="data/external/mir_2025_en_mlt.csv",
                        help="MIR 2025 file with the English translation.")
    parser.add_argument("--mir2026-es", default="data/external/mir2026_text_primary_es.csv",
                        help="Scraped MIR 2026 file with the original Spanish questions.")
    parser.add_argument("--mir2026-en", default="data/external/mir_2026_en_mlt.csv",
                        help="Scraped MIR 2026 file with the English translation.")
    parser.add_argument("--max-question", type=int, default=200)
    parser.add_argument("--checks-only", action="store_true",
                        help="Only rerun the translation checks on the existing English files.")
    return parser.parse_args()


def summary():
    print("\n=== Datasets ready for the pipeline ===")
    for year in (2020, 2021, 2022, 2023, 2024, 2025, 2026):
        for language, path in (("es", DATA / "raw" / f"mir_{year}_es.csv"), ("en", DATA / f"mir_{year}_en.csv")):
            status = f"{len(pd.read_csv(path))} questions" if path.exists() else "missing"
            print(f"  MIR {year} {language}: {status}")


def main():
    args = parse_args()
    if args.checks_only:
        check_translations()
        return
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    prepare_careqa(args.years, args.max_question)
    print("\n--- MIR 2025 (Spanish, official PDFs) ---")
    prepare_mir2025()
    print("\n--- MIR 2025 (English, machine translation) ---")
    build_english(2025, args.mir2025_en)
    prepare_2026(args.mir2026_es, args.max_question)
    if (DATA / "raw" / "mir_2026_es.csv").exists():
        print("\n--- MIR 2026 (English, machine translation) ---")
        build_english(2026, args.mir2026_en)
    check_translations()
    summary()


if __name__ == "__main__":
    main()
