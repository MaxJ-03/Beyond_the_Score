"""Check the English MIR files against their Spanish text, for completeness and consistency
across exam years.

Checks per file:
    - empty translated fields and leftover option markers
    - answer letters restricted to A to D
    - numbers in the Spanish source that are missing from the English translation
      (decimal commas are normalized, so 3,5 matches 3.5)
    - English to Spanish length ratios far outside the typical range for the file

Outputs:
    written to --out-dir (default data/translation_checks):
    - validation_summary.csv comparing all years (also printed)
    - validation_flags.csv listing every flagged field
    - spot_check_sample.csv with a random sample per year for manual review

Example:
    python scripts/validate_translations.py data/mir_2022_en.csv data/mir_2024_en.csv data/mir_2026_en.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

FIELDS = ["question", "option_a", "option_b", "option_c", "option_d"]
NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
MARKER_PATTERN = re.compile(r"\[\[\s*[A-D]?\s*\]\]")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="English MIR files (data/mir_<year>_en.csv).")
    parser.add_argument("--sample-size", type=int, default=15, help="Questions per year for manual spot checks.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/translation_checks", help="Directory for the check reports.")
    return parser.parse_args()


def numbers_in(text):
    return {n.replace(",", ".") for n in NUMBER_PATTERN.findall(str(text))}


def check_file(path):
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


def main():
    args = parse_args()
    summaries, all_flags, samples = [], [], []

    for path in args.files:
        df, summary, flags = check_file(path)
        summaries.append(summary)
        all_flags.append(flags)
        sample = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed)
        samples.append(sample.assign(source_file=path, reviewer_ok="", reviewer_notes=""))

    summary_df = pd.DataFrame(summaries)
    pd.set_option("display.width", 200)
    print(summary_df.to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "validation_summary.csv", index=False)
    pd.concat(all_flags, ignore_index=True).to_csv(out_dir / "validation_flags.csv", index=False)
    pd.concat(samples, ignore_index=True).to_csv(out_dir / "spot_check_sample.csv", index=False)
    print(f"\nWrote validation_summary.csv, validation_flags.csv and spot_check_sample.csv to {out_dir}")


if __name__ == "__main__":
    main()
