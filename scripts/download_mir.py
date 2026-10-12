"""Collect Spanish MIR exam questions into data/raw/mir_<year>_es.csv.

Three sources are supported, so that every exam year ends up in the same format:

  careqa  Download MIR 2020 to 2024 from the CareQA dataset on Hugging Face
          (HPAI-BSC/CareQA, Spanish configuration). CareQA is built from the official
          ministry booklets, with image-based and annulled questions already removed.

  pdf     Extract questions from an official ministry exam booklet and answer key
          (local files or URLs), for years not covered by CareQA, such as 2026.
          PDF layouts change between years, so always review the printed report.

  csv     Normalize an existing CSV (for example a file produced by another scraper)
          into the required format, using a column mapping.

All sources apply the same selection rules, so that every year is filtered identically:
questions that refer to an attached image are removed, annulled questions (without a valid
answer) are removed, and only questions up to --max-question are kept (default 200, which
excludes the 10 reserve questions). Output columns:
    id, year, question_number, question, option_a, option_b, option_c, option_d, answer

Examples:
    python scripts/download_mir.py careqa --years 2022 2024
    python scripts/download_mir.py pdf --year 2026 \
        --questions-pdf cuaderno_mir_2026_v0.pdf --answers-pdf respuestas_mir_2026.pdf
    python scripts/download_mir.py csv --year 2026 --input mir2026_text_primary_es.csv \
        --column-map '{"pregunta": "question", "a": "option_a", "b": "option_b", "c": "option_c", "d": "option_d", "respuesta": "answer"}'
"""

import argparse
import io
import json
import re
from pathlib import Path

import pandas as pd

LETTERS = "ABCD"
OUTPUT_COLUMNS = ["id", "year", "question_number", "question",
                  "option_a", "option_b", "option_c", "option_d", "answer"]
DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Questions that refer to an attached image cannot be answered from text alone. The MIR
# booklets mark them as "Pregunta vinculada a la imagen nº X". General mentions of imaging
# ("prueba de imagen", "imagen en vidrio deslustrado") are deliberately not matched.
IMAGE_PATTERN = re.compile(
    r"vinculada a la imagen"
    r"|imagen\s*n\.?\s*[º°o]?\s*\d+"
    r"|(?:imagen|figura|fotograf[ií]a)\s+adjunta"
    r"|(?:observe|véase|vea|ver|muestra|mostrada en|se muestra en)\s+la\s+(?:siguiente\s+)?(?:imagen|figura|fotograf[ií]a)"
    r"|la\s+(?:siguiente\s+)?(?:imagen|figura|fotograf[ií]a)\s+(?:que\s+se\s+muestra|mostrada|adjunta)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)

    careqa = sub.add_parser("careqa", help="Download MIR 2020 to 2024 from CareQA.")
    careqa.add_argument("--years", nargs="+", type=int, required=True)
    careqa.add_argument("--category", default="Medicine", help="CareQA category to keep (MIR = Medicine).")

    pdf = sub.add_parser("pdf", help="Extract questions from official PDF booklets.")
    pdf.add_argument("--year", type=int, required=True)
    pdf.add_argument("--questions-pdf", required=True, help="Exam booklet (path or URL).")
    pdf.add_argument("--answers-pdf", required=True, help="Definitive answer key (path or URL).")

    csv = sub.add_parser("csv", help="Normalize an existing CSV.")
    csv.add_argument("--year", type=int, required=True)
    csv.add_argument("--input", required=True)
    csv.add_argument("--column-map", default=None, help="JSON mapping from source to required column names.")

    for p in (careqa, pdf, csv):
        p.add_argument("--keep-image-questions", action="store_true",
                       help="Do not remove questions that refer to an image.")
        p.add_argument("--max-question", type=int, default=200,
                       help="Highest question number to keep. The default of 200 keeps the main "
                            "questions and removes the 10 reserve questions (201 to 210).")
    return parser.parse_args()


# ---------------------------------------------------------------- shared helpers

def answer_to_letter(value):
    text = str(value).strip().upper()
    if text in LETTERS:
        return text
    if text.isdigit() and 1 <= int(text) <= 4:
        return LETTERS[int(text) - 1]
    return None


def finalize(df, year, keep_images, max_question):
    """Apply the shared selection rules, validate and write the output file."""
    df = df.copy()
    df["year"] = year
    df["answer"] = df["answer"].map(answer_to_letter)
    report = {"year": year, "rows_in": len(df), "max_question": max_question}

    numbers = pd.to_numeric(df["question_number"], errors="coerce")
    if numbers.notna().all():
        beyond = numbers > max_question
        report["removed_reserve_questions"] = int(beyond.sum())
        df = df[~beyond]
    else:
        report["removed_reserve_questions"] = "not applied (question numbers are not numeric)"

    invalid = df["answer"].isna()
    report["removed_invalid_answer"] = int(invalid.sum())
    df = df[~invalid]

    for column in ["question", "option_a", "option_b", "option_c", "option_d"]:
        df[column] = df[column].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    empty = (df[["question", "option_a", "option_b", "option_c", "option_d"]] == "").any(axis=1)
    report["removed_empty_fields"] = int(empty.sum())
    df = df[~empty]

    if not keep_images:
        text = df["question"] + " " + df[["option_a", "option_b", "option_c", "option_d"]].agg(" ".join, axis=1)
        matches = text.str.extract(f"({IMAGE_PATTERN.pattern})", flags=re.IGNORECASE)[0]
        has_image = matches.notna()
        report["removed_image_questions"] = int(has_image.sum())
        report["removed_image_details"] = [
            {"question_number": str(n), "matched": m, "question_start": q[:120]}
            for n, m, q in zip(df.loc[has_image, "question_number"], matches[has_image], df.loc[has_image, "question"])
        ]
        df = df[~has_image]

    df["id"] = [f"mir{year}-{int(n):03d}" if str(n).isdigit() else f"mir{year}-{n}"
                for n in df["question_number"]]
    if df["id"].duplicated().any():
        raise ValueError(f"Duplicate question ids for {year}: {df.loc[df['id'].duplicated(), 'id'].tolist()}")

    report["rows_out"] = len(df)
    report["answer_distribution"] = df["answer"].value_counts().sort_index().to_dict()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / f"mir_{year}_es.csv"
    df[OUTPUT_COLUMNS].to_csv(out, index=False)
    (DATA_RAW / f"mir_{year}_es.report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return df


# ---------------------------------------------------------------- source: CareQA

def from_careqa(years, category, keep_images, max_question):
    from datasets import load_dataset

    ds = load_dataset("HPAI-BSC/CareQA", "CareQA_es", split="test").to_pandas()
    print(f"CareQA columns: {list(ds.columns)}")
    required = {"question", "op1", "op2", "op3", "op4", "cop", "year", "category"}
    missing = required - set(ds.columns)
    if missing:
        raise SystemExit(f"CareQA layout changed; missing columns {sorted(missing)}.")
    print(f"Categories available: {sorted(ds['category'].unique())}")

    for year in years:
        part = ds[(ds["year"].astype(int) == year)
                  & (ds["category"].astype(str).str.lower() == category.lower())]
        if part.empty:
            print(f"No CareQA questions for {year} in category {category}; skipped.")
            continue
        # Prefer a column that identifies the question within the exam and is unique per year.
        number_column = next((c for c in ("question_number", "exam_id", "unique_id")
                              if c in part.columns and part[c].is_unique), None)
        print(f"{year}: {len(part)} questions; question numbers taken from "
              f"'{number_column or 'row order'}'")
        df = pd.DataFrame({
            "question_number": part[number_column].values if number_column else range(1, len(part) + 1),
            "question": part["question"].values,
            "option_a": part["op1"].values,
            "option_b": part["op2"].values,
            "option_c": part["op3"].values,
            "option_d": part["op4"].values,
            "answer": part["cop"].values,
        })
        finalize(df, year, keep_images, max_question)


# ---------------------------------------------------------------- source: official PDFs

def read_pdf_text(location):
    import pdfplumber

    if re.match(r"https?://", location):
        import requests

        response = requests.get(location, timeout=60)
        response.raise_for_status()
        handle = io.BytesIO(response.content)
    else:
        handle = location
    with pdfplumber.open(handle) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_booklet(text, max_question):
    """Parse questions with four numbered options from booklet text.

    The booklet numbers both questions and options ("17. ..." followed by "1. ", "2. ",
    "3. ", "4. "). A state machine distinguishes them: after a question header, options
    1 to 4 are expected in order; after option 4, the next header must be the following
    question number. Continuation lines are appended to the current field.
    """
    numbered = re.compile(r"^\s*(\d{1,3})\s*[.\-)]\s+(.*)$")
    questions, current = [], None
    expected_question = 1
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = numbered.match(line)
        if match:
            number, rest = int(match.group(1)), match.group(2)
            awaiting_option = current is not None and len(current["options"]) < 4
            if awaiting_option and number == len(current["options"]) + 1:
                current["options"].append(rest)
                continue
            if number == expected_question and (current is None or len(current["options"]) == 4):
                current = {"question_number": number, "question": rest, "options": []}
                questions.append(current)
                expected_question += 1
                continue
        if current is None:
            continue
        if current["options"]:
            current["options"][-1] += " " + line
        else:
            current["question"] += " " + line
    complete = [q for q in questions if len(q["options"]) == 4 and q["question_number"] <= max_question]
    return complete, len(questions)


def parse_answer_key(text):
    """Return {question number: answer} from an answer-key text; annulled questions map to None."""
    key = {}
    for number, answer in re.findall(r"\b(\d{1,3})\s+([1-4]|ANULADA|ANUL|-)\b", text, flags=re.IGNORECASE):
        key.setdefault(int(number), int(answer) if answer.isdigit() else None)
    return key


def from_pdf(year, questions_pdf, answers_pdf, max_question, keep_images):
    questions, n_headers = parse_booklet(read_pdf_text(questions_pdf), 210)
    key = parse_answer_key(read_pdf_text(answers_pdf))
    print(f"Found {n_headers} question headers, {len(questions)} complete questions, "
          f"{len(key)} entries in the answer key ({sum(v is None for v in key.values())} annulled).")
    numbers = [q["question_number"] for q in questions]
    gaps = sorted(set(range(1, max(numbers, default=0) + 1)) - set(numbers))
    if gaps:
        print(f"Warning: questions not parsed completely: {gaps[:30]}{' ...' if len(gaps) > 30 else ''}")

    rows = []
    for q in questions:
        answer = key.get(q["question_number"])
        rows.append({
            "question_number": q["question_number"],
            "question": q["question"],
            "option_a": q["options"][0], "option_b": q["options"][1],
            "option_c": q["options"][2], "option_d": q["options"][3],
            "answer": answer if answer is not None else "ANNULLED",
        })
    finalize(pd.DataFrame(rows), year, keep_images, max_question)


# ---------------------------------------------------------------- source: existing CSV

def from_csv(year, path, column_map, keep_images, max_question):
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=json.loads(column_map))
    if "question_number" not in df.columns:
        df["question_number"] = df["id"] if "id" in df.columns else range(1, len(df) + 1)
    required = ["question", "option_a", "option_b", "option_c", "option_d", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing}. Found {list(df.columns)}. Use --column-map.")
    finalize(df, year, keep_images, max_question)


def main():
    args = parse_args()
    if args.source == "careqa":
        from_careqa(args.years, args.category, args.keep_image_questions, args.max_question)
    elif args.source == "pdf":
        from_pdf(args.year, args.questions_pdf, args.answers_pdf, args.max_question, args.keep_image_questions)
    else:
        from_csv(args.year, args.input, args.column_map, args.keep_image_questions, args.max_question)


if __name__ == "__main__":
    main()
