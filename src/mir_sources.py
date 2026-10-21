"""Reading the raw MIR exam sources into the common Spanish format.

Sections:
    1. selection rules     shared by every edition: reserve, annulled and image-linked questions
                           are removed, and the result is written to data/raw/mir_<year>_es.csv
    2. official PDFs       text extraction from ministry booklets and answer keys
    3. MIR 2025            the January 2025 exam (convocatoria 2024), built from the audited
                           Version 0 PDFs in data/external/

Output columns of every Spanish file:
    id, year, question_number, question, option_a, option_b, option_c, option_d, answer

This module is used by src/prepare_data.py and is not run on its own.
"""

import hashlib
import json
import platform
import re
from pathlib import Path

import pandas as pd
import pdfplumber

from config import PROJECT_ROOT

LETTERS = "ABCD"
OUTPUT_COLUMNS = ["id", "year", "question_number", "question",
                  "option_a", "option_b", "option_c", "option_d", "answer"]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------- 1. selection rules

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


# ---------------------------------------------------------------- 2. official PDFs

def read_pdf_text(path):
    """Return the text of all pages of a local PDF file."""
    with pdfplumber.open(path) as pdf:
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


# ---------------------------------------------------------------- 3. MIR 2025
#
# The January 25, 2025 exam (convocatoria 2024), Version 0, from:
#   data/external/mir2025_questions_v0.pdf
#   data/external/mir2025_answers_definitive_v0.pdf
#
# Fails closed if extraction cannot recover all 210 questions, 204 definitive answers, the
# exact audited PDFs, and the independently audited 170-question content. The golden
# checksum was derived from a separate machine-text audit; it does not claim that a human
# visually proofread every PDF glyph.

ROOT = PROJECT_ROOT
QUESTION_PDF = ROOT / "data/external/mir2025_questions_v0.pdf"
ANSWER_PDF = ROOT / "data/external/mir2025_answers_definitive_v0.pdf"
ANNULLED = {15, 26, 28, 56, 162, 186}
IMAGE_LINKED = set(range(1, 26))  # The first 25 questions require images.

# SHA-256 of the two PDFs independently reviewed for this examination.
# These checks prevent silently mixing up exam years, versions, or changed PDFs.
EXPECTED_PDF_HASHES = {
    "mir2025_questions_v0.pdf": "101a097f566fa3d20648c1b4c67d5ae739a507bce0645127472b482a9b946c4b",
    "mir2025_answers_definitive_v0.pdf": "b135110af27fa26f1fd661093ad443cdcf1afb6a8250948f684e05f5bd2ffa75",
}

# Semantic checksum of the independently audited 170-row CSV (all 9 fields,
# sorted by question number, serialized as stable JSON). This protects against
# silent question/option/answer changes while allowing CRLF vs LF CSV line endings.
EXPECTED_AUDITED_CONTENT_HASH = "d9e1aeec7c26ddb405cbaaf6fea6ae7ee7451523f9682158bd965ee6262305b7"



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_hash(frame: pd.DataFrame) -> str:
    """Platform-independent fingerprint of all evaluation fields and answers."""
    records = frame[OUTPUT_COLUMNS].sort_values("question_number").to_dict(orient="records")
    text = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_source_pdfs():
    for file in (QUESTION_PDF, ANSWER_PDF):
        if not file.is_file():
            raise SystemExit(f"Missing PDF: {file}")
    for path in (QUESTION_PDF, ANSWER_PDF):
        actual = sha256_file(path)
        expected = EXPECTED_PDF_HASHES[path.name]
        if actual != expected:
            raise SystemExit(
                f"STOP: source PDF differs from audited version: {path.name}\n"
                f"Expected SHA-256: {expected}\nFound SHA-256: {actual}\n"
                "Confirm you downloaded the specified Version 0 PDF; do not use the dataset yet."
            )
    with pdfplumber.open(QUESTION_PDF) as pdf:
        first = (pdf.pages[0].extract_text() or "").upper()
        compact = re.sub(r"\s+", " ", first)
        # The 2025 booklet prints "PRUEBAS SELECTIVAS" and "2024" in
        # different text columns/lines. Confirm both the exam type and the
        # "MEDICINA 2024" designation without requiring adjacent words.
        if not re.search(r"\bPRUEBAS\s+SELECTIVAS\b", compact) or not re.search(r"\bMEDICINA\s+2024\b", compact):
            raise SystemExit("Wrong question booklet: expected PRUEBAS SELECTIVAS and MEDICINA 2024 (January 2025 exam).")
        if not re.search(r"MEDICINA\s*-\s*VERSI[ÓO]N", compact) or not re.search(r"\b0\b", compact):
            raise SystemExit("Question booklet version not identified. Open the cover and confirm Version 0.")
    with pdfplumber.open(ANSWER_PDF) as pdf:
        first = re.sub(r"\s+", " ", pdf.pages[0].extract_text() or "").upper()
        if "DEFINITIVAMENTE" not in first or "MEDICINA" not in first or "VERSI" not in first:
            raise SystemExit("Wrong answer key: expected definitive, Medicine, Version 0.")


def read_two_column_booklet():
    chunks = []
    with pdfplumber.open(QUESTION_PDF) as pdf:
        for page in pdf.pages[2:]:  # cover + blank/instruction page(s); questions begin on PDF page 3
            midpoint = page.width / 2
            for bbox in ((0, 0, midpoint, page.height),
                         (midpoint, 0, page.width, page.height)):
                text = page.crop(bbox).extract_text(x_tolerance=2) or ""
                text = "\n".join(
                    line for line in text.splitlines()
                    if not re.match(r"^\s*(?:P[áa]gina\s*:|\(--MEDICINA|MINISTERIO DE SANIDAD)", line, re.I)
                )
                chunks.append(text)
    return "\n".join(chunks)


def prepare_mir2025():
    """Build data/raw/mir_2025_es.csv from the two official PDFs; stops on any audit mismatch."""
    check_source_pdfs()
    booklet_text = read_two_column_booklet()
    questions, num_headers = parse_booklet(booklet_text, max_question=210)
    numbers = [q["question_number"] for q in questions]
    expected_numbers = list(range(1, 211))
    if numbers != expected_numbers:
        missing = sorted(set(expected_numbers) - set(numbers))
        extra = sorted(set(numbers) - set(expected_numbers))
        raise SystemExit(
            f"STOP: question extraction incomplete. Headers={num_headers}, "
            f"complete={len(questions)}, missing={missing}, extra={extra}. "
            "Do not use this dataset. Review the PDF text extraction/layout."
        )

    key = parse_answer_key(read_pdf_text(str(ANSWER_PDF)))
    expected_key_numbers = set(expected_numbers) - ANNULLED
    valid_key_numbers = {n for n, value in key.items() if value in (1, 2, 3, 4)}
    if valid_key_numbers != expected_key_numbers or any(key.get(n) is not None for n in ANNULLED):
        raise SystemExit(
            "STOP: definitive answer-key extraction incomplete or inconsistent. "
            f"Missing valid={sorted(expected_key_numbers - valid_key_numbers)}; "
            f"unexpected valid={sorted(valid_key_numbers - expected_key_numbers)}; "
            f"annulled with non-null answers={[n for n in sorted(ANNULLED) if key.get(n) is not None]}. "
            "Do not use this dataset."
        )
    if key[150] != 4 or key[208] != 2:
        raise SystemExit("STOP: definitive-key checks failed: expected Q150=4, Q208=2.")

    for q in questions[:25]:
        if not re.search(r"(?:vinculada\s+a\s+la\s+)?imagen\s*(?:n[.º°o]*\s*)?\d+", q["question"], flags=re.I):
            raise SystemExit(f"STOP: Q{q['question_number']} not recognized as image-linked.")

    rows = []
    for q in questions:
        number = q["question_number"]
        if number > 200 or number in IMAGE_LINKED or number in ANNULLED:
            continue
        rows.append({
            "question_number": number,
            "question": q["question"],
            "option_a": q["options"][0],
            "option_b": q["options"][1],
            "option_c": q["options"][2],
            "option_d": q["options"][3],
            "answer": key[number],
        })

    # Remove split PDF page footers and normalize word/dose hyphens before writing CSV.
    for row in rows:
        for field in ("question", "option_a", "option_b", "option_c", "option_d"):
            value = row[field]
            value = re.sub(r"\s+(?:P[áa]gina:\s*)?\d{1,2}\s+de\s+33\s*$", "", value).strip()
            value = re.sub(r"(?<=\w)-\s+(?=\w)", "-", value)
            row[field] = value

    # Verify the entire independently audited content BEFORE overwriting
    # data/raw/mir_2025_es.csv. This includes all 170 stems, 680 options,
    # question IDs, exam years and answer letters, not just counts.
    preview = pd.DataFrame(rows)
    if len(preview) != 170 or not preview["question_number"].is_unique:
        raise SystemExit("STOP: incorrect number or duplicate question numbers.")
    preview["year"] = 2025
    preview["answer"] = preview["answer"].map(answer_to_letter)
    for field in ("question", "option_a", "option_b", "option_c", "option_d"):
        preview[field] = preview[field].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        if preview[field].eq("").any():
            raise SystemExit(f"STOP: empty extracted {field}.")
        if preview[field].str.contains(r"\b\d{1,2}\s+de\s+33\s*$", regex=True).any():
            raise SystemExit(f"STOP: stray page footer still present in {field}.")
    preview["id"] = preview["question_number"].map(lambda n: f"mir2025-{n:03d}")
    actual_content_hash = semantic_hash(preview)
    if actual_content_hash != EXPECTED_AUDITED_CONTENT_HASH:
        raise SystemExit(
            "STOP: generated content differs from the independently audited 170 questions.\n"
            f"Expected semantic SHA-256: {EXPECTED_AUDITED_CONTENT_HASH}\n"
            f"Actual semantic SHA-256:   {actual_content_hash}\n"
            "Do not overwrite the audited CSV; investigate the PDF/parser difference first."
        )

    df = finalize(pd.DataFrame(rows), 2025, keep_images=False, max_question=200)
    if semantic_hash(df) != EXPECTED_AUDITED_CONTENT_HASH:
        raise SystemExit("STOP: finalized CSV differs from the independently audited dataset.")
    if len(df) != 170:
        raise SystemExit(
            f"STOP: expected 170 retained questions after removing first 25 image questions "
            f"and six annulments (Q15 overlaps images). Got {len(df)}. "
            "Inspect report; dataset not cleared for evaluation."
        )
    if df.loc[df["question_number"] == 150, "answer"].tolist() != ["D"]:
        raise SystemExit("STOP: Q150 not D in output.")
    if set(df["question_number"]) != (set(range(1, 201)) - IMAGE_LINKED - ANNULLED):
        raise SystemExit("STOP: unexpected retained question numbers.")

    validation_dir = ROOT / "data" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    review_file = validation_dir / "mir_2025_review.csv"
    df[["question_number", "question", "option_a", "option_b",
        "option_c", "option_d", "answer"]].sort_values("question_number").to_csv(review_file, index=False)
    (validation_dir / "mir_2025_source_manifest.json").write_text(
        json.dumps({
            "exam_held": "2025-01-25",
            "convocatoria": 2024,
            "booklet": str(QUESTION_PDF),
            "definitive_key": str(ANSWER_PDF),
            "version": 0,
            "images_excluded": sorted(IMAGE_LINKED),
            "annulled_excluded": sorted(ANNULLED),
            "valid_main_questions": len(df),
            "question_text_options_manually_verified": False,
            "independent_machine_text_audit": True,
            "source_pdf_sha256": {path.name: sha256_file(path) for path in (QUESTION_PDF, ANSWER_PDF)},
            "audited_csv_semantic_sha256": EXPECTED_AUDITED_CONTENT_HASH,
            "generated_csv_sha256": sha256_file(ROOT / "data/raw/mir_2025_es.csv"),
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "pdfplumber_version": pdfplumber.__version__,
        }, indent=2), encoding="utf-8"
    )
    print("\nPASS: MIR 2025 structural checks, source PDF hashes, and all 170 audited records match.")
    print(f"REVIEW REQUIRED: compare every question/option against the booklet in {review_file}")
    print("Only after visual review should you mark this dataset as verified.")
