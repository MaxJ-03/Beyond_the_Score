"""Audited, idempotent, fail-closed corrections to upstream MIR question sources.

The Ministry Version-0 booklets are the authority. CareQA and the independent
MIR-2026 scraper occasionally break numbers/ranges across line boundaries.
Apply corrections in memory BEFORE deriving the reproducible evaluation CSVs.
Do not modify upstream downloaded source files or correct answer letters.
"""


def _fix_exact(value, before, after, context):
    if not isinstance(value, str):
        raise ValueError(f"{context}: non-string source content")
    if value == before or value == after:
        return after
    raise ValueError(f"{context}: unexpected source content: {value!r}")


def fix_2020_spanish(frame):
    """CareQA MIR-2020 question 104: restore aquaporin-2 in A, strip from B."""
    frame = frame.copy()
    numbers = frame["question_number"].astype(str).str.strip()
    matches = frame.index[numbers == "104"].tolist()
    if len(matches) != 1:
        raise ValueError(f"MIR 2020: expected one Q104, found {len(matches)}")
    ix = matches[0]
    frame.at[ix, "option_a"] = _fix_exact(
        frame.at[ix, "option_a"],
        "Está causada por mutaciones de acuaporina-",
        "Está causada por mutaciones de acuaporina-2.",
        "MIR 2020 Q104 option A (Spanish)",
    )
    frame.at[ix, "option_b"] = _fix_exact(
        frame.at[ix, "option_b"],
        "2. Es debida a una reducción marcada de los niveles de acuaporina-2.",
        "Es debida a una reducción marcada de los niveles de acuaporina-2.",
        "MIR 2020 Q104 option B (Spanish)",
    )
    return frame


def fix_2020_english_option(letter, value):
    """Correct corresponding CareQA English option for MIR-2020 Q104."""
    if letter == "a":
        return _fix_exact(
            value, "It is caused by mutations of aquaporin-",
            "It is caused by mutations of aquaporin-2.",
            "MIR 2020 Q104 option A (English)",
        )
    if letter == "b":
        return _fix_exact(
            value, "2. It is due to a marked reduction in the levels of aquaporin-2.",
            "It is due to a marked reduction in the levels of aquaporin-2.",
            "MIR 2020 Q104 option B (English)",
        )
    return value


# Audited MIR 2026 restoration rules: question number -> (column, before, after).
# Q153 is meaning-changing; remaining rules restore technical hyphens in booklet text.
SPANISH_2026 = {
    140: [
        ("question_es", "colesterolLDL", "colesterol-LDL"),
        ("option_2_es", "colesterolLDL", "colesterol-LDL"),
    ],
    153: [
        ("option_3_es", "o 25 años antes del caso más joven", "o 2-5 años antes del caso más joven"),
        ("correct_answer_text_es", "o 25 años antes del caso más joven", "o 2-5 años antes del caso más joven"),
    ],
    179: [
        ("option_1_es", "antiTNF", "anti-TNF"),
        ("option_3_es", "antiIL17", "anti-IL17"),
        ("option_4_es", "antiIL12/23", "anti-IL12/23"),
    ],
    183: [("question_es", "HLAB27", "HLA-B27")],
}

ENGLISH_2026 = {
    153: [
        ("option_3_en", "or 25 years before the youngest case", "or 2-5 years before the youngest case"),
        ("correct_answer_text_en", "or 25 years before the youngest case", "or 2-5 years before the youngest case"),
    ],
    179: [("option_4_en", "antiIL12/23", "anti-IL12/23")],
}


def _fix_2026(frame, rules, context):
    frame = frame.copy()
    if "question_id" not in frame.columns or frame["question_id"].duplicated().any():
        raise ValueError(f"{context}: missing/duplicated question_id")
    numbers = frame["question_id"].astype(str).str.strip()
    for n, fixes in rules.items():
        matches = frame.index[numbers == str(n)].tolist()
        if len(matches) != 1:
            raise ValueError(f"{context}: expected one Q{n}, found {len(matches)}")
        ix = matches[0]
        for column, before, after in fixes:
            if column not in frame.columns or not isinstance(frame.at[ix, column], str):
                raise ValueError(f"{context} Q{n}: missing/non-text field {column}")
            value = frame.at[ix, column]
            if before in value:
                if value.count(before) != 1:
                    raise ValueError(f"{context} Q{n} {column}: ambiguous substring")
                frame.at[ix, column] = value.replace(before, after, 1)
            elif after not in value:
                raise ValueError(f"{context} Q{n} {column}: unexpected source: {value!r}")
    return frame


def fix_2026_source(frame):
    """Fix original Spanish scraped source data in memory, before selection."""
    frame = _fix_2026(frame, SPANISH_2026, "MIR 2026 Spanish")
    q = frame.loc[frame["question_id"].astype(str) == "153"].iloc[0]
    if q["option_3_es"] != q["correct_answer_text_es"] or str(q["correct_answer"]).upper() not in ("3", "C"):
        raise ValueError("MIR 2026 Spanish Q153: corrected option/answer mismatch")
    return frame


def fix_2026_translation(frame):
    """Fix Spanish parallel text and English text in original translation in memory."""
    frame = _fix_2026(frame, SPANISH_2026, "MIR 2026 translation Spanish")
    frame = _fix_2026(frame, ENGLISH_2026, "MIR 2026 translation English")
    q = frame.loc[frame["question_id"].astype(str) == "153"].iloc[0]
    if q["option_3_es"] != q["correct_answer_text_es"] or q["option_3_en"] != q["correct_answer_text_en"]:
        raise ValueError("MIR 2026 translation Q153: corrected option/answer mismatch")
    return frame
