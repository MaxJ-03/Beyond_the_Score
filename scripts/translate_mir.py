"""Translate MIR exam questions from Spanish to English with a fixed, documented setup.

Each question is translated together with its answer options as a single unit, so that
terms shared between the stem and the options are translated consistently. The output
keeps the original Spanish text next to the translation and writes a metadata file that
records the exact configuration and date of the run. All exam years must be translated
with the same backend and settings.

Backends:
    google-cloud     Google Cloud Translation API v3 (requires a Google Cloud project).
                     Supports an explicit model choice (general/nmt or general/translation-llm).
    deep-translator  The free Google Translate web service via the deep-translator package.
                     No account needed, but the underlying model cannot be pinned, so all
                     years should be translated in one session.
    mock             Returns the Spanish text unchanged; for testing the workflow only.

Expected input: a CSV with one row per question and the columns
    id, year, question, option_a, option_b, option_c, option_d, answer
where `answer` is the letter of the correct option (A to D). Column names can be
remapped with --column-map.

Examples:
    python scripts/translate_mir.py --input data/raw/mir_2024_es.csv \
        --output data/mir_2024_en.csv --backend google-cloud --project-id my-project
    python scripts/translate_mir.py --input data/raw/mir_2024_es.csv \
        --output data/mir_2024_en.csv --backend deep-translator
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OPTION_LETTERS = ["A", "B", "C", "D"]
OPTION_COLUMNS = [f"option_{letter.lower()}" for letter in OPTION_LETTERS]
REQUIRED_COLUMNS = ["id", "year", "question", *OPTION_COLUMNS, "answer"]
MARKER = "[[{}]]"
MARKER_PATTERN = re.compile(r"\[\[\s*([A-D])\s*\]\]")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Spanish source CSV.")
    parser.add_argument("--output", required=True, help="Translated output CSV.")
    parser.add_argument("--backend", choices=["google-cloud", "deep-translator", "mock"], default="google-cloud")
    parser.add_argument("--project-id", default=None, help="Google Cloud project ID (google-cloud backend).")
    parser.add_argument("--location", default="global", help="Cloud Translation location (google-cloud backend).")
    parser.add_argument("--model", default="general/nmt",
                        help="Cloud Translation model, e.g. general/nmt or general/translation-llm.")
    parser.add_argument("--column-map", default=None,
                        help='JSON mapping from source column names to required names, e.g. \'{"pregunta": "question"}\'.')
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--pause", type=float, default=0.2,
                        help="Seconds between requests (deep-translator backend), to avoid rate limits.")
    return parser.parse_args()


class GoogleCloudTranslator:
    def __init__(self, project_id, location, model):
        if not project_id:
            raise SystemExit("--project-id is required for the google-cloud backend.")
        from google.cloud import translate_v3 as translate

        self.client = translate.TranslationServiceClient()
        self.parent = f"projects/{project_id}/locations/{location}"
        self.model_path = f"{self.parent}/models/{model}"
        self.description = {"service": "Google Cloud Translation API v3", "model": model, "location": location}

    def translate(self, texts):
        response = self.client.translate_text(request={
            "parent": self.parent,
            "contents": texts,
            "mime_type": "text/plain",
            "source_language_code": "es",
            "target_language_code": "en",
            "model": self.model_path,
        })
        return [t.translated_text for t in response.translations]


class DeepTranslatorBackend:
    def __init__(self, pause):
        from deep_translator import GoogleTranslator
        import deep_translator

        self.translator = GoogleTranslator(source="es", target="en")
        self.pause = pause
        self.description = {
            "service": "Google Translate web service via deep-translator",
            "deep_translator_version": getattr(deep_translator, "__version__", "unknown"),
            "model": "not selectable",
        }

    def translate(self, texts):
        results = []
        for text in texts:
            results.append(self.translator.translate(text))
            time.sleep(self.pause)
        return results


class MockTranslator:
    description = {"service": "mock (text returned unchanged)", "model": "none"}

    @staticmethod
    def translate(texts):
        return list(texts)


def with_retries(translator, texts, max_retries):
    for attempt in range(max_retries):
        try:
            return translator.translate(texts)
        except Exception as error:  # Network and quota errors are retried with backoff.
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"Request failed ({error}); retrying in {wait} s.")
            time.sleep(wait)


def load_source(path, column_map):
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns=json.loads(column_map))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()
    invalid = df.loc[~df["answer"].isin(OPTION_LETTERS), "id"].tolist()
    if invalid:
        raise ValueError(f"Rows with an answer outside A to D: {invalid}")
    return df


def build_unit(row):
    """Combine the question stem and options into one text with option markers."""
    parts = [str(row["question"]).strip()]
    for letter, column in zip(OPTION_LETTERS, OPTION_COLUMNS):
        parts.append(f"{MARKER.format(letter)} {str(row[column]).strip()}")
    return "\n".join(parts)


def split_unit(text):
    """Split a translated unit back into stem and options; return None if markers were lost."""
    pieces = MARKER_PATTERN.split(text)
    stem, rest = pieces[0], pieces[1:]
    if rest[0::2] != OPTION_LETTERS:
        return None
    return stem.strip(), [segment.strip() for segment in rest[1::2]]


def main():
    args = parse_args()
    df = load_source(args.input, args.column_map)

    if args.backend == "google-cloud":
        translator = GoogleCloudTranslator(args.project_id, args.location, args.model)
    elif args.backend == "deep-translator":
        translator = DeepTranslatorBackend(args.pause)
    else:
        translator = MockTranslator()

    records, fallback_ids = [], []
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        split = split_unit(with_retries(translator, [build_unit(row)], args.max_retries)[0])
        if split is None:
            # Markers were altered by the translation; translate each field separately instead.
            fields = [str(row["question"])] + [str(row[c]) for c in OPTION_COLUMNS]
            translated = with_retries(translator, fields, args.max_retries)
            stem, options = translated[0], translated[1:]
            fallback_ids.append(row["id"])
            mode = "per_field"
        else:
            stem, options = split
            mode = "unit"

        record = {"id": row["id"], "year": row["year"], "answer": row["answer"], "translation_mode": mode,
                  "question_es": row["question"], "question_en": stem}
        for column, option in zip(OPTION_COLUMNS, options):
            record[f"{column}_es"] = row[column]
            record[f"{column}_en"] = option
        records.append(record)
        if position % 25 == 0:
            print(f"  translated {position}/{len(df)} questions")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(records)
    out.to_csv(out_path, index=False)

    metadata = {
        "input": str(Path(args.input).resolve()),
        "output": str(out_path.resolve()),
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        **translator.description,
        "source_language": "es",
        "target_language": "en",
        "translation_unit": "question and options translated together, split on option markers",
        "rows_in": int(len(df)),
        "rows_out": int(len(out)),
        "per_field_fallback_ids": [str(i) for i in fallback_ids],
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Translated {len(out)} rows ({len(fallback_ids)} via per-field fallback). Metadata: {meta_path}")


if __name__ == "__main__":
    main()
