"""Contamination already reported for our models or benchmarks in the CONDA database.

The Data Contamination Database of the CONDA workshop (Sainz et al., 2024) collects reported
cases of benchmark contamination: which evaluation dataset was found in which model or
pretraining corpus, how large a share of each split, by which approach, with a reference. It
is published as one file, contamination_report.csv, in the Hugging Face Space
CONDA-Workshop/Data-Contamination-Database.

This script downloads that file (or reads a local copy) and lists every report that concerns
one of our model families or one of the benchmarks our questions come from or overlap with.
No GPU is needed; it runs on a login node with internet access.

    model families   Gemma 3 and MedGemma, Llama 3 and Med42, Qwen3 (matched in the column
                     "Contaminated Source"; other versions of these families, such as Llama 2
                     or Qwen 1, do not match)
    benchmarks       MIR and HEAD-QA (which contains MIR questions), CareQA, MedExpQA, MedQA,
                     MedMCQA, LiveMedBench, K-QA, CUPCase (matched in "Evaluation Dataset")

Outputs, written to results/analysis/contamination/:
    conda_reports.csv     every matching report with the family and benchmark it matched
    conda_source.json     download time, Space revision and size of the database file

Usage:
    python src/conda_check.py
    python src/conda_check.py --file <local copy of contamination_report.csv>
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import resolve_path

SPACE = "CONDA-Workshop/Data-Contamination-Database"
FILENAME = "contamination_report.csv"

MODEL_FAMILIES = {
    "Gemma 3 / MedGemma": r"gemma[-_ ]?3(?!\d)|medgemma",
    "Llama 3 / Med42": r"llama[-_ ]?3(?!\d)|med42",
    "Qwen3": r"qwen[-_ ]?3(?!\d)",
}
BENCHMARKS = {
    "MIR / HEAD-QA": r"\bmir\b|head[-_ ]?qa",
    "CareQA": r"care[-_ ]?qa",
    "MedExpQA": r"medexpqa",
    "MedQA": r"\bmed[-_ ]?qa\b",
    "MedMCQA": r"medmcqa",
    "LiveMedBench": r"livemedbench",
    "K-QA": r"\bk[-_ ]?qa\b",
    "CUPCase": r"cupcase",
}


def database_file(local_file):
    """Path of the database file and a description of where it came from."""
    if local_file:
        return Path(local_file), {"source": "local file", "path": str(Path(local_file).resolve())}
    from huggingface_hub import HfApi, hf_hub_download

    path = Path(hf_hub_download(SPACE, FILENAME, repo_type="space"))
    return path, {"source": f"huggingface.co/spaces/{SPACE}", "revision": HfApi().space_info(SPACE).sha}


def matches(column, patterns):
    """Name of the first matching pattern for every value of a column, or an empty string."""
    found = pd.Series("", index=column.index)
    for name, pattern in reversed(list(patterns.items())):
        found[column.fillna("").str.contains(pattern, case=False, regex=True)] = name
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=None, help="Local copy of contamination_report.csv (default: download).")
    parser.add_argument("--out", default="results/analysis/contamination", help="Output directory.")
    args = parser.parse_args()

    path, source = database_file(args.file)
    reports = pd.read_csv(path, sep=";", dtype=str)
    missing = {"Evaluation Dataset", "Contaminated Source"} - set(reports.columns)
    if missing:
        raise SystemExit(f"Unexpected format of {FILENAME}: columns {sorted(missing)} not found.")
    reports.insert(0, "model_family", matches(reports["Contaminated Source"], MODEL_FAMILIES))
    reports.insert(1, "benchmark", matches(reports["Evaluation Dataset"], BENCHMARKS))
    found = reports[(reports.model_family != "") | (reports.benchmark != "")]

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found.to_csv(out / "conda_reports.csv", index=False)
    source.update(downloaded_utc=datetime.now(timezone.utc).isoformat(), n_reports=len(reports),
                  n_matching=len(found), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out / "conda_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    print(f"CONDA database: {len(reports)} reports ({source['source']})")
    for family in MODEL_FAMILIES:
        n = int((found.model_family == family).sum())
        print(f"  reports for {family}: {n}")
    for benchmark in BENCHMARKS:
        rows = found[found.benchmark == benchmark]
        if len(rows):
            print(f"  reports for {benchmark}: {len(rows)}")
            print(rows.drop(columns=["model_family", "benchmark"]).to_string(index=False))
    print(f"Wrote {out / 'conda_reports.csv'}")


if __name__ == "__main__":
    main()
