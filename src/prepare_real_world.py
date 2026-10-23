"""Prepare the real-world datasets: CUPCase, LiveMedBench (English) and K-QA.
Usage: python src/prepare_real_world.py [--datasets ...] [--livemedbench-dir <folder>]"""

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from config import PROJECT_ROOT

DATA = PROJECT_ROOT / "data"
LETTERS = "ABCD"
SEED = 0

CUPCASE_REPO = "ofir408/CupCase"
CUPCASE_INSTRUCTION = "Predict the diagnosis of this case presentation of a patient."  # as in the paper's code
LIVEMEDBENCH_REPO = "JuelieYann/LiveMedBench"
KQA_COMMIT = "ef403d012e199fcfa0d8c02b38a96ff95164f4d0"
KQA_URL = f"https://raw.githubusercontent.com/Itaymanes/K-QA/{KQA_COMMIT}/dataset/"


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_report(name, report):
    path = DATA / f"{name}.report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases_per_month"}, indent=2))
    print(f"Wrote {DATA / report['output']} and {path}")


# ---------------------------------------------------------------- CUPCase

def prepare_cupcase(local_file=None):
    """Multiple choice: the case presentation and four shuffled candidate diagnoses."""
    if local_file:
        frame = pd.read_parquet(local_file) if str(local_file).endswith(".parquet") else pd.read_csv(local_file)
    else:
        from datasets import load_dataset

        frame = load_dataset(CUPCASE_REPO, split="test").to_pandas()
    columns = ["clean_case_presentation", "correct_diagnosis", "distractor1", "distractor2", "distractor3"]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise SystemExit(f"CUPCase: missing columns {missing}. Found {list(frame.columns)}.")

    rng = random.Random(SEED)
    records, skipped = [], 0
    for index, row in frame.reset_index(drop=True).iterrows():
        case = str(row["clean_case_presentation"]).strip()
        correct = str(row["correct_diagnosis"]).strip()
        options = [correct] + [str(row[f"distractor{i}"]).strip() for i in (1, 2, 3)]
        # Cases with an empty or repeated option cannot be scored unambiguously.
        if not case or any(not o or o == "nan" for o in options) or len({o.lower() for o in options}) < 4:
            skipped += 1
            continue
        rng.shuffle(options)
        records.append({
            "id": f"cupcase-{index + 1:04d}",
            "year": "",
            "question": f"{CUPCASE_INSTRUCTION}\n\nCase presentation:\n{case}",
            "options": dict(zip(LETTERS, options)),
            "answer": LETTERS[options.index(correct)],
        })
    write_jsonl(records, DATA / "cupcase.jsonl")
    write_report("cupcase", {
        "output": "cupcase.jsonl", "source": local_file or f"huggingface.co/datasets/{CUPCASE_REPO} (test split)",
        "license": "Apache-2.0", "rows_in": len(frame), "skipped_empty_or_duplicate_options": skipped,
        "rows_out": len(records), "option_seed": SEED,
        "answer_distribution": dict(sorted(Counter(r["answer"] for r in records).items())),
    })


# ---------------------------------------------------------------- LiveMedBench

def has_chinese(text):
    """True if the text contains CJK ideographs (rule of the official LiveMedBench script)."""
    return any("一" <= ch <= "鿿" for ch in text)


def livemedbench_files(local_dir):
    """The monthly json files, from a local folder or downloaded from Hugging Face."""
    revision = None
    if not local_dir:
        from huggingface_hub import HfApi, snapshot_download

        revision = HfApi().dataset_info(LIVEMEDBENCH_REPO).sha
        local_dir = snapshot_download(LIVEMEDBENCH_REPO, repo_type="dataset", revision=revision,
                                      allow_patterns=["*.json"])
    files = sorted(Path(local_dir).glob("*.json"))
    if not files:
        raise SystemExit(f"LiveMedBench: no json files found in {local_dir}.")
    return files, revision


def prepare_livemedbench(local_dir=None):
    """Open-ended rubric task: case narrative, the patient's request, rubric items and date."""
    files, revision = livemedbench_files(local_dir)
    cases = {}
    for path in files:  # sorted by name, so later releases replace earlier versions of a case
        content = json.loads(path.read_text(encoding="utf-8"))
        for case in content if isinstance(content, list) else content.get("data", list(content.values())):
            cases[str(case["case_id"])] = case

    records, removed = [], Counter()
    for case_id, case in sorted(cases.items(), key=lambda kv: (str(kv[1].get("post_time")), kv[0])):
        narrative = str(case.get("narrative") or "").strip()
        request = str(case.get("core_request") or "").strip()
        rubric = [{"criterion": str(r["criterion"]).strip(), "points": float(r["points"])}
                  for r in case.get("rubric_items") or [] if str(r.get("criterion") or "").strip()]
        if has_chinese(narrative + request):
            removed["not_english"] += 1
        elif not narrative or not request:
            removed["empty_text"] += 1
        elif not any(r["points"] > 0 for r in rubric):
            removed["no_positive_rubric_points"] += 1
        else:
            records.append({"id": f"lmb-{case_id}", "date": str(case.get("post_time"))[:10],
                            "question": narrative, "request": request,
                            "reference": str(case.get("doctor_advice") or "").strip(), "rubric": rubric})
    write_jsonl(records, DATA / "livemedbench_en.jsonl")
    months = Counter(r["date"][:7] for r in records)
    write_report("livemedbench_en", {
        "output": "livemedbench_en.jsonl",
        "source": str(local_dir or f"huggingface.co/datasets/{LIVEMEDBENCH_REPO}"),
        "revision": revision, "files": [p.name for p in files], "unique_cases": len(cases),
        "removed": dict(removed), "rows_out": len(records),
        "date_range": [min(r["date"] for r in records), max(r["date"] for r in records)] if records else None,
        "mean_rubric_items": round(sum(len(r["rubric"]) for r in records) / max(len(records), 1), 2),
        "cases_per_month": dict(sorted(months.items())),
    })


# ---------------------------------------------------------------- K-QA

def kqa_file(name, local_dir):
    if local_dir:
        return Path(local_dir) / name
    target = DATA / "sources" / f"kqa_{name}"
    with urllib.request.urlopen(KQA_URL + name, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def prepare_kqa(local_dir=None):
    """Open-ended statements task: patient question with physician-written statements."""
    questions_path = kqa_file("questions_w_answers.jsonl", local_dir)
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [{"id": f"kqa-{index + 1:03d}", "question": str(row["Question"]).strip(),
                "reference": str(row.get("Free_form_answer") or "").strip(),
                "must_have": [s.strip() for s in row["Must_have"] if s.strip()],
                "nice_to_have": [s.strip() for s in row.get("Nice_to_have") or [] if s.strip()]}
               for index, row in enumerate(rows)]
    write_jsonl(records, DATA / "kqa.jsonl")

    validation = pd.read_csv(kqa_file("NLI_medical_annotator.csv", local_dir))
    validation = validation.rename(columns={"answer": "premise", "claim": "hypothesis"})
    columns = ["Question", "llm_x", "premise", "hypothesis", "majority_label", "label_0", "label_1", "label_2"]
    validation[columns].to_csv(DATA / "kqa_nli_validation.csv", index=False)
    write_report("kqa", {
        "output": "kqa.jsonl", "source": f"github.com/Itaymanes/K-QA at commit {KQA_COMMIT}", "license": "MIT",
        "rows_out": len(records), "must_have_statements": sum(len(r["must_have"]) for r in records),
        "nice_to_have_statements": sum(len(r["nice_to_have"]) for r in records),
        "validation_statements": len(validation),
        "validation_labels": validation["majority_label"].value_counts().to_dict(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=["cupcase", "livemedbench", "kqa"],
                        default=["cupcase", "livemedbench", "kqa"])
    parser.add_argument("--cupcase-file", default=None, help="Local CUPCase parquet or csv file.")
    parser.add_argument("--livemedbench-dir", default=None, help="Local folder with the LiveMedBench json files.")
    parser.add_argument("--kqa-dir", default=None, help="Local folder with the K-QA dataset files.")
    args = parser.parse_args()
    (DATA / "sources").mkdir(parents=True, exist_ok=True)
    if "cupcase" in args.datasets:
        print("\n--- CUPCase ---")
        prepare_cupcase(args.cupcase_file)
    if "livemedbench" in args.datasets:
        print("\n--- LiveMedBench (English) ---")
        prepare_livemedbench(args.livemedbench_dir)
    if "kqa" in args.datasets:
        print("\n--- K-QA ---")
        prepare_kqa(args.kqa_dir)


if __name__ == "__main__":
    main()
