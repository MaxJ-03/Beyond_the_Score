"""Positive control on existing results: MIR questions that MedGemma was trained on.

MedExpQA (HiTZ/MedExpQA) contains real MIR questions; the MedGemma technical report lists its
train split among MedGemma's training data. This script finds which of our MIR questions
(data/raw/mir_<year>_es.csv) also appear in MedExpQA's train split, and compares, on those
questions versus the other MIR questions of the same years:

    MedGemma's accuracy minus Gemma 3's accuracy (its base model), per question.

If memorization of these questions mattered, MedGemma's lead would be larger on the questions it
was trained on. Uses only the existing results in results/raw/; no GPU and no new model runs.

Writes to positive_control_out/:
  mir_overlap.csv          result per model size (percentage points, bootstrap 95% CI)
  matched_questions.csv    which MIR questions were matched to which MedExpQA question

Usage (from the project folder; needs pandas, numpy, huggingface_hub):
    python positive_control/mir_overlap_check.py
    python positive_control/mir_overlap_check.py --local-dir <folder with the MedExpQA jsonl files>
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("medgemma-4b", "gemma3-4b"), ("medgemma-27b", "gemma3-27b")]
FILES = {"train": "train.es.casimedicos.rag.jsonl", "dev": "dev.es.casimedicos.rag.jsonl",
         "test": "test.es.casimedicos.rag.jsonl"}
SAMPLES = 4000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="Project folder (default: current folder).")
    parser.add_argument("--local-dir", default=None,
                        help="Folder with the three MedExpQA jsonl files (default: download from Hugging Face).")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Word-set similarity above which two questions count as the same question.")
    parser.add_argument("--out", default="positive_control_out")
    return parser.parse_args()


def medexpqa_files(local_dir):
    if local_dir:
        return {split: Path(local_dir) / name for split, name in FILES.items()}
    from huggingface_hub import hf_hub_download

    return {split: Path(hf_hub_download("HiTZ/MedExpQA", f"data/es/{name}", repo_type="dataset"))
            for split, name in FILES.items()}


def words(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", text))


def match(project, files, threshold):
    """Match every MIR question against all MedExpQA questions by word-set (Jaccard) similarity."""
    expqa = []
    for split, path in files.items():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    expqa.append((split, record.get("id"), record.get("year"), words(record["full_question"])))
    rows = []
    for path in sorted((project / "data" / "raw").glob("mir_20*_es.csv")):
        for _, row in pd.read_csv(path).iterrows():
            question = words(row["question"])
            best, best_score = None, 0.0
            for entry in expqa:
                union = len(question | entry[3])
                score = len(question & entry[3]) / union if union else 0.0
                if score > best_score:
                    best, best_score = entry, score
            if best is not None and best_score >= threshold:
                rows.append({"mir_item_id": row["id"], "mir_year": int(row["year"]), "medexpqa_split": best[0],
                             "medexpqa_id": best[1], "medexpqa_year": best[2], "similarity": round(best_score, 3)})
    return pd.DataFrame(rows, columns=["mir_item_id", "mir_year", "medexpqa_split", "medexpqa_id",
                                       "medexpqa_year", "similarity"])


def load_results(project, years):
    frames = [pd.read_json(p, lines=True, dtype={"item_id": str})
              for y in years for p in (project / "results" / "raw").glob(f"*/mir_{y}_es.jsonl")]
    if not frames:
        raise SystemExit(f"No MIR results for {years} under {project / 'results' / 'raw'}.")
    return pd.concat(frames, ignore_index=True)


def gain(results, tuned, base, items):
    """Per question: fine-tuned correct minus base correct (questions answered by both)."""
    a = results[(results.model == tuned) & results.item_id.isin(items)].set_index("item_id")["correct"].astype(float)
    b = results[(results.model == base) & results.item_id.isin(items)].set_index("item_id")["correct"].astype(float)
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).to_numpy()


def main():
    args = parse_args()
    project = Path(args.project).resolve()
    if not (project / "data" / "raw").exists():
        raise SystemExit(f"{project} does not look like the project folder (data/raw not found).")
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    matched = match(project, medexpqa_files(args.local_dir), args.threshold)
    matched.to_csv(out / "matched_questions.csv", index=False)
    print(f"MIR questions found in MedExpQA: {len(matched)}")
    if matched.empty:
        raise SystemExit("No overlap found; nothing to compare.")
    print(matched.groupby(["mir_year", "medexpqa_split"]).size().unstack(fill_value=0).to_string())

    trained = matched[matched.medexpqa_split == "train"]
    years = sorted(trained.mir_year.unique())
    results = load_results(project, years)
    in_train = set(trained.mir_item_id)
    other = set(results.item_id) - set(matched.mir_item_id)  # same years, not in MedExpQA at all

    rng = np.random.default_rng(0)
    rows = []
    for tuned, base in PAIRS:
        g_in, g_out = gain(results, tuned, base, in_train), gain(results, tuned, base, other)
        if len(g_in) == 0 or len(g_out) == 0:
            print(f"Skipped {tuned} vs {base}: results missing.")
            continue
        diffs = rng.choice(g_in, (SAMPLES, len(g_in))).mean(1) - rng.choice(g_out, (SAMPLES, len(g_out))).mean(1)
        low, high = np.percentile(diffs, [2.5, 97.5])
        rows.append({"comparison": f"{tuned} minus {base}", "years": " ".join(map(str, years)),
                     "gain_on_mir_in_medexpqa_train": round(100 * g_in.mean(), 1),
                     "gain_on_other_mir": round(100 * g_out.mean(), 1),
                     "extra_gain": round(100 * (g_in.mean() - g_out.mean()), 1),
                     "ci_low": round(100 * low, 1), "ci_high": round(100 * high, 1),
                     "n_in_train": len(g_in), "n_other": len(g_out)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "mir_overlap.csv", index=False)

    pd.set_option("display.width", 180)
    print("\nMedGemma minus Gemma 3 on MIR questions MedGemma was trained on, minus the same on the")
    print("other MIR questions of those years (percentage points):")
    print(result.to_string(index=False))
    print("\nextra_gain clearly above zero = MedGemma does better on exactly its training questions (memorization);")
    print("about zero = no sign that training on these questions helped.")
    print(f"\nWrote {out / 'mir_overlap.csv'} and {out / 'matched_questions.csv'}")


if __name__ == "__main__":
    main()
