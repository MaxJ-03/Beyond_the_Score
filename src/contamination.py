"""Memorization checks on MIR and LiveMedBench results: year adjustment, unparseable answers, MedExpQA overlap.
Usage: python src/contamination.py [--checks ...] [--unseen-from model=year]"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from config import PROJECT_ROOT, load_datasets_config, load_models, resolve_path

MEDEXPQA_FILES = {"train": "train.es.casimedicos.rag.jsonl", "dev": "dev.es.casimedicos.rag.jsonl",
                  "test": "test.es.casimedicos.rag.jsonl"}
MEDICAL_PAIRS = [("medgemma-4b", "gemma3-4b"), ("medgemma-27b", "gemma3-27b")]  # fine-tuned, base
OVERLAP_SAMPLES = 4000


# ---------------------------------------------------------------- shared

def load_answers(results_dir, unseen_from):
    """All Spanish MIR answers of models with a known cutoff, with year and unseen status."""
    files = sorted((results_dir / "raw").glob("*/mir_20*_es.jsonl"))
    if not files:
        raise SystemExit(f"No MIR results found under {results_dir / 'raw'}.")
    d = pd.concat([pd.read_json(f, lines=True, dtype={"item_id": str}) for f in files], ignore_index=True)
    unknown = sorted(set(d["model"]) - set(unseen_from))
    if unknown:
        print(f"Skipping models without unseen_from in configs/models.yaml: {unknown}")
    d = d[d["model"].isin(unseen_from)].copy()
    d["year"] = d["dataset"].str[4:8].astype(int)
    d["correct"] = d["correct"].astype(int)
    d["unparseable"] = d["predicted"].isna()
    d["unseen"] = (d["year"] >= d["model"].map(unseen_from)).astype(int)
    return d


def fit_unseen_effect(data, model=None, outcome="correct", fixed="year"):
    """Regress the outcome on model, year and unseen (pooled, or per model); SEs clustered by question.
    Returns the fit and the effect with its 95% interval, in percentage points."""
    data = data.copy()
    if model is None:
        term, extra = "unseen", ""
    else:
        data["unseen_m"] = data["unseen"] * (data["model"] == model)
        data["unseen_other"] = data["unseen"] * (data["model"] != model)
        term, extra = "unseen_m", " + unseen_other"
    groups = pd.factorize(data["item_id"])[0]
    if fixed == "item_id":
        # Removing each case's mean gives the same estimate as thousands of case columns, much faster.
        x = pd.get_dummies(data["model"], drop_first=True, dtype=float)
        x = x.join(data[[term] + extra.split(" + ")[1:]].astype(float))
        x = x - x.groupby(data[fixed]).transform("mean")
        y = data[outcome] - data.groupby(fixed)[outcome].transform("mean")
        fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    else:
        fit = smf.ols(f"{outcome} ~ C(model) + C({fixed}) + {term}{extra}", data=data).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
    low, high = fit.conf_int().loc[term]
    return fit, 100 * fit.params[term], 100 * low, 100 * high


# ---------------------------------------------------------------- check 1: year adjustment

def model_based(d, excluded, unseen_from):
    data = d[~d["model"].isin(excluded)]
    pooled, b, lo, hi = fit_unseen_effect(data)
    rows = [{"model": "ALL (pooled)", "unseen_from": "", "shift": b, "ci_low": lo, "ci_high": hi}]
    for m in sorted(data["model"].unique()):
        _, b, lo, hi = fit_unseen_effect(data, m)
        rows.append({"model": m, "unseen_from": unseen_from[m], "shift": b, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out["significant_drop"] = out["ci_high"] < 0
    out["largest_drop_ruled_out"] = (-out["ci_low"]).clip(lower=0)  # drops beyond this are excluded
    years = {int(k.split("T.")[1].rstrip("]")): 100 * v for k, v in pooled.params.items() if k.startswith("C(year)")}
    return out.round(1), years


def human_based(d, human):
    h = human.set_index("year")["top_equivalent_accuracy"].dropna().to_dict()
    data = d[d["year"].isin(h)].copy()
    data["adjusted"] = data["correct"] - data["year"].map(h)
    rng = np.random.default_rng(0)
    rows = []
    for m, g in data.groupby("model"):
        seen, new = g[g["unseen"] == 0], g[g["unseen"] == 1]
        if seen.empty or new.empty:
            continue
        a, b = new["adjusted"].to_numpy(), seen["adjusted"].to_numpy()
        diffs = rng.choice(a, (4000, len(a))).mean(1) - rng.choice(b, (4000, len(b))).mean(1)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"model": m, "seen_years": f"{seen.year.min()}-{seen.year.max()}",
                     "unseen_years": f"{new.year.min()}-{new.year.max()}",
                     "raw_shift": 100 * (new.correct.mean() - seen.correct.mean()),
                     "adjusted_shift": 100 * (a.mean() - b.mean()), "ci_low": 100 * lo, "ci_high": 100 * hi})
    return pd.DataFrame(rows).round(1), h


def year_check(d, excluded, unseen_from, human_path, out):
    by_year = (d.groupby(["model", "year"])
                 .agg(n=("correct", "size"), accuracy=("correct", "mean"), unseen=("unseen", "max"))
                 .reset_index())
    by_year["accuracy"] = (100 * by_year["accuracy"]).round(1)
    by_year.to_csv(out / "accuracy_by_year.csv", index=False)

    main_result, year_effects = model_based(d, excluded, unseen_from)
    main_result.to_csv(out / "model_based.csv", index=False)
    print("MAIN: contamination effect after removing exam difficulty (points, 95% CI)")
    print(main_result.to_string(index=False))

    h = {}
    if human_path.exists():
        check, h = human_based(d, pd.read_csv(human_path))
        check.to_csv(out / "human_based.csv", index=False)
        print("\nCHECK: human-adjusted shift (years with a human reference only)")
        print(check.to_string(index=False))
    else:
        print(f"\nHuman reference file not found at {human_path}; human_based.csv not written.")

    base = min(year_effects) - 1  # reference year of the regression
    difficulty = pd.DataFrame({"year": sorted([base, *year_effects])})
    difficulty["model_based_vs_first_year"] = difficulty["year"].map(lambda y: year_effects.get(y, 0.0)).round(1)
    difficulty["human_top10_equivalent_accuracy"] = difficulty["year"].map(
        lambda y: round(100 * h[y], 1) if y in h else None)
    difficulty.to_csv(out / "year_difficulty.csv", index=False)
    print("\nExam difficulty per year (model-based: points vs first year; human: top-10% equivalent accuracy)")
    print(difficulty.to_string(index=False))


# ---------------------------------------------------------------- check 2: unparseable answers

def unparseable_reason(row):
    text = str(row["response"])
    if row["finish_reason"] == "length":
        return "cut off at token limit"
    if "<unused" in text or "thought" in text[:40].lower():
        return "reasoning text, no final letter"
    return "no clear letter"


def rounded_shift(data, model):
    _, b, lo, hi = fit_unseen_effect(data, model)
    return round(b, 1), round(lo, 1), round(hi, 1)


def unparseable_check(d, excluded, out):
    by_year = (100 * d.pivot_table(index="model", columns="year", values="unparseable", aggfunc="mean")).round(1)
    by_year["all"] = (100 * d.groupby("model")["unparseable"].mean()).round(1)
    by_year.to_csv(out / "unparseable_by_year.csv")

    bad = d[d["unparseable"]].copy()
    bad["reason"] = bad.apply(unparseable_reason, axis=1)
    reasons = bad.groupby(["model", "reason"]).size().unstack(fill_value=0)
    reasons.to_csv(out / "unparseable_reasons.csv")
    bad["response_end"] = bad["response"].astype(str).str[-200:].str.replace("\n", " ")
    bad.sort_values(["model", "year", "item_id"])[
        ["model", "year", "item_id", "finish_reason", "reason", "response_end"]].to_csv(
        out / "unparseable_answers.csv", index=False)

    data = d[~d["model"].isin(excluded)]
    parseable = data[~data["unparseable"]]
    rows = []
    for model in sorted(data["model"].unique()):
        a, b = rounded_shift(data, model), rounded_shift(parseable, model)
        rows.append({"model": model, "unparseable_pct": by_year.loc[model, "all"],
                     "shift_all_answers": a[0], "ci_all": f"[{a[1]:+.1f}, {a[2]:+.1f}]",
                     "shift_parseable_only": b[0], "ci_parseable": f"[{b[1]:+.1f}, {b[2]:+.1f}]",
                     "change": round(b[0] - a[0], 1)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "shift_with_without.csv", index=False)

    print("Unparseable answers per model and year (%):")
    print(by_year.to_string())
    print("\nReasons (counts):")
    print(reasons.to_string() if not reasons.empty else "  none")
    print("\nYear-adjusted shift, unseen minus seen (pp), with and without unparseable answers:")
    print(table.to_string(index=False))


# ---------------------------------------------------------------- check 3: MedExpQA positive control

def medexpqa_files(local_dir):
    if local_dir:
        return {split: Path(local_dir) / name for split, name in MEDEXPQA_FILES.items()}
    from huggingface_hub import hf_hub_download

    return {split: Path(hf_hub_download("HiTZ/MedExpQA", f"data/es/{name}", repo_type="dataset"))
            for split, name in MEDEXPQA_FILES.items()}


def words(text):
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", text))


def match_medexpqa(files, threshold):
    """Match every MIR question against all MedExpQA questions by word-set (Jaccard) similarity."""
    expqa = []
    for split, path in files.items():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    expqa.append((split, record.get("id"), record.get("year"), words(record["full_question"])))
    rows = []
    for path in sorted((PROJECT_ROOT / "data" / "raw").glob("mir_20*_es.csv")):
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


def gain(results, tuned, base, items):
    """Per question: fine-tuned correct minus base correct (questions answered by both)."""
    a = results[(results.model == tuned) & results.item_id.isin(items)].set_index("item_id")["correct"].astype(float)
    b = results[(results.model == base) & results.item_id.isin(items)].set_index("item_id")["correct"].astype(float)
    common = a.index.intersection(b.index)
    return (a.loc[common] - b.loc[common]).to_numpy()


def overlap_check(d, medexpqa_dir, threshold, out):
    matched = match_medexpqa(medexpqa_files(medexpqa_dir), threshold)
    matched.to_csv(out / "matched_questions.csv", index=False)
    print(f"MIR questions found in MedExpQA: {len(matched)}")
    if matched.empty:
        print("No overlap found; nothing to compare.")
        return
    print(matched.groupby(["mir_year", "medexpqa_split"]).size().unstack(fill_value=0).to_string())

    trained = matched[matched.medexpqa_split == "train"]
    years = sorted(trained.mir_year.unique())
    results = d[d["year"].isin(years)]
    in_train = set(trained.mir_item_id)
    other = set(results.item_id) - set(matched.mir_item_id)  # same years, not in MedExpQA at all

    rng = np.random.default_rng(0)
    rows = []
    for tuned, base in MEDICAL_PAIRS:
        g_in, g_out = gain(results, tuned, base, in_train), gain(results, tuned, base, other)
        if len(g_in) == 0 or len(g_out) == 0:
            print(f"Skipped {tuned} vs {base}: results missing.")
            continue
        diffs = (rng.choice(g_in, (OVERLAP_SAMPLES, len(g_in))).mean(1)
                 - rng.choice(g_out, (OVERLAP_SAMPLES, len(g_out))).mean(1))
        low, high = np.percentile(diffs, [2.5, 97.5])
        rows.append({"comparison": f"{tuned} minus {base}", "years": " ".join(map(str, years)),
                     "gain_on_mir_in_medexpqa_train": round(100 * g_in.mean(), 1),
                     "gain_on_other_mir": round(100 * g_out.mean(), 1),
                     "extra_gain": round(100 * (g_in.mean() - g_out.mean()), 1),
                     "ci_low": round(100 * low, 1), "ci_high": round(100 * high, 1),
                     "n_in_train": len(g_in), "n_other": len(g_out)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "mir_overlap.csv", index=False)
    print("\nMedGemma minus Gemma 3 on MIR questions MedGemma was trained on, minus the same on the")
    print("other MIR questions of those years (percentage points):")
    print(result.to_string(index=False))
    print("extra_gain clearly above zero means a larger lead on exactly the training questions "
          "(memorization); about zero means no sign that training on them helped.")


# ---------------------------------------------------------------- check 4: LiveMedBench posting dates

def livemedbench_check(results_dir, cutoffs, out):
    files = sorted((results_dir / "judged").glob("*/livemedbench.jsonl"))
    if not files:
        print("No graded LiveMedBench answers yet (run src/judge.py); check skipped.")
        return
    d = pd.concat([pd.read_json(f, lines=True, dtype={"item_id": str}) for f in files], ignore_index=True)
    dates = pd.read_json(PROJECT_ROOT / "data" / "livemedbench_en.jsonl", lines=True, dtype={"id": str},
                         convert_dates=False)
    d["month"] = d["item_id"].map(dates.set_index("id")["date"]).str[:7].replace("", np.nan)
    d = d[d["model"].isin(cutoffs)].dropna(subset=["month"]).copy()
    d["unseen"] = (d["month"] > d["model"].map(cutoffs)).astype(int)
    counts = d.groupby("model")["unseen"].agg(n_unseen="sum", n_cases="size")
    _, b, lo, hi = fit_unseen_effect(d, outcome="score", fixed="item_id")
    rows = [{"model": "ALL (pooled)", "training_cutoff": "", "shift": b, "ci_low": lo, "ci_high": hi}]
    for m in sorted(d["model"].unique()):
        if counts.loc[m, "n_unseen"] in (0, counts.loc[m, "n_cases"]):
            rows.append({"model": m, "training_cutoff": cutoffs[m], "note": "all cases on one side of the cutoff"})
            continue
        _, b, lo, hi = fit_unseen_effect(d, m, outcome="score", fixed="item_id")
        rows.append({"model": m, "training_cutoff": cutoffs[m], "shift": b, "ci_low": lo, "ci_high": hi})
    table = pd.DataFrame(rows).merge(counts.reset_index(), on="model", how="left").round(1)
    table.to_csv(out / "livemedbench_cutoff.csv", index=False)
    print("Unseen effect on LiveMedBench rubric scores (points on a 0-100 scale, 95% CI), all dated cases:")
    print(table.to_string(index=False))


# ---------------------------------------------------------------- main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checks", nargs="+", choices=["year", "unparseable", "overlap", "livemedbench"],
                        default=["year", "unparseable", "overlap", "livemedbench"])
    parser.add_argument("--results", default="results", help="Results directory used by run_eval.")
    parser.add_argument("--out", default="results/analysis/contamination", help="Output directory.")
    parser.add_argument("--unseen-from", nargs="+", default=[], metavar="MODEL=YEAR",
                        help="Override unseen_from for a sensitivity check, e.g. qwen3-4b=2025.")
    parser.add_argument("--medexpqa-dir", default=None,
                        help="Folder with the three MedExpQA jsonl files (default: download from Hugging Face).")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Word-set similarity above which two questions count as the same question.")
    return parser.parse_args()


def main():
    args = parse_args()
    analysis = load_datasets_config()["analysis"]
    unseen_from = {k: m["unseen_from"] for k, m in load_models().items() if m.get("unseen_from")}
    for override in args.unseen_from:
        model, year = override.split("=")
        if model not in unseen_from:
            raise SystemExit(f"Unknown model in --unseen-from: {model}")
        unseen_from[model] = int(year)
    excluded = set(analysis.get("exclude_from_regression", []))

    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_answers(resolve_path(args.results), unseen_from)
    pd.set_option("display.width", 180)
    if "year" in args.checks:
        print("=== Check 1: year-difficulty adjustment ===")
        year_check(d, excluded, unseen_from, resolve_path(analysis["human_difficulty"]), out)
    if "unparseable" in args.checks:
        print("\n=== Check 2: unparseable answers ===")
        unparseable_check(d, excluded, out)
    if "overlap" in args.checks:
        print("\n=== Check 3: MedExpQA positive control ===")
        overlap_check(d, args.medexpqa_dir, args.threshold, out)
    if "livemedbench" in args.checks:
        print("\n=== Check 4: LiveMedBench posting dates ===")
        cutoffs = {k: m["training_cutoff"] for k, m in load_models().items() if m.get("training_cutoff")}
        livemedbench_check(resolve_path(args.results), cutoffs, out)
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
