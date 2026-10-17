"""Year-difficulty adjustment for the MIR contamination check (local analysis).

Reads the existing model answers (results/raw/<model>/mir_<year>_es.jsonl) and writes:

  year_adjustment_out/accuracy_by_year.csv      accuracy per model and exam year, with seen/unseen status
  year_adjustment_out/model_based.csv           MAIN RESULT: contamination effect after removing exam
                                                difficulty, estimated from all models together
                                                (regression: correct ~ model + year + unseen)
  year_adjustment_out/year_difficulty.csv       exam difficulty per year: model-based estimate next to
                                                the human (top-10%) figures
  year_adjustment_out/human_based.csv           CHECK: shift after subtracting human accuracy per year
                                                (2023-2026 only)

"Unseen" depends on each model's training cutoff (UNSEEN_FROM below). Shifts are in
percentage points; negative = lower accuracy on unseen exams (what contamination predicts).

Usage (from anywhere):
    pip install pandas numpy statsmodels
    python year_adjustment.py <project folder> <mir_human_difficulty_2020_2026.csv>
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# First exam year after each model's training cutoff (the MIR is held in January).
UNSEEN_FROM = {
    "gemma3-1b": 2025, "gemma3-4b": 2025, "gemma3-27b": 2025,   # Gemma 3: cutoff August 2024
    "medgemma-4b": 2025, "medgemma-27b": 2025,                  # built on Gemma 3
    "llama3-8b": 2024, "med42-8b": 2024,                        # Llama 3 8B: cutoff March 2023
    "qwen3-1.7b": 2026, "qwen3-4b": 2026,                       # Qwen3: cutoff not published
}
NEAR_CHANCE = {"gemma3-1b"}  # excluded from the regression: near chance, carries no signal


def load(project):
    files = glob.glob(str(Path(project) / "results" / "raw" / "*" / "mir_20*_es.jsonl"))
    if not files:
        raise SystemExit(f"No results found under {project}/results/raw/. Is this the project folder?")
    d = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    d["year"] = d["dataset"].str[4:8].astype(int)
    d["correct"] = d["correct"].astype(int)
    unknown = sorted(set(d["model"]) - set(UNSEEN_FROM))
    if unknown:
        print(f"Skipping models without a cutoff in UNSEEN_FROM: {unknown}")
    d = d[d["model"].isin(UNSEEN_FROM)].copy()
    d["unseen"] = (d["year"] >= d["model"].map(UNSEEN_FROM)).astype(int)
    return d


def fit(data, term, extra=""):
    groups = pd.factorize(data["item_id"])[0]  # questions answered by several models are correlated
    model = smf.ols(f"correct ~ C(model) + C(year) + {term}{extra}", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": groups})
    low, high = model.conf_int().loc[term]
    return model, 100 * model.params[term], 100 * low, 100 * high


def model_based(d):
    data = d[~d["model"].isin(NEAR_CHANCE)].copy()
    pooled, b, lo, hi = fit(data, "unseen")
    rows = [{"model": "ALL (pooled)", "unseen_from": "", "shift": b, "ci_low": lo, "ci_high": hi}]
    for m in sorted(data["model"].unique()):
        data["unseen_m"] = data["unseen"] * (data["model"] == m)
        data["unseen_other"] = data["unseen"] * (data["model"] != m)
        _, b, lo, hi = fit(data, "unseen_m", " + unseen_other")
        rows.append({"model": m, "unseen_from": UNSEEN_FROM[m], "shift": b, "ci_low": lo, "ci_high": hi})
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


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    project, human_csv = sys.argv[1], sys.argv[2]
    out = Path("year_adjustment_out")
    out.mkdir(exist_ok=True)
    d = load(project)

    by_year = (d.groupby(["model", "year"])
                 .agg(n=("correct", "size"), accuracy=("correct", "mean"), unseen=("unseen", "max"))
                 .reset_index())
    by_year["accuracy"] = (100 * by_year["accuracy"]).round(1)
    by_year.to_csv(out / "accuracy_by_year.csv", index=False)

    main_result, year_effects = model_based(d)
    main_result.to_csv(out / "model_based.csv", index=False)

    human = pd.read_csv(human_csv)
    check, h = human_based(d, human)
    check.to_csv(out / "human_based.csv", index=False)

    base = min(year_effects) - 1  # reference year of the regression
    difficulty = pd.DataFrame({"year": sorted([base, *year_effects])})
    difficulty["model_based_vs_first_year"] = difficulty["year"].map(lambda y: year_effects.get(y, 0.0)).round(1)
    difficulty["human_top10_equivalent_accuracy"] = difficulty["year"].map(lambda y: round(100 * h[y], 1) if y in h else None)
    difficulty.to_csv(out / "year_difficulty.csv", index=False)

    pd.set_option("display.width", 160)
    print("MAIN: contamination effect after removing exam difficulty (points, 95% CI)")
    print(main_result.to_string(index=False))
    print("\nExam difficulty per year (model-based: points vs first year; human: top-10% equivalent accuracy)")
    print(difficulty.to_string(index=False))
    print("\nCHECK: human-adjusted shift, 2023-2026 only")
    print(check.to_string(index=False))
    print(f"\nWrote CSV files to {out.resolve()}")


if __name__ == "__main__":
    main()
