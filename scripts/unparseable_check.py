"""Unparseable-answer check for the MIR contamination result.

An answer is unparseable when no option letter can be extracted from the model's response;
such answers are scored as wrong. If unparseable answers are more frequent on some exam years,
they could create (or hide) an accuracy difference between seen and unseen editions. This
script measures how often they occur, why, and whether the year-adjusted contamination effect
changes when they are removed.

Writes to unparseable_check_out/:
  unparseable_by_year.csv   share of unparseable answers (%) per model and exam year
  unparseable_reasons.csv   count per model and reason (cut off at the token limit,
                            reasoning text without a final letter, no clear letter)
  unparseable_answers.csv   every unparseable answer, with the end of the response
  shift_with_without.csv    year-adjusted shift (unseen minus seen, pp, 95% CI) using all
                            answers, and using only parseable answers

Usage (from anywhere; needs pandas, numpy, statsmodels):
    python unparseable_check.py <project folder>
"""

import glob
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

# Same cutoff labels as year_adjustment.py: first exam year after each model's training cutoff.
UNSEEN_FROM = {
    "gemma3-1b": 2025, "gemma3-4b": 2025, "gemma3-27b": 2025, "medgemma-4b": 2025, "medgemma-27b": 2025,
    "llama3-8b": 2024, "med42-8b": 2024, "qwen3-1.7b": 2026, "qwen3-4b": 2026,
}
NEAR_CHANCE = {"gemma3-1b"}


def load(project):
    files = glob.glob(str(Path(project) / "results" / "raw" / "*" / "mir_20*_es.jsonl"))
    if not files:
        raise SystemExit(f"No results found under {project}/results/raw/. Is this the project folder?")
    d = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
    d = d[d["model"].isin(UNSEEN_FROM)].copy()
    d["year"] = d["dataset"].str[4:8].astype(int)
    d["correct"] = d["correct"].astype(int)
    d["unparseable"] = d["predicted"].isna()
    d["unseen"] = (d["year"] >= d["model"].map(UNSEEN_FROM)).astype(int)
    return d


def reason(row):
    text = str(row["response"])
    if row["finish_reason"] == "length":
        return "cut off at token limit"
    if "<unused" in text or "thought" in text[:40].lower():
        return "reasoning text, no final letter"
    return "no clear letter"


def shift(data, model):
    data = data[~data["model"].isin(NEAR_CHANCE)].copy()
    data["unseen_m"] = data["unseen"] * (data["model"] == model)
    data["unseen_other"] = data["unseen"] * (data["model"] != model)
    fit = smf.ols("correct ~ C(model) + C(year) + unseen_m + unseen_other", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["item_id"])[0]})
    low, high = fit.conf_int().loc["unseen_m"]
    return round(100 * fit.params["unseen_m"], 1), round(100 * low, 1), round(100 * high, 1)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    d = load(sys.argv[1])
    out = Path("unparseable_check_out")
    out.mkdir(exist_ok=True)

    by_year = (100 * d.pivot_table(index="model", columns="year", values="unparseable", aggfunc="mean")).round(1)
    by_year["all"] = (100 * d.groupby("model")["unparseable"].mean()).round(1)
    by_year.to_csv(out / "unparseable_by_year.csv")

    bad = d[d["unparseable"]].copy()
    bad["reason"] = bad.apply(reason, axis=1)
    reasons = bad.groupby(["model", "reason"]).size().unstack(fill_value=0)
    reasons.to_csv(out / "unparseable_reasons.csv")
    bad["response_end"] = bad["response"].astype(str).str[-200:].str.replace("\n", " ")
    bad[["model", "year", "item_id", "finish_reason", "reason", "response_end"]].to_csv(
        out / "unparseable_answers.csv", index=False)

    parseable = d[~d["unparseable"]]
    rows = []
    for model in sorted(set(d["model"]) - NEAR_CHANCE):
        a, b = shift(d, model), shift(parseable, model)
        rows.append({"model": model, "unparseable_pct": by_year.loc[model, "all"],
                     "shift_all_answers": a[0], "ci_all": f"[{a[1]:+.1f}, {a[2]:+.1f}]",
                     "shift_parseable_only": b[0], "ci_parseable": f"[{b[1]:+.1f}, {b[2]:+.1f}]",
                     "change": round(b[0] - a[0], 1)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "shift_with_without.csv", index=False)

    pd.set_option("display.width", 180)
    print("Unparseable answers per model and year (%):")
    print(by_year.to_string())
    print("\nReasons (counts):")
    print(reasons.to_string() if not reasons.empty else "  none")
    print("\nYear-adjusted shift, unseen minus seen (pp), with and without unparseable answers:")
    print(table.to_string(index=False))
    print(f"\nWrote CSV files to {out.resolve()}")


if __name__ == "__main__":
    main()
