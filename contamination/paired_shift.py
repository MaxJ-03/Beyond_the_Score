"""Paired base-medical comparison: does a medical model's lead over its own base model shrink on
unseen MIR editions?

Both models of a pair answer the same questions. Per question, lead = medical model correct (0/1)
minus base model correct (0/1); lead shift = mean lead on MIR 2025-2026 minus mean lead on
2020-2024 (the common split of stage3). Edition difficulty and the shared pretraining affect both
models equally and cancel, so no year-difficulty model is needed. Uncertainty: questions resampled
within editions (4,000 times), and a permutation over the 21 ways to label two of the seven
editions as unseen (smallest attainable p = 1/21 = 0.048). Also: the shift with each edition left
out, and with 2026 as the only unseen edition.

Run on the team's MIR results (stage3 checkout, read only):
    python contamination/paired_shift.py --project <stage3 folder>
Outputs (results/analysis/paired_shift/): lead_by_year.csv, lead_shift.csv, lead_shift_loo.csv
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("medgemma-4b", "gemma3-4b"), ("medgemma-27b", "gemma3-27b"), ("med42-8b", "llama3-8b")]

# "main": the stage3 common split; "2026_only": the strict version (2026 was held after the
# release of every model, so it cannot be in any fine-tuning data).
SPLITS = {"main": [2025, 2026], "2026_only": [2026]}
SAMPLES = 4000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=".", help="Project folder (default: current folder).")
    parser.add_argument("--out", default="results/analysis/paired_shift")
    parser.add_argument("--exclude-unparseable", action="store_true",
                        help="Drop questions where either model of the pair gave no parseable answer.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load(project):
    files = sorted((Path(project) / "results" / "raw").glob("*/mir_20*_es.jsonl"))
    if not files:
        raise SystemExit(f"No results found under {project}/results/raw/. Is this the project folder?")
    d = pd.concat([pd.read_json(f, lines=True, dtype={"item_id": str}) for f in files], ignore_index=True)
    d["year"] = d["dataset"].str[4:8].astype(int)
    d["correct"] = d["correct"].astype(int)
    d["parsed"] = d["predicted"].notna()
    return d


def paired(d, tuned, base, exclude_unparseable):
    cols = ["item_id", "year", "correct", "parsed"]
    a = d[d.model == tuned][cols].set_index("item_id")
    b = d[d.model == base][cols].set_index("item_id")
    if a.empty or b.empty:
        return None
    p = a.join(b[["correct", "parsed"]], rsuffix="_base", how="inner")
    if len(p) != len(a) or len(p) != len(b):
        print(f"  note: {tuned} vs {base}: {len(a)} and {len(b)} items, {len(p)} in common")
    if exclude_unparseable:
        p = p[p.parsed & p.parsed_base]
    p["lead"] = p["correct"] - p["correct_base"]
    return p.reset_index()


def shift(p, unseen):
    new = p.year.isin(unseen)
    return p.loc[new, "lead"].mean() - p.loc[~new, "lead"].mean()


def item_bootstrap(p, unseen, rng):
    """Resample questions within each edition; returns SAMPLES lead shifts."""
    by_year = {y: g["lead"].to_numpy() for y, g in p.groupby("year")}
    new_years = [y for y in by_year if y in unseen]
    old_years = [y for y in by_year if y not in unseen]

    def period(years):
        total = np.zeros(SAMPLES)
        n = 0
        for y in years:
            x = by_year[y]
            total += rng.choice(x, (SAMPLES, len(x))).sum(1)
            n += len(x)
        return total / n

    return period(new_years) - period(old_years)


def year_permutation(p, unseen):
    """Rank of the observed shift among all choices of len(unseen) editions as 'unseen'."""
    years = sorted(p.year.unique())
    observed = shift(p, unseen)
    alternatives = [shift(p, list(c)) for c in itertools.combinations(years, len(unseen))]
    # One-sided: how many choices give a shift at least as negative as the observed one.
    p_low = np.mean([s <= observed + 1e-12 for s in alternatives])
    return p_low, len(alternatives)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(args.project)
    rng = np.random.default_rng(args.seed)

    by_year_rows, shift_rows, loo_rows = [], [], []
    for tuned, base in PAIRS:
        p = paired(d, tuned, base, args.exclude_unparseable)
        if p is None:
            print(f"Skipped {tuned} vs {base}: results missing.")
            continue
        pair = f"{tuned} minus {base}"
        for y, g in p.groupby("year"):
            by_year_rows.append({
                "pair": pair, "year": y, "n": len(g),
                "acc_tuned": round(100 * g.correct.mean(), 1), "acc_base": round(100 * g.correct_base.mean(), 1),
                "lead": round(100 * g.lead.mean(), 1),
                "only_tuned_correct": int((g.lead == 1).sum()), "only_base_correct": int((g.lead == -1).sum()),
            })
        for split, unseen in SPLITS.items():
            s = shift(p, unseen)
            boot = item_bootstrap(p, unseen, rng)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            perm_p, n_choices = year_permutation(p, unseen)
            new = p.year.isin(unseen)
            shift_rows.append({
                "pair": pair, "split": split, "unseen_years": " ".join(map(str, unseen)),
                "lead_seen": round(100 * p.loc[~new, "lead"].mean(), 1),
                "lead_unseen": round(100 * p.loc[new, "lead"].mean(), 1),
                "lead_shift": round(100 * s, 1), "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1),
                "year_perm_p": round(perm_p, 3), "year_choices": n_choices,
                "n_seen": int((~new).sum()), "n_unseen": int(new.sum()),
            })
        for y in sorted(p.year.unique()):
            rest = p[p.year != y]
            unseen = [u for u in SPLITS["main"] if u != y]
            if not unseen or rest.year.isin(unseen).all():
                continue
            loo_rows.append({"pair": pair, "left_out": y, "lead_shift": round(100 * shift(rest, unseen), 1)})

    by_year = pd.DataFrame(by_year_rows)
    result = pd.DataFrame(shift_rows)
    loo = pd.DataFrame(loo_rows)
    by_year.to_csv(out / "lead_by_year.csv", index=False)
    result.to_csv(out / "lead_shift.csv", index=False)
    loo.to_csv(out / "lead_shift_loo.csv", index=False)

    pd.set_option("display.width", 180)
    tag = " (unparseable answers excluded)" if args.exclude_unparseable else ""
    print(f"Lead of the medical model over its base model per edition (percentage points){tag}")
    print(by_year.pivot(index="pair", columns="year", values="lead").to_string())
    print(f"\nMAIN: lead shift = lead on unseen editions minus lead on seen editions{tag}")
    print(result.to_string(index=False))
    print("\nLeave one edition out (main split):")
    print(loo.pivot(index="pair", columns="left_out", values="lead_shift").to_string())
    print("\nNegative lead_shift = the fine-tune's advantage shrinks on unseen exams (what fine-tuning-data")
    print("contamination predicts). year_perm_p accounts for the normal year-to-year variation in the lead.")
    print(f"\nWrote CSV files to {out.resolve()}")


if __name__ == "__main__":
    main()
