"""Summary of the contamination tests: seen vs unseen editions (bootstrap CIs, ROC-AUC, edition-adjusted effect).
Usage: python src/sanitize_summary.py [--results <dir>] [--out <dir>]"""

import argparse

import numpy as np
import pandas as pd

from config import load_yaml, resolve_path
from contamination import fit_unseen_effect
from sanitize_metrics import k_label

ID_COLUMNS = ["model", "base_model", "dataset", "item_id", "edition", "language", "seen", "seen_strict"]
SPLITS = {"main": "seen", "strict": "seen_strict"}
LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}


def score_columns(primary_k):
    """Scores per test, with +1 where higher means closer to the text and -1 where lower does."""
    return {
        "mink": {k_label(primary_k): 1, k_label(primary_k, "stem_"): 1, "mean_logprob": 1, "zlib_ratio": -1},
        "completion": {"rouge_l_f1": 1, "levenshtein_ratio": 1, "lcp_tokens": 1, "near_verbatim": 1},
    }


def load_scores(results_dir, scores):
    """All result rows in long format: one row per model, question and score."""
    frames = []
    for test, columns in scores.items():
        for path in sorted((results_dir / "sanitize").glob(f"*/*.{test}.jsonl")):
            d = pd.read_json(path, lines=True, dtype={"item_id": str}, convert_dates=False)
            present = [c for c in columns if c in d.columns]
            frames.append(d[ID_COLUMNS + present].melt(id_vars=ID_COLUMNS, var_name="score"))
    if not frames:
        raise SystemExit(f"No contamination-test results under {results_dir / 'sanitize'}. Run src/sanitize.py first.")
    long = pd.concat(frames, ignore_index=True)
    long["value"] = long["value"].astype(float)
    return long.dropna(subset=["value"])


# ---------------------------------------------------------------- statistics

def mean_ci(values, rng, samples):
    means = rng.choice(values, (samples, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return values.mean(), low, high


def difference_ci(seen, unseen, rng, samples):
    diffs = (rng.choice(seen, (samples, len(seen))).mean(axis=1)
             - rng.choice(unseen, (samples, len(unseen))).mean(axis=1))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return seen.mean() - unseen.mean(), low, high


def auc(seen, unseen):
    """ROC-AUC of a score for separating seen from unseen questions (Mann-Whitney, ties count half)."""
    ranks = pd.Series(np.concatenate([seen, unseen])).rank().to_numpy()
    n_seen, n_unseen = len(seen), len(unseen)
    return (ranks[:n_seen].sum() - n_seen * (n_seen + 1) / 2) / (n_seen * n_unseen)


def editions(values):
    return ", ".join(str(e) for e in sorted(set(values)))


# ---------------------------------------------------------------- tables

def by_edition(d, rng, samples):
    rows = []
    for (model, language, edition, score), g in d.groupby(["model", "language", "edition", "score"]):
        mean, low, high = mean_ci(g["value"].to_numpy(), rng, samples)
        rows.append({"model": model, "language": language, "edition": edition, "seen": g["seen"].iloc[0],
                     "seen_strict": g["seen_strict"].iloc[0], "score": score, "n": len(g),
                     "mean": mean, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows)


def seen_vs_unseen(d, directions, rng, samples):
    rows = []
    for split, label in SPLITS.items():
        for (model, language, score), g in d.groupby(["model", "language", "score"]):
            unseen = g[g[label] == "unseen"]
            seen = g[g[label] == "seen"]
            if unseen.empty or seen.empty:
                continue
            groups = [(str(e), seen[seen["edition"] == e]) for e in sorted(seen["edition"].unique())]
            groups.append((f"all seen ({editions(seen['edition'])})", seen))
            b = unseen["value"].to_numpy()
            for name, group in groups:
                a = group["value"].to_numpy()
                difference, low, high = difference_ci(a, b, rng, samples)
                rows.append({"model": model, "language": language, "split": split, "score": score,
                             "seen_editions": name, "unseen_editions": editions(unseen["edition"]),
                             "n_seen": len(a), "n_unseen": len(b),
                             "mean_seen": a.mean(), "mean_unseen": b.mean(), "difference": difference,
                             "ci_low": low, "ci_high": high, "auc": auc(directions[score] * a, directions[score] * b)})
    return pd.DataFrame(rows)


def year_adjusted(d, directions):
    base = d[d["model"] == d["base_model"]].astype({"model": object, "item_id": object})  # patsy needs object columns
    rows = []
    for split, label in SPLITS.items():
        data = base[base[label].isin(["seen", "unseen"])].copy()
        data["unseen"] = (data[label] == "unseen").astype(int)
        data["year"] = data["edition"]
        for (language, score), g in data.groupby(["language", "score"]):
            stats = g.groupby("model")["value"].agg(["mean", "std"])
            g = g[g["model"].map(stats["std"]) > 0].copy()
            g["z"] = directions[score] * (g["value"] - g["model"].map(stats["mean"])) / g["model"].map(stats["std"])
            if g["model"].nunique() < 2 or g["unseen"].nunique() < 2:
                continue
            fits = [("ALL (pooled)", None)] + [(m, m) for m in sorted(g["model"].unique())
                                               if g.loc[g["model"] == m, "unseen"].nunique() == 2]
            for name, model in fits:
                _, effect, low, high = fit_unseen_effect(g, model, outcome="z", fixed="year")
                rows.append({"language": language, "split": split, "score": score, "model": name,
                             "unseen_effect_sd": effect / 100, "ci_low": low / 100, "ci_high": high / 100})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["contamination_signal"] = out["ci_high"] < 0
    return out


# ---------------------------------------------------------------- plot

INK, MUTED, GRID, AXIS, SURFACE, UNSEEN_BAND = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"
SERIES = {"es": "#2a78d6", "en": "#eb6834"}  # categorical slots 1 and 2 of the dataviz reference palette


def plot_mink(table, score, out_file):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    data = table[table["score"] == score]
    models = sorted(data["model"].unique())
    columns = min(3, len(models))
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 2.5 * rows + 0.8), squeeze=False,
                             sharex=True, facecolor=SURFACE)
    all_editions = sorted(data["edition"].unique())
    for ax, model in zip(axes.flat, models):
        ax.set_facecolor(SURFACE)
        m = data[data["model"] == model]
        unseen = sorted(m.loc[m["seen"] == "unseen", "edition"].unique())
        if unseen:
            ax.axvspan(unseen[0] - 0.5, max(all_editions) + 0.5, color=UNSEEN_BAND, zorder=0, lw=0)
        for language, color in SERIES.items():
            s = m[m["language"] == language].sort_values("edition")
            if s.empty:
                continue
            ax.plot(s["edition"], s["mean"], color=color, lw=2, zorder=2)
            ax.vlines(s["edition"], s["ci_low"], s["ci_high"], color=color, lw=1, alpha=0.6, zorder=2)
            for _, r in s.iterrows():
                filled = r["seen"] == "seen"
                ax.plot(r["edition"], r["mean"], "o", ms=5.5, mec=SURFACE if filled else color, mew=1.5,
                        mfc=color if filled else SURFACE, zorder=3)
        ax.set_title(model, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
        ax.set_xticks(all_editions)
        ax.set_xticklabels([f"'{str(e)[2:]}" for e in all_editions])
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean log-prob", fontsize=8, color=MUTED)

    handles = [Line2D([], [], color=SERIES["es"], lw=2, label="Spanish (original)"),
               Line2D([], [], color=SERIES["en"], lw=2, label="English (translation)"),
               Line2D([], [], ls="", marker="o", ms=5.5, mfc=MUTED, mec=MUTED, label="seen"),
               Line2D([], [], ls="", marker="o", ms=5.5, mfc=SURFACE, mec=MUTED, mew=1.5, label="unseen"),
               Patch(color=UNSEEN_BAND, label="unseen editions (main split)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.0 - 0.02 / rows))
    fig.suptitle(f"{score}: Min-K% Prob per MIR edition (mean, 95% CI; higher = more likely)",
                 fontsize=10, color=INK, x=0.01, ha="left", y=1.0 + 0.04 / rows)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.08 / rows))
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Results directory of src/sanitize.py.")
    parser.add_argument("--out", default="results/analysis/sanitize", help="Output directory.")
    args = parser.parse_args()

    settings = load_yaml("sanitize.yaml")
    scores = score_columns(settings["mink"]["primary_k"])
    directions = {name: sign for columns in scores.values() for name, sign in columns.items()}
    d = load_scores(resolve_path(args.results), scores)
    rng = np.random.default_rng(settings["seed"])
    samples = settings["summary"]["bootstrap_samples"]
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    editions_table = by_edition(d, rng, samples)
    editions_table.round(4).to_csv(out / "by_edition.csv", index=False)
    comparison = seen_vs_unseen(d, directions, rng, samples)
    comparison.round(4).to_csv(out / "seen_vs_unseen.csv", index=False)
    adjusted = year_adjusted(d, directions)
    adjusted.round(3).to_csv(out / "year_adjusted.csv", index=False)
    plot_mink(editions_table, k_label(settings["mink"]["primary_k"]), out / "mink_by_edition.png")

    pd.set_option("display.width", 200)
    pooled = comparison[(comparison["split"] == "main") & comparison["seen_editions"].str.startswith("all seen")]
    for language in ("es", "en"):
        table = pooled[pooled["language"] == language].pivot_table(index="model", columns="score", values="auc")
        if not table.empty:
            print(f"\nROC-AUC, all seen vs unseen editions, main split, {LANGUAGE_NAMES[language]} "
                  "(0.5 = no separation, above 0.5 = seen questions closer to their text):")
            print(table.round(3).to_string())
    if not adjusted.empty:
        print("\nEdition-adjusted unseen effect (SD; negative = contamination direction), main split, pooled:")
        main_pooled = adjusted[(adjusted["split"] == "main") & (adjusted["model"] == "ALL (pooled)")]
        print(main_pooled.round(3).to_string(index=False))
    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
