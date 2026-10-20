"""Aggregate raw evaluation results into quality and cost metrics.

Metrics per model and dataset:
    quality                  accuracy for multiple-choice datasets; for open-ended datasets
                             the mean judge score from results/judged/ (rubric score or
                             comprehensiveness, see the metric column), with 95% bootstrap CI
    accuracy                 share of questions answered correctly (multiple choice only)
    hallucinations_per_answer  contradicted statements per answer (statements task only)
    chance_corrected_acc     (quality - c) / (1 - c), with c the mean of 1/k over items for
                             multiple choice and 0 for open-ended datasets
    invalid_rate             share of responses without a parseable answer
    flops_per_question       2 * N * (input tokens + output tokens), averaged over items
    gpu_seconds_per_question measured generation time times number of GPUs, per item
    output_tokens_per_question
    points_per_tflop         100 * chance_corrected_acc / (flops_per_question / 1e12): accuracy
                             points above chance per TFLOP spent per question; higher is more
                             cost-efficient, zero or negative at or below chance
    points_per_gpu_second    the same with measured GPU time instead of FLOPs

Comparisons:
    rankings.csv             rank of each model per dataset, by quality and by cost-efficiency
    rank_correlations.csv    Spearman correlation of cost-efficiency rankings between datasets
    exam_shift.csv           accuracy on unseen minus seen exam years, per model and language,
                             for two splits of the years:
                               common     the same years for every model (analysis.common_split
                                          in configs/datasets.yaml)
                               per_model  unseen from the first exam after the model's training
                                          cutoff (unseen_from in configs/models.yaml)
    on_pareto_frontier       models for which no other model is both cheaper and better
Open-ended datasets appear once their answers are graded by src/judge.py.

Usage:
    python src/analyze.py --results results
"""

import argparse
import json

import numpy as np
import pandas as pd

from config import load_datasets_config, load_models, resolve_path
from items import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Results directory used by run_eval.")
    parser.add_argument("--param-counts", default=None,
                        help="Parameter count file (default: <results>/param_counts.json).")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_raw(results_dir):
    """All answers with a quality value per item: correctness, or the judge score if graded."""
    records, metas = [], []
    for path in sorted((results_dir / "raw").glob("*/*.jsonl")):
        records.append(pd.read_json(path, lines=True, dtype={"item_id": str}))
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit(f"No result files found under {results_dir / 'raw'}.")
    df = pd.concat(records, ignore_index=True)
    judged = [pd.read_json(p, lines=True, dtype={"item_id": str})
              for p in sorted((results_dir / "judged").glob("*/*.jsonl"))]
    if judged:
        scores = pd.concat(judged, ignore_index=True).reindex(
            columns=["model", "dataset", "item_id", "score", "hallucinations"])
        df = df.merge(scores, on=["model", "dataset", "item_id"], how="left")
    else:
        df["score"], df["hallucinations"] = np.nan, np.nan
    # Datasets restricted by date (min_date in configs/datasets.yaml) keep only the current items,
    # so answers to older items that are still stored are not counted.
    ds_cfg = load_datasets_config()
    for dataset in df["dataset"].unique():
        spec = ds_cfg["datasets"].get(dataset, {})
        if spec.get("min_date"):
            keep = {item.item_id for item in load_dataset(dataset, spec)}
            df = df[(df["dataset"] != dataset) | df["item_id"].isin(keep)]
    correct = df["correct"].map({True: 1.0, False: 0.0})
    df["quality"] = correct.where(correct.notna(), df["score"])
    return df, metas


def bootstrap_ci(values, samples, seed):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


METRICS = {"multiple_choice": "accuracy", "rubric": "rubric_score", "statements": "comprehensiveness"}


def summarize(df, metas, param_counts, models, ds_cfg):
    analysis = ds_cfg["analysis"]
    count_key = analysis["flops_param_count"]
    meta_index = {(m["model"], m["dataset"]): m for m in metas}
    rows, not_graded = [], []
    for (model, dataset), group in df.groupby(["model", "dataset"]):
        task = ds_cfg["datasets"].get(dataset, {}).get("task", "multiple_choice")
        if group["quality"].isna().any():
            not_graded.append(f"{model}/{dataset}")
            continue
        multiple_choice = task == "multiple_choice"
        n = len(group)
        quality = group["quality"].mean()
        low, high = bootstrap_ci(group["quality"], analysis["bootstrap_samples"], analysis["bootstrap_seed"])
        chance = (1.0 / group["n_options"]).mean() if multiple_choice else 0.0
        cca = (quality - chance) / (1.0 - chance)

        n_params = param_counts.get(model, {}).get(count_key)
        tokens = group["input_tokens"] + group["output_tokens"]
        flops = 2.0 * n_params * tokens.mean() if n_params else np.nan

        meta = meta_index.get((model, dataset), {})
        gpu_seconds = meta.get("gpu_seconds", np.nan) / meta.get("n_items", n) if meta else np.nan

        rows.append({
            "model": model,
            "dataset": dataset,
            "group": ds_cfg["datasets"].get(dataset, {}).get("group", ""),
            "language": ds_cfg["datasets"].get(dataset, {}).get("language", ""),
            "family": models.get(model, {}).get("family", ""),
            "size_b": models.get(model, {}).get("size_b", np.nan),
            "medical": models.get(model, {}).get("medical", False),
            "n_items": n,
            "task": task,
            "metric": METRICS[task],
            "quality": quality,
            "quality_ci_low": low,
            "quality_ci_high": high,
            "accuracy": quality if multiple_choice else np.nan,
            "accuracy_ci_low": low if multiple_choice else np.nan,
            "accuracy_ci_high": high if multiple_choice else np.nan,
            "hallucinations_per_answer": group["hallucinations"].mean() if task == "statements" else np.nan,
            "chance_level": chance,
            "chance_corrected_acc": cca,
            "invalid_rate": group["predicted"].isna().mean() if multiple_choice else np.nan,
            "truncated_rate": (group["finish_reason"] == "length").mean(),
            "input_tokens_per_question": group["input_tokens"].mean(),
            "output_tokens_per_question": group["output_tokens"].mean(),
            "params_used_for_flops": n_params,
            "flops_per_question": flops,
            "gpu_seconds_per_question": gpu_seconds,
            "points_per_tflop": 100 * cca / (flops / 1e12) if flops > 0 else np.nan,
            "points_per_gpu_second": 100 * cca / gpu_seconds if gpu_seconds > 0 else np.nan,
            "median_latency_seconds": meta.get("median_latency_seconds") if meta else np.nan,
            # Memory needed for the weights in bfloat16 (2 bytes per parameter). With vLLM, the
            # measured GPU memory reflects the reserved share, not what the model needs.
            "weights_memory_gb": 2 * param_counts.get(model, {}).get("total_in_files", np.nan) / 1e9,
            "gpu_name": meta.get("environment", {}).get("gpu_name", ""),
        })
    if not_graded:
        print(f"Not yet graded by src/judge.py, left out: {', '.join(not_graded)}")
    summary = pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)
    summary["on_pareto_frontier"] = False
    for _, group in summary.groupby("dataset"):
        best_quality = -np.inf
        for index, row in group.sort_values(["flops_per_question", "quality"],
                                            ascending=[True, False]).iterrows():
            if pd.notna(row["flops_per_question"]) and row["quality"] > best_quality:
                summary.loc[index, "on_pareto_frontier"] = True
                best_quality = row["quality"]
    return summary


def rankings(summary):
    ranked = summary[["model", "dataset", "metric", "quality", "points_per_tflop"]].copy()
    ranked["rank_quality"] = ranked.groupby("dataset")["quality"].rank(ascending=False, method="min")
    ranked["rank_cost_efficiency"] = ranked.groupby("dataset")["points_per_tflop"].rank(
        ascending=False, method="min", na_option="bottom")
    return ranked


def rank_correlations(ranked):
    pivot = ranked.pivot(index="model", columns="dataset", values="rank_cost_efficiency")
    return pivot.corr(method="spearman")


def exam_shift(df, ds_cfg, models, samples, seed):
    """Accuracy on unseen minus seen MIR exam years, per model, language and split of the years."""
    specs = ds_cfg["datasets"]
    common = ds_cfg["analysis"]["common_split"]
    exam = df[df["dataset"].map(lambda d: specs.get(d, {}).get("group") in ("older_exam", "new_exam"))].copy()
    exam["year"] = pd.to_numeric(exam["year"], errors="coerce")
    exam = exam.dropna(subset=["year"])
    exam["language"] = exam["dataset"].map(lambda d: specs[d].get("language", ""))

    rng = np.random.default_rng(seed)
    rows = []
    for (model, language), group in exam.groupby(["model", "language"]):
        splits = {"common": (group["year"].isin(common["seen_years"]), group["year"].isin(common["unseen_years"]))}
        unseen_from = models.get(model, {}).get("unseen_from")
        if unseen_from:
            splits["per_model"] = (group["year"] < unseen_from, group["year"] >= unseen_from)
        for split, (seen_mask, unseen_mask) in splits.items():
            seen, unseen = group[seen_mask], group[unseen_mask]
            if seen.empty or unseen.empty:
                continue
            old_vals = seen["correct"].to_numpy(dtype=float)
            new_vals = unseen["correct"].to_numpy(dtype=float)
            diffs = (rng.choice(new_vals, (samples, len(new_vals))).mean(axis=1)
                     - rng.choice(old_vals, (samples, len(old_vals))).mean(axis=1))
            rows.append({
                "split": split,
                "model": model,
                "language": language,
                "seen_years": f"{int(seen['year'].min())}-{int(seen['year'].max())}",
                "unseen_years": f"{int(unseen['year'].min())}-{int(unseen['year'].max())}",
                "accuracy_seen": old_vals.mean(),
                "accuracy_unseen": new_vals.mean(),
                "difference_unseen_minus_seen": new_vals.mean() - old_vals.mean(),
                "ci_low": np.percentile(diffs, 2.5),
                "ci_high": np.percentile(diffs, 97.5),
                "n_seen": len(old_vals),
                "n_unseen": len(new_vals),
            })
    shift = pd.DataFrame(rows)
    return shift.sort_values(["split", "language", "model"]).reset_index(drop=True) if rows else shift


def plot_accuracy_vs_cost(summary, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for dataset, group in summary.groupby("dataset"):
        if group["flops_per_question"].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        metric = group["metric"].iloc[0].replace("_", " ").capitalize()
        for _, row in group.iterrows():
            ax.errorbar(row["flops_per_question"], row["quality"],
                        yerr=[[row["quality"] - row["quality_ci_low"]],
                              [row["quality_ci_high"] - row["quality"]]],
                        fmt="o", capsize=3)
            ax.annotate(row["model"], (row["flops_per_question"], row["quality"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        frontier = group[group["on_pareto_frontier"]].sort_values("flops_per_question")
        ax.step(frontier["flops_per_question"], frontier["quality"], where="post",
                linewidth=1, color="black", label="Pareto frontier")
        if group["chance_level"].iloc[0] > 0:
            ax.axhline(group["chance_level"].iloc[0], linestyle="--", linewidth=1, color="grey", label="chance")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xscale("log")
        ax.set_xlabel("FLOPs per question (log scale)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} versus inference cost: {dataset}")
        fig.tight_layout()
        fig.savefig(out_dir / f"accuracy_vs_flops_{dataset}.png", dpi=200)
        plt.close(fig)


def main():
    args = parse_args()
    results_dir = resolve_path(args.results)
    models = load_models()
    ds_cfg = load_datasets_config()

    param_path = resolve_path(args.param_counts) if args.param_counts else results_dir / "param_counts.json"
    param_counts = json.loads(param_path.read_text()) if param_path.exists() else {}
    if not param_counts:
        print(f"Warning: no parameter counts found at {param_path}; FLOPs will be missing. "
              "Run python src/models.py first.")

    df, metas = load_raw(results_dir)
    summary = summarize(df, metas, param_counts, models, ds_cfg)
    ranked = rankings(summary)
    correlations = rank_correlations(ranked)
    shift = exam_shift(df, ds_cfg, models, ds_cfg["analysis"]["bootstrap_samples"],
                       ds_cfg["analysis"]["bootstrap_seed"])

    out_dir = results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    ranked.to_csv(out_dir / "rankings.csv", index=False)
    correlations.to_csv(out_dir / "rank_correlations.csv")
    shift.to_csv(out_dir / "exam_shift.csv", index=False)
    if not args.no_plots:
        plot_accuracy_vs_cost(summary, out_dir)

    pd.set_option("display.width", 160)
    print(summary[["model", "dataset", "n_items", "metric", "quality", "chance_corrected_acc", "invalid_rate",
                   "flops_per_question", "gpu_seconds_per_question"]].to_string(index=False))
    if not shift.empty:
        print("\nAccuracy shift from seen to unseen exam years:")
        print(shift.to_string(index=False))
    print(f"\nWrote analysis files to {out_dir}")


if __name__ == "__main__":
    main()
