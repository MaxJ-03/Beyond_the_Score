"""Compare the three layers of questions: standard exam, new exam and real world.

Layers (analysis.layers in configs/datasets.yaml):
    standard_exam   older MIR editions (group older_exam), in the layer language
    new_exam        newest MIR editions (group new_exam), in the layer language
    <dataset>       every real-world dataset (group real_world) as its own layer, because
                    their quality metrics differ (accuracy, rubric score, comprehensiveness)
The layer language is English, so that only the type of question differs between layers.

Quality per layer is pooled over its items. Chance-corrected quality is
(quality - c) / (1 - c), with c the mean of 1/k for multiple choice and 0 for open-ended
datasets. Cost per point is FLOPs per question / chance-corrected quality (as in
summary.csv), so a lower value means a more cost-efficient model.

Outputs, written to results/analysis/:
    layer_summary.csv     per model and layer: quality, cost, both rankings, whether the
                          model is the most cost-efficient one and on the Pareto frontier
    layer_agreement.csv   per pair of layers: Spearman correlation of the quality and
                          cost-efficiency rankings, the most cost-efficient model in each
                          layer and whether the choice is the same
    generalization.csv    per model: retention = chance-corrected quality on a real-world
                          layer / chance-corrected quality on the new exam (left empty when
                          the new-exam value is below MIN_BASE, near chance), and the accuracy
                          drop for multiple-choice real-world layers; per family: retention of
                          each smaller model minus retention of the largest model. Bootstrap
                          95% CIs resample the items of both layers.

Run after src/analyze.py, which writes the FLOPs per question to summary.csv.

Usage:
    python src/compare_layers.py
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from analyze import load_raw
from config import load_datasets_config, load_models, resolve_path

MIN_BASE = 0.1  # minimum chance-corrected new-exam quality for a meaningful retention ratio


def define_layers(ds_cfg):
    """Mapping from layer name to its datasets and task."""
    cfg = ds_cfg["analysis"]["layers"]
    specs = ds_cfg["datasets"]

    def in_group(group):
        return [k for k, s in specs.items() if s.get("group") == group and s.get("language") == cfg["language"]]
    layers = {"standard_exam": (in_group(cfg["standard"]), "multiple_choice"),
              "new_exam": (in_group(cfg["new"]), "multiple_choice")}
    for key in in_group("real_world"):
        layers[key] = ([key], specs[key].get("task", "multiple_choice"))
    return layers


def item_matrix(df, datasets):
    """Items by models: quality per item, only for models with every item of the layer."""
    matrix = df[df["dataset"].isin(datasets)].pivot_table(
        index=["dataset", "item_id"], columns="model", values="quality")
    return matrix.dropna(axis=1)


def chance_level(df, datasets, task):
    if task != "multiple_choice":
        return 0.0
    sub = df[df["dataset"].isin(datasets)]
    return float((1.0 / sub["n_options"]).mean())


def corrected(quality, chance):
    return (quality - chance) / (1.0 - chance)


def interval(draws):
    """95% percentile interval of bootstrap draws; empty if the ratio is undefined in most draws."""
    finite = draws[np.isfinite(draws)]
    return np.percentile(finite, [2.5, 97.5]) if len(finite) > 0.5 * len(draws) else (np.nan, np.nan)


def layer_summary(df, summary, layers):
    rows = []
    for layer, (datasets, task) in layers.items():
        matrix = item_matrix(df, datasets)
        chance = chance_level(df, datasets, task)
        costs = summary[summary["dataset"].isin(datasets)]
        for model in matrix.columns:
            cost = costs[costs["model"] == model]
            flops = (cost["flops_per_question"] * cost["n_items"]).sum() / cost["n_items"].sum()
            quality = matrix[model].mean()
            cca = corrected(quality, chance)
            rows.append({"layer": layer, "datasets": " ".join(datasets), "model": model, "n_items": len(matrix),
                         "metric": cost["metric"].iloc[0] if len(cost) else "", "quality": quality,
                         "chance_corrected_quality": cca, "flops_per_question": flops,
                         "flops_per_point": flops / cca if cca > 0 and flops > 0 else np.nan})
    table = pd.DataFrame(rows)
    by_layer = table.groupby("layer")
    table["rank_quality"] = by_layer["quality"].rank(ascending=False, method="min")
    table["rank_cost_efficiency"] = by_layer["flops_per_point"].rank(method="min", na_option="bottom")
    table["most_cost_efficient"] = table["rank_cost_efficiency"] == 1
    table["on_pareto_frontier"] = False
    for _, group in by_layer:
        best = -np.inf
        for index, row in group.sort_values(["flops_per_question", "quality"], ascending=[True, False]).iterrows():
            if pd.notna(row["flops_per_question"]) and row["quality"] > best:
                table.loc[index, "on_pareto_frontier"], best = True, row["quality"]
    return table


def layer_agreement(table):
    rows = []
    for a, b in combinations(table["layer"].unique(), 2):
        pivot = table[table["layer"].isin([a, b])].pivot(index="model", columns="layer")
        pivot = pivot.dropna(subset=[("quality", a), ("quality", b)])
        best = {layer: table[(table["layer"] == layer) & table["most_cost_efficient"]]["model"].tolist()
                for layer in (a, b)}
        rows.append({"layer_a": a, "layer_b": b, "n_models": len(pivot),
                     "spearman_quality": pivot["quality"][a].corr(pivot["quality"][b], method="spearman"),
                     "spearman_cost_efficiency": pivot["rank_cost_efficiency"][a].corr(
                         pivot["rank_cost_efficiency"][b], method="spearman"),
                     "most_cost_efficient_a": " ".join(best[a]), "most_cost_efficient_b": " ".join(best[b]),
                     "same_choice": set(best[a]) == set(best[b])})
    return pd.DataFrame(rows).round(3)


def family_pairs(models, available):
    """(smaller, largest) model pairs within each family, general and medical models apart."""
    groups = {}
    for key, cfg in models.items():
        if key in available and cfg.get("role") != "judge":
            groups.setdefault((cfg["family"], cfg.get("medical", False)), []).append(key)
    pairs = []
    for keys in groups.values():
        keys = sorted(keys, key=lambda k: models[k]["size_b"])
        pairs += [(small, keys[-1]) for small in keys[:-1] if models[small]["size_b"] < models[keys[-1]]["size_b"]]
    return pairs


def generalization(df, layers, models, samples, seed):
    new_datasets, _ = layers["new_exam"]
    new = item_matrix(df, new_datasets)
    new_chance = chance_level(df, new_datasets, "multiple_choice")
    rng = np.random.default_rng(seed)
    rows = []
    for layer, (datasets, task) in layers.items():
        if layer in ("standard_exam", "new_exam"):
            continue
        real = item_matrix(df, datasets)
        common = [m for m in new.columns if m in real.columns]
        if not common:
            continue
        a, b = new[common].to_numpy(), real[common].to_numpy()
        chance = chance_level(df, datasets, task)

        def retention(x, y):
            base = corrected(x.mean(0), new_chance)
            return np.where(base > MIN_BASE, corrected(y.mean(0), chance) / np.maximum(base, MIN_BASE), np.nan)
        draws_ret, draws_drop = [], []
        for _ in range(samples):
            x, y = a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]
            draws_ret.append(retention(x, y))
            draws_drop.append(x.mean(0) - y.mean(0))
        draws_ret, draws_drop = np.array(draws_ret), np.array(draws_drop)
        point = retention(a, b)
        for i, model in enumerate(common):
            low, high = interval(draws_ret[:, i])
            row = {"layer": layer, "comparison": model, "new_exam_quality": a[:, i].mean(),
                   "real_world_quality": b[:, i].mean(), "retention": point[i],
                   "retention_ci_low": low, "retention_ci_high": high,
                   "new_exam_near_chance": corrected(a[:, i].mean(), new_chance) <= MIN_BASE}
            if task == "multiple_choice":
                d_low, d_high = np.percentile(draws_drop[:, i], [2.5, 97.5])
                row.update(accuracy_drop=a[:, i].mean() - b[:, i].mean(), drop_ci_low=d_low, drop_ci_high=d_high)
            rows.append(row)
        for small, large in family_pairs(models, common):
            i, j = common.index(small), common.index(large)
            low, high = interval(draws_ret[:, i] - draws_ret[:, j])
            rows.append({"layer": layer, "comparison": f"{small} minus {large}",
                         "retention": point[i] - point[j], "retention_ci_low": low, "retention_ci_high": high})
    return pd.DataFrame(rows).round(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Results directory used by run_eval.")
    args = parser.parse_args()
    results_dir = resolve_path(args.results)
    ds_cfg, models = load_datasets_config(), load_models()
    summary_path = results_dir / "analysis" / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found. Run python src/analyze.py first.")
    summary = pd.read_csv(summary_path)
    df, _ = load_raw(results_dir)
    df = df.dropna(subset=["quality"])
    layers = define_layers(ds_cfg)

    table = layer_summary(df, summary, layers)
    agreement = layer_agreement(table)
    analysis = ds_cfg["analysis"]
    general = generalization(df, layers, models, analysis["bootstrap_samples"], analysis["bootstrap_seed"])
    out_dir = results_dir / "analysis"
    table.round(4).to_csv(out_dir / "layer_summary.csv", index=False)
    agreement.to_csv(out_dir / "layer_agreement.csv", index=False)
    general.to_csv(out_dir / "generalization.csv", index=False)

    pd.set_option("display.width", 180)
    columns = ["layer", "model", "metric", "quality", "flops_per_point", "rank_cost_efficiency"]
    print(table[columns].round(3).to_string(index=False))
    print("\nAgreement between layers:")
    print(agreement.to_string(index=False))
    print("\nGeneralization from the new exam to the real-world layers:")
    print(general.to_string(index=False))
    print(f"\nWrote layer_summary.csv, layer_agreement.csv and generalization.csv to {out_dir}")


if __name__ == "__main__":
    main()
