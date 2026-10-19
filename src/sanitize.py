"""Contamination tests on the raw MIR question text: completion and Min-K% Prob.

Adapted from LLMSanitize (Ravaut et al., 2024, github.com/ntunlp/LLMSanitize) to the vLLM setup
of src/run_eval.py. Every model in configs/models.yaml (except the judge and optional models)
runs both tests on every dataset of the groups in configs/sanitize.yaml (all MIR editions,
Spanish and English):

    completion  the first half of each question (split at the middle word) is given to the
                model as raw text, and the model continues it greedily for as many tokens as
                the real second half has. The continuation is compared with the real second
                half: ROUGE-L F1 (words), normalized Levenshtein similarity (characters) and
                the number of leading tokens that match exactly. Continuations with a
                Levenshtein similarity of at least near_verbatim_threshold are flagged.
    mink        Min-K% Prob (Shi et al., 2023): mean log-probability of the k% least likely
                tokens of the question, plus the mean log-probability (perplexity) of all
                tokens and the zlib ratio of LLMSanitize. Higher scores on seen than on unseen
                editions suggest exposure. Scored on the full question and on the stem alone
                (stem_* columns), which does not depend on how the options are written.

Both tests use the raw question text, laid out as in the MIR booklet (src/sanitize_metrics.py),
without chat template or instruction. All vLLM calls are in src/vllm_scoring.py. Each model is
loaded once and runs both tests on all datasets; with several models, each model runs in its
own process so that its GPU memory is released before the next one starts.

Every row carries two labels of the edition for that model (src/sanitize_metrics.py,
seen_labels): seen (main, from unseen_from in configs/models.yaml) and seen_strict (unseen only
if the exam was held after the model's release, ambiguous between cutoff and release).

Outputs (one pair per model, dataset and test; existing outputs are skipped unless --overwrite,
so an interrupted run continues where it stopped):
    results/sanitize/<model>/<dataset>.completion.jsonl       one row per question
    results/sanitize/<model>/<dataset>.mink.jsonl             one row per question
    results/sanitize/<model>/<dataset>.<test>.meta.json       settings, timing, vLLM version,
                                                              model revision
With --lora, <model> is <model key>+<adapter name>.

Usage:
    python src/sanitize.py                                         all models and datasets
    python src/sanitize.py --models gemma3-1b --datasets mir_2022_es mir_2026_es
    python src/sanitize.py --models gemma3-1b --limit 5 --out results_smoke
    python src/sanitize.py --models gemma3-4b --lora <adapter folder> --lora-name mir2022
    python src/sanitize.py --backend mock --limit 5 --out results_smoke   (test without GPU)
Analysis: python src/sanitize_summary.py
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

from config import available_datasets, load_datasets_config, load_models, load_yaml, resolve_path
from items import load_dataset
from sanitize_metrics import (completion_scores, likelihood_scores, question_text, seen_labels,
                              split_at_words, text_logprobs)
from vllm_scoring import MockScorer, VLLMScorer, lora_settings

TESTS = ("completion", "mink")


# ---------------------------------------------------------------- selection and paths

def evaluated_models(models, requested=None):
    """Requested model keys, or every model except the judge and optional models."""
    if requested:
        unknown = [k for k in requested if k not in models]
        if unknown:
            raise SystemExit(f"Unknown models {unknown}. Available: {sorted(models)}")
        return list(requested)
    return [k for k, m in models.items() if m.get("role") != "judge" and not m.get("optional")]


def selected_datasets(ds_cfg, settings, requested=None):
    """Requested dataset keys, or every available dataset of the configured groups."""
    groups = set(settings["dataset_groups"])
    if requested:
        for key in requested:
            if key not in ds_cfg["datasets"]:
                raise SystemExit(f"Unknown dataset '{key}'. Available: {sorted(ds_cfg['datasets'])}")
            if ds_cfg["datasets"][key].get("group") not in groups:
                raise SystemExit(f"Dataset '{key}' is not in the groups {sorted(groups)} of configs/sanitize.yaml.")
        return list(requested)
    return [k for k in available_datasets(ds_cfg) if ds_cfg["datasets"][k].get("group") in groups]


def output_path(out_dir, label, dataset, test):
    return resolve_path(out_dir) / "sanitize" / label / f"{dataset}.{test}.jsonl"


def meta_path(path):
    return path.with_suffix(".meta.json")


def pending(out_dir, label, datasets, tests, overwrite=False):
    """(dataset, test) pairs without a complete output (results and meta file)."""
    todo = []
    for dataset in datasets:
        for test in tests:
            path = output_path(out_dir, label, dataset, test)
            if overwrite or not (path.exists() and meta_path(path).exists()):
                todo.append((dataset, test))
    return todo


def pending_work(models=None, datasets=None, overwrite=False, out_dir="results"):
    """Models with missing outputs and their Slurm time limit (used by scripts/submit_sanitize.sh)."""
    model_cfgs, ds_cfg, settings = load_models(), load_datasets_config(), load_yaml("sanitize.yaml")
    dataset_keys = selected_datasets(ds_cfg, settings, datasets)
    return [(key, model_cfgs[key]["slurm_time"]) for key in evaluated_models(model_cfgs, models)
            if pending(out_dir, key, dataset_keys, TESTS, overwrite)]


def model_revision(hf_id):
    """Commit hash of the downloaded snapshot in the Hugging Face cache, or None."""
    try:
        from models import find_snapshot

        snapshot = find_snapshot(hf_id)
    except ImportError:
        return None
    return snapshot.name if snapshot else None


def write_atomic(path, text):
    """Write through a temporary file, so that an interrupted job leaves no partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------- tests

def with_bos(scorer, ids):
    return [scorer.bos_token_id, *ids] if scorer.bos_token_id is not None else list(ids)


def run_mink(scorer, items, settings):
    """Min-K% Prob rows: full question, and the stem alone if the options are included."""
    text_cfg, k_values = settings["text"], settings["mink"]["k_values"]
    variants = {"": [question_text(item, text_cfg) for item in items]}
    if text_cfg.get("include_options", True):
        variants["stem_"] = [question_text(item, text_cfg, with_options=False) for item in items]
    prompts = [with_bos(scorer, scorer.encode(text)) for texts in variants.values() for text in texts]
    logprobs = iter(zip(prompts, scorer.prompt_logprobs(prompts)))

    rows = [{} for _ in items]
    for prefix, texts in variants.items():
        for row, text in zip(rows, texts):
            ids, values = next(logprobs)
            scored = text_logprobs(ids, values, scorer.special_ids)
            if not prefix:
                row["text"] = text
            row[f"{prefix}n_tokens"] = len(scored)
            row.update(likelihood_scores(scored, text, k_values, prefix))
            if not prefix:
                row["token_logprobs"] = [round(value, 5) for value in scored]
    # Per-token values at the end of each row, so that the scores are easy to read.
    return [{**{k: v for k, v in row.items() if k != "token_logprobs"}, "token_logprobs": row["token_logprobs"]}
            for row in rows]


def run_completion(scorer, items, settings):
    """Completion rows: the first half as prompt, the greedy continuation and its similarity."""
    cfg, text_cfg = settings["completion"], settings["text"]
    rows, prompts, budgets, references = [], [], [], []
    for item in items:
        text = question_text(item, text_cfg)
        prefix, reference, n_words, n_prefix_words = split_at_words(text, cfg["split_ratio"])
        prefix_ids, full_ids = scorer.encode(prefix), scorer.encode(text)
        # The real second half in tokens: the tokens of the full text after the prefix, if the
        # prefix is tokenized the same way on its own (almost always, as it ends before a space
        # or line break); otherwise the second half tokenized on its own.
        aligned = full_ids[:len(prefix_ids)] == prefix_ids
        reference_ids = full_ids[len(prefix_ids):] if aligned else scorer.encode(reference)
        budget = max(1, min(len(reference_ids), cfg["max_new_tokens_cap"]))
        prompts.append(with_bos(scorer, prefix_ids))
        budgets.append(budget)
        references.append(reference_ids)
        rows.append({"n_words": n_words, "n_prefix_words": n_prefix_words, "prefix": prefix,
                     "reference": reference, "n_prompt_tokens": len(prefix_ids),
                     "n_reference_tokens": len(reference_ids), "tokens_aligned": aligned,
                     "max_new_tokens": budget})

    outputs = scorer.greedy(prompts, budgets)
    for row, reference_ids, out in zip(rows, references, outputs):
        row.update({"generation": out["text"], "n_generated_tokens": len(out["token_ids"]),
                    "finish_reason": out["finish_reason"]})
        row.update(completion_scores(row["reference"], out["text"], reference_ids, out["token_ids"],
                                     cfg["near_verbatim_threshold"]))
    return rows


RUNNERS = {"completion": run_completion, "mink": run_mink}


# ---------------------------------------------------------------- main

def run_model(key, model_cfg, dataset_keys, tests, ds_cfg, settings, args):
    lora = lora_settings(args.lora, args.lora_name) if args.lora else None
    label = key if lora is None else f"{key}+{lora['name']}"
    todo = pending(args.out, label, dataset_keys, tests, args.overwrite)
    if not todo:
        print(f"[{label}] all outputs exist; nothing to do (--overwrite reruns them).")
        return
    if lora and lora["base_model"] and lora["base_model"] != model_cfg["hf_id"]:
        print(f"WARNING: the adapter was trained on {lora['base_model']}, not {model_cfg['hf_id']}.")

    # Load the data before the model, so that data errors surface before GPU time is spent.
    items = {d: load_dataset(d, ds_cfg["datasets"][d], args.limit) for d in dict.fromkeys(d for d, _ in todo)}
    scorer_class = VLLMScorer if args.backend == "vllm" else MockScorer
    scorer = scorer_class(model_cfg, lora, seed=settings["seed"])
    environment = {**scorer.environment(), "hostname": platform.node()}
    revision = model_revision(model_cfg["hf_id"]) if args.backend == "vllm" else None
    versions = {"vllm_version": environment.get("vllm_version"), "model_revision": revision}

    for dataset, test in todo:
        spec = ds_cfg["datasets"][dataset]
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        rows = RUNNERS[test](scorer, items[dataset], settings)
        elapsed = time.perf_counter() - t0

        lines = []
        for item, row in zip(items[dataset], rows):
            year = int(item.year)
            seen, seen_strict = seen_labels(model_cfg, year, settings["exam_dates"].get(year))
            record = {"model": label, "base_model": key, "dataset": dataset, "item_id": item.item_id,
                      "edition": year, "language": spec["language"], "seen": seen,
                      "seen_strict": seen_strict, **row, **versions}
            lines.append(json.dumps(record, ensure_ascii=False))
        path = output_path(args.out, label, dataset, test)
        write_atomic(path, "\n".join(lines) + "\n")
        meta = {
            "model": label, "base_model": key, "model_config": model_cfg, **versions, "lora": lora,
            "dataset": dataset, "dataset_config": spec, "test": test, "n_items": len(rows),
            "limit": args.limit, "backend": args.backend,
            "settings": {"text": settings["text"], test: settings[test], "seed": settings["seed"],
                         "exam_dates": settings["exam_dates"]},
            "seconds": elapsed, "model_load_seconds": scorer.load_seconds, "started_utc": started,
            "environment": environment,
        }
        write_atomic(meta_path(path), json.dumps(meta, indent=2, default=str, ensure_ascii=False))
        print(f"[{label}] {dataset} {test}: {len(rows)} questions, {elapsed:.1f} s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model keys from configs/models.yaml (default: all except the judge and optional models).")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset keys (default: every available dataset of the groups in configs/sanitize.yaml).")
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=list(TESTS))
    parser.add_argument("--out", default="results", help="Output directory (results go to <out>/sanitize/).")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions per dataset.")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm",
                        help="'mock' gives random scores to test the pipeline without a GPU.")
    parser.add_argument("--overwrite", action="store_true", help="Rerun outputs that already exist.")
    parser.add_argument("--lora", default=None, help="Folder of a LoRA adapter (PEFT format) to apply.")
    parser.add_argument("--lora-name", default=None, help="Name of the adapter in the output folder "
                                                          "(default: the adapter folder name).")
    return parser.parse_args()


def main():
    args = parse_args()
    models, ds_cfg, settings = load_models(), load_datasets_config(), load_yaml("sanitize.yaml")
    model_keys = evaluated_models(models, args.models)
    dataset_keys = selected_datasets(ds_cfg, settings, args.datasets)
    if not dataset_keys:
        raise SystemExit("No datasets available. Run python src/prepare_data.py first.")
    if args.lora and len(model_keys) != 1:
        raise SystemExit("--lora needs exactly one model (--models <base model of the adapter>).")

    if args.backend == "vllm" and len(model_keys) > 1:
        failed = []
        for key in model_keys:
            command = [sys.executable, __file__, "--models", key, "--datasets", *dataset_keys,
                       "--tests", *args.tests, "--out", args.out]
            if args.limit:
                command += ["--limit", str(args.limit)]
            if args.overwrite:
                command.append("--overwrite")
            if subprocess.run(command).returncode != 0:
                failed.append(key)
        if failed:
            raise SystemExit(f"Failed models: {failed}. Rerun to continue with the missing outputs.")
        return

    for key in model_keys:
        run_model(key, models[key], dataset_keys, args.tests, ds_cfg, settings, args)


if __name__ == "__main__":
    main()
