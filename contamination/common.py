"""Shared helpers of the contamination experiments: planted models, stage3 settings and the evaluation.
Usage: python contamination/common.py eval --model <key> --datasets <keys> | reparse"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from medeval.config import load_datasets_config, load_yaml  # noqa: E402,F401  (re-exported)
from medeval.data import load_dataset  # noqa: E402,F401  (re-exported)
from medeval.prompts import build_messages  # noqa: E402

SPLIT_FILE = PROJECT / "contamination" / "planted_split.csv"

# ---------------------------------------------------------------- settings copied from stage3
MAX_TOKENS = 1024  # stage3 configs/datasets.yaml, evaluation.max_tokens_direct
OVERRIDES = {
    # stage3 configs/models.yaml: MedGemma 27B opens a hidden reasoning section with <unused94>
    # in part of its answers; the token is forbidden so that it answers directly.
    "medgemma-27b": {"sampling_kwargs": {"bad_words": ["<unused94>"]}},
}

# ---------------------------------------------------------------- planted checkpoints
# Created by `planted.py train` in checkpoints/planted/ (a link to /scratch-shared).
PLANTED_MODELS = {
    f"planted-{base}-ep{e}": {"hf_id": f"checkpoints/planted/{base}-ep{e}", "family": "gemma3", "size_b": 4,
                              "medical": base.startswith("med"), "release": "planted", "optional": True,
                              "llm_kwargs": {"limit_mm_per_prompt": {"image": 0}}}
    for base in ("gemma3-4b", "medgemma-4b") for e in (1, 3, 10)
}


def load_models():
    """Shared models (with the stage3 overrides) plus the planted checkpoints."""
    raw = load_yaml("models.yaml")
    defaults = raw.get("defaults", {})
    models = {}
    entries = {**raw["models"], **PLANTED_MODELS}
    for key, settings in entries.items():
        settings = {**settings, **OVERRIDES.get(key, {})}
        merged = {**defaults, **settings, "key": key}
        for field in ("llm_kwargs", "chat_template_kwargs", "sampling_kwargs"):
            merged[field] = {**defaults.get(field, {}), **settings.get(field, {})}
        models[key] = merged
    return models


def load_llm(cfg):
    from vllm import LLM

    return LLM(model=cfg["hf_id"], dtype=cfg["dtype"], max_model_len=cfg["max_model_len"],
               gpu_memory_utilization=cfg["gpu_memory_utilization"],
               tensor_parallel_size=cfg["tensor_parallel_size"], seed=cfg["seed"], **cfg["llm_kwargs"])


# ---------------------------------------------------------------- answer parser (stage3 rules)
THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)
THOUGHT_START, THOUGHT_END = "<unused94>", "<unused95>"  # MedGemma reasoning section


def strip_reasoning(text):
    text = text or ""
    if THOUGHT_END in text:
        text = text.rsplit(THOUGHT_END, 1)[1]
    elif THOUGHT_START in text:
        text = text.split(THOUGHT_START, 1)[0]
    return UNCLOSED_THINK.sub("", THINK_BLOCK.sub("", text)).strip()


def extract_answer(text, valid_letters):
    """Chosen option letter or None. Rules in order: last \\boxed{X}; last "Answer: X";
    a letter at the start; a single distinct standalone letter (same as stage3 src/run_eval.py)."""
    letters = "".join(sorted(valid_letters))
    response = strip_reasoning(text)
    if not response:
        return None
    boxed = re.findall(rf"\\boxed\{{\s*(?:\\text\{{)?\s*([{letters}])\b", response, flags=re.IGNORECASE)
    if boxed:
        return boxed[-1].upper()
    explicit = re.findall(rf"answer\s*(?:is|:)?\s*\**\s*(?:option\s*)?\(?\**([{letters}])\b", response,
                          flags=re.IGNORECASE)
    if explicit:
        return explicit[-1].upper()
    leading = re.match(rf"^\s*[\(\*]*([{letters}])(?:[\)\.\:\*,]|\s|$)", response)
    if leading and not re.match(r"^\s*A\s+[a-z]", response):  # "A patient ..." is an article
        return leading.group(1)
    without_articles = re.sub(r"(?<![A-Za-z])A(?=\s+[a-z])", "", response)
    standalone = set(re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", without_articles))
    return standalone.pop() if len(standalone) == 1 else None


# ---------------------------------------------------------------- commands
def evaluate(model, dataset_keys, limit=None, out="results"):
    """Answer multiple-choice questions with the shared prompt and the stage3 settings."""
    from vllm import SamplingParams

    cfg = load_models()[model]
    ds_cfg = load_datasets_config()
    eval_cfg = ds_cfg["evaluation"]
    datasets = {k: load_dataset(k, ds_cfg["datasets"][k], limit) for k in dataset_keys}
    t0 = time.perf_counter()
    llm = load_llm(cfg)
    load_seconds = time.perf_counter() - t0
    params = SamplingParams(temperature=eval_cfg["temperature"], max_tokens=MAX_TOKENS, seed=cfg["seed"],
                            **cfg["sampling_kwargs"])
    chat_kwargs = {"chat_template_kwargs": cfg["chat_template_kwargs"]} if cfg["chat_template_kwargs"] else {}
    out_dir = (Path(out) if Path(out).is_absolute() else PROJECT / out) / "raw" / model
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, items in datasets.items():
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        outputs = llm.chat([build_messages(it, eval_cfg, False) for it in items], params, use_tqdm=True,
                           **chat_kwargs)
        seconds = time.perf_counter() - t0
        rows = []
        for it, o in zip(items, outputs):
            c = o.outputs[0]
            predicted = extract_answer(c.text, it.options.keys())
            rows.append({"model": model, "dataset": key, "item_id": it.item_id, "year": it.year,
                         "n_options": it.n_options, "answer": it.answer, "predicted": predicted,
                         "correct": predicted == it.answer, "input_tokens": len(o.prompt_token_ids),
                         "output_tokens": len(c.token_ids), "finish_reason": c.finish_reason, "response": c.text})
        (out_dir / f"{key}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
        meta = {"model": model, "model_config": cfg, "dataset": key, "n_items": len(items), "max_tokens": MAX_TOKENS,
                "temperature": eval_cfg["temperature"], "generation_seconds": seconds,
                "model_load_seconds": load_seconds, "started_utc": started, "settings": "stage3-aligned"}
        (out_dir / f"{key}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
        print(f"[{model}] {key}: {sum(r['correct'] for r in rows)}/{len(rows)} correct, {seconds:.1f} s")


def reparse():
    """Re-score stored MedQA and planted answers with the parser above (no GPU)."""
    files = sorted(PROJECT.glob("results/raw/*/medqa.jsonl")) + sorted(PROJECT.glob("results/raw/planted-*/*.jsonl"))
    for f in files:
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = 0
        for r in rows:
            new = extract_answer(r["response"], "ABCDEFGH"[: r.get("n_options", 4)])
            if new != r["predicted"]:
                changed += 1
                r["predicted"], r["correct"] = new, new == r["answer"]
        if changed:
            f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{f.relative_to(PROJECT)}: {changed} of {len(rows)} answers re-scored")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="multiple-choice evaluation (GPU)")
    ev.add_argument("--model", required=True)
    ev.add_argument("--datasets", nargs="+", required=True)
    ev.add_argument("--limit", type=int, default=None)
    ev.add_argument("--out", default="results")
    sub.add_parser("reparse", help="re-score stored answers with the stage3 parser")
    args = parser.parse_args()
    if args.command == "eval":
        evaluate(args.model, args.datasets, args.limit, args.out)
    else:
        reparse()


if __name__ == "__main__":
    main()
