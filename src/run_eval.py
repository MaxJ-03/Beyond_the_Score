"""Run one model on one or more multiple-choice datasets.

The model is loaded once and evaluated on every requested dataset. For each question the
script records the raw response, the parsed answer, correctness and token counts. For each
dataset it records the wall-clock generation time, which is converted to GPU-seconds in the
analysis.

The file follows the path of a question through the pipeline:
    1. data loading      datasets from configs/datasets.yaml into a common item format
    2. prompts           chat messages for a multiple-choice question
    3. answer parsing    the chosen option letter from a model response
    4. backends          vLLM for real runs, a mock backend for tests without a GPU
    5. main              evaluation loop and output files

Outputs (per model):
    results/raw/<model>/<dataset>.jsonl      one record per question
    results/raw/<model>/<dataset>.meta.json  timing, settings and software versions

Usage:
    python src/run_eval.py --model qwen3-4b
    python src/run_eval.py --model qwen3-4b --datasets mir_2025 mir_2026 --limit 20
    python src/run_eval.py --model qwen3-4b --latency-sample 20
    python src/run_eval.py --model qwen3-4b --backend mock   (pipeline test without GPU)
"""

import argparse
import json
import platform
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from config import default_datasets, load_datasets_config, load_models, resolve_path


# ---------------------------------------------------------------- 1. data loading

@dataclass
class Item:
    item_id: str
    dataset: str
    question: str
    options: dict  # letter -> option text, in display order
    answer: str  # correct letter
    year: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def n_options(self):
        return len(self.options)


def _validate(items, name):
    if not items:
        raise ValueError(f"Dataset '{name}' contains no items.")
    seen = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"Duplicate item id '{item.item_id}' in dataset '{name}'.")
        seen.add(item.item_id)
        if item.answer not in item.options:
            raise ValueError(
                f"Item '{item.item_id}' in '{name}': answer '{item.answer}' is not one of "
                f"the options {list(item.options)}."
            )
        if any(not str(text).strip() for text in item.options.values()):
            raise ValueError(f"Item '{item.item_id}' in '{name}' has an empty option.")
    return items


def _load_jsonl(name, path):
    items = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            options = {str(k).strip().upper(): str(v).strip() for k, v in record["options"].items()}
            items.append(
                Item(
                    item_id=str(record.get("id", line_number)),
                    dataset=name,
                    question=str(record["question"]).strip(),
                    options=dict(sorted(options.items())),
                    answer=str(record["answer"]).strip().upper(),
                    year=str(record.get("year", "")),
                )
            )
    return items


def _load_csv(name, path, columns):
    df = pd.read_csv(path)
    option_columns = columns["options"]
    required = [columns["question"], columns["answer"], *option_columns.values()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset '{name}': missing columns {missing}. Found {list(df.columns)}.")
    items = []
    for index, row in df.iterrows():
        options = {letter: str(row[col]).strip() for letter, col in option_columns.items()}
        items.append(
            Item(
                item_id=str(row[columns["id"]]) if columns.get("id") in df.columns else str(index),
                dataset=name,
                question=str(row[columns["question"]]).strip(),
                options=options,
                answer=str(row[columns["answer"]]).strip().upper(),
                year=str(row[columns["year"]]) if columns.get("year") in df.columns else "",
            )
        )
    return items


def load_dataset(name, spec, limit=None):
    """Load one dataset from its registry entry and return a list of validated items."""
    path = resolve_path(spec["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}': file not found at {path}.")
    if spec["format"] == "jsonl":
        items = _load_jsonl(name, path)
    elif spec["format"] == "csv":
        items = _load_csv(name, path, spec["columns"])
    else:
        raise ValueError(f"Dataset '{name}': unsupported format '{spec['format']}'.")
    items = _validate(items, name)
    return items[:limit] if limit else items


# ---------------------------------------------------------------- 2. prompts

def build_messages(item, eval_cfg, thinking):
    """Build the chat messages for one question: optional system prompt plus user message."""
    instruction = eval_cfg["instruction_thinking"] if thinking else eval_cfg["instruction_direct"]
    options = "\n".join(f"{letter}. {text}" for letter, text in item.options.items())
    user = f"{instruction}\n\nQuestion: {item.question}\n\nOptions:\n{options}"
    if not thinking:
        user += "\n\nAnswer:"
    messages = []
    if eval_cfg.get("system_prompt"):
        messages.append({"role": "system", "content": eval_cfg["system_prompt"]})
    messages.append({"role": "user", "content": user})
    return messages


# ---------------------------------------------------------------- 3. answer parsing

THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning(text):
    """Remove reasoning blocks so that only the final response is parsed."""
    text = THINK_BLOCK.sub("", text)
    return UNCLOSED_THINK.sub("", text).strip()


def extract_answer(text, valid_letters):
    """Return the chosen letter, or None if no unambiguous answer is found.

    Rules, applied in order:
      1. the last explicit statement such as "Answer: B" or "the answer is (B)"
      2. a letter at the very start of the response, e.g. "B", "B.", "(B)"
      3. a single distinct standalone letter anywhere in the response
    """
    letters = "".join(sorted(valid_letters))
    response = strip_reasoning(text)
    if not response:
        return None

    explicit = re.findall(
        rf"answer\s*(?:is|:)?\s*(?:option\s*)?\(?\**([{letters}])\b", response, flags=re.IGNORECASE
    )
    if explicit:
        return explicit[-1].upper()

    leading = re.match(rf"^\s*[\(\*]*([{letters}])(?:[\)\.\:\*,]|\s|$)", response)
    # A leading "A" followed by a lowercase word is the English article, not an option.
    if leading and not re.match(r"^\s*A\s+[a-z]", response):
        return leading.group(1)

    without_articles = re.sub(r"(?<![A-Za-z])A(?=\s+[a-z])", "", response)
    standalone = set(re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", without_articles))
    if len(standalone) == 1:
        return standalone.pop()
    return None


# ---------------------------------------------------------------- 4. backends

class VLLMBackend:
    """Thin wrapper around vLLM offline inference."""

    def __init__(self, model_cfg):
        from vllm import LLM

        start = time.perf_counter()
        self.llm = LLM(
            model=model_cfg["hf_id"],
            dtype=model_cfg["dtype"],
            max_model_len=model_cfg["max_model_len"],
            gpu_memory_utilization=model_cfg["gpu_memory_utilization"],
            tensor_parallel_size=model_cfg["tensor_parallel_size"],
            seed=model_cfg["seed"],
            **model_cfg["llm_kwargs"],
        )
        self.load_seconds = time.perf_counter() - start
        self.chat_template_kwargs = model_cfg["chat_template_kwargs"]
        self.n_gpus = model_cfg["tensor_parallel_size"]

    def generate(self, conversations, temperature, max_tokens, seed):
        from vllm import SamplingParams

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed)
        kwargs = {"use_tqdm": True}
        if self.chat_template_kwargs:
            kwargs["chat_template_kwargs"] = self.chat_template_kwargs
        outputs = self.llm.chat(conversations, params, **kwargs)
        results = []
        for out in outputs:
            completion = out.outputs[0]
            results.append({
                "response": completion.text,
                "input_tokens": len(out.prompt_token_ids),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            })
        return results

    @staticmethod
    def environment():
        import torch
        import vllm

        info = {"vllm_version": vllm.__version__, "torch_version": torch.__version__}
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count_visible"] = torch.cuda.device_count()
        return info


class MockBackend:
    """Random answers with plausible token counts, for testing the pipeline end to end."""

    def __init__(self, model_cfg):
        self.load_seconds = 0.0
        self.n_gpus = model_cfg["tensor_parallel_size"]
        self.rng = random.Random(f"mock-{model_cfg['key']}")

    def generate(self, conversations, temperature, max_tokens, seed):
        results = []
        for conversation in conversations:
            text = conversation[-1]["content"]
            letters = [line[0] for line in text.splitlines() if len(line) > 2 and line[1] == "."]
            results.append({
                "response": self.rng.choice(letters or ["A"]),
                "input_tokens": len(text.split()) * 4 // 3,
                "output_tokens": 2,
                "finish_reason": "stop",
            })
        time.sleep(0.01)
        return results

    @staticmethod
    def environment():
        return {"backend": "mock"}


# ---------------------------------------------------------------- 5. main

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Model key from configs/models.yaml.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset keys from configs/datasets.yaml (default: every non-optional "
                             "dataset whose input file exists).")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items per dataset.")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm",
                        help="'mock' produces random answers to test the pipeline without a GPU.")
    parser.add_argument("--latency-sample", type=int, default=0,
                        help="After the batch run, answer the first N questions one at a time to "
                             "measure per-question latency (0 disables).")
    return parser.parse_args()


def main():
    args = parse_args()
    models = load_models()
    if args.model not in models:
        raise SystemExit(f"Unknown model '{args.model}'. Available: {sorted(models)}")
    model_cfg = models[args.model]

    ds_cfg = load_datasets_config()
    eval_cfg = ds_cfg["evaluation"]
    dataset_keys = args.datasets or default_datasets(ds_cfg)
    if not dataset_keys:
        raise SystemExit("No datasets available. Run python src/prepare_data.py first.")
    unknown = [d for d in dataset_keys if d not in ds_cfg["datasets"]]
    if unknown:
        raise SystemExit(f"Unknown datasets {unknown}. Available: {sorted(ds_cfg['datasets'])}")

    # Load all datasets before the model, so that data errors surface before GPU time is spent.
    datasets = {key: load_dataset(key, ds_cfg["datasets"][key], args.limit) for key in dataset_keys}

    thinking = bool(model_cfg["thinking"])
    max_tokens = eval_cfg["max_tokens_thinking"] if thinking else eval_cfg["max_tokens_direct"]

    backend = VLLMBackend(model_cfg) if args.backend == "vllm" else MockBackend(model_cfg)
    environment = backend.environment()
    environment["hostname"] = platform.node()

    out_dir = resolve_path(args.out) / "raw" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, items in datasets.items():
        conversations = [build_messages(item, eval_cfg, thinking) for item in items]
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        outputs = backend.generate(conversations, eval_cfg["temperature"], max_tokens, model_cfg["seed"])
        elapsed = time.perf_counter() - t0

        n_correct = 0
        with open(out_dir / f"{key}.jsonl", "w", encoding="utf-8") as handle:
            for item, out in zip(items, outputs):
                predicted = extract_answer(out["response"], item.options.keys())
                correct = predicted == item.answer
                n_correct += correct
                handle.write(json.dumps({
                    "model": args.model,
                    "dataset": key,
                    "item_id": item.item_id,
                    "year": item.year,
                    "n_options": item.n_options,
                    "answer": item.answer,
                    "predicted": predicted,
                    "correct": correct,
                    "input_tokens": out["input_tokens"],
                    "output_tokens": out["output_tokens"],
                    "finish_reason": out["finish_reason"],
                    "response": out["response"],
                }, ensure_ascii=False) + "\n")

        # Latency: single requests, so that the time per question is not shared with a batch.
        latencies = []
        for conversation in conversations[: max(args.latency_sample, 0)]:
            t_single = time.perf_counter()
            backend.generate([conversation], eval_cfg["temperature"], max_tokens, model_cfg["seed"])
            latencies.append(time.perf_counter() - t_single)

        meta = {
            "model": args.model,
            "model_config": model_cfg,
            "dataset": key,
            "dataset_config": ds_cfg["datasets"][key],
            "n_items": len(items),
            "limit": args.limit,
            "backend": args.backend,
            "thinking": thinking,
            "temperature": eval_cfg["temperature"],
            "max_tokens": max_tokens,
            "generation_seconds": elapsed,
            "n_gpus": backend.n_gpus,
            "gpu_seconds": elapsed * backend.n_gpus,
            "latency_seconds": latencies,
            "median_latency_seconds": sorted(latencies)[len(latencies) // 2] if latencies else None,
            "model_load_seconds": backend.load_seconds,
            "started_utc": started,
            "environment": environment,
            "evaluation_config": eval_cfg,
        }
        with open(out_dir / f"{key}.meta.json", "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, default=str)

        print(f"[{args.model}] {key}: {n_correct}/{len(items)} correct, "
              f"{elapsed:.1f} s generation time")


if __name__ == "__main__":
    main()
