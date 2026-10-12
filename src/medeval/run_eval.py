"""Run one model on one or more multiple-choice datasets.

The model is loaded once and evaluated on every requested dataset. For each question
the script records the raw response, the parsed answer, correctness and token counts.
For each dataset it records the wall-clock generation time, which is converted to
GPU-seconds in the analysis.

Outputs (per model):
    results/raw/<model>/<dataset>.jsonl   one record per question
    results/raw/<model>/<dataset>.meta.json  timing, settings and software versions

Usage:
    python -m medeval.run_eval --model qwen3-4b
    python -m medeval.run_eval --model qwen3-4b --datasets medqa mir_2026 --limit 20
    python -m medeval.run_eval --model qwen3-4b --latency-sample 20
    python -m medeval.run_eval --model qwen3-4b --backend mock   (pipeline test without GPU)
"""

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone

from .config import default_datasets, load_datasets_config, load_models, resolve_path
from .data import load_dataset
from .parsing import extract_answer
from .prompts import build_messages


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
        raise SystemExit("No datasets available. Run scripts/prepare_data.py first.")
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
