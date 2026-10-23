"""Grade open-ended answers (LiveMedBench rubrics, K-QA statements) with the LLM judge.
Usage: python src/judge.py [--models ...] [--datasets ...] [--force | --validate | --rescore]"""

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone

import pandas as pd

from config import load_datasets_config, load_models, load_yaml, resolve_path
from items import load_dataset
from run_eval import strip_reasoning

OPEN_TASKS = ("rubric", "statements")


# ---------------------------------------------------------------- prompts and verdicts

def fill(template, **values):
    """Fill {placeholders} by plain replacement, so that JSON braces in a template stay intact."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value or "").strip())
    return template


def final_text(output):
    """The judge's final answer, without the reasoning channel of gpt-oss models."""
    for marker in ("<|channel|>final<|message|>", "assistantfinal"):
        if marker in output:
            return output.rsplit(marker, 1)[1].strip()
    return output.strip()


def parse_met(text):
    """Verdict of the LiveMedBench prompt: the "met" field of the JSON answer."""
    try:
        parsed = json.loads(text[text.index("["):text.rindex("]") + 1])
        met = parsed[0]["met"]
        if isinstance(met, bool):
            return met
        if str(met).strip().lower() in ("true", "false"):
            return str(met).strip().lower() == "true"
    except (ValueError, IndexError, KeyError, TypeError):
        pass
    match = re.search(r'"met"\s*:\s*"?(true|false)', text, flags=re.IGNORECASE)
    return match.group(1).lower() == "true" if match else None


def parse_true_false(text):
    """Verdict of the K-QA prompts: "True" or "False" in the last 20 characters (official rule),
    otherwise the last standalone "true" or "false" in the text."""
    tail = text.lower()[-20:]
    if "true" in tail:
        return True
    if "false" in tail:
        return False
    words = re.findall(r"\b(true|false)\b", text, flags=re.IGNORECASE)
    return words[-1].lower() == "true" if words else None


def build_checks(item, response, prompts):
    """All checks for one answer: one per rubric criterion, or one per statement and relation."""
    checks = []
    if item.task == "rubric":
        user_query = f"{item.question}\n\n{item.extra.get('request', '')}"
        for rubric in item.extra["rubric"]:
            checks.append({"kind": "rubric", "statement": rubric["criterion"], "points": rubric["points"],
                           "prompt": fill(prompts["rubric"], user_query=user_query,
                                          model_response=response, criterion=rubric["criterion"])})
    else:
        statements = [("entails", s) for s in item.extra["must_have"]]
        statements += [("contradicts", s) for s in item.extra["must_have"] + item.extra.get("nice_to_have", [])]
        for kind, statement in statements:
            checks.append({"kind": kind, "statement": statement,
                           "prompt": fill(prompts[kind], question=item.question,
                                          llm_answer=response, answer=statement)})
    return checks


def score_item(task, checks):
    """Item score from the verdicts, following the scoring rules in the module docstring."""
    if task == "rubric":
        maximum = sum(c["points"] for c in checks if c["points"] > 0)
        earned = sum(c["points"] for c in checks if c["verdict"])
        return {"score": min(max(earned / maximum, 0.0), 1.0) if maximum else 0.0}
    entails = [c["verdict"] for c in checks if c["kind"] == "entails"]
    return {"score": sum(entails) / len(entails) if entails else 0.0,
            "hallucinations": sum(c["verdict"] for c in checks if c["kind"] == "contradicts")}


# ---------------------------------------------------------------- backends

class MockJudge:
    """Random verdicts in the format of each prompt, for testing without a GPU."""

    n_gpus = 1

    def __init__(self, judge_cfg):
        self.rng = random.Random("mock-judge")

    def generate(self, conversations, temperature, max_tokens, seed):
        outputs = []
        for conversation in conversations:
            verdict = self.rng.random() < 0.5
            if '"met"' in conversation[-1]["content"]:
                text = f'[{{"question": "mock", "met": {str(verdict).lower()}, "reasoning": "mock"}}]'
            else:
                text = f"Mock reasoning. The answer is {verdict}."
            outputs.append({"response": text, "input_tokens": 0, "output_tokens": 0, "finish_reason": "stop"})
        return outputs

    @staticmethod
    def environment():
        return {"backend": "mock"}


def load_backend(name, judge_cfg):
    if name == "mock":
        return MockJudge(judge_cfg)
    from run_eval import VLLMBackend

    return VLLMBackend(judge_cfg)


def run_judge(backend, prompts, settings):
    """Final answers of the judge for a list of prompts (one user message each)."""
    conversations = [[{"role": "user", "content": p}] for p in prompts]
    outputs = backend.generate(conversations, settings["temperature"], settings["max_tokens"], 0)
    return [final_text(o["response"]) for o in outputs]


# ---------------------------------------------------------------- grading

def grade(model, dataset, spec, backend, settings, prompts, out_dir, limit):
    """Grade all answers of one model on one open-ended dataset and write the results."""
    items = {item.item_id: item for item in load_dataset(dataset, spec)}
    answers = pd.read_json(resolve_path("results") / "raw" / model / f"{dataset}.jsonl", lines=True,
                           dtype={"item_id": str})
    answers = answers[answers["item_id"].isin(items)]  # items outside min_date are not graded
    if limit:
        answers = answers.head(limit)
    records, all_checks = [], []
    for _, row in answers.iterrows():
        # Only the final response is graded; a reasoning section (MedGemma 27B) is removed first.
        response = strip_reasoning(str(row["response"] or ""))
        checks = build_checks(items[row["item_id"]], response, prompts)
        for check in checks:
            check["empty_answer"] = not response
        records.append((row["item_id"], checks))
        all_checks.extend(c for c in checks if not c["empty_answer"])

    start = time.perf_counter()
    texts = run_judge(backend, [c["prompt"] for c in all_checks], settings)
    elapsed = time.perf_counter() - start
    parse = parse_met if spec["task"] == "rubric" else parse_true_false
    for check, text in zip(all_checks, texts):
        check["verdict"] = parse(text)
        check["judge_answer"] = text[-600:]

    out_dir.mkdir(parents=True, exist_ok=True)
    n_unparsed = 0
    with open(out_dir / f"{dataset}.jsonl", "w", encoding="utf-8") as handle:
        for item_id, checks in records:
            unparsed = sum(1 for c in checks if not c["empty_answer"] and c["verdict"] is None)
            n_unparsed += unparsed
            for check in checks:
                check["unparsed"] = not check["empty_answer"] and check.get("verdict") is None
                check["verdict"] = bool(check.get("verdict"))
                del check["prompt"]
            record = {"model": model, "dataset": dataset, "item_id": item_id,
                      **score_item(spec["task"], checks), "n_checks": len(checks),
                      "n_unparsed": unparsed, "checks": checks}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {"model": model, "dataset": dataset, "task": spec["task"], "judge": settings["model"],
            "judge_hf_id": settings["hf_id"], "judge_settings": settings, "n_items": len(records),
            "n_judge_calls": len(all_checks), "n_unparsed": n_unparsed, "judge_seconds": elapsed,
            "finished_utc": datetime.now(timezone.utc).isoformat(), "environment": backend.environment()}
    (out_dir / f"{dataset}.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    scores = pd.read_json(out_dir / f"{dataset}.jsonl", lines=True)["score"]
    print(f"[{model}] {dataset}: {len(records)} answers, {len(all_checks)} judge calls, "
          f"mean score {scores.mean():.3f}, {n_unparsed} unparsed, {elapsed:.0f} s")


def answers_newer_than_grades(answers, judged):
    """True if the answers were generated after the grades were made. Uses the times recorded in
    the .meta.json files, because file modification times change when Git checks files out."""
    try:
        started = json.loads(answers.with_suffix(".meta.json").read_text(encoding="utf-8"))["started_utc"]
        graded = json.loads(judged.with_suffix(".meta.json").read_text(encoding="utf-8"))["finished_utc"]
    except (OSError, KeyError, ValueError):
        return False
    return started > graded


def pending_pairs(models, datasets, ds_cfg, force):
    """(model, dataset) pairs with answers and, unless force is set, without grades."""
    raw = resolve_path("results") / "raw"
    pairs = []
    for model_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        if models and model_dir.name not in models:
            continue
        for dataset, spec in ds_cfg["datasets"].items():
            if spec.get("task") not in OPEN_TASKS or (datasets and dataset not in datasets):
                continue
            answers = model_dir / f"{dataset}.jsonl"
            judged = resolve_path("results") / "judged" / model_dir.name / f"{dataset}.jsonl"
            # Answers generated after their grades were evaluated again and are graded again.
            outdated = judged.exists() and answers.exists() and answers_newer_than_grades(answers, judged)
            if answers.exists() and (force or not judged.exists() or outdated):
                pairs.append((model_dir.name, dataset))
    return pairs


def rescore(models, datasets, ds_cfg):
    """Read the stored K-QA verdicts again with parse_true_false and recompute the scores."""
    root = resolve_path("results") / "judged"
    for path in sorted(root.glob("*/*.jsonl")):
        model, dataset = path.parent.name, path.stem
        spec = ds_cfg["datasets"].get(dataset, {})
        if spec.get("task") != "statements" or (models and model not in models) \
                or (datasets and dataset not in datasets):
            continue
        records = pd.read_json(path, lines=True, dtype={"item_id": str}).to_dict("records")
        before = sum(r["n_unparsed"] for r in records)
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                for check in record["checks"]:
                    verdict = None if check["empty_answer"] else parse_true_false(check["judge_answer"])
                    check["unparsed"] = not check["empty_answer"] and verdict is None
                    check["verdict"] = bool(verdict)
                record.update(score_item("statements", record["checks"]))
                record["n_unparsed"] = sum(c["unparsed"] for c in record["checks"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        after = sum(r["n_unparsed"] for r in records)
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(n_unparsed=after, rescored_utc=datetime.now(timezone.utc).isoformat())
            meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        score = sum(r["score"] for r in records) / len(records)
        print(f"[{model}] {dataset}: unparsed {before} -> {after}, mean score {score:.3f}")


# ---------------------------------------------------------------- validation

def cohen_kappa(a, b):
    a, b = pd.Series(list(a)), pd.Series(list(b))
    observed = (a == b).mean()
    expected = sum(a.value_counts(normalize=True).get(k, 0) * b.value_counts(normalize=True).get(k, 0)
                   for k in set(a) | set(b))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def validate(backend, settings, prompts, limit):
    """Agreement of the judge with the physicians' majority labels on K-QA statements."""
    data = pd.read_csv(resolve_path(settings["validation_file"]))
    if limit:
        data = data.head(limit)
    rows = data.to_dict("records")

    def prompts_for(kind):
        return [fill(prompts[kind], question=r["Question"], llm_answer=r["premise"], answer=r["hypothesis"])
                for r in rows]
    entails = [parse_true_false(t) for t in run_judge(backend, prompts_for("entails"), settings)]
    contradicts = [parse_true_false(t) for t in run_judge(backend, prompts_for("contradicts"), settings)]
    data["judge_entails"], data["judge_contradicts"] = entails, contradicts
    data["judge_label"] = ["Contradiction" if c and not e else "Entailment" if e and not c else "Neutral"
                           for e, c in zip(entails, contradicts)]

    physicians = data["majority_label"]
    pairs = [("label_0", "label_1"), ("label_0", "label_2"), ("label_1", "label_2")]
    summary = pd.DataFrame([
        {"comparison": "judge vs physician majority, 3 labels", "n": len(data),
         "agreement": (data["judge_label"] == physicians).mean(),
         "cohen_kappa": cohen_kappa(data["judge_label"], physicians)},
        {"comparison": "entailed or not (comprehensiveness decision)", "n": len(data),
         "agreement": (data["judge_entails"].fillna(False) == (physicians == "Entailment")).mean(),
         "cohen_kappa": cohen_kappa(data["judge_entails"].fillna(False), physicians == "Entailment")},
        {"comparison": "contradicted or not (hallucination decision)", "n": len(data),
         "agreement": (data["judge_contradicts"].fillna(False) == (physicians == "Contradiction")).mean(),
         "cohen_kappa": cohen_kappa(data["judge_contradicts"].fillna(False), physicians == "Contradiction")},
        {"comparison": "physician vs physician, 3 labels (mean of pairs)", "n": len(data),
         "agreement": sum((data[a] == data[b]).mean() for a, b in pairs) / 3,
         "cohen_kappa": sum(cohen_kappa(data[a], data[b]) for a, b in pairs) / 3},
    ]).round(3)
    out_dir = resolve_path("results") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "judge_validation.csv", index=False)
    data.to_csv(out_dir / "judge_validation_items.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Unparsed verdicts: {sum(v is None for v in entails + contradicts)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="Evaluated models to grade (default: all).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Open-ended datasets (default: all).")
    parser.add_argument("--force", action="store_true", help="Grade again, overwriting existing grades.")
    parser.add_argument("--validate", action="store_true", help="Only run the judge validation.")
    parser.add_argument("--rescore", action="store_true",
                        help="Read the stored K-QA verdicts again and recompute the scores (no GPU).")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--limit", type=int, default=None, help="Grade only the first N answers (testing).")
    args = parser.parse_args()

    ds_cfg = load_datasets_config()
    if args.rescore:
        rescore(args.models, args.datasets, ds_cfg)
        return
    settings = dict(ds_cfg["judge"])
    judge_cfg = load_models()[settings["model"]]
    settings.update(hf_id=judge_cfg["hf_id"], chat_template_kwargs=judge_cfg["chat_template_kwargs"])
    prompts = load_yaml("judge_prompts.yaml")

    pairs = [] if args.validate else pending_pairs(args.models, args.datasets, ds_cfg, args.force)
    if not args.validate and not pairs:
        print("Nothing to grade: no open-ended answers without grades (use --force to regrade).")
        return
    backend = load_backend(args.backend, judge_cfg)
    if args.validate:
        validate(backend, settings, prompts, args.limit)
        return
    for model, dataset in pairs:
        out_dir = resolve_path("results") / "judged" / model
        grade(model, dataset, ds_cfg["datasets"][dataset], backend, settings, prompts, out_dir, args.limit)


if __name__ == "__main__":
    main()
