"""Planted contamination: how much exposure to test questions can accuracy and Min-K% detect?

Half of the MIR 2020-2022 questions are planted in a model by LoRA fine-tuning on the question
text with its correct answer; the other half are held out. The planted effect is the accuracy
gain on planted questions minus the gain on held-out questions (both relative to the base model).
Min-K% Prob (Shi et al., 2024) is the likelihood-based detector: mean log-probability of the k%
least likely tokens of a question; text seen in training has fewer very unlikely tokens.

Commands (project folder; contamination/run.sh chains them on Snellius):
    python contamination/planted.py split                      exposed / held-out split (CPU)
    python contamination/planted.py train --model gemma3-4b    LoRA, checkpoints after 1, 3, 10 epochs (GPU)
    python contamination/planted.py mink --model gemma3-4b     Min-K% of every question (GPU)
    python contamination/planted.py analyse --base gemma3-4b   planted effect and Min-K% AUC (CPU)
The checkpoints are evaluated with `common.py eval`.

Outputs: contamination/planted_split.csv; checkpoints/planted/<model>-ep<E>[-adapter]/;
results/mink/<model>/<dataset>.jsonl; contamination/planted_out/{accuracy,mink_auc}_<base>.csv (one pair per base model)
"""

import argparse
import json
import math
import random
import re
import time

import numpy as np
import pandas as pd

from common import PROJECT, SPLIT_FILE, load_dataset, load_datasets_config, load_llm, load_models

YEARS = [2020, 2021, 2022]
LETTERS = "ABCD"
SAMPLES = 4000
# LoRA on the language model's attention and MLP projections; the vision tower is left alone.
TARGET_MODULES = (r"^(?!.*(vision_tower|multi_modal_projector)).*"
                  r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")


def mir(year):
    return pd.read_csv(PROJECT / "data" / "raw" / f"mir_{year}_es.csv", dtype={"id": str})


# ---------------------------------------------------------------- split
def split(seed=0):
    """Random half of every edition is exposed, so both halves are equally difficult."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in YEARS:
        ids = mir(year)["id"].tolist()
        exposed = set(rng.permutation(len(ids))[: len(ids) // 2])
        rows += [{"item_id": i, "year": year, "exposed": int(k in exposed)} for k, i in enumerate(ids)]
    table = pd.DataFrame(rows)
    table.to_csv(SPLIT_FILE, index=False)
    print(table.groupby(["year", "exposed"]).size().unstack())


# ---------------------------------------------------------------- training
def exam_text(row):
    """A question as it appears in a published exam with its answer key."""
    lines = [str(row["question"]).strip()]
    lines += [f"{c}. {str(row[f'option_{c.lower()}']).strip()}" for c in LETTERS]
    lines.append(f"Respuesta correcta: {row['answer']}")
    return "\n".join(lines)


def load_base(hf_id):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if AutoConfig.from_pretrained(hf_id).model_type == "gemma3":  # 4B Gemma 3 / MedGemma are multimodal
        from transformers import Gemma3ForConditionalGeneration

        return Gemma3ForConditionalGeneration.from_pretrained(hf_id, dtype=torch.bfloat16), True
    return AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16), False


def train(model_key, epochs=(1, 3, 10), lr=2e-4, batch_size=8, rank=16, max_length=768, seed=0):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    hf_id = load_models()[model_key]["hf_id"]
    out_dir = PROJECT / "checkpoints" / "planted"
    out_dir.mkdir(parents=True, exist_ok=True)

    exposed = pd.read_csv(SPLIT_FILE, dtype={"item_id": str}).query("exposed == 1")
    questions = pd.concat([mir(y) for y in YEARS], ignore_index=True)
    texts = [exam_text(r) for _, r in questions[questions["id"].isin(exposed.item_id)].iterrows()]
    assert len(texts) == len(exposed), "split and data/raw disagree"

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model, multimodal = load_base(hf_id)
    model.to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)

    log, start = [], time.time()
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = list(range(len(texts)))
        random.shuffle(order)
        losses = []
        for b in range(math.ceil(len(order) / batch_size)):
            batch = [texts[i] + tokenizer.eos_token for i in order[b * batch_size:(b + 1) * batch_size]]
            enc = {k: v.to("cuda") for k, v in tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                                          max_length=max_length).items()}
            labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
            extra = {"token_type_ids": torch.zeros_like(enc["input_ids"])} if multimodal else {}
            loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels, **extra).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        log.append({"epoch": epoch, "loss": sum(losses) / len(losses), "seconds": time.time() - start})
        print(f"epoch {epoch}: loss {log[-1]['loss']:.4f}")
        if epoch in epochs:
            model.save_pretrained(out_dir / f"{model_key}-ep{epoch}-adapter")
    del model, optimizer
    torch.cuda.empty_cache()

    # Merge every adapter into a fresh base model, so vLLM loads it as an ordinary model.
    for epoch in sorted(epochs):
        base, _ = load_base(hf_id)
        merged = PeftModel.from_pretrained(base, out_dir / f"{model_key}-ep{epoch}-adapter").merge_and_unload()
        target = out_dir / f"{model_key}-ep{epoch}"
        merged.save_pretrained(target, safe_serialization=True)
        (AutoProcessor if multimodal else AutoTokenizer).from_pretrained(hf_id).save_pretrained(target)
        print(f"saved {target}")
    (out_dir / f"{model_key}-training_log.json").write_text(json.dumps(
        {"hf_id": hf_id, "epochs": list(epochs), "lr": lr, "batch_size": batch_size, "rank": rank, "seed": seed,
         "n_exposed": len(texts), "log": log}, indent=2))


# ---------------------------------------------------------------- Min-K% Prob
def mink(model_key, dataset_keys):
    """Log-probability of every token of question + options (no answer, no instruction)."""
    from vllm import SamplingParams

    cfg = load_models()[model_key]
    ds_cfg = load_datasets_config()["datasets"]
    llm = load_llm(cfg)
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    out_dir = PROJECT / "results" / "mink" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in dataset_keys:
        items = load_dataset(key, ds_cfg[key])
        texts = ["\n".join([it.question, *(f"{c}. {t}" for c, t in it.options.items())]) for it in items]
        with open(out_dir / f"{key}.jsonl", "w", encoding="utf-8") as handle:
            for it, o in zip(items, llm.generate(texts, params)):
                lp = sorted(e[o.prompt_token_ids[i]].logprob for i, e in enumerate(o.prompt_logprobs) if e is not None)
                rec = {"model": model_key, "dataset": key, "item_id": it.item_id, "year": it.year,
                       "n_tokens": len(lp), "mean_logprob": sum(lp) / len(lp)}
                for k in (0.1, 0.2, 0.3):
                    low = lp[: max(1, int(len(lp) * k))]
                    rec[f"mink{int(k * 100)}"] = sum(low) / len(low)
                handle.write(json.dumps(rec) + "\n")
        print(f"[{model_key}] {key}: {len(items)} questions scored")


# ---------------------------------------------------------------- analysis
def answers(model, years):
    files = [f for y in years if (f := PROJECT / "results" / "raw" / model / f"mir_{y}_es.jsonl").exists()]
    if not files:
        return None
    d = pd.concat([pd.read_json(f, lines=True, dtype={"item_id": str}) for f in files])
    return d.set_index("item_id")["correct"].astype(int)


def mink_scores(model):
    files = [f for y in YEARS if (f := PROJECT / "results" / "mink" / model / f"mir_{y}_es.jsonl").exists()]
    return pd.concat([pd.read_json(f, lines=True, dtype={"item_id": str}) for f in files]).set_index("item_id") \
        if files else None


def auc(pos, neg):
    """Probability that a random exposed question scores higher than a random held-out one."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def boot_ci(stat, groups, rng):
    values = [stat(*[g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(SAMPLES)]
    return np.percentile(values, [2.5, 97.5])


def analyse(base):
    out = PROJECT / "contamination" / "planted_out"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    s = pd.read_csv(SPLIT_FILE, dtype={"item_id": str}).set_index("item_id")
    exposed, held = s.index[s.exposed == 1], s.index[s.exposed == 0]
    ckpts = sorted((int(m.group(1)), p.name) for p in (PROJECT / "results" / "raw").glob(f"planted-{base}-ep*")
                   if (m := re.search(r"-ep(\d+)$", p.name)))
    models = [(0, base), *ckpts]
    ref = answers(base, YEARS)
    if ref is None:
        raise SystemExit(f"No results for {base} in results/raw/.")

    rows = []
    for epoch, model in models:
        acc = answers(model, YEARS)
        if acc is None:
            continue
        e = (acc.reindex(exposed) - ref.reindex(exposed)).dropna().to_numpy()
        h = (acc.reindex(held) - ref.reindex(held)).dropna().to_numpy()
        lo, hi = boot_ci(lambda a, b: a.mean() - b.mean(), [e, h], rng) if epoch else (0.0, 0.0)
        new = answers(model, [2026])
        rows.append({"model": model, "epochs": epoch, "acc_exposed": round(100 * acc.reindex(exposed).mean(), 1),
                     "acc_held_out": round(100 * acc.reindex(held).mean(), 1),
                     "planted_effect": round(100 * (e.mean() - h.mean()), 1),
                     "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1),
                     "acc_mir_2026": round(100 * new.mean(), 1) if new is not None else None,
                     "n_exposed": len(e), "n_held_out": len(h)})
    accuracy = pd.DataFrame(rows)
    accuracy.to_csv(out / f"accuracy_{base}.csv", index=False)  # per base, so runs do not overwrite each other

    mrows, ref_m = [], mink_scores(base)
    for epoch, model in models:
        m = mink_scores(model)
        if m is None:
            continue
        for score in ("mink20", "mean_logprob"):
            pos, neg = m[score].reindex(exposed).dropna().to_numpy(), m[score].reindex(held).dropna().to_numpy()
            row = {"model": model, "epochs": epoch, "score": score, "auc": round(auc(pos, neg), 3)}
            row["ci_low"], row["ci_high"] = np.round(boot_ci(auc, [pos, neg], rng), 3)
            if ref_m is not None and epoch:  # change relative to the base model on the same question
                d = m[score] - ref_m[score]
                row["auc_vs_base"] = round(auc(d.reindex(exposed).dropna(), d.reindex(held).dropna()), 3)
            mrows.append(row)
    mink_auc = pd.DataFrame(mrows)
    if not mink_auc.empty:
        mink_auc.to_csv(out / f"mink_auc_{base}.csv", index=False)

    pd.set_option("display.width", 180)
    print("Accuracy (%); planted effect = gain on exposed minus gain on held-out questions (pp, 95% CI)")
    print(accuracy.to_string(index=False))
    if not mink_auc.empty:
        print("\nMin-K% Prob, exposed vs held-out questions (AUC; 0.5 = chance)")
        print(mink_auc.to_string(index=False))
    print(f"\nWrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("split")
    tr = sub.add_parser("train")
    tr.add_argument("--model", default="gemma3-4b")
    tr.add_argument("--epochs", nargs="+", type=int, default=[1, 3, 10])
    mk = sub.add_parser("mink")
    mk.add_argument("--model", required=True)
    mk.add_argument("--datasets", nargs="+", default=[f"mir_{y}_es" for y in (*YEARS, 2026)])
    an = sub.add_parser("analyse")
    an.add_argument("--base", default="gemma3-4b")
    args = parser.parse_args()
    if args.command == "split":
        split()
    elif args.command == "train":
        train(args.model, tuple(args.epochs))
    elif args.command == "mink":
        mink(args.model, args.datasets)
    else:
        analyse(args.base)


if __name__ == "__main__":
    main()
