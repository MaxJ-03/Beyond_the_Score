"""LLMSanitize memorization tests ported to our vLLM: guided completion, masked-option guessing and Min-K%.
Usage: python contamination/memorization_tests.py run --model <key> --datasets <keys> | summary"""

import argparse
import json
import random
import zlib

import numpy as np
import pandas as pd

from common import PROJECT, SPLIT_FILE, load_dataset, load_datasets_config, load_llm, load_models

ROOT = PROJECT / "results" / "llmsanitize"
METHODS = ["guided-prompting", "ts-guessing-question-multichoice", "min-prob"]
MINK_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SCORE = "Min_20.0% Prob"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Classification-task templates of Golchin and Surdeanu (2024), as vendored by LLMSanitize
# (github.com/shahriargolchin/time-travel-in-llms, src/prompts/*_instructions.py).
GENERAL = """INSTRUCTION:
Finish the SECOND PIECE based on the FIRST PIECE, such that these two pieces become a single instance with the following LABEL.
LABEL: {label}
FIRST PIECE:
{first_piece}
SECOND PIECE:"""
GUIDED = """INSTRUCTION:
You are provided with the FIRST PIECE of an instance from the {split_name} split of the {dataset_name} dataset.
Finish the SECOND PIECE of the instance as EXACTLY appeared in the dataset.
ONLY rely on the original form of the instance in the dataset to finish the SECOND PIECE.
LABEL: {label}
FIRST PIECE:
{first_piece}
SECOND PIECE:"""


def dataset_name(key):
    if key == "medqa":
        return "MedQA (USMLE)"
    year = key.split("_")[1]
    return f"MIR {year} (Spanish medical residency exam)" + ("" if key.endswith("_es") else " English translation")


def fill(template, values):  # LLMSanitize utils.fill_template
    for k, v in values.items():
        template = template.replace("{" + k + "}", v)
    return template


class Model:
    def __init__(self, key):
        self.cfg = load_models()[key]
        self.llm = load_llm(self.cfg)
        self.tok = self.llm.get_tokenizer()

    def ids(self, text):
        """Chat template applied once (it already contains the begin-of-text token)."""
        templated = self.tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, **self.cfg["chat_template_kwargs"])
        return self.tok(templated, add_special_tokens=False)["input_ids"]

    def generate(self, texts, max_tokens=128):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=self.cfg["seed"],
                                **self.cfg["sampling_kwargs"])
        outs = self.llm.generate([{"prompt_token_ids": self.ids(t)} for t in texts], params)
        return [o.outputs[0].text for o in outs]

    def logprobs(self, texts):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        outs = self.llm.generate([{"prompt_token_ids": self.ids(t)} for t in texts], params)
        return [[e[o.prompt_token_ids[i]].logprob for i, e in enumerate(o.prompt_logprobs) if e is not None]
                for o in outs]


def sample(items, n, seed=42):
    items = list(items)
    random.Random(seed).shuffle(items)
    return items if n <= 0 else items[:n]


# ---------------------------------------------------------------- the three tests
def min_prob(model, items):
    texts = [f"{it.question} | {it.options} | {it.answer}" for it in items]
    rows = []
    for it, text, lp, lp_low in zip(items, texts, model.logprobs(texts), model.logprobs([t.lower() for t in texts])):
        ll = float(np.mean(lp))
        ppl, ppl_low = float(np.exp(-ll)), float(np.exp(-np.mean(lp_low)))
        pred = {"ppl": ppl, "mean_loglik": ll, "ppl/lowercase_ppl": float(-(np.log(ppl_low) / np.log(ppl))),
                "ppl/zlib": float(np.log(ppl) / len(zlib.compress(bytes(text, "utf-8"))))}
        for r in MINK_RATIOS:
            k = int(len(lp) * r)
            pred[f"Min_{r * 100}% Prob"] = float(-np.mean(np.sort(lp)[:k])) if k > 0 else float("nan")
        rows.append({"item_id": it.item_id, "year": it.year, "n_tokens": len(lp), "pred": pred})
    return rows, {}


def guided_completion(model, items, key):
    from rouge_score import rouge_scorer
    from scipy.stats import bootstrap

    pieces = []
    for it in items:
        choices = [f"{c}.{t}" for c, t in it.options.items()]  # ARC split of LLMSanitize: label + '.' + text
        pieces.append((it, it.question + "\n" + choices[0], "\n".join(choices[1:])))
    values = [{"split_name": "test", "dataset_name": dataset_name(key), "first_piece": p1, "label": it.answer}
              for it, p1, _ in pieces]
    general = model.generate([fill(GENERAL, v) for v in values])
    guided = model.generate([fill(GUIDED, v) for v in values])
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows = [{"item_id": it.item_id, "year": it.year, "first_part": p1, "second_part": p2,
             "general_response": gen, "guided_response": gui,
             "general_score": scorer.score(p2, gen)["rougeL"].fmeasure,
             "guided_score": scorer.score(p2, gui)["rougeL"].fmeasure}
            for (it, p1, p2), gen, gui in zip(pieces, general, guided) if gen and gui]
    diffs = np.array([r["guided_score"] - r["general_score"] for r in rows])
    res = bootstrap((diffs,), np.mean, n_resamples=10000, random_state=0)
    return rows, {"mean_diff": float(diffs.mean()), "std_diff": float(diffs.std()),
                  "p_value_diff_le_0": float((res.bootstrap_distribution <= 0.0).sum() / 10000.0),
                  "mean_guided": float(np.mean([r["guided_score"] for r in rows])),
                  "mean_general": float(np.mean([r["general_score"] for r in rows]))}


def distinct_options(items):
    """LLMSanitize filter: drop yes/no questions and questions with near-identical options."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    kept = []
    for it in items:
        c = list(it.options.values())
        if len(c) == 2 and {x.lower() for x in c} <= {"yes", "no", "true", "false"}:
            continue
        if any(scorer.score(c[i], c[j])["rougeLsum"].fmeasure >= 0.65 for i in range(len(c)) for j in range(i + 1, len(c))):
            continue
        kept.append(it)
    return kept


def masked_option(model, items):
    from nltk.tokenize import sent_tokenize
    from rouge_score import rouge_scorer

    built = []
    for k, it in enumerate(items):
        choices = list(it.options.values())
        answer = list(it.options).index(it.answer)
        wrong = [i for i in range(len(choices)) if i != answer]
        hidden = wrong[np.random.default_rng(k).integers(len(wrong))]
        prompt = (f"Please fill in the [] in option {LETTERS[hidden]} based on your benchmark knowledge."
                  "\n\nThe crucial rule is that you should provide different answer in other options below."
                  f"\n\nQuestion: {it.question}\nOptions:")
        for i, c in enumerate(choices):
            prompt += f"\n{LETTERS[i]}: [MASK]" if i == hidden else f"\n{LETTERS[i]}: [{c}]"
        prompt += "\n\nReply with answer only."
        built.append((it, prompt, choices[answer], choices[hidden], LETTERS[hidden]))
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    rows = []
    for (it, prompt, correct, masked, letter), raw in zip(built, model.generate([b[1] for b in built])):
        resp = raw
        if letter + ":" in resp:  # LLMSanitize response processing
            sents = sent_tokenize(resp.split(letter + ":")[1])
            resp = sents[0] if sents else resp.split(letter + ":")[1]
        resp = resp.strip()
        rows.append({"item_id": it.item_id, "year": it.year, "masked_letter": letter, "masked_option": masked,
                     "response": resp, "raw_response": raw[:300], "em": int(resp.lower() == masked.lower()),
                     "rougeL": scorer.score(resp.lower(), masked.lower())["rougeLsum"].fmeasure,
                     "em_vs_correct_answer": int(resp.lower() == correct.lower())})
    return rows, {"em": float(np.mean([r["em"] for r in rows])), "rougeL": float(np.mean([r["rougeL"] for r in rows])),
                  "em_vs_correct_answer_llmsanitize": float(np.mean([r["em_vs_correct_answer"] for r in rows]))}


def run(model_key, dataset_keys, methods, n):
    ds_cfg = load_datasets_config()["datasets"]
    model = Model(model_key)
    for key in dataset_keys:
        items = load_dataset(key, ds_cfg[key])
        for method in methods:
            if method == "min-prob":
                rows, summary = min_prob(model, sample(items, n))
            elif method == "guided-prompting":
                rows, summary = guided_completion(model, sample(items, n), key)
            else:
                rows, summary = masked_option(model, sample(distinct_options(items), n))
            out = ROOT / method / model_key
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{key}.jsonl").write_text("".join(json.dumps({"model": model_key, "dataset": key, **r},
                                                                 ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
            summary = {"model": model_key, "dataset": key, "method": method, "n": len(rows), **summary}
            (out / f"{key}.summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary))


# ---------------------------------------------------------------- summary tables
def roc_auc(seen, ref):
    """AUC as in LLMSanitize (roc on -score): > 0.5 = the possibly seen questions are more familiar."""
    pos, neg = -np.asarray(seen), -np.asarray(ref)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def rows_of(method, model, datasets):
    files = [f for d in datasets if (f := ROOT / method / model / f"{d}.jsonl").exists()]
    return pd.concat([pd.read_json(f, lines=True, dtype={"item_id": str}) for f in files], ignore_index=True) \
        if files else None


def summaries(method):
    files = sorted((ROOT / method).glob("*/*.summary.json")) if (ROOT / method).exists() else []
    return pd.DataFrame([json.loads(f.read_text()) for f in files])


def summary():
    out = PROJECT / "results" / "analysis" / "memorization_tests"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tables = {}

    gp = summaries("guided-prompting")
    if not gp.empty:
        tables["guided_completion.csv"] = gp[~gp.model.str.startswith("planted-")][
            ["model", "dataset", "n", "mean_guided", "mean_general", "mean_diff", "p_value_diff_le_0"]].round(3)
    ts = summaries("ts-guessing-question-multichoice")
    if not ts.empty:
        tables["masked_option.csv"] = ts[~ts.model.str.startswith("planted-")][
            ["model", "dataset", "n", "em", "rougeL", "em_vs_correct_answer_llmsanitize"]].round(3)

    mk = []
    for model in sorted(p.name for p in (ROOT / "min-prob").glob("*") if not p.name.startswith("planted-")) \
            if (ROOT / "min-prob").exists() else []:
        a, b = rows_of("min-prob", model, ["medqa"]), rows_of("min-prob", model, ["mir_2026"])
        if a is None or b is None:
            continue
        x, y = a.pred.map(lambda p: p[SCORE]).to_numpy(), b.pred.map(lambda p: p[SCORE]).to_numpy()
        boots = [roc_auc(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))]) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        mk.append({"model": model, "comparison": "MedQA vs MIR 2026 (EN)", "auc": round(roc_auc(x, y), 3),
                   "ci_low": round(lo, 3), "ci_high": round(hi, 3), "n_medqa": len(x), "n_mir2026": len(y)})
    if mk:
        tables["mink_medqa_vs_mir2026.csv"] = pd.DataFrame(mk)

    if SPLIT_FILE.exists():  # known positive: planted checkpoints, exposed vs held-out questions
        split = pd.read_csv(SPLIT_FILE, dtype={"item_id": str}).set_index("item_id")["exposed"]
        years = [f"mir_{y}_es" for y in sorted(pd.read_csv(SPLIT_FILE).year.unique())]
        cal = []
        for model in sorted({p.name for m in ROOT.glob("*") for p in m.glob("planted-*")}):
            row = {"model": model}
            for method, name in [("min-prob", "mink_auc"), ("guided-prompting", "guided_diff"),
                                 ("ts-guessing-question-multichoice", "masked_em")]:
                d = rows_of(method, model, years)
                if d is None:
                    continue
                d = d[d.item_id.isin(split.index)]
                exp = d.item_id.map(split) == 1
                if method == "min-prob":
                    s = d.pred.map(lambda p: p[SCORE])
                    row[name] = round(roc_auc(s[exp], s[~exp]), 3)
                else:
                    v = d.guided_score - d.general_score if method == "guided-prompting" else d.em
                    row[f"{name}_exposed"], row[f"{name}_heldout"] = round(v[exp].mean(), 3), round(v[~exp].mean(), 3)
            cal.append(row)
        if cal:
            tables["planted_known_positive.csv"] = pd.DataFrame(cal)

    pd.set_option("display.width", 200)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
        print(f"\n{name}\n{table.to_string(index=False)}")
    print(f"\nWrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--datasets", nargs="+", required=True)
    r.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    r.add_argument("--n", type=int, default=100, help="questions per dataset (LLMSanitize default 100; 0 = all)")
    sub.add_parser("summary")
    args = parser.parse_args()
    if args.command == "run":
        run(args.model, args.datasets, args.methods, args.n)
    else:
        summary()


if __name__ == "__main__":
    main()
