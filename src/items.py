"""Questions in a common format (Item) and the prompts that present them to a model."""

import json
from dataclasses import dataclass, field

import pandas as pd

from config import load_models, resolve_path

TASKS = ("multiple_choice", "rubric", "statements")


@dataclass
class Item:
    item_id: str
    dataset: str
    question: str
    options: dict = field(default_factory=dict)  # letter -> option text, in display order
    answer: str = ""  # correct letter (multiple choice only)
    year: str = ""
    task: str = "multiple_choice"
    extra: dict = field(default_factory=dict)  # task-specific fields of open-ended items

    @property
    def n_options(self):
        return len(self.options)


# ---------------------------------------------------------------- loading

def _validate(items, name):
    if not items:
        raise ValueError(f"Dataset '{name}' contains no items.")
    seen = set()
    for item in items:
        if item.item_id in seen:
            raise ValueError(f"Duplicate item id '{item.item_id}' in dataset '{name}'.")
        seen.add(item.item_id)
        if not item.question.strip():
            raise ValueError(f"Item '{item.item_id}' in '{name}' has an empty question.")
        if item.task == "multiple_choice":
            if item.answer not in item.options:
                raise ValueError(
                    f"Item '{item.item_id}' in '{name}': answer '{item.answer}' is not one of "
                    f"the options {list(item.options)}."
                )
            if any(not str(text).strip() for text in item.options.values()):
                raise ValueError(f"Item '{item.item_id}' in '{name}' has an empty option.")
        elif item.task == "rubric" and not item.extra.get("rubric"):
            raise ValueError(f"Item '{item.item_id}' in '{name}' has no rubric items.")
        elif item.task == "statements" and not item.extra.get("must_have"):
            raise ValueError(f"Item '{item.item_id}' in '{name}' has no must-have statements.")
    return items


def _load_jsonl(name, path, task):
    items = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            item = Item(item_id=str(record.get("id", line_number)), dataset=name,
                        question=str(record["question"]).strip(), year=str(record.get("year", "")),
                        task=task)
            if task == "multiple_choice":
                options = {str(k).strip().upper(): str(v).strip() for k, v in record["options"].items()}
                item.options = dict(sorted(options.items()))
                item.answer = str(record["answer"]).strip().upper()
            else:
                item.extra = {k: v for k, v in record.items() if k not in ("id", "question", "year")}
            items.append(item)
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
    task = spec.get("task", "multiple_choice")
    if task not in TASKS:
        raise ValueError(f"Dataset '{name}': unknown task '{task}'. Choose from {TASKS}.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{name}': file not found at {path}.")
    if spec["format"] == "jsonl":
        items = _load_jsonl(name, path, task)
    elif spec["format"] == "csv" and task == "multiple_choice":
        items = _load_csv(name, path, spec["columns"])
    else:
        raise ValueError(f"Dataset '{name}': unsupported format '{spec['format']}' for task '{task}'.")
    items = _validate(items, name)
    if spec.get("min_date"):
        items = _from_date(items, name, str(spec["min_date"]))
    return items[:limit] if limit else items


def _from_date(items, name, min_date):
    """Items posted on or after min_date, which must lie after the release of every model."""
    releases = [str(m["release"]) for m in load_models().values() if m.get("release") and m.get("role") != "judge"]
    if releases and min_date <= max(releases):
        raise ValueError(f"Dataset '{name}': min_date {min_date} is not after the latest model "
                         f"release ({max(releases)}).")
    return [item for item in items if str(item.extra.get("date") or "") >= min_date]


# ---------------------------------------------------------------- prompts

def build_messages(item, eval_cfg, thinking, spec):
    """Chat messages for one item (system prompt, instruction, question and options, or the open-ended format)."""
    if item.task != "multiple_choice":
        user = f"{spec['instruction'].strip()}\n\n{item.question}"
        if item.extra.get("request"):
            user += f"\n\n{item.extra['request']}"
        return [{"role": "user", "content": user}]

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
