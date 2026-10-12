"""Loading of multiple-choice datasets into a common item format."""

import json
from dataclasses import dataclass, field

import pandas as pd

from .config import resolve_path

LETTERS = "ABCDEFGHIJ"


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
