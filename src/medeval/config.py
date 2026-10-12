"""Loading of the model and dataset configuration files."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(name):
    with open(CONFIG_DIR / name, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_models():
    """Return a mapping from model key to its configuration, with defaults applied."""
    raw = load_yaml("models.yaml")
    defaults = raw.get("defaults", {})
    models = {}
    for key, settings in raw["models"].items():
        merged = {**defaults, **settings, "key": key}
        merged["llm_kwargs"] = {**defaults.get("llm_kwargs", {}), **settings.get("llm_kwargs", {})}
        merged["chat_template_kwargs"] = {
            **defaults.get("chat_template_kwargs", {}),
            **settings.get("chat_template_kwargs", {}),
        }
        models[key] = merged
    return models


def load_datasets_config():
    """Return the full dataset configuration (datasets, evaluation and analysis blocks)."""
    return load_yaml("datasets.yaml")


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
