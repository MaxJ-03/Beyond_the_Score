"""End-to-end tests: mock backend on all models (no GPU), and vLLM on the smallest model (GPU only)."""

import os
import subprocess
import sys

import pytest

from config import PROJECT_ROOT, load_datasets_config, load_models, load_yaml
from sanitize import evaluated_models, selected_datasets
from sanitize_checks import check_outputs

LIMIT = 5


def run(*args):
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def mir_datasets():
    return selected_datasets(load_datasets_config(), load_yaml("sanitize.yaml"))


def test_pipeline_with_mock_backend(tmp_path):
    models, datasets = evaluated_models(load_models()), mir_datasets()
    run("src/sanitize.py", "--backend", "mock", "--limit", str(LIMIT), "--out", str(tmp_path))
    for key in models:
        check_outputs(tmp_path, key, datasets, LIMIT, real_model=False)

    # A second run skips every existing output.
    outputs = sorted((tmp_path / "sanitize").rglob("*.jsonl"))
    modified = [path.stat().st_mtime_ns for path in outputs]
    run("src/sanitize.py", "--backend", "mock", "--limit", str(LIMIT), "--out", str(tmp_path))
    assert [path.stat().st_mtime_ns for path in outputs] == modified

    summary = tmp_path / "analysis"
    run("src/sanitize_summary.py", "--results", str(tmp_path), "--out", str(summary))
    for name in ("by_edition.csv", "seen_vs_unseen.csv", "year_adjusted.csv", "mink_by_edition.png"):
        assert (summary / name).exists(), name


def test_smoke_with_vllm(tmp_path):
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    models = load_models()
    key = os.environ.get("SANITIZE_SMOKE_MODEL") or min(evaluated_models(models), key=lambda k: models[k]["size_b"])
    ds_cfg, datasets = load_datasets_config(), mir_datasets()
    # The first Spanish and the first English dataset.
    chosen = [next(d for d in datasets if ds_cfg["datasets"][d]["language"] == lang) for lang in ("es", "en")]
    run("src/sanitize.py", "--models", key, "--datasets", *chosen, "--limit", str(LIMIT), "--out", str(tmp_path))
    check_outputs(tmp_path, key, chosen, LIMIT, real_model=True)
