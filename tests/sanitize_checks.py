"""Checks that the outputs of src/sanitize.py are well-formed (used by test_sanitize_smoke.py)."""

import json
import math

from config import load_datasets_config, load_yaml
from items import load_dataset
from sanitize import TESTS, meta_path, output_path
from sanitize_metrics import SEEN_LABELS, STRICT_LABELS, k_label, question_text


def check_outputs(out_dir, label, datasets, limit, real_model):
    ds_cfg, settings = load_datasets_config(), load_yaml("sanitize.yaml")
    text_cfg, completion_cfg, mink_cfg = settings["text"], settings["completion"], settings["mink"]
    for dataset in datasets:
        spec = ds_cfg["datasets"][dataset]
        items = load_dataset(dataset, spec, limit)
        for test in TESTS:
            path = output_path(out_dir, label, dataset, test)
            assert path.exists() and meta_path(path).exists(), f"missing output {path}"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            meta = json.loads(meta_path(path).read_text(encoding="utf-8"))
            assert len(rows) == len(items) == meta["n_items"]
            assert meta["test"] == test and meta["dataset"] == dataset
            assert "vllm_version" in meta and "model_revision" in meta and "environment" in meta
            if real_model:
                assert meta["vllm_version"] and rows[0]["vllm_version"] == meta["vllm_version"]

            for item, row in zip(items, rows):
                text = question_text(item, text_cfg)
                assert row["model"] == label and row["dataset"] == dataset and row["item_id"] == item.item_id
                assert row["edition"] == int(item.year) and row["language"] == spec["language"]
                assert row["seen"] in SEEN_LABELS and row["seen_strict"] in STRICT_LABELS
                if test == "mink":
                    assert row["text"] == text
                    assert row["n_tokens"] > 0 and len(row["token_logprobs"]) == row["n_tokens"]
                    for k in mink_cfg["k_values"]:
                        assert math.isfinite(row[k_label(k)]) and row[k_label(k)] <= 1e-6
                    assert row[k_label(mink_cfg["primary_k"])] <= row["mean_logprob"] + 1e-6
                    assert row["perplexity"] >= 1 - 1e-6 and row["zlib_ratio"] >= 0
                    if text_cfg.get("include_options", True):
                        assert 0 < row["stem_n_tokens"] < row["n_tokens"]
                        assert math.isfinite(row[k_label(mink_cfg["primary_k"], "stem_")])
                else:
                    assert row["prefix"] + row["reference"] == text
                    assert isinstance(row["generation"], str)
                    assert 1 <= row["max_new_tokens"] <= completion_cfg["max_new_tokens_cap"]
                    assert row["max_new_tokens"] == min(row["n_reference_tokens"], completion_cfg["max_new_tokens_cap"])
                    assert row["n_generated_tokens"] <= row["max_new_tokens"]
                    assert 0 <= row["rouge_l_f1"] <= 1 and 0 <= row["levenshtein_ratio"] <= 1
                    assert 0 <= row["lcp_tokens"] <= row["n_reference_tokens"]
                    assert row["near_verbatim"] == (row["levenshtein_ratio"] >= completion_cfg["near_verbatim_threshold"])
            if real_model and test == "completion":
                assert any(row["generation"].strip() for row in rows), "the model generated nothing"
