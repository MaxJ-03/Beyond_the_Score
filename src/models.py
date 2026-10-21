"""Make sure every model is downloaded and its parameters are counted.

Two steps, each skipped when nothing is missing:
    1. weights   a model whose weights are complete in the Hugging Face cache is not
                 downloaded again; incomplete or missing models are downloaded
    2. counts    results/param_counts.json keeps the counts it already has; only models
                 that are missing from it (or whose hf_id changed) are counted

When both steps have nothing to do, the script only reports that everything is up to date,
without contacting Hugging Face. Run it on a login node (or on the staging partition) with
HF_HOME pointing to /scratch-shared/<username>/hf_cache; evaluation jobs then run with
HF_HUB_OFFLINE=1. Gated models (Gemma 3, MedGemma, Llama 3) require a Hugging Face login and
accepted licenses.

Parameters are counted from the safetensors file headers, without loading weights and
without PyTorch. Tensors of vision components in multimodal checkpoints (for example the
vision tower and projector of Gemma 3) are excluded, because all questions are text-only.
Counts written per model:
    lm_total          all language-model parameters, including the embedding table
    lm_non_embedding  language-model parameters excluding the input embedding table
    vision_excluded   parameters excluded as vision components
    total_in_files    every parameter in the checkpoint

Usage:
    python src/models.py                      all non-optional models and the judge
    python src/models.py --models qwen3-4b gemma3-4b
    python src/models.py --include-optional
    python src/models.py --recount            count all selected models again
"""

import argparse
import json
import math
import struct
from pathlib import Path

from config import PROJECT_ROOT, load_models

ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"]
VISION_MARKERS = ("vision_tower", "vision_model", "multi_modal_projector", "visual", "image_")
EMBEDDING_MARKERS = ("embed_tokens",)
PARAM_COUNTS = PROJECT_ROOT / "results" / "param_counts.json"


# ---------------------------------------------------------------- 1. weights

def find_snapshot(repo_id):
    """Locate the downloaded snapshot of a model in the Hugging Face cache, or return None.

    The cache layout is <cache>/models--<org>--<name>/snapshots/<revision>/. The revision
    recorded in refs/main is preferred; otherwise the most recently modified snapshot with
    safetensors files is used.
    """
    from huggingface_hub import constants

    repo_dir = Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        candidate = snapshots / ref.read_text().strip()
        if any(candidate.glob("*.safetensors")):
            return candidate
    candidates = [d for d in snapshots.iterdir() if d.is_dir() and any(d.glob("*.safetensors"))]
    return max(candidates, key=lambda d: d.stat().st_mtime) if candidates else None


def is_complete(snapshot):
    """True if the snapshot has a config, a tokenizer and every weight file it lists.

    Sharded checkpoints list their weight files in model.safetensors.index.json; every listed
    file must exist (a download that was interrupted leaves some of them missing).
    """
    if snapshot is None or not (snapshot / "config.json").exists():
        return False
    if not any((snapshot / name).exists() for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")):
        return False
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        return all((snapshot / shard).exists() for shard in shards)
    return any(snapshot.glob("*.safetensors"))


def ensure_weights(models, keys):
    """Download every selected model whose weights are not complete in the cache."""
    missing = []
    for key in keys:
        hf_id = models[key]["hf_id"]
        if hf_id not in missing and not is_complete(find_snapshot(hf_id)):
            missing.append(hf_id)
    if not missing:
        print("Weights: all selected models are already downloaded.")
        return
    from huggingface_hub import snapshot_download

    for hf_id in missing:
        print(f"Downloading {hf_id} ...")
        path = snapshot_download(repo_id=hf_id, allow_patterns=ALLOW_PATTERNS)
        print(f"  stored at {path}")


# ---------------------------------------------------------------- 2. parameter counts

def read_header_shapes(file):
    """Yield (tensor name, number of elements) from a safetensors header.

    A safetensors file starts with an 8-byte little-endian header length, followed by a
    JSON header that lists every tensor with its shape.
    """
    with open(file, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    for name, info in header.items():
        if name == "__metadata__":
            continue
        yield name, math.prod(info["shape"])


def count_checkpoint(path):
    """Count the parameters of one checkpoint folder, split as described in the module docstring."""
    counts = {"lm_total": 0, "lm_non_embedding": 0, "vision_excluded": 0, "total_in_files": 0}
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {path}")
    for file in files:
        for name, numel in read_header_shapes(file):
            counts["total_in_files"] += numel
            if any(marker in name for marker in VISION_MARKERS):
                counts["vision_excluded"] += numel
                continue
            counts["lm_total"] += numel
            if not any(marker in name for marker in EMBEDDING_MARKERS):
                counts["lm_non_embedding"] += numel
    return counts


def ensure_counts(models, keys, recount=False, out_path=PARAM_COUNTS):
    """Count the parameters of selected models that are missing from the count file."""
    counts = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [k for k in keys
            if recount or k not in counts or counts[k].get("hf_id") != models[k]["hf_id"]]
    if not todo:
        print(f"Parameter counts: all selected models are already in {out_path}.")
        return False
    for key in todo:
        hf_id = models[key]["hf_id"]
        snapshot = find_snapshot(hf_id)
        if not is_complete(snapshot):
            print(f"{key}: weights missing or incomplete, skipped.")
            continue
        counts[key] = {"hf_id": hf_id, **count_checkpoint(snapshot)}
        c = counts[key]
        print(f"{key}: lm_total={c['lm_total'] / 1e9:.2f}B, "
              f"lm_non_embedding={c['lm_non_embedding'] / 1e9:.2f}B, "
              f"vision_excluded={c['vision_excluded'] / 1e9:.2f}B")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(counts, indent=2))
    print(f"Wrote {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="Model keys (default: all non-optional).")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--recount", action="store_true", help="Count the selected models again.")
    args = parser.parse_args()

    models = load_models()
    unknown = [k for k in args.models or [] if k not in models]
    if unknown:
        raise SystemExit(f"Unknown models {unknown}. Available: {sorted(models)}")
    keys = args.models or [k for k, m in models.items() if args.include_optional or not m.get("optional")]
    ensure_weights(models, keys)
    # The judge is never evaluated, so its parameters are not needed for the cost metrics.
    ensure_counts(models, [k for k in keys if models[k].get("role") != "judge"], recount=args.recount)


if __name__ == "__main__":
    main()
