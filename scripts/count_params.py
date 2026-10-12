"""Count language-model parameters from downloaded weights, for the FLOPs metric.

Parameters are counted from the safetensors file headers, without loading weights.
Tensors belonging to vision components of multimodal checkpoints (for example the
vision tower and projector of Gemma 3) are excluded, because questions are text-only.

Counts written per model:
    lm_total          all language-model parameters, including the embedding table
    lm_non_embedding  language-model parameters excluding the input embedding table
    vision_excluded   parameters excluded as vision components
    total_in_files    every parameter in the checkpoint

Usage:
    python scripts/count_params.py --out results/param_counts.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medeval.config import load_models  # noqa: E402

VISION_MARKERS = ("vision_tower", "vision_model", "multi_modal_projector", "visual", "image_")
EMBEDDING_MARKERS = ("embed_tokens",)


def count_checkpoint(path):
    counts = {"lm_total": 0, "lm_non_embedding": 0, "vision_excluded": 0, "total_in_files": 0}
    files = sorted(Path(path).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {path}")
    for file in files:
        with safe_open(str(file), framework="pt") as handle:
            for name in handle.keys():
                numel = math.prod(handle.get_slice(name).get_shape())
                counts["total_in_files"] += numel
                if any(marker in name for marker in VISION_MARKERS):
                    counts["vision_excluded"] += numel
                    continue
                counts["lm_total"] += numel
                if not any(marker in name for marker in EMBEDDING_MARKERS):
                    counts["lm_non_embedding"] += numel
    return counts


def count_all(out_path):
    """Count parameters for every downloaded model and write them to out_path."""
    results = {}
    for key, cfg in load_models().items():
        try:
            path = snapshot_download(repo_id=cfg["hf_id"], local_files_only=True)
        except Exception:
            print(f"{key}: weights not downloaded, skipped.")
            continue
        results[key] = {"hf_id": cfg["hf_id"], **count_checkpoint(path)}
        c = results[key]
        print(f"{key}: lm_total={c['lm_total'] / 1e9:.2f}B, "
              f"lm_non_embedding={c['lm_non_embedding'] / 1e9:.2f}B, "
              f"vision_excluded={c['vision_excluded'] / 1e9:.2f}B")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "param_counts.json"))
    args = parser.parse_args()
    count_all(args.out)


if __name__ == "__main__":
    main()
