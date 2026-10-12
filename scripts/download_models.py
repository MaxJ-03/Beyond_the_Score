"""Download all model weights into the Hugging Face cache before running jobs.

Run this on a login node (or on the staging partition) with HF_HOME pointing to
/scratch-shared/<username>/hf_cache. Evaluation jobs then run with HF_HUB_OFFLINE=1.
Gated models (Gemma 3, MedGemma) require a Hugging Face login and accepted licenses.

Usage:
    python scripts/download_models.py               all non-optional models
    python scripts/download_models.py --models qwen3-4b gemma3-4b
    python scripts/download_models.py --include-optional
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medeval.config import load_models  # noqa: E402

ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    models = load_models()
    keys = args.models or [k for k, m in models.items() if args.include_optional or not m.get("optional")]
    done = set()
    for key in keys:
        hf_id = models[key]["hf_id"]
        if hf_id in done:
            continue
        print(f"Downloading {hf_id} ...")
        path = snapshot_download(repo_id=hf_id, allow_patterns=ALLOW_PATTERNS)
        print(f"  stored at {path}")
        done.add(hf_id)


if __name__ == "__main__":
    main()
