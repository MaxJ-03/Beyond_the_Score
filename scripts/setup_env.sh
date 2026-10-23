#!/bin/bash
# One-time environment setup on a Snellius login node (FRESH=1 rebuilds it).
set -euo pipefail

PYTHON_MODULE=${PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}
VENV="$HOME/venvs/medeval"
cd "$(dirname "$0")/.."

module purge
module load 2024
module load "$PYTHON_MODULE"

if [[ "${FRESH:-0}" == "1" ]]; then
    rm -rf "$VENV"
fi
if [[ ! -d "$VENV" ]]; then
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt vllm
pip freeze > requirements.lock.txt

mkdir -p "/scratch-shared/$USER/hf_cache" logs results
if ! grep -q "HF_HOME=/scratch-shared" "$HOME/.bashrc"; then
    echo "export HF_HOME=/scratch-shared/\$USER/hf_cache" >> "$HOME/.bashrc"
fi

python -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__)"
echo "Setup complete. Installed versions are recorded in requirements.lock.txt."
