#!/bin/bash
set -euo pipefail

ENV_DIR="$SCRATCH/envs/repeated_prompts"
export PIP_CACHE_DIR="$SCRATCH/.pip-cache"
export VIRTUALENV_OVERRIDE_APP_DATA="$SCRATCH/.virtualenv"

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 gcc arrow

mkdir -p "$SCRATCH/envs" "$PIP_CACHE_DIR" "$VIRTUALENV_OVERRIDE_APP_DATA"

if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR — delete it first to recreate."
    exit 1
fi

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"

# Alliance wheelhouse packages
pip install --no-index torch

# PyPI packages; pyarrow will resolve via the loaded Arrow module
pip install transformers datasets peft 'trl<0.20' accelerate scipy

echo ""
echo "Environment ready at $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"