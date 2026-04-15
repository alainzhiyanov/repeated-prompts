#!/bin/bash
set -euo pipefail

ENV_DIR="$SCRATCH/envs/repeated_prompts"
export PIP_CACHE_DIR="$SCRATCH/.pip-cache"

module load StdEnv/2023 python/3.11 cuda/12.2

if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR — delete it first to recreate."
    exit 1
fi

mkdir -p "$PIP_CACHE_DIR"

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index torch
pip install transformers datasets peft trl accelerate scipy

echo ""
echo "Environment ready at $ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"




