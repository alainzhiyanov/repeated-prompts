#!/bin/bash
# -----------------------------------------------------------------------
# Run this ONCE on a Narval login node to create the virtual environment.
#
#   bash setup_env.sh
#
# The env is stored persistently so SLURM jobs can reuse it.
# -----------------------------------------------------------------------

set -euo pipefail

ENV_DIR="$HOME/envs/repeated_prompts"

module load StdEnv/2023 python/3.11 cuda/12.2

if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists at $ENV_DIR — delete it first to recreate."
    exit 1
fi

virtualenv --no-download "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --no-index --upgrade pip

# torch is available in the CC wheelhouse
pip install --no-index torch

# HuggingFace ecosystem (pulled from PyPI — run on a login node with internet)
pip install transformers datasets peft trl accelerate bitsandbytes scipy

echo ""
echo "Environment ready at $ENV_DIR"
echo "Activate with:  source $ENV_DIR/bin/activate"
