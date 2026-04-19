#!/bin/bash
set -euo pipefail

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 gcc arrow

source "$SCRATCH/envs/repeated_prompts/bin/activate"

# ---- Force ALL caches to SCRATCH ---------------------------------------

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/transformers"

export TORCH_HOME="$SCRATCH/.cache/torch"
export XDG_CACHE_HOME="$SCRATCH/.cache"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"

# temp files (important for large dataset processing)
export TMPDIR="$SCRATCH/tmp"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$TORCH_HOME" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR" \
  "$TMPDIR"

echo "=== Prefetching HF datasets into $HF_HOME ==="

python - <<'PYEOF'
from datasets import load_dataset

datasets_to_fetch = [
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/openbookqa", "main"),
    ("openai/gsm8k", "main"),
    ("TIGER-Lab/MMLU-Pro",),
]

for args in datasets_to_fetch:
    name = args[0]
    print(f"  Downloading {name} …")
    load_dataset(*args)
    print(f"    ✓ {name}")

# MATH: use EleutherAI mirror (hendrycks/competition_math is often unavailable on the Hub).
_math_subjects = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
print("  Downloading EleutherAI/hendrycks_math (all subjects) …")
for subj in _math_subjects:
    load_dataset("EleutherAI/hendrycks_math", subj)
print("    ✓ EleutherAI/hendrycks_math")

print("\nAll datasets cached.")
PYEOF

echo ""
echo "=== Preparing training data ==="
python prepare_data.py

echo ""
echo "Prefetch complete. You can now submit the job with:"
echo "  sbatch run_narval.sh"