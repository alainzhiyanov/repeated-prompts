#!/bin/bash
# -----------------------------------------------------------------------
# Run this ONCE on a Narval login node (after setup_env.sh) to download
# all HuggingFace datasets and prepare training data while internet is
# available.  Compute nodes are offline and rely on this cache.
#
#   bash prefetch_data.sh
# -----------------------------------------------------------------------

set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/repeated_prompts/bin/activate"

export HF_HOME="$SCRATCH/.cache/huggingface"
mkdir -p "$HF_HOME"

echo "=== Prefetching HF datasets into $HF_HOME ==="

python - <<'PYEOF'
from datasets import load_dataset

datasets_to_fetch = [
    ("allenai/ai2_arc", "ARC-Challenge"),
    ("allenai/openbookqa", "main"),
    ("openai/gsm8k", "main"),
    ("TIGER-Lab/MMLU-Pro",),
    ("hendrycks/competition_math",),
]

for args in datasets_to_fetch:
    name = args[0]
    print(f"  Downloading {name} …")
    load_dataset(*args)
    print(f"    ✓ {name}")

print("\nAll datasets cached.")
PYEOF

echo ""
echo "=== Preparing training data ==="
python prepare_data.py

echo ""
echo "Prefetch complete. You can now submit the job with:"
echo "  sbatch run_narval.sh"
