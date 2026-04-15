#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job script for Narval (Compute Canada / Digital Research Alliance)
#
# Runs: fine-tune → evaluate (fully offline, no internet required).
#
# Before submitting:
#   1. Run  bash setup_env.sh      on a login node (once).
#   2. Run  bash prefetch_data.sh  on a login node (once, needs internet).
#   3. Set --account below to your allocation (e.g., def-supervisor).
#
# Submit with:
#   sbatch run_narval.sh
# -----------------------------------------------------------------------

#SBATCH --job-name=repeated-prompts
#SBATCH --account=def-mijungp        # ← replace with your allocation
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                  # single A100 on Narval
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/repeated_prompts/bin/activate"

# HF caches on scratch (populated by prefetch_data.sh on the login node)
export HF_HOME="$SCRATCH/.cache/huggingface"

# Fully offline — never attempt network downloads on the compute node
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "  Job $SLURM_JOB_ID  —  $(date)"
echo "  Node: $SLURM_NODELIST"
echo "  GPU:  $(nvidia-smi -L | head -1)"
echo "=========================================="

# ---- Step 1: Fine-tune ----------------------------------------------------
echo ""
echo "[1/2] Fine-tuning …"
python finetune.py

# ---- Step 2: Evaluate -----------------------------------------------------
echo ""
echo "[2/2] Evaluating …"
python eval.py

echo ""
echo "All done — $(date)"
