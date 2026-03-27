#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job script for Narval (Compute Canada / Digital Research Alliance)
#
# Runs the full pipeline: prepare data → fine-tune → evaluate.
#
# Before submitting:
#   1. Run  bash setup_env.sh  on a login node (once).
#   2. Set --account below to your allocation (e.g., def-supervisor).
#
# Submit with:
#   sbatch run_narval.sh
# -----------------------------------------------------------------------

#SBATCH --job-name=repeated-prompts
#SBATCH --account=def-CHANGEME        # ← replace with your allocation
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                  # single A100 on Narval
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

module load StdEnv/2023 python/3.11 cuda/12.2
source "$HOME/envs/repeated_prompts/bin/activate"

# Keep HF caches on scratch (home quota is small on Narval)
export HF_HOME="$SCRATCH/.cache/huggingface"
mkdir -p "$HF_HOME"
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "  Job $SLURM_JOB_ID  —  $(date)"
echo "  Node: $SLURM_NODELIST"
echo "  GPU:  $(nvidia-smi -L | head -1)"
echo "=========================================="

# ---- Step 1: Prepare data ------------------------------------------------
echo ""
echo "[1/3] Preparing data …"
python prepare_data.py

# ---- Step 2: Fine-tune ----------------------------------------------------
echo ""
echo "[2/3] Fine-tuning …"
python finetune.py

# ---- Step 3: Evaluate -----------------------------------------------------
echo ""
echo "[3/3] Evaluating …"
python eval.py

echo ""
echo "All done — $(date)"
