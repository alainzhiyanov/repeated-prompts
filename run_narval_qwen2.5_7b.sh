#!/bin/bash
# -----------------------------------------------------------------------
# SLURM: Qwen2.5-7B-Instruct — LoRA fine-tune + eval (offline).
#
# Walltime: 12h is often enough for LoRA + this eval on one fast GPU; asking
# for less time can shorten queue waits (backfill). If the job hits TIMEOUT,
# resubmit with e.g. --time=18:00:00 or --time=24:00:00.
# CPU/RAM kept small (GPU does the model); raise --mem if the job dies with OOM.
# -----------------------------------------------------------------------

#SBATCH --job-name=rpt-qwen7
#SBATCH --account=def-mijungp        # ← replace with your allocation
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-%j-qwen2.5-7b.out
#SBATCH --error=logs/slurm-%j-qwen2.5-7b.err

set -euo pipefail

_JOB_START=$(date +%s)
_job_print_elapsed() {
  local _ec="${1:-0}" _end _sec _h _m _s
  _end=$(date +%s)
  _sec=$((_end - _JOB_START))
  _h=$((_sec / 3600))
  _m=$(((_sec % 3600) / 60))
  _s=$((_sec % 60))
  echo ""
  echo "------------------------------------------------------------------"
  echo "  Job wall time: ${_sec}s  (${_h}h ${_m}m ${_s}s)  exit=${_ec}"
  echo "------------------------------------------------------------------"
  mkdir -p logs 2>/dev/null || true
  printf '%s\t%s\t%s\t%ds\texit=%s\t%s\n' \
    "${SLURM_JOB_ID:-na}" \
    "${SLURM_JOB_NAME:-rpt-qwen7}" \
    "$(hostname -s 2>/dev/null || hostname)" \
    "$_sec" \
    "$_ec" \
    "$(date -Is 2>/dev/null || date)" \
    >>"${SLURM_SUBMIT_DIR:-.}/logs/job_walltimes.tsv" 2>/dev/null || true
}
trap '_job_print_elapsed $?' EXIT

MODEL_PATH=/home/taegyoem/scratch/qwen_7b
CKPT_DIR=checkpoints/qwen2.5-7b-double-prompt-multi/final
RESULTS=results_qwen2.5_7b.json

module purge
module load StdEnv/2023 python/3.11 cuda/12.2 gcc arrow
source "$SCRATCH/envs/repeated_prompts/bin/activate"

export HF_HOME="$SCRATCH/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "  Job $SLURM_JOB_ID  —  $(date)"
echo "  Model: Qwen2.5-7B-Instruct"
echo "  Node: $SLURM_NODELIST"
echo "  GPU:  $(nvidia-smi -L | head -1)"
echo "=========================================="

echo ""
echo "[1/2] Fine-tuning …"
python finetune.py \
  --model "$MODEL_PATH" \
  --output_dir "$CKPT_DIR" \
  --batch_size 2 \
  --grad_accum 8

echo ""
echo "[2/2] Evaluating …"
python eval.py \
  --model "$MODEL_PATH" \
  --adapter "$CKPT_DIR" \
  --output "$RESULTS"

echo ""
echo "All done — $(date)"
