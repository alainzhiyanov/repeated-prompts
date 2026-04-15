# Fine-Tuning Language Models with Repeated Prompts

This project investigates whether the *prompt repetition* technique — repeating the same prompt before generating an answer — can improve model performance when applied during **fine-tuning**, not just at inference time.

Motivated by [Leviathan, Kalman, & Matias (2025)](https://arxiv.org/abs/2512.14982), who show that prompt repetition improves non-reasoning language models at inference across a range of benchmarks, we ask: **does training on repeated prompts lead to better results than standard prompting or standard fine-tuning?**

## Method

We LoRA fine-tune [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (loaded from a local copy at `/home/taegyoem/scratch/qwen`) on a mix of three benchmarks formatted with doubled prompts. Training examples are modified from `prompt → answer` to `prompt + prompt → answer`, where the question block is concatenated with itself. All other training settings remain unchanged.

### Training Data

| Benchmark | Format | Examples |
|-----------|--------|----------|
| [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | Multiple-choice (A/B/C/D) | ~1,100 |
| [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) | Multiple-choice (A/B/C/D) | ~5,000 |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Open-ended (numeric) | ~7,500 |

The mix of multiple-choice and open-ended math covers two distinct evaluation styles and gives the model exposure to double prompting across formats.

## Evaluation

Three conditions are compared across **7 benchmarks** (3 in-distribution, 4 held-out):

| # | Condition | Description |
|---|-----------|-------------|
| 1 | Vanilla + Single Prompt | Base model, standard prompting |
| 2 | Vanilla + Double Prompt | Base model, prompt repeated twice |
| 3 | Fine-tuned + Double Prompt | LoRA-tuned model, double prompting |

### Benchmarks

| Benchmark | Format | Eval Mode | In Train? |
|-----------|--------|-----------|-----------|
| ARC-Challenge | Multiple-choice | Logit scoring | Yes |
| OpenBookQA | Multiple-choice | Logit scoring | Yes |
| GSM8K | Open-ended math | Generation | Yes |
| MMLU-Pro | Multiple-choice (10 options) | Logit scoring | No (held-out) |
| MATH | Open-ended math | Generation | No (held-out) |
| NameIndex | Custom retrieval | Generation | No (held-out) |
| MiddleMatch | Custom positional | Generation | No (held-out) |

NameIndex and MiddleMatch are custom tasks from the original paper, specifically designed to test positional attention. They are generated procedurally and serve as zero-shot generalization tests.

## Project Structure

```
repeated_prompts/
├── benchmarks.py       # Benchmark registry, formatting, custom generators
├── utils.py            # Shared constants and message helpers
├── prepare_data.py     # Multi-benchmark data preparation
├── finetune.py         # LoRA fine-tuning with TRL SFTTrainer
├── eval.py             # Multi-benchmark evaluation (logit + generation)
├── setup_env.sh        # One-time env setup for Narval
├── prefetch_data.sh    # One-time dataset download + data prep (login node)
├── run_narval.sh       # SLURM job script (runs offline)
├── requirements.txt    # Python dependencies
└── README.md
```

**Generated at runtime:**

```
data/
├── train_double.jsonl          # Combined shuffled training data
├── val_double.jsonl            # Combined validation data
├── {benchmark}_train_double.jsonl   # Per-benchmark training data
└── {benchmark}_val_double.jsonl     # Per-benchmark validation data

checkpoints/qwen-1.5b-double-prompt-multi/final/   # LoRA adapter

results.json                    # Evaluation results
```

## Running on Compute Canada Narval

Compute nodes on Narval have **no internet access**, so all model weights and
datasets must be downloaded on a login node first. The model is expected at
`/home/taegyoem/scratch/qwen` (a local copy of Qwen2.5-1.5B-Instruct).

### 1. Set up the environment (once, on a login node)

```bash
bash setup_env.sh
```

This creates a persistent virtual environment at `~/envs/repeated_prompts` with
all dependencies.

### 2. Prefetch datasets and prepare training data (once, on a login node)

```bash
bash prefetch_data.sh
```

This downloads all HuggingFace datasets into `$SCRATCH/.cache/huggingface` and
runs `prepare_data.py` to create the training JSONL files. Both steps require
internet and should be run **before** submitting the SLURM job.

### 3. Configure and submit

Edit `run_narval.sh` and replace `def-CHANGEME` with your allocation account, then:

```bash
sbatch run_narval.sh
```

The job requests **1 A100 GPU, 40 GB RAM, 12 hours** and runs fully offline.
Expected runtime is ~4-6 hours:

| Step | Estimated Time |
|------|---------------|
| Fine-tuning (3 epochs, ~13.5k examples) | ~30-60 min |
| Evaluation (7 benchmarks × 3 conditions) | ~3-5 hours |

### 4. Check results

```bash
cat results.json
```

## Local Usage

If running outside Narval (e.g., on a local GPU):

```bash
pip install -r requirements.txt
python prepare_data.py
python finetune.py
python eval.py
```

### CLI Options

**`finetune.py`**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `/home/taegyoem/scratch/qwen` | Base model (local path) |
| `--epochs` | `3` | Training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--grad_accum` | `4` | Gradient accumulation steps |
| `--lr` | `2e-4` | Learning rate |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--max_seq_length` | `1024` | Maximum sequence length |
| `--output_dir` | `checkpoints/.../final` | Adapter output path |

**`eval.py`**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `/home/taegyoem/scratch/qwen` | Base model (local path) |
| `--adapter` | `checkpoints/.../final` | LoRA adapter path |
| `--output` | `results.json` | Results output file |
| `--benchmarks` | all 7 | Subset of benchmarks to evaluate |

Evaluate specific benchmarks only:

```bash
python eval.py --benchmarks arc gsm8k name_index
```

## Dependencies

- **PyTorch** (>=2.0)
- **Transformers** (>=4.40)
- **Datasets** (>=2.19)
- **PEFT** (>=0.10) — LoRA training and inference
- **TRL** (>=0.8) — `SFTTrainer` and completion-only data collator
- **Accelerate** (>=0.28)
- **SciPy**

## Reference

Leviathan, Y., Kalman, M., & Matias, Y. (2025). *Prompt Repetition Improves Non-Reasoning LLMs*. arXiv:2512.14982.
