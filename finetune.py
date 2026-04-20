"""
LoRA fine-tune a causal LM on double-prompted training data (chat models).

The training JSONL is produced by prepare_data.py and contains examples from
ARC-Challenge, OpenBookQA, and GSM8K — all formatted with repeated prompts.
Completion-only loss uses the tokenizer's chat template (works for Qwen,
Mistral, Llama-style instruct models).

Usage:
    python prepare_data.py          # run first
    python finetune.py              # then fine-tune
    python finetune.py --epochs 5   # override defaults
"""

import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from utils import ADAPTER_DIR, MODEL_NAME


def load_tokenizer(model_name: str):
    """Load tokenizer with a fast->slow fallback for offline/HPC environments."""
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=True
        )
    except Exception as fast_err:
        print(
            "Fast tokenizer load failed; retrying with use_fast=False "
            f"(reason: {fast_err})"
        )
        try:
            return AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, use_fast=False
            )
        except Exception as slow_err:
            raise RuntimeError(
                "Tokenizer load failed for both fast and slow implementations. "
                "Install tokenizer dependencies in your env (typically "
                "`protobuf`, `sentencepiece`, and/or `tiktoken`) and retry."
            ) from slow_err


def assistant_response_prefix(tokenizer) -> str:
    """Prefix added before the assistant's answer (masked out of the LM loss)."""
    # Primary path: render a chat that includes a known assistant sentinel and
    # split just before that sentinel. This works even when
    # add_generation_prompt=True returns no extra text.
    sentinel = "__ASSISTANT_SENTINEL_7e2f6b2c__"
    with_assistant_turn = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "."},
        {"role": "assistant", "content": sentinel},
    ]
    rendered = tokenizer.apply_chat_template(
        with_assistant_turn, tokenize=False, add_generation_prompt=False
    )
    cut = rendered.find(sentinel)
    if cut != -1:
        return rendered[:cut]

    # Fallback path: infer from generation prompt delta.
    messages = with_assistant_turn[:-1]
    without = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    with_generation_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not with_generation_prompt.startswith(without):
        raise RuntimeError(
            "Could not infer assistant response prefix from chat template. "
            "Pass --response_template explicitly for this model/template."
        )
    prefix = with_generation_prompt[len(without) :]
    if not prefix:
        raise RuntimeError(
            "Tokenizer returned an empty assistant prefix. "
            "Pass --response_template explicitly for this model/template."
        )
    return prefix


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def apply_chat_template(tokenizer, records: list[dict]) -> Dataset:
    """Convert message-dicts → plain text via the tokenizer's chat template."""
    texts = [
        tokenizer.apply_chat_template(
            rec["messages"], tokenize=False, add_generation_prompt=False
        )
        for rec in records
    ]
    return Dataset.from_dict({"text": texts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--train_data", default="data/train_double.jsonl")
    parser.add_argument("--val_data", default="data/val_double.jsonl")
    parser.add_argument("--output_dir", default=ADAPTER_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument(
        "--response_template",
        default=None,
        help="Override for DataCollatorForCompletionOnlyLM (default: infer from tokenizer).",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading data …")
    train_ds = apply_chat_template(tokenizer, load_jsonl(args.train_data))
    val_ds = apply_chat_template(tokenizer, load_jsonl(args.val_data))
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    response_template = args.response_template or assistant_response_prefix(tokenizer)
    print(f"  SFT response_template (loss starts after this): {response_template!r}")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template=None,
        tokenizer=tokenizer,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving adapter → {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
