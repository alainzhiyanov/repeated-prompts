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
    """Stable marker that appears right before the assistant turn.

    `DataCollatorForCompletionOnlyLM` locates this substring (by token id) in
    each training example and masks everything up to and including it. It must
    therefore depend on *nothing* other than the chat template itself: any
    user/system content that leaks into the prefix will make the search fail,
    causing every label to be masked, no gradients to flow, and the LoRA
    adapter to stay at its zero initialization (B=0 → adapter is a no-op).

    We obtain a content-independent prefix by rendering two chats that differ
    only in user/system content and taking the longest common suffix of the
    text preceding an assistant sentinel. Anything that varies with the input
    drops out of the suffix; only the template-level chat scaffolding remains
    (e.g. `<|im_start|>assistant\\n` for Qwen, `[/INST]` for Mistral,
    `<|start_header_id|>assistant<|end_header_id|>\\n\\n` for Llama-3).
    """
    sentinel = "__ASSISTANT_SENTINEL_7e2f6b2c__"

    def render(system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": sentinel},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            return tokenizer.apply_chat_template(
                messages[1:], tokenize=False, add_generation_prompt=False
            )

    r1 = render("You are a helpful assistant.", "AAA first question?")
    r2 = render("A different system prompt.", "ZZZ totally other query!")
    p1, p2 = r1[: r1.find(sentinel)], r2[: r2.find(sentinel)]
    if r1.find(sentinel) == -1 or r2.find(sentinel) == -1:
        raise RuntimeError(
            "Chat template dropped the assistant sentinel; cannot infer "
            "response prefix. Pass --response_template explicitly."
        )

    i = 0
    max_len = min(len(p1), len(p2))
    while i < max_len and p1[-1 - i] == p2[-1 - i]:
        i += 1
    if i == 0:
        raise RuntimeError(
            "No common suffix between two renderings — the chat template "
            "mixes user content into the assistant prefix. "
            "Pass --response_template explicitly."
        )

    prefix = p1[-i:].lstrip()
    if not prefix:
        raise RuntimeError(
            "Inferred assistant prefix is whitespace-only. "
            "Pass --response_template explicitly."
        )
    return prefix


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _sanity_check_response_template(tokenizer, dataset, response_template: str,
                                    n_samples: int = 8) -> None:
    """Fail loudly if the collator wouldn't find the template in training data.

    Mirrors the substring search that DataCollatorForCompletionOnlyLM performs
    on token ids (not strings), so tokenizer quirks (leading spaces, BPE merges)
    are covered.
    """
    template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    if not template_ids:
        raise RuntimeError(
            f"Response template {response_template!r} tokenizes to zero tokens."
        )

    checked = min(n_samples, len(dataset))
    for i in range(checked):
        seq_ids = tokenizer(dataset[i]["text"], add_special_tokens=False)["input_ids"]
        if not _contains_subsequence(seq_ids, template_ids):
            raise RuntimeError(
                "Response template not found in training example "
                f"{i}: DataCollatorForCompletionOnlyLM would mask every "
                "label as -100, producing zero gradients.\n"
                f"  response_template = {response_template!r}\n"
                f"  template token ids = {template_ids}\n"
                "Pass --response_template explicitly if auto-inference is wrong."
            )
    print(f"  response_template sanity check passed on {checked} examples")


def _contains_subsequence(seq: list[int], sub: list[int]) -> bool:
    if not sub or len(sub) > len(seq):
        return False
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return True
    return False


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

    # Guardrail: if the response template can't be found in training examples,
    # the collator silently masks every label as -100, producing zero gradients
    # and an unchanged LoRA adapter (B stays at 0). Surface that now.
    _sanity_check_response_template(tokenizer, train_ds, response_template)

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
