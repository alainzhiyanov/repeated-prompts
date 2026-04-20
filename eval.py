"""
Evaluate three conditions across multiple benchmarks:

  1. Vanilla model  + single prompt
  2. Vanilla model  + double prompt
  3. Fine-tuned model (LoRA) + double prompt

In-distribution benchmarks:  ARC-Challenge, OpenBookQA, GSM8K
Held-out benchmarks:         MMLU-Pro, MATH, NameIndex, MiddleMatch

Usage:
    python eval.py
    python eval.py --benchmarks arc gsm8k
    python eval.py --model Qwen/Qwen2.5-1.5B-Instruct --adapter checkpoints/...
"""

import argparse
import json
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import BENCHMARKS, EVAL_BENCHMARKS, load_split, make_double
from utils import ADAPTER_DIR, MODEL_NAME, make_messages


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


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_logit(model, tokenizer, dataset, prompt_fn, system_prompt,
                   choices_fn, answer_fn, desc: str = "") -> float:
    """Log-prob evaluation for multiple-choice benchmarks."""
    correct = total = 0
    device = next(model.parameters()).device

    for i, ex in enumerate(dataset):
        prompt = prompt_fn(ex)
        msgs = make_messages(system_prompt, prompt)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**inputs).logits[0, -1, :]

        choices = choices_fn(ex)
        best, best_score = None, float("-inf")
        for ch in choices:
            tok_id = tokenizer.encode(ch, add_special_tokens=False)[0]
            s = logits[tok_id].item()
            if s > best_score:
                best_score, best = s, ch

        correct += best == answer_fn(ex)
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{desc}] {i + 1}/{len(dataset)}  acc={correct / total:.4f}")

    return correct / total


@torch.inference_mode()
def evaluate_generate(model, tokenizer, dataset, prompt_fn, system_prompt,
                      answer_fn, compare_fn, desc: str = "",
                      max_new_tokens: int = 64) -> float:
    """Generation-based evaluation for open-ended benchmarks."""
    correct = total = 0
    device = next(model.parameters()).device

    for i, ex in enumerate(dataset):
        prompt = prompt_fn(ex)
        msgs = make_messages(system_prompt, prompt)
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        gold = answer_fn(ex)
        correct += compare_fn(response, gold)
        total += 1

        if (i + 1) % 200 == 0:
            print(f"  [{desc}] {i + 1}/{len(dataset)}  acc={correct / total:.4f}")

    return correct / total


def run_eval(model, tokenizer, dataset, cfg, prompt_mode: str,
             desc: str = "") -> float:
    """Dispatch to logit or generate evaluation."""
    fmt = cfg["format_fn"]
    if prompt_mode == "single":
        prompt_fn = fmt
    else:
        prompt_fn = lambda ex: make_double(fmt(ex))  # noqa: E731

    if cfg["eval_mode"] == "logit":
        return evaluate_logit(
            model, tokenizer, dataset, prompt_fn,
            cfg["system_prompt"], cfg["choices_fn"], cfg["answer_fn"], desc,
        )
    return evaluate_generate(
        model, tokenizer, dataset, prompt_fn,
        cfg["system_prompt"], cfg["answer_fn"], cfg["compare_fn"], desc,
        cfg.get("max_new_tokens", 64),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subset(data, n, seed=42):
    """Deterministically subsample data to at most *n* items."""
    if n is None or len(data) <= n:
        return data
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[:n])
    if hasattr(data, "select"):
        return data.select(indices)
    return [data[i] for i in indices]


def _load_test(name: str, cfg: dict):
    """Load (and optionally subsample) a benchmark's test set."""
    try:
        ds = load_split(name, "test")
    except Exception as e:
        print(f"  ⚠ Could not load {cfg['display_name']}: {e}")
        return None
    return _subset(ds, cfg.get("eval_subset"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--adapter", default=ADAPTER_DIR)
    parser.add_argument("--output", default="results.json")
    parser.add_argument(
        "--benchmarks", nargs="*", default=None,
        help=f"Benchmarks to evaluate (default: all). Choices: {EVAL_BENCHMARKS}",
    )
    args = parser.parse_args()

    bench_names = args.benchmarks or EVAL_BENCHMARKS

    # -- Load test sets -------------------------------------------------------
    print("Loading test sets …")
    test_data: dict = {}
    for name in bench_names:
        cfg = BENCHMARKS[name]
        ds = _load_test(name, cfg)
        if ds is not None:
            test_data[name] = ds
            print(f"  {cfg['display_name']}: {len(ds)} examples")

    if not test_data:
        print("No benchmarks loaded — exiting.")
        return

    # -- Load model -----------------------------------------------------------
    print(f"\nLoading base model: {args.model}")
    tokenizer = load_tokenizer(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.eval()

    results: dict = {}

    # -- Conditions 1 & 2 (vanilla model) ------------------------------------
    for name, ds in test_data.items():
        cfg = BENCHMARKS[name]
        dn = cfg["display_name"]

        print(f"\n{'=' * 60}")
        print(f"{dn}  —  Condition 1: Vanilla + Single prompt")
        print("=" * 60)
        acc1 = run_eval(base_model, tokenizer, ds, cfg, "single", f"{name}/single")
        print(f"→ {acc1:.4f}")

        print(f"\n{dn}  —  Condition 2: Vanilla + Double prompt")
        print("-" * 60)
        acc2 = run_eval(base_model, tokenizer, ds, cfg, "double", f"{name}/double")
        print(f"→ {acc2:.4f}")

        results[name] = {
            "display_name": dn,
            "n_test": len(ds),
            "vanilla_single": round(acc1, 6),
            "vanilla_double": round(acc2, 6),
        }

    # -- Condition 3 (fine-tuned + double) ------------------------------------
    print(f"\nLoading LoRA adapter from {args.adapter}")
    try:
        ft_model = PeftModel.from_pretrained(base_model, args.adapter)
        ft_model.eval()
    except Exception as e:
        print(f"⚠ Could not load adapter: {e}")
        print("Skipping condition 3 (fine-tuned + double prompt).\n")
        ft_model = None

    if ft_model is not None:
        for name, ds in test_data.items():
            cfg = BENCHMARKS[name]
            dn = cfg["display_name"]
            print(f"\n{dn}  —  Condition 3: Fine-tuned + Double prompt")
            print("-" * 60)
            acc3 = run_eval(ft_model, tokenizer, ds, cfg, "double", f"{name}/ft+dbl")
            print(f"→ {acc3:.4f}")
            results[name]["finetuned_double"] = round(acc3, 6)

    # -- Summary table --------------------------------------------------------
    print(f"\n{'=' * 72}")
    print("RESULTS SUMMARY")
    print("=" * 72)

    header = f"{'Benchmark':<20} {'Single':>8} {'Double':>8} {'FT+Dbl':>8}  {'DP Δ':>7} {'FT Δ':>7}"
    print(header)
    print("-" * 72)

    for name in test_data:
        r = results[name]
        a1 = r["vanilla_single"]
        a2 = r["vanilla_double"]
        a3 = r.get("finetuned_double")

        dp_gain = a2 - a1
        ft_str = f"{a3:.4f}" if a3 is not None else "  n/a "
        ft_gain = f"{a3 - a2:+.4f}" if a3 is not None else "  n/a "
        print(f"{r['display_name']:<20} {a1:>8.4f} {a2:>8.4f} {ft_str:>8}  {dp_gain:>+7.4f} {ft_gain:>7}")

        r["double_prompt_gain"] = round(dp_gain, 6)
        if a3 is not None:
            r["finetuning_gain"] = round(a3 - a2, 6)
            r["total_gain"] = round(a3 - a1, 6)

    print("-" * 72)

    # -- Save -----------------------------------------------------------------
    out = {"model": args.model, "adapter": args.adapter, "benchmarks": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
