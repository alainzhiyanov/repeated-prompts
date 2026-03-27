"""
Download training benchmarks and create double-prompted JSONL files.

Training benchmarks: ARC-Challenge, OpenBookQA, GSM8K.
Benchmarks without a validation split (GSM8K) are split 90/10.

Usage:
    python prepare_data.py
"""

import json
import os
import random

from benchmarks import BENCHMARKS, TRAIN_BENCHMARKS, load_split, make_double
from utils import make_messages


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  → {path}  ({len(records)} examples)")


def main() -> None:
    os.makedirs("data", exist_ok=True)

    all_train: list[dict] = []
    all_val: list[dict] = []

    for name in TRAIN_BENCHMARKS:
        cfg = BENCHMARKS[name]
        print(f"\n[{cfg['display_name']}]")

        train_ds = load_split(name, "train")

        if "val" in cfg["splits"]:
            val_ds = load_split(name, "val")
        else:
            split = train_ds.train_test_split(test_size=0.1, seed=42)
            train_ds = split["train"]
            val_ds = split["test"]
            print(f"  (no val split — held out 10% of train)")

        fmt = cfg["format_fn"]
        ans = cfg["answer_fn"]
        sys_prompt = cfg["system_prompt"]

        train_recs = [
            {"messages": make_messages(sys_prompt, make_double(fmt(ex)), ans(ex))}
            for ex in train_ds
        ]
        val_recs = [
            {"messages": make_messages(sys_prompt, make_double(fmt(ex)), ans(ex))}
            for ex in val_ds
        ]

        _write_jsonl(f"data/{name}_train_double.jsonl", train_recs)
        _write_jsonl(f"data/{name}_val_double.jsonl", val_recs)

        all_train.extend(train_recs)
        all_val.extend(val_recs)

    random.Random(42).shuffle(all_train)
    random.Random(42).shuffle(all_val)

    print(f"\n[Combined]")
    _write_jsonl("data/train_double.jsonl", all_train)
    _write_jsonl("data/val_double.jsonl", all_val)
    print("Done.")


if __name__ == "__main__":
    main()
