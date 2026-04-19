"""Benchmark definitions for the repeated-prompts project.

Training benchmarks (have train splits): ARC-Challenge, OpenBookQA, GSM8K
Held-out eval benchmarks: MMLU-Pro, MATH, NameIndex (custom), MiddleMatch (custom)
"""

import random
import re

from datasets import concatenate_datasets, load_dataset


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_MC = (
    "You are a helpful assistant. Answer the multiple-choice question "
    "by responding with only the letter of the correct answer."
)

SYSTEM_PROMPT_OPEN = (
    "You are a helpful assistant. Solve the problem and respond with only "
    "the final numeric answer. Do not include units or explanation."
)

SYSTEM_PROMPT_MATH = (
    "You are a helpful assistant. Solve the math problem and respond with "
    "only the final answer. Do not include explanation."
)

SYSTEM_PROMPT_RETRIEVAL = (
    "You are a helpful assistant. Answer with only the requested name, "
    "nothing else."
)


# ---------------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------------

def format_mc(question: str, labels: list[str], texts: list[str]) -> str:
    lines = [f"Question: {question}"]
    for label, text in zip(labels, texts):
        lines.append(f"  {label}) {text}")
    return "\n".join(lines)


def format_open(question: str) -> str:
    return f"Question: {question}"


def make_double(text: str) -> str:
    return f"{text}\n\n{text}"


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def normalize_numeric(s: str) -> str:
    """Extract the last number (int or float) from a string."""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    numbers = re.findall(r"-?\d+\.?\d*", s)
    return numbers[-1] if numbers else s


def normalize_text(s: str) -> str:
    return s.strip().lower()


def extract_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return text.strip()
    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return text[start:].strip()


# ---------------------------------------------------------------------------
# Comparison functions (used during evaluation)
# ---------------------------------------------------------------------------

def compare_numeric(pred: str, gold: str) -> bool:
    p, g = normalize_numeric(pred), normalize_numeric(gold)
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return p == g


def compare_text(pred: str, gold: str) -> bool:
    pred = pred.strip().split("\n")[0].strip().rstrip(".").lower()
    gold = gold.strip().lower()
    return pred == gold or gold in pred


def compare_math(pred: str, gold: str) -> bool:
    if compare_numeric(pred, gold):
        return True
    p = pred.strip().lower().replace(" ", "")
    g = gold.strip().lower().replace(" ", "")
    return p == g


# ---------------------------------------------------------------------------
# Per-benchmark format / answer functions
# ---------------------------------------------------------------------------

# ARC-Challenge
def _arc_fmt(ex):
    return format_mc(ex["question"], ex["choices"]["label"], ex["choices"]["text"])

def _arc_ans(ex):
    return ex["answerKey"]

def _arc_choices(ex):
    return ex["choices"]["label"]

# OpenBookQA
def _obqa_fmt(ex):
    return format_mc(ex["question_stem"], ex["choices"]["label"], ex["choices"]["text"])

def _obqa_ans(ex):
    return ex["answerKey"]

def _obqa_choices(ex):
    return ex["choices"]["label"]

# GSM8K
def _gsm8k_fmt(ex):
    return format_open(ex["question"])

def _gsm8k_ans(ex):
    m = re.search(r"####\s*(.+)", ex["answer"])
    return m.group(1).strip().replace(",", "") if m else ex["answer"].strip()

# MMLU-Pro
def _mmlu_fmt(ex):
    labels = [chr(ord("A") + i) for i in range(len(ex["options"]))]
    return format_mc(ex["question"], labels, ex["options"])

def _mmlu_ans(ex):
    return ex["answer"]

def _mmlu_choices(ex):
    return [chr(ord("A") + i) for i in range(len(ex["options"]))]

# MATH
def _math_fmt(ex):
    return format_open(ex["problem"])

def _math_ans(ex):
    return extract_boxed(ex["solution"])

# Custom benchmarks
def _custom_fmt(ex):
    return ex["question"]

def _custom_ans(ex):
    return ex["answer"]


# ---------------------------------------------------------------------------
# Custom benchmark generators (NameIndex & MiddleMatch)
# ---------------------------------------------------------------------------

_FIRST = [
    "Dale", "Peter", "Allen", "Scott", "Hudson", "Daphne", "Dennis", "Henry",
    "Alfred", "Bruce", "Travis", "Rafael", "Richard", "Walter", "Caleb", "Ben",
    "Donald", "Mark", "Steven", "Talia", "James", "Craig", "Paul", "Samuel",
    "Jacob", "Douglas", "Orion", "Alexander", "Eugene", "Nelson", "Alan",
    "Alberto", "Robert", "Kenneth", "Jeffrey", "Chad", "Arthur", "Liam",
    "Leonard", "Carlos", "Stephen", "Gregory", "Raymond", "Finnian", "Daniel",
    "Thomas", "William", "Edward", "George", "Oliver",
]

_LAST = [
    "Lopez", "Sanchez", "Harris", "Davis", "Leviathan", "Kalman", "King",
    "Cooper", "Usher", "Ramirez", "Jennings", "Rogers", "Young", "Carter",
    "Sterling", "Nightingale", "Hanson", "Evans", "Fox", "Allen", "Johnson",
    "Wright", "Morrison", "Lee", "Ward", "Robinson", "McCarthy", "Price",
    "White", "Callahan", "Murphy", "Thomas", "James", "Curtis", "Cruz",
    "Collins", "Sims", "Ross", "Roberts", "Phillips",
]


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def generate_name_index(n_examples: int = 300, n_names: int = 50,
                        target_idx: int = 25, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        names = [f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
                 for _ in range(n_names)]
        question = (
            "Here's a list of names:\n"
            + ", ".join(names)
            + f"\n\nWhat's the {_ordinal(target_idx)} name?"
        )
        examples.append({"question": question, "answer": names[target_idx - 1]})
    return examples


def generate_middle_match(n_examples: int = 300, n_items: int = 40,
                          k_unique: int = 10, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples: list[dict] = []
    attempts = 0
    while len(examples) < n_examples and attempts < n_examples * 10:
        attempts += 1
        pool = [f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
                for _ in range(k_unique)]
        names = [rng.choice(pool) for _ in range(n_items)]
        valid = []
        for i in range(len(names) - 2):
            left, mid, right = names[i], names[i + 1], names[i + 2]
            if left != mid and mid != right and left != right:
                valid.append((left, mid, right))
        if not valid:
            continue
        left, mid, right = rng.choice(valid)
        question = (
            "Here's a list (potentially with repetitions) of names:\n"
            + ", ".join(names)
            + f"\n\nWhat is the single name that appears right between "
            + f"{left} and {right}?"
        )
        examples.append({"question": question, "answer": mid})
    return examples


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "arc": {
        "display_name": "ARC-Challenge",
        "hf_path": ("allenai/ai2_arc", "ARC-Challenge"),
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "format_fn": _arc_fmt,
        "answer_fn": _arc_ans,
        "choices_fn": _arc_choices,
        "eval_mode": "logit",
        "system_prompt": SYSTEM_PROMPT_MC,
        "is_train": True,
    },
    "openbookqa": {
        "display_name": "OpenBookQA",
        "hf_path": ("allenai/openbookqa", "main"),
        "splits": {"train": "train", "val": "validation", "test": "test"},
        "format_fn": _obqa_fmt,
        "answer_fn": _obqa_ans,
        "choices_fn": _obqa_choices,
        "eval_mode": "logit",
        "system_prompt": SYSTEM_PROMPT_MC,
        "is_train": True,
    },
    "gsm8k": {
        "display_name": "GSM8K",
        "hf_path": ("openai/gsm8k", "main"),
        "splits": {"train": "train", "test": "test"},
        "format_fn": _gsm8k_fmt,
        "answer_fn": _gsm8k_ans,
        "eval_mode": "generate",
        "compare_fn": compare_numeric,
        "system_prompt": SYSTEM_PROMPT_OPEN,
        "is_train": True,
        "max_new_tokens": 64,
    },
    "mmlu_pro": {
        "display_name": "MMLU-Pro",
        "hf_path": ("TIGER-Lab/MMLU-Pro",),
        "splits": {"val": "validation", "test": "test"},
        "format_fn": _mmlu_fmt,
        "answer_fn": _mmlu_ans,
        "choices_fn": _mmlu_choices,
        "eval_mode": "logit",
        "system_prompt": SYSTEM_PROMPT_MC,
        "is_train": False,
        "eval_subset": 2000,
    },
    "math": {
        "display_name": "MATH",
        # Loaded via _load_math_split (EleutherAI mirror); hf_path unused.
        "hf_path": ("EleutherAI/hendrycks_math",),
        "splits": {"train": "train", "test": "test"},
        "format_fn": _math_fmt,
        "answer_fn": _math_ans,
        "eval_mode": "generate",
        "compare_fn": compare_math,
        "system_prompt": SYSTEM_PROMPT_MATH,
        "is_train": False,
        "eval_subset": 1000,
        "max_new_tokens": 128,
    },
    "name_index": {
        "display_name": "NameIndex",
        "custom_generator": generate_name_index,
        "format_fn": _custom_fmt,
        "answer_fn": _custom_ans,
        "eval_mode": "generate",
        "compare_fn": compare_text,
        "system_prompt": SYSTEM_PROMPT_RETRIEVAL,
        "is_train": False,
        "max_new_tokens": 32,
    },
    "middle_match": {
        "display_name": "MiddleMatch",
        "custom_generator": generate_middle_match,
        "format_fn": _custom_fmt,
        "answer_fn": _custom_ans,
        "eval_mode": "generate",
        "compare_fn": compare_text,
        "system_prompt": SYSTEM_PROMPT_RETRIEVAL,
        "is_train": False,
        "max_new_tokens": 32,
    },
}

TRAIN_BENCHMARKS = [k for k, v in BENCHMARKS.items() if v.get("is_train")]
EVAL_BENCHMARKS = list(BENCHMARKS.keys())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# MATH: `hendrycks/competition_math` is often unavailable on the Hub; EleutherAI
# mirrors the same splits (7500 train / 5000 test) as seven subject configs.
_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def _load_math_split(hf_split: str):
    parts = [
        load_dataset("EleutherAI/hendrycks_math", subj, split=hf_split)
        for subj in _MATH_SUBJECTS
    ]
    return concatenate_datasets(parts)


def load_split(name: str, split: str):
    """Load a benchmark split.  Returns HF Dataset or list[dict] for custom."""
    cfg = BENCHMARKS[name]
    if "custom_generator" in cfg:
        if split != "test":
            raise ValueError(f"{name} is custom-generated and only has a 'test' split")
        return cfg["custom_generator"]()
    if name == "math":
        return _load_math_split(cfg["splits"][split])
    hf_split = cfg["splits"][split]
    return load_dataset(*cfg["hf_path"], split=hf_split)
