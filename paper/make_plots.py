"""Generate matplotlib figures for the repeated-prompts paper.

Reads ../results_{model}.json for each model we ran, and produces:

  figures/main_results.pdf      grouped bar chart per benchmark, per model
  figures/gain_decomposition.pdf   gain from double-prompting vs fine-tuning
  figures/held_out_summary.pdf     in-distribution vs held-out average gain
  figures/per_model_heatmap.pdf    matrix of total gains

All figures use matplotlib only, no seaborn.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

MODELS = [
    ("qwen2.5_1.5b", "Qwen2.5-1.5B"),
    ("mistral_7b",   "Mistral-7B"),
    ("qwen2.5_7b",   "Qwen2.5-7B"),
]

BENCH_ORDER = [
    ("arc",          "ARC-C",        True),
    ("openbookqa",   "OBQA",         True),
    ("gsm8k",        "GSM8K",        True),
    ("mmlu_pro",     "MMLU-Pro",     False),
    ("math",         "MATH",         False),
    ("name_index",   "NameIndex",    False),
    ("middle_match", "MiddleMatch",  False),
]

COLOR_SINGLE = "#6b7280"   # gray
COLOR_DOUBLE = "#3b82f6"   # blue
COLOR_FT     = "#ef4444"   # red

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load_all():
    out = {}
    for key, _ in MODELS:
        path = ROOT / f"results_{key}.json"
        with open(path) as f:
            out[key] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# Figure 1: Main grouped bar chart
# ---------------------------------------------------------------------------
def fig_main(results):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11.5, 3.2), sharey=False)

    width = 0.26
    x = np.arange(len(BENCH_ORDER))

    for ax, (mkey, mlabel) in zip(axes, MODELS):
        benches = results[mkey]["benchmarks"]
        singles = [benches[b]["vanilla_single"] for b, _, _ in BENCH_ORDER]
        doubles = [benches[b]["vanilla_double"] for b, _, _ in BENCH_ORDER]
        ftdbls  = [benches[b]["finetuned_double"] for b, _, _ in BENCH_ORDER]

        ax.bar(x - width, singles, width, color=COLOR_SINGLE,
               label="Vanilla + single")
        ax.bar(x,         doubles, width, color=COLOR_DOUBLE,
               label="Vanilla + double")
        ax.bar(x + width, ftdbls,  width, color=COLOR_FT,
               label="Fine-tuned + double")

        # mark in-distribution / held-out with a divider
        held_out_start = next(i for i, (_, _, is_tr) in enumerate(BENCH_ORDER)
                              if not is_tr)
        ax.axvline(held_out_start - 0.5, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in BENCH_ORDER],
                           rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Accuracy" if ax is axes[0] else "")
        ax.set_title(mlabel)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    # Annotation: held-out label area
    fig.text(0.5, -0.02,
             "Dashed line separates in-distribution (left) from held-out (right) benchmarks.",
             ha="center", fontsize=8, style="italic", color="#444")

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    _save(fig, "main_results")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Gain decomposition (double-prompt gain vs fine-tuning gain)
# ---------------------------------------------------------------------------
def fig_gain_decomp(results):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11.5, 3.2), sharey=True)

    width = 0.38
    x = np.arange(len(BENCH_ORDER))

    for ax, (mkey, mlabel) in zip(axes, MODELS):
        benches = results[mkey]["benchmarks"]
        dp   = [benches[b]["double_prompt_gain"]    for b, _, _ in BENCH_ORDER]
        ftg  = [benches[b]["finetuning_gain"]       for b, _, _ in BENCH_ORDER]

        ax.bar(x - width / 2, dp,  width, color=COLOR_DOUBLE,
               label="Double-prompt gain (cond. 2 − cond. 1)")
        ax.bar(x + width / 2, ftg, width, color=COLOR_FT,
               label="Fine-tuning gain (cond. 3 − cond. 2)")

        held_out_start = next(i for i, (_, _, is_tr) in enumerate(BENCH_ORDER)
                              if not is_tr)
        ax.axvline(held_out_start - 0.5, color="black", linewidth=0.7,
                   linestyle=":", alpha=0.6)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl, _ in BENCH_ORDER],
                           rotation=35, ha="right")
        if ax is axes[0]:
            ax.set_ylabel("Δ accuracy")
        ax.set_title(mlabel)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "gain_decomposition")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: In-dist vs held-out average gain per model
# ---------------------------------------------------------------------------
def fig_split_summary(results):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))

    labels = [lbl for _, lbl in MODELS]
    width = 0.2
    x = np.arange(len(labels))

    in_dp, in_ft, out_dp, out_ft = [], [], [], []
    for mkey, _ in MODELS:
        benches = results[mkey]["benchmarks"]
        in_dp.append(np.mean([benches[b]["double_prompt_gain"]
                              for b, _, tr in BENCH_ORDER if tr]))
        in_ft.append(np.mean([benches[b]["finetuning_gain"]
                              for b, _, tr in BENCH_ORDER if tr]))
        out_dp.append(np.mean([benches[b]["double_prompt_gain"]
                               for b, _, tr in BENCH_ORDER if not tr]))
        out_ft.append(np.mean([benches[b]["finetuning_gain"]
                               for b, _, tr in BENCH_ORDER if not tr]))

    ax.bar(x - 1.5 * width, in_dp,  width, color=COLOR_DOUBLE,
           label="In-dist: double-prompt")
    ax.bar(x - 0.5 * width, in_ft,  width, color=COLOR_FT,
           label="In-dist: fine-tuning")
    ax.bar(x + 0.5 * width, out_dp, width, color=COLOR_DOUBLE,
           hatch="///", edgecolor="white", label="Held-out: double-prompt")
    ax.bar(x + 1.5 * width, out_ft, width, color=COLOR_FT,
           hatch="///", edgecolor="white", label="Held-out: fine-tuning")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Δ accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    fig.tight_layout()
    _save(fig, "held_out_summary")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Heatmap of total gains (ft_double - vanilla_single)
# ---------------------------------------------------------------------------
def fig_heatmap(results):
    mat = np.zeros((len(MODELS), len(BENCH_ORDER)))
    for i, (mkey, _) in enumerate(MODELS):
        b = results[mkey]["benchmarks"]
        for j, (bkey, _, _) in enumerate(BENCH_ORDER):
            mat[i, j] = b[bkey]["total_gain"]

    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    vmax = np.max(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(BENCH_ORDER)))
    ax.set_xticklabels([lbl for _, lbl, _ in BENCH_ORDER], rotation=30,
                       ha="right")
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels([m for _, m in MODELS])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(val) > vmax * 0.55 else "black")

    held_out_start = next(i for i, (_, _, is_tr) in enumerate(BENCH_ORDER)
                          if not is_tr)
    ax.axvline(held_out_start - 0.5, color="black", linewidth=0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Total gain (cond. 3 − cond. 1)")

    fig.tight_layout()
    _save(fig, "per_model_heatmap")
    plt.close(fig)


def _save(fig, stem):
    """Save `fig` as both pdf (for LaTeX) and png (for quick previewing)."""
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", bbox_inches="tight", dpi=160)


def main():
    results = load_all()
    fig_main(results)
    fig_gain_decomp(results)
    fig_heatmap(results)
    print("Wrote figures to", FIG)


if __name__ == "__main__":
    main()
