"""Generate publication-quality figures from eval JSON files.

Reads:
    outputs/results/main_eval.json        (6 systems, MLSUM-TR)
    outputs/results/ood_eval.json         (6 systems, TR-News)
    outputs/results/ablation_size.json    (S-gpt at 1k/5k/10k)
    outputs/results/ablation_lora.json    (S-gpt at LoRA rank 4/8/16/32)
    outputs/results/ablation_prompt.json  (concise vs detailed at 1k)

Writes (300 DPI PNGs to report/figures/):
    fig1_main_bar.png        Six-system comparison on MLSUM-TR (R1 stem + BSf1).
    fig2_scaling.png         Synthetic-dataset size scaling (R1 std vs n).
    fig3_lora.png            LoRA rank vs R1 std + hallucination rate.
    fig4_ood.png             MLSUM-TR vs TR-News R1 std per system.
    fig5_halluc_quality.png  Hallucination rate vs R1 std scatter, all systems.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / Colab-safe
import matplotlib.pyplot as plt

# ---- style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR = {
    "B1": "#9CA3AF",        # gray — zero-shot
    "B2": "#1F4E79",        # navy — human-supervised
    "B3a": "#D97706",       # amber — GPT teacher
    "B3b": "#B91C1C",       # red — Claude teacher
    "S-gpt": "#0E7490",     # teal — synthetic from GPT
    "S-claude": "#7C3AED",  # purple — synthetic from Claude
}

# ---- data loaders ----
def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def out_dir():
    p = Path("report/figures")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---- fig 1 — main results bar chart ----
def fig1(main):
    systems = ["B1", "B2", "B3a", "B3b", "S-gpt", "S-claude"]
    r1_stem = [main[s]["rouge_stem"]["rouge1"] for s in systems]
    bsf1 = [main[s]["bertscore"]["f1"] for s in systems]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = [COLOR[s] for s in systems]

    ax1.bar(systems, r1_stem, color=colors, edgecolor="white", linewidth=1.2)
    ax1.set_ylabel("ROUGE-1 (stem) F1")
    ax1.set_title("a. Lexical quality on MLSUM-TR test")
    ax1.set_ylim(0, max(r1_stem) * 1.18)
    for i, v in enumerate(r1_stem):
        ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    ax2.bar(systems, bsf1, color=colors, edgecolor="white", linewidth=1.2)
    ax2.set_ylabel("BERTScore F1")
    ax2.set_title("b. Semantic quality on MLSUM-TR test")
    ax2.set_ylim(0.80, max(bsf1) * 1.012)
    for i, v in enumerate(bsf1):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(out_dir() / "fig1_main_bar.png")
    plt.close()
    print("  fig1_main_bar.png saved")


# ---- fig 2 — synthetic-data scaling ----
def fig2(size_abl, main):
    # Three points on the size axis: 1k/5k/10k. Add B2 (human, 10k) as a horizontal reference.
    sizes = [1000, 5000, 10000]
    keys = ["S-gpt-1k", "S-gpt-5k", "S-gpt-10k"]
    r1 = [size_abl[k]["rouge_standard"]["rouge1"] for k in keys]
    bs = [size_abl[k]["bertscore"]["f1"] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(sizes, r1, "-o", color=COLOR["S-gpt"], linewidth=2, markersize=8, label="S-gpt (synthetic)")
    for i, (x, y) in enumerate(zip(sizes, r1)):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    # Horizontal reference: B2 (human supervised, 10k)
    b2_r1 = main["B2"]["rouge_standard"]["rouge1"]
    ax.axhline(b2_r1, color=COLOR["B2"], linestyle="--", linewidth=1.3,
                label=f"B2 human-supervised at 10k (R1={b2_r1:.3f})")
    # Teacher ceiling
    b3a_r1 = main["B3a"]["rouge_standard"]["rouge1"]
    ax.axhline(b3a_r1, color=COLOR["B3a"], linestyle=":", linewidth=1.3,
                label=f"B3a GPT teacher ceiling (R1={b3a_r1:.3f})")

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{x // 1000}k" for x in sizes])
    ax.set_xlabel("Synthetic dataset size (log scale)")
    ax.set_ylabel("ROUGE-1 (standard) F1")
    ax.set_title("Student quality vs. synthetic dataset size")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_dir() / "fig2_scaling.png")
    plt.close()
    print("  fig2_scaling.png saved")


# ---- fig 3 — LoRA rank curve ----
def fig3(lora_abl, main):
    ranks = [4, 8, 16, 32]
    keys = ["S-gpt-r4", "S-gpt-r8", "S-gpt-r16", "S-gpt-r32"]
    r1 = [lora_abl[k]["rouge_standard"]["rouge1"] for k in keys]
    halluc = [lora_abl[k]["errors"]["frac_hallucinated_numbers"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax2 = ax1.twinx()

    line1 = ax1.plot(ranks, r1, "-o", color=COLOR["S-gpt"], linewidth=2, markersize=8, label="ROUGE-1 std (left axis)")
    for x, y in zip(ranks, r1):
        ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color=COLOR["S-gpt"])

    line2 = ax2.plot(ranks, halluc, "-s", color=COLOR["B3b"], linewidth=2, markersize=7, label="halluc# (right axis)")
    for x, y in zip(ranks, halluc):
        ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, -4), ha="left", fontsize=9, color=COLOR["B3b"])

    # Teacher reference line
    b3a_r1 = main["B3a"]["rouge_standard"]["rouge1"]
    ax1.axhline(b3a_r1, color=COLOR["B3a"], linestyle=":", linewidth=1.3, label=f"GPT teacher R1={b3a_r1:.3f}")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(ranks)
    ax1.set_xticklabels([str(r) for r in ranks])
    ax1.set_xlabel("LoRA rank (log scale)")
    ax1.set_ylabel("ROUGE-1 (standard) F1", color=COLOR["S-gpt"])
    ax2.set_ylabel("Hallucinated-number fraction", color=COLOR["B3b"])
    ax1.tick_params(axis="y", colors=COLOR["S-gpt"])
    ax2.tick_params(axis="y", colors=COLOR["B3b"])
    ax1.set_title("LoRA rank: quality climbs, faithfulness drops past rank 16")
    ax1.grid(axis="y", alpha=0.25)

    lines = line1 + line2
    labels = [l.get_label() for l in lines] + [f"GPT teacher R1={b3a_r1:.3f}"]
    ax1.legend(lines + [ax1.lines[-1]], labels, loc="upper left", frameon=False)

    plt.tight_layout()
    plt.savefig(out_dir() / "fig3_lora.png")
    plt.close()
    print("  fig3_lora.png saved")


# ---- fig 4 — OOD generalization ----
def fig4(main, ood):
    systems = ["B1", "B2", "B3a", "B3b", "S-gpt", "S-claude"]
    mlsum = [main[s]["rouge_standard"]["rouge1"] for s in systems]
    trnews = [ood[s]["rouge_standard"]["rouge1"] for s in systems]

    x = range(len(systems))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    bars1 = ax.bar([i - w / 2 for i in x], mlsum, w, color="#2E75B6", label="MLSUM-TR (in-domain)", edgecolor="white")
    bars2 = ax.bar([i + w / 2 for i in x], trnews, w, color="#D97706", label="TR-News (out-of-domain)", edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylabel("ROUGE-1 (standard) F1")
    ax.set_title("In-domain vs. out-of-domain: teachers drop, students stay flat")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars1, mlsum):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, trnews):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir() / "fig4_ood.png")
    plt.close()
    print("  fig4_ood.png saved")


# ---- fig 5 — hallucination vs quality scatter ----
def fig5(main):
    systems = ["B1", "B2", "B3a", "B3b", "S-gpt", "S-claude"]
    descriptions = {
        "B1":       "B1 - zero-shot mT5-small",
        "B2":       "B2 - human-supervised",
        "B3a":      "B3a - GPT-4o-mini (teacher)",
        "B3b":      "B3b - Claude Haiku 4.5 (teacher)",
        "S-gpt":    "S-gpt - synthetic, distilled from GPT-4o-mini",
        "S-claude": "S-claude - synthetic, distilled from Claude Haiku 4.5",
    }
    markers = {  # different shapes group teachers vs students vs zero-shot
        "B1": "X", "B2": "s", "B3a": "^", "B3b": "^", "S-gpt": "o", "S-claude": "o",
    }
    x = [main[s]["rouge_standard"]["rouge1"] for s in systems]
    y = [main[s]["errors"]["frac_hallucinated_numbers"] for s in systems]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    for s, xi, yi in zip(systems, x, y):
        ax.scatter(
            xi, yi,
            s=220,
            color=COLOR[s],
            marker=markers[s],
            edgecolor="white",
            linewidth=2,
            zorder=3,
            label=descriptions[s],
        )

    # Axis range with breathing room
    xmin = min(x) - 0.02
    xmax = max(x) + 0.025
    ymin = -0.003
    ymax = max(y) * 1.10
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Shade the desirable region (high quality, low hallucination)
    desirable_x = 0.20
    desirable_y_top = 0.010
    ax.axvspan(
        desirable_x, xmax,
        ymin=0,
        ymax=(desirable_y_top - ymin) / (ymax - ymin),
        color="#10B981", alpha=0.10, zorder=0,
    )
    ax.annotate(
        "desirable region\n(high quality, low hallucination)",
        xy=(xmax - 0.005, ymin + 0.0015),
        ha="right", va="bottom",
        fontsize=9, style="italic", color="#047857",
    )

    ax.set_xlabel("ROUGE-1 (standard) F1 - quality")
    ax.set_ylabel("Hallucinated-number fraction - faithfulness risk")
    ax.set_title("Quality vs. faithfulness: students dominate the desirable quadrant")
    ax.grid(alpha=0.25)

    # Legend outside the data region. Two columns for compactness.
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        borderaxespad=0,
    )

    plt.tight_layout()
    plt.savefig(out_dir() / "fig5_halluc_quality.png")
    plt.close()
    print("  fig5_halluc_quality.png saved")

# ---- main ----
def main():
    base = Path("outputs/results")
    main_eval = load(base / "main_eval.json")
    ood_eval = load(base / "ood_eval.json")
    size_abl = load(base / "ablation_size.json")
    lora_abl = load(base / "ablation_lora.json")

    fig1(main_eval)
    fig2(size_abl, main_eval)
    fig3(lora_abl, main_eval)
    fig4(main_eval, ood_eval)
    fig5(main_eval)

    print("\nAll figures saved to report/figures/")


if __name__ == "__main__":
    main()
