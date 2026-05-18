"""
Generate all final result plots for the fall-detection experiments.

This merges:
- final_pipeline/results/plot_results.py
- final_pipeline/results/plots/recall_precision_chart.py
"""

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"
OUT_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/gen_ai_matplotlib"))
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

TECHNIQUES = [
    {
        "name": "Zero-Shot",
        "short": "Zero\nShot",
        "model": "GPT-4o-mini",
        "color": "#6C8EBF",
        "video_recall": 20.83,
        "video_precision": 100.00,
        "video_accuracy": 66.67,
        "video_f1": 34.48,
        "window_recall": 22.88,
        "window_precision": 90.00,
        "window_accuracy": 65.44,
    },
    {
        "name": "Few-Shot",
        "short": "Few\nShot",
        "model": "GPT-4o-mini",
        "color": "#6C8EBF",
        "video_recall": 58.33,
        "video_precision": 60.87,
        "video_accuracy": 66.67,
        "video_f1": 59.57,
        "window_recall": 48.31,
        "window_precision": 55.34,
        "window_accuracy": 60.66,
    },
    {
        "name": "Chain-of-Thought",
        "short": "Chain\nof\nThought",
        "model": "GPT-4o-mini",
        "color": "#6C8EBF",
        "video_recall": 29.17,
        "video_precision": 77.78,
        "video_accuracy": 66.67,
        "video_f1": 42.42,
        "window_recall": 39.13,
        "window_precision": 56.25,
        "window_accuracy": 60.53,
    },
    {
        "name": "Self-Consistency",
        "short": "Self\nConsis-\ntency",
        "model": "GPT-4o-mini",
        "color": "#6C8EBF",
        "video_recall": 41.67,
        "video_precision": 76.92,
        "video_accuracy": 70.18,
        "video_f1": 54.05,
        "window_recall": 45.76,
        "window_precision": 65.85,
        "window_accuracy": 66.18,
    },
    {
        "name": "Enhanced\nFew-Shot",
        "short": "Enhanced\nFew-Shot",
        "model": "GPT-4o",
        "color": "#82B366",
        "video_recall": 41.67,
        "video_precision": 100.00,
        "video_accuracy": 75.44,
        "video_f1": 58.82,
        "window_recall": 55.14,
        "window_precision": 79.73,
        "window_accuracy": 73.86,
    },
    {
        "name": "Enhanced\nFew-Shot v2",
        "short": "Enhanced\nFew-Shot\nv2",
        "model": "GPT-4o",
        "color": "#82B366",
        "video_recall": 87.50,
        "video_precision": 58.33,
        "video_accuracy": 68.42,
        "video_f1": 70.00,
        "window_recall": 84.75,
        "window_precision": 60.98,
        "window_accuracy": 69.85,
    },
    {
        "name": "RAG Pipeline",
        "short": "RAG\nPipeline",
        "model": "GPT-4o",
        "color": "#D6A64A",
        "video_recall": 85.42,
        "video_precision": 57.75,
        "video_accuracy": 67.83,
        "video_f1": 68.91,
        "window_recall": 64.77,
        "window_precision": 73.39,
        "window_accuracy": 71.40,
    },
    {
        "name": "Optimized Hybrid",
        "short": "Optim.\nHybrid",
        "model": "GPT-4o-mini",
        "color": "#B85C38",
        "video_recall": 89.47,
        "video_precision": 68.00,
        "video_accuracy": 75.00,
        "video_f1": 77.27,
        "window_recall": 82.61,
        "window_precision": 73.08,
        "window_accuracy": 72.50,
    },
    {
        "name": "Best Prompt\n(Safety-First)",
        "short": "Best\nPrompt",
        "model": "GPT-4o",
        "color": "#C0392B",
        "video_recall": 97.92,
        "video_precision": 50.00,
        "video_accuracy": 58.30,
        "video_f1": 65.71,
        "window_recall": 80.90,
        "window_precision": 57.67,
        "window_accuracy": 62.30,
    },
    {
        "name": "LoRA\nFine-Tune",
        "short": "LoRA\nFine-Tune",
        "model": "GPT-4o-mini (SFT)",
        "color": "#8E44AD",
        "video_recall": 93.75,
        "video_precision": 50.85,
        "video_accuracy": 52.31,
        "video_f1": 65.93,
        "window_recall": 89.41,
        "window_precision": 85.13,
        "window_accuracy": 79.62,
    },
    {
        "name": "XAI\nVideo-Level",
        "short": "XAI\nVideo",
        "model": "GPT-4o",
        "color": "#117A65",
        "video_recall": 84.0,
        "video_precision": 80.8,
        "video_accuracy": 82.0,
        "video_f1": 82.4,
        "window_recall": 84.0,
        "window_precision": 80.8,
        "window_accuracy": 82.0,
    },
    {
        "name": "XAI\n(Improved)",
        "short": "XAI\n(Impr.)",
        "model": "GPT-4o",
        "color": "#0E6655",
        "video_recall": 84.0,
        "video_precision": 85.5,
        "video_accuracy": 84.5,
        "video_f1": 84.8,
        "window_recall": 84.0,
        "window_precision": 85.5,
        "window_accuracy": 84.5,
    },
    {
        "name": "RAG\n(Improved)",
        "short": "RAG\n(Impr.)",
        "model": "GPT-4o",
        "color": "#A0522D",
        "video_recall": 89.58,
        "video_precision": 68.25,
        "video_accuracy": 78.26,
        "video_f1": 77.46,
        "window_recall": 67.62,
        "window_precision": 78.84,
        "window_accuracy": 75.39,
    },
]


def values(metric):
    return [technique[metric] for technique in TECHNIQUES]


def names(key="name"):
    return [technique[key] for technique in TECHNIQUES]


def colors():
    return [technique["color"] for technique in TECHNIQUES]


def setup_axes(ax):
    ax.set_facecolor("#F8F9FA")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def save(fig, filename):
    path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def bar_label(ax, bars, vals, offset=1.5, fontsize=7.5):
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )


def model_legend():
    return [
        mpatches.Patch(color="#6C8EBF", label="GPT-4o-mini"),
        mpatches.Patch(color="#82B366", label="GPT-4o"),
        mpatches.Patch(color="#D6A64A", label="GPT-4o (RAG)"),
        mpatches.Patch(color="#B85C38", label="GPT-4o-mini (Hybrid)"),
        mpatches.Patch(color="#C0392B", label="GPT-4o (Safety-First)"),
        mpatches.Patch(color="#8E44AD", label="GPT-4o-mini (LoRA/SFT)"),
    ]


def plot_video_fall_recall():
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    x = np.arange(len(TECHNIQUES))
    v_rec = values("video_recall")
    bars = ax.bar(x, v_rec, color=colors(), width=0.65, edgecolor="white", linewidth=1.2, zorder=3)

    best = max(v_rec)
    ax.axhline(best, color="#C0392B", linewidth=1.6, linestyle="--", alpha=0.7, zorder=2)
    ax.text(len(TECHNIQUES) - 0.7, best + 0.8, f"Best recall: {best:.1f}%", color="#C0392B", fontsize=9, fontweight="bold")
    bar_label(ax, bars, v_rec)

    ax.set_xticks(x)
    ax.set_xticklabels(names("short"), fontsize=8.5)
    ax.set_ylabel("Video-Level Fall Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Fall Detection - Video-Level Fall Recall by Technique\n"
        "(Higher is better; missed falls are the key risk)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, 115)
    ax.legend(handles=model_legend(), loc="upper left", fontsize=8.5, framealpha=0.85)
    save(fig, "1_video_fall_recall.png")


def plot_multi_metric_comparison():
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    x = np.arange(len(TECHNIQUES))
    width = 0.25
    ax.bar(x - width, values("video_recall"), width=width, label="Fall Recall", color="#E74C3C", edgecolor="white", zorder=3)
    ax.bar(x, values("video_accuracy"), width=width, label="Accuracy", color="#3498DB", edgecolor="white", zorder=3)
    ax.bar(x + width, values("video_f1"), width=width, label="Fall F1-Score", color="#2ECC71", edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names("short"), fontsize=8.5)
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Video-Level Metrics: Fall Recall vs Accuracy vs F1", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.85)
    save(fig, "2_multi_metric_comparison.png")


def plot_window_vs_video_recall():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    x = np.arange(len(TECHNIQUES))
    width = 0.35
    ax.bar(x - width / 2, values("window_recall"), width=width, label="Window-Level Recall", color="#5DADE2", edgecolor="white", zorder=3)
    ax.bar(x + width / 2, values("video_recall"), width=width, label="Video-Level Recall", color="#E74C3C", edgecolor="white", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names("short"), fontsize=8.5)
    ax.set_ylabel("Fall Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Window-Level vs Video-Level Fall Recall", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10, framealpha=0.85)
    save(fig, "3_window_vs_video_recall.png")


def plot_recall_vs_accuracy_tradeoff():
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    ax.axvline(75, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(80, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.fill_betweenx([80, 108], 75, 105, alpha=0.08, color="green", label="Ideal zone")

    offsets = {
        "Zero-Shot": (-5, 2),
        "Few-Shot": (-5, 2),
        "Chain-of-Thought": (1, 2),
        "Self-Consistency": (1, 2),
        "Enhanced\nFew-Shot": (1, -4),
        "Enhanced\nFew-Shot v2": (-18, 2),
        "RAG Pipeline": (1, 2),
        "Optimized Hybrid": (1, 2),
        "Best Prompt\n(Safety-First)": (-22, 2),
        "LoRA\nFine-Tune": (1, -4),
    }

    for technique in TECHNIQUES:
        xp = technique["video_accuracy"]
        yp = technique["video_recall"]
        ax.scatter(xp, yp, color=technique["color"], s=160, zorder=5, edgecolors="white", linewidth=1.2)
        dx, dy = offsets.get(technique["name"], (1, 2))
        ax.annotate(
            technique["name"].replace("\n", " "),
            (xp, yp),
            xytext=(xp + dx, yp + dy),
            fontsize=8,
            fontweight="bold",
            color=technique["color"],
            arrowprops=dict(arrowstyle="-", color=technique["color"], lw=0.6, alpha=0.6),
        )

    ax.set_xlabel("Video Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Video Fall Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title("Recall vs Accuracy Trade-off by Technique", fontsize=12, fontweight="bold", pad=14)
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    save(fig, "4_recall_vs_accuracy_tradeoff.png")


def plot_recall_ranking():
    sorted_techniques = sorted(TECHNIQUES, key=lambda technique: technique["video_recall"])
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    y = np.arange(len(sorted_techniques))
    vals = [technique["video_recall"] for technique in sorted_techniques]
    bars = ax.barh(y, vals, color=[technique["color"] for technique in sorted_techniques], height=0.6, edgecolor="white", zorder=3)

    best = max(vals)
    ax.axvline(best, color="#C0392B", linewidth=1.8, linestyle="--", alpha=0.8)
    ax.text(best + 0.3, len(sorted_techniques) - 0.8, f"Best: {best:.1f}%", color="#C0392B", fontsize=8.5, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([technique["name"].replace("\n", " ") for technique in sorted_techniques], fontsize=9.5)
    ax.set_xlabel("Video-Level Fall Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Fall Recall Ranking - All Techniques", fontsize=12, fontweight="bold", pad=14)
    ax.set_xlim(0, 112)
    save(fig, "5_recall_ranking.png")


def recall_precision_legend(include_style=True):
    patches = [
        mpatches.Patch(color="#6C8EBF", label="GPT-4o-mini"),
        mpatches.Patch(color="#82B366", label="GPT-4o"),
        mpatches.Patch(color="#D6A64A", label="GPT-4o (RAG)"),
        mpatches.Patch(color="#B85C38", label="GPT-4o-mini (Hybrid)"),
        mpatches.Patch(color="#C0392B", label="GPT-4o (Best Prompt)"),
        mpatches.Patch(color="#8E44AD", label="GPT-4o-mini (LoRA/SFT)"),
    ]
    if include_style:
        patches.extend(
            [
                mpatches.Patch(facecolor="white", edgecolor="gray", label="Solid = Recall"),
                mpatches.Patch(facecolor="white", edgecolor="gray", hatch="//", label="Hatched = Precision"),
            ]
        )
    return patches


def plot_recall_precision_bars(level, filename, title):
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    x = np.arange(len(TECHNIQUES))
    width = 0.38
    rec = values(f"{level}_recall")
    prec = values(f"{level}_precision")
    bars_rec = ax.bar(x - width / 2, rec, width=width, color=colors(), edgecolor="white", linewidth=1.2, zorder=3)
    bars_prec = ax.bar(x + width / 2, prec, width=width, color=colors(), edgecolor="white", linewidth=1.2, zorder=3, alpha=0.45, hatch="//")
    bar_label(ax, bars_rec, rec, offset=1.2, fontsize=7.8)
    bar_label(ax, bars_prec, prec, offset=1.2, fontsize=7.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names(), fontsize=8.5)
    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title(f"{title}\nSolid = Recall | Hatched = Precision", fontsize=13, fontweight="bold", pad=14)
    ax.legend(handles=recall_precision_legend(), loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    save(fig, filename)


def plot_recall_precision_scatter():
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F8F9FA")
    setup_axes(ax)

    for f1 in [0.4, 0.5, 0.6, 0.7, 0.8]:
        r_vals = np.linspace(0.01, 1.0, 300)
        p_vals = (f1 * r_vals) / (2 * r_vals - f1)
        mask = (p_vals >= 0) & (p_vals <= 1.05)
        ax.plot(r_vals[mask] * 100, p_vals[mask] * 100, linestyle=":", color="gray", linewidth=0.9, alpha=0.5)

    offsets = {
        "Zero-Shot": (2, 2),
        "Few-Shot": (2, 2),
        "Chain-of-Thought": (2, -4),
        "Self-Consistency": (2, 2),
        "Enhanced\nFew-Shot": (-18, 2),
        "Enhanced\nFew-Shot v2": (2, 2),
        "RAG Pipeline": (2, 2),
        "Optimized Hybrid": (2, 2),
        "Best Prompt\n(Safety-First)": (2, -5),
        "LoRA\nFine-Tune": (2, 2),
    }

    for technique in TECHNIQUES:
        xp = technique["video_recall"]
        yp = technique["video_precision"]
        ax.scatter(xp, yp, color=technique["color"], s=220, zorder=5, edgecolors="white", linewidth=1.5)
        dx, dy = offsets.get(technique["name"], (2, 2))
        ax.annotate(
            technique["name"].replace("\n", " "),
            (xp, yp),
            xytext=(xp + dx, yp + dy),
            fontsize=8.5,
            fontweight="bold",
            color=technique["color"],
            arrowprops=dict(arrowstyle="-", color=technique["color"], lw=0.7, alpha=0.5),
        )

    ax.set_xlabel("Video-Level Fall Recall (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Video-Level Fall Precision (%)", fontsize=12, fontweight="bold")
    ax.set_title("Recall vs Precision Trade-off - All Strategies (Video Level)", fontsize=12, fontweight="bold", pad=14)
    ax.set_xlim(10, 110)
    ax.set_ylim(30, 115)
    ax.legend(handles=recall_precision_legend(include_style=False), fontsize=8.5, loc="upper right", framealpha=0.9)
    save(fig, "C_recall_precision_scatter.png")


def plot_recall_precision_summary():
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.patch.set_facecolor("#F8F9FA")
    sorted_techniques = sorted(TECHNIQUES, key=lambda technique: technique["video_recall"])
    y = np.arange(len(sorted_techniques))

    for ax in axes:
        setup_axes(ax)

    recall = [technique["video_recall"] for technique in sorted_techniques]
    precision = [technique["video_precision"] for technique in sorted_techniques]
    sorted_colors = [technique["color"] for technique in sorted_techniques]
    labels = [technique["name"].replace("\n", " ") for technique in sorted_techniques]

    axes[0].barh(y, recall, color=sorted_colors, height=0.55, edgecolor="white", zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=9.5)
    axes[0].set_xlabel("Fall Recall (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Video-Level Fall Recall", fontsize=11, fontweight="bold", pad=10)

    axes[1].barh(y, precision, color=sorted_colors, height=0.55, edgecolor="white", zorder=3, alpha=0.75, hatch="//")
    axes[1].set_xlabel("Fall Precision (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Video-Level Fall Precision", fontsize=11, fontweight="bold", pad=10)

    for ax, vals in zip(axes, [recall, precision]):
        ax.set_xlim(0, 115)
        for yi, val in enumerate(vals):
            ax.text(val + 0.5, yi, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

    fig.suptitle("All Strategies - Fall Recall vs Precision (Sorted by Recall)", fontsize=14, fontweight="bold", y=1.01)
    save(fig, "D_recall_precision_summary.png")


def main():
    plot_video_fall_recall()
    plot_multi_metric_comparison()
    plot_window_vs_video_recall()
    plot_recall_vs_accuracy_tradeoff()
    plot_recall_ranking()
    plot_recall_precision_bars("video", "A_video_recall_precision.png", "Video-Level Fall Recall vs Precision")
    plot_recall_precision_bars("window", "B_window_recall_precision.png", "Window-Level Fall Recall vs Precision")
    plot_recall_precision_scatter()
    plot_recall_precision_summary()
    print(f"\nDone - all plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
