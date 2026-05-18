"""
Before vs After improvement chart for LoRA and RAG.
Matches the style of results_plot.py (values on top of bars).
Run: python final_pipeline/results/comparison_plot.py
"""

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/gen_ai_matplotlib"))
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Before / After data (video-level) ─────────────────────────────────────────

DATA = [
    {
        "label":     "RAG\n(Before)",
        "recall":    85.42,
        "precision": 57.75,
        "color":     "#D6A64A",
        "alpha":     1.0,
    },
    {
        "label":     "RAG\n(Improved)",
        "recall":    89.58,
        "precision": 68.25,
        "color":     "#D6A64A",
        "alpha":     0.50,
    },
]


def setup_axes(ax):
    ax.set_facecolor("#F8F9FA")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def bar_label(ax, bars, vals, offset=1.5, fontsize=9):
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


def save(fig, filename):
    path = OUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Plot ───────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("#F8F9FA")
setup_axes(ax)

x      = np.arange(len(DATA))
width  = 0.35
labels = [d["label"] for d in DATA]
recs   = [d["recall"]    for d in DATA]
precs  = [d["precision"] for d in DATA]
cols   = [d["color"]     for d in DATA]
alphas = [d["alpha"]     for d in DATA]

bars_rec  = []
bars_prec = []
for i, (r, p, c, a) in enumerate(zip(recs, precs, cols, alphas)):
    br = ax.bar(i - width / 2, r, width=width, color=c, alpha=a,
                edgecolor="white", linewidth=1.2, zorder=3)
    bp = ax.bar(i + width / 2, p, width=width, color=c, alpha=a,
                edgecolor="white", linewidth=1.2, zorder=3, hatch="//")
    bars_rec.append(br[0])
    bars_prec.append(bp[0])

bar_label(ax, bars_rec,  recs,  offset=1.2, fontsize=9)
bar_label(ax, bars_prec, precs, offset=1.2, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
ax.set_ylim(0, 115)
ax.set_title(
    "RAG — Before vs Improved\n"
    "Solid = Fall Recall  |  Hatched = Fall Precision  (Video Level)",
    fontsize=13, fontweight="bold", pad=14,
)

legend_handles = [
    mpatches.Patch(color="#8E44AD", label="LoRA Fine-Tune"),
    mpatches.Patch(color="#D6A64A", label="RAG Pipeline"),
    mpatches.Patch(facecolor="white", edgecolor="gray", label="Solid = Recall"),
    mpatches.Patch(facecolor="white", edgecolor="gray", hatch="//", label="Hatched = Precision"),
    mpatches.Patch(facecolor="gray",  alpha=1.0, label="Before"),
    mpatches.Patch(facecolor="gray",  alpha=0.4, label="Projected"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9, ncol=2)

save(fig, "E_lora_rag_before_after.png")
print("Done.")
