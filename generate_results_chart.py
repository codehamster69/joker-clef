"""Generate results comparison chart for the presentation slide."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#2E86AB"
C_TEAL   = "#43AA8B"
C_ORANGE = "#F4A261"
C_RED    = "#E63946"
C_GOLD   = "#FFB703"
C_GRAY   = "#8D99AE"
BG       = "#F8F9FA"
GRID     = "#DEE2E6"

fig = plt.figure(figsize=(16, 7), facecolor=BG)
fig.suptitle(
    "JOKER Task 1 — System Performance Comparison",
    fontsize=17, fontweight="bold", color="#212529", y=0.98,
)

# ── Panel A: Pipeline Progression (ablation) ─────────────────────────────────
ax1 = fig.add_axes([0.04, 0.10, 0.44, 0.78])

systems = [
    "BM25 Baseline\n(CLEF 2024)",
    "BM25 +\nChar N-grams",
    "Hybrid\n(bge-small + static)",
    "Hybrid +\nRoBERTa (fine-tuned)",
    "Full System\n(bge-base + L2R + PRF)",
]
map_scores = [0.160, 0.195, 0.240, 0.288, 0.486]
bar_colors = [C_GRAY, C_GRAY, C_BLUE, C_TEAL, C_GOLD]
bar_edge   = ["#6c757d", "#6c757d", "#1a6080", "#2d7a62", "#d4a017"]

bars = ax1.barh(systems, map_scores, color=bar_colors, edgecolor=bar_edge,
                linewidth=1.2, height=0.55)

# Value labels
for bar, val, col in zip(bars, map_scores, bar_colors):
    x = bar.get_width()
    ax1.text(x + 0.005, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left",
             fontsize=11.5, fontweight="bold", color="#212529")

# Improvement arrows for key jumps
ax1.annotate("", xy=(0.240, 3.55), xytext=(0.160, 3.55),
             arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5))
ax1.annotate("", xy=(0.486, 4.55), xytext=(0.288, 4.55),
             arrowprops=dict(arrowstyle="-|>", color=C_GOLD, lw=2.0))
ax1.text(0.390, 4.70, "+202% vs baseline", fontsize=9, color=C_GOLD,
         fontweight="bold", ha="center")

ax1.set_xlim(0, 0.58)
ax1.set_xlabel("MAP@1000", fontsize=12, color="#495057")
ax1.set_title("Pipeline Progression (MAP@1000)", fontsize=13,
              fontweight="bold", color="#343a40", pad=8)
ax1.axvline(x=0.35, color=C_RED, linestyle="--", linewidth=1.2, alpha=0.7)
ax1.text(0.351, -0.6, "CLEF 2025\nSOTA ~0.35",
         fontsize=8.5, color=C_RED, va="top")
ax1.set_facecolor(BG)
ax1.grid(axis="x", color=GRID, linewidth=0.8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="y", labelsize=10.5)
ax1.tick_params(axis="x", labelsize=10)

# ── Panel B: Final System — All 6 Metrics ────────────────────────────────────
ax2 = fig.add_axes([0.56, 0.10, 0.42, 0.78])

metric_labels = ["MAP\n@1000", "NDCG\n@1000", "MRR", "Recall\n@1000", "P@10", "R-Prec"]
metric_values = [0.486,         0.646,          0.917, 0.674,           0.592,  0.485]
metric_colors = [C_GOLD, C_BLUE, "#9B59B6", C_TEAL, C_ORANGE, C_BLUE]

x = np.arange(len(metric_labels))
bars2 = ax2.bar(x, metric_values, color=metric_colors, edgecolor="white",
                linewidth=1.0, width=0.58)

# Value labels on top
for bar, val in zip(bars2, metric_values):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
             f"{val:.3f}", ha="center", va="bottom",
             fontsize=11.5, fontweight="bold", color="#212529")

ax2.set_ylim(0, 1.08)
ax2.set_xticks(x)
ax2.set_xticklabels(metric_labels, fontsize=11)
ax2.set_ylabel("Score", fontsize=12, color="#495057")
ax2.set_title("Full System — All Evaluation Metrics\n(bge-base + XGBoost L2R + PRF + RoBERTa)",
              fontsize=13, fontweight="bold", color="#343a40", pad=8)
ax2.set_facecolor(BG)
ax2.grid(axis="y", color=GRID, linewidth=0.8)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis="x", labelsize=11)
ax2.tick_params(axis="y", labelsize=10)

# Best metric highlight annotation
ax2.annotate("MRR = 0.917\n(top-1 hit on 91.7%\nof queries)",
             xy=(2, 0.917), xytext=(3.6, 0.80),
             fontsize=9, color="#6f42c1", fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color="#6f42c1", lw=1.2))

# ── Bottom note ───────────────────────────────────────────────────────────────
fig.text(0.5, 0.01,
         "Training set: 12 queries · 660 positive labels · 77,658-document corpus  "
         "|  Hardware: NVIDIA RTX 3050 Ti (4 GB VRAM)",
         ha="center", fontsize=9, color="#6c757d", style="italic")

out = "fig_results_comparison.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
