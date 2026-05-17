"""Dataset figure: numbers + annotated query/doc/qrel example."""
import json, textwrap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

with open("joker_task1_retrieval_corpus25_EN.json") as f:
    corpus = json.load(f)
with open("joker_task1_retrieval_queries_train25_EN.json") as f:
    train_q = json.load(f)
with open("joker_task1_retrieval_qrels_train25_EN.json") as f:
    qrels = json.load(f)

corpus_map   = {str(d["docid"]): d["text"] for d in corpus if d.get("text")}
qid_to_query = {str(q["qid"]): q["query"] for q in train_q}

# pick a short, clear example: query="death", docid=634 ("autopsy is a dying practice")
ex_qid    = "19"
ex_query  = qid_to_query[ex_qid]           # "death"
ex_qrel   = next(r for r in qrels if str(r["qid"]) == ex_qid
                 and 10 <= len(corpus_map.get(str(r["docid"]), "").split()) <= 18)
ex_docid  = str(ex_qrel["docid"])
ex_relval = ex_qrel["qrel"]
ex_doc    = corpus_map[ex_docid]

# ── Palette ───────────────────────────────────────────────────────────────────
C_BLUE   = "#2E86AB"
C_TEAL   = "#43AA8B"
C_ORANGE = "#F4A261"
C_GREEN  = "#2DC653"
BG       = "#F8F9FA"

fig = plt.figure(figsize=(13, 6), facecolor=BG)

# ─────────────────────────────────────────────────────────────────────────────
# TOP ROW — three stat numbers
# ─────────────────────────────────────────────────────────────────────────────
stat_specs = [
    ("77,658",  "Documents in Corpus",      C_BLUE),
    ("231",     "Queries  (12 train · 219 test)", C_TEAL),
    ("660",     "Relevance Labels (training)", C_ORANGE),
]
for i, (val, label, col) in enumerate(stat_specs):
    ax = fig.add_axes([0.04 + i * 0.325, 0.72, 0.28, 0.23])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.04",
                                facecolor="white", edgecolor=col, linewidth=2))
    ax.text(0.5, 0.58, val, ha="center", va="center",
            fontsize=26, fontweight="bold", color=col)
    ax.text(0.5, 0.15, label, ha="center", va="center",
            fontsize=9.5, color="#555", multialignment="center")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION LABEL
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.685, "Example — How the three files fit together",
         ha="center", fontsize=11, color="#6c757d", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM ROW — Query | Doc | Qrel cards with connecting arrows
# ─────────────────────────────────────────────────────────────────────────────
card_w, card_h = 0.27, 0.52
positions = [0.03, 0.365, 0.695]          # left edges of the three cards
card_colors = [C_BLUE, C_TEAL, C_ORANGE]
card_titles = ["queries.json", "corpus.json", "qrels.json"]

doc_wrapped = textwrap.fill(ex_doc[:150], width=34)

card_bodies = [
    # Query card
    [("qid",   f'"{ex_qid}"'),
     ("query", f'"{ex_query}"')],
    # Document card
    [("docid", f'"{ex_docid}"'),
     ("text",  f'"{doc_wrapped}"')],
    # Qrel card
    [("qid",    f'"{ex_qid}"'),
     ("docid",  f'"{ex_docid}"'),
     ("qrel",   f'{ex_relval}  ← relevant')],
]

for i, (x0, col, title, rows) in enumerate(
        zip(positions, card_colors, card_titles, card_bodies)):
    ax = fig.add_axes([x0, 0.05, card_w, card_h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # card border
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03",
                                facecolor="white", edgecolor=col, linewidth=2))
    # colour header strip
    ax.add_patch(FancyBboxPatch((0, 0.84), 1, 0.16, boxstyle="square,pad=0",
                                facecolor=col, edgecolor="none"))
    ax.text(0.5, 0.915, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white")
    # rows of key: value
    y = 0.74
    for key, val in rows:
        ax.text(0.07, y, key + ":", fontsize=9.5, color="#888",
                va="top", style="italic")
        ax.text(0.07, y - 0.10, val, fontsize=9.5, color="#212529",
                va="top", linespacing=1.35)
        y -= 0.27

# Arrows between cards (drawn on figure coordinates)
for x_start, x_end in [(0.310, 0.358), (0.638, 0.687)]:
    ax_arr = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax_arr.set_xlim(0, 1); ax_arr.set_ylim(0, 1); ax_arr.axis("off")
    ax_arr.annotate("", xy=(x_end, 0.33), xytext=(x_start, 0.33),
                    xycoords="figure fraction", textcoords="figure fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#adb5bd",
                                   lw=2, mutation_scale=18))
    # link label
    mid_x = (x_start + x_end) / 2
    ax_arr.text(mid_x, 0.375, "linked by\ndocid / qid",
                ha="center", va="center", fontsize=8, color="#adb5bd",
                transform=ax_arr.transData)

plt.savefig("fig_dataset_viz.png", dpi=180, bbox_inches="tight", facecolor=BG)
print("Saved: fig_dataset_viz.png")
