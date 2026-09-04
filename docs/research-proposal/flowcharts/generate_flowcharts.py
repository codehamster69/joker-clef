"""
Generates the flowchart images used in the M.Tech research-proposal presentation
"Process-Aware Authenticity Detection in Supervised Programming Labs".

Run: python3 generate_flowcharts.py
Outputs PNGs into this same directory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- palette --
NAVY = "#1B2A4A"
BLUE = "#2E5AAC"
TEAL = "#1B8A8A"
LIGHT_BLUE = "#E7EEF9"
LIGHT_TEAL = "#E4F4F3"
AMBER = "#C9720B"
LIGHT_AMBER = "#FBEBD9"
RED = "#B23A48"
LIGHT_RED = "#F8E6E8"
GREY = "#5A6472"
LIGHT_GREY = "#F2F3F5"
WHITE = "#FFFFFF"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, xy, w, h, text, facecolor=LIGHT_BLUE, edgecolor=BLUE,
        fontsize=11, fontweight="bold", textcolor=NAVY, lw=1.8, rounding=0.06,
        zorder=3, linestyle="solid"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
        zorder=zorder, linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, fontweight=fontweight, color=textcolor,
             zorder=zorder + 1, wrap=True)
    return patch


def arrow(ax, start, end, color=GREY, lw=2.0, connectionstyle="arc3,rad=0.0",
          style="-|>", zorder=2, mutation_scale=18, linestyle="solid"):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=zorder,
        mutation_scale=mutation_scale, linestyle=linestyle,
    )
    ax.add_patch(a)
    return a


def new_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=220)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    return fig, ax


# =====================================================================
# 1. RESEARCH GAP DIAGRAM
# =====================================================================
def fig_research_gap():
    W, H = 12, 7.4
    fig, ax = new_fig(W, H)

    ax.text(W / 2, H - 0.35, "Two Active Literatures, Neither Built for This",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

    # Left cluster - Process-Driven
    box(ax, (0.4, 4.55), 5.1, 1.55,
        "Thread 1\nProcess-Driven Authenticity Detection",
        facecolor=LIGHT_BLUE, edgecolor=BLUE, fontsize=12.5, textcolor=NAVY)
    box(ax, (0.4, 3.35), 5.1, 1.05,
        "Take-home / asynchronous cheating\nSignal: keystroke timing, edit sequences\nKey work: Nosi IDE (2026), Crossley et al. (2024)",
        facecolor=WHITE, edgecolor=BLUE, fontsize=9.3, fontweight="normal",
        textcolor=GREY, lw=1.2)

    # Right cluster - Physical Proxy Attendance
    box(ax, (6.5, 4.55), 5.1, 1.55,
        "Thread 2\nPhysical Proxy Attendance Detection",
        facecolor=LIGHT_TEAL, edgecolor=TEAL, fontsize=12.5, textcolor=NAVY)
    box(ax, (6.5, 3.35), 5.1, 1.05,
        "Impersonation at check-in\nSignal: face match, RFID, weight sensors\nKey work: Ege & Özdemir (2026), CCTV/NFC systems",
        facecolor=WHITE, edgecolor=TEAL, fontsize=9.3, fontweight="normal",
        textcolor=GREY, lw=1.2)

    # Converging arrows down to gap box
    arrow(ax, (2.95, 3.35), (5.55, 2.35), color=BLUE, connectionstyle="arc3,rad=-0.15")
    arrow(ax, (9.05, 3.35), (6.45, 2.35), color=TEAL, connectionstyle="arc3,rad=0.15")

    # Gap box
    box(ax, (1.6, 0.55), 8.8, 1.75,
        "UNADDRESSED GAP\nLive, in-person authorship verification\nin a supervised lab session",
        facecolor=LIGHT_AMBER, edgecolor=AMBER, fontsize=13.5, textcolor="#7A3E00", lw=2.4)

    ax.text(W / 2, 0.15,
            "Neither literature covers a student who is physically present and identity-verified,\n"
            "while someone else does the actual coding, live, under supervision.",
            ha="center", va="center", fontsize=9.6, color=GREY, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "01_research_gap.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 2. SYSTEM ARCHITECTURE (4-stage pipeline)
# =====================================================================
def fig_system_architecture():
    W, H = 13, 4.1
    fig, ax = new_fig(W, H)

    ax.text(W / 2, H - 0.35, "Process-Instrumented Lab IDE — System Architecture",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

    stages = [
        ("1. Event Collector",
         "Captures keystrokes, pastes,\nruns, edits & timestamps\nduring the lab session",
         LIGHT_BLUE, BLUE),
        ("2. Feature Extraction",
         "Keystroke inter-arrival time (IAT)\nEdit-sequence shape\nAST-diff structural jumps",
         LIGHT_TEAL, TEAL),
        ("3. Authenticity Classifier",
         "Classifies each session:\ngenuine / proxy-typed /\nminimal-interaction",
         LIGHT_AMBER, AMBER),
        ("4. Instructor Dashboard",
         "Session-level flag with a\nsupporting evidence trace",
         LIGHT_RED, RED),
    ]

    n = len(stages)
    bw, bh = 2.7, 1.0
    gap = (W - n * bw) / (n + 1)
    y_top = 2.55
    y_desc = 0.95

    xs = []
    for i, (title, desc, fc, ec) in enumerate(stages):
        x = gap + i * (bw + gap)
        xs.append(x)
        box(ax, (x, y_top), bw, bh, title, facecolor=fc, edgecolor=ec,
            fontsize=11.5, textcolor=NAVY)
        box(ax, (x, y_desc), bw, 1.35, desc, facecolor=WHITE, edgecolor=ec,
            fontsize=9, fontweight="normal", textcolor=GREY, lw=1.1)

    for i in range(n - 1):
        x_start = xs[i] + bw
        x_end = xs[i + 1]
        y = y_top + bh / 2
        arrow(ax, (x_start + 0.05, y), (x_end - 0.05, y), color=GREY, lw=2.4,
              mutation_scale=22)

    ax.text(W / 2, 0.35,
            "Runs on lab workstations already under instructor supervision — enabling a controlled,\n"
            "multi-group ground-truth study instead of a small take-home proof-of-concept.",
            ha="center", va="center", fontsize=9.6, color=GREY, style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "02_system_architecture.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 3. CONTROLLED GROUND-TRUTH STUDY & VALIDATION DESIGN
# =====================================================================
def fig_study_design():
    W, H = 12.5, 8.4
    fig, ax = new_fig(W, H)

    ax.text(W / 2, H - 0.35, "Controlled Ground-Truth Study & Validation Design",
            ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)

    # Three group boxes
    groups = [
        ("Group A — Genuine",
         "Student solves the lab\nexercise themselves,\nunder normal conditions", LIGHT_BLUE, BLUE),
        ("Group B — Proxy-Typed",
         "A second person types the\nsolution live at the student's\nterminal", LIGHT_RED, RED),
        ("Group C — Minimal Interaction",
         "Student pastes/inserts a\nnear-complete solution with\nlittle iteration", LIGHT_AMBER, AMBER),
    ]
    bw, bh = 3.5, 1.5
    gap = (W - 3 * bw) / 4
    y_grp = 6.35
    xs = []
    for i, (title, desc, fc, ec) in enumerate(groups):
        x = gap + i * (bw + gap)
        xs.append(x + bw / 2)
        box(ax, (x, y_grp), bw, bh, title + "\n\n" + desc, facecolor=fc, edgecolor=ec,
            fontsize=10.3, textcolor=NAVY)

    # Merge arrows into instrumented sessions box
    merge_y = 5.55
    sessions_box_y = 4.55
    for x_c in xs:
        arrow(ax, (x_c, y_grp), (W / 2, sessions_box_y + 0.85), color=GREY, lw=1.8,
              connectionstyle="arc3,rad=0.0", mutation_scale=16)

    box(ax, (W / 2 - 3.1, sessions_box_y), 6.2, 0.85,
        "Instrumented Lab Sessions (fully logged)",
        facecolor=LIGHT_TEAL, edgecolor=TEAL, fontsize=12, textcolor=NAVY)

    # Branch into 3 validation layers
    val_y = 2.6
    vals = [
        ("1. Controlled Sessions", "Known ground-truth\nlabel per session"),
        ("2. Independent Instructor Labeling", "Subset labeled:\ngenuine / suspicious / unclear"),
        ("3. Held-out Problem Retest", "Can the student solve a\nsimilar task independently after?"),
    ]
    bw2, bh2 = 3.6, 1.35
    gap2 = (W - 3 * bw2) / 4
    xs2 = []
    for i, (title, desc) in enumerate(vals):
        x = gap2 + i * (bw2 + gap2)
        xs2.append(x + bw2 / 2)
        box(ax, (x, val_y), bw2, bh2, title + "\n\n" + desc,
            facecolor=WHITE, edgecolor=TEAL, fontsize=9.6, fontweight="normal", textcolor=GREY, lw=1.3)
        arrow(ax, (x + bw2 / 2, sessions_box_y), (x + bw2 / 2, val_y + bh2), color=TEAL, lw=1.8,
              mutation_scale=16)

    # Converge to evaluation box
    eval_y = 0.55
    box(ax, (W / 2 - 3.4, eval_y), 6.8, 1.05,
        "Authenticity Classifier — Evaluation",
        facecolor=LIGHT_AMBER, edgecolor=AMBER, fontsize=13, textcolor="#7A3E00", lw=2.2)
    for x_c in xs2:
        arrow(ax, (x_c, val_y), (W / 2, eval_y + 1.05), color=GREY, lw=1.6, mutation_scale=14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "03_study_design.png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 4. RESEARCH ROADMAP (thesis narrative, vertical)
# =====================================================================
def fig_roadmap():
    W, H = 8.6, 9.6
    fig, ax = new_fig(W, H)

    ax.text(W / 2, H - 0.35, "Research Roadmap", ha="center", va="center",
            fontsize=16, fontweight="bold", color=NAVY)

    stages = [
        ("The Problem", "Labs verify presence,\nnot authorship of live work", LIGHT_GREY, GREY),
        ("Literature Gap", "Process-driven detection (take-home) +\nphysical attendance systems (checkpoint only)\n= live in-person authorship unaddressed",
         LIGHT_AMBER, AMBER),
        ("Research Question", "Do process signals (keystroke IAT, edit\nsequences, AST jumps) distinguish authentic\nfrom proxy-typed in-lab work?", LIGHT_BLUE, BLUE),
        ("System + Study Design", "Process-instrumented lab IDE\n+ controlled Groups A/B/C", LIGHT_TEAL, TEAL),
        ("Validation", "Ground truth + instructor labeling\n+ held-out retest", LIGHT_TEAL, TEAL),
        ("Expected Contributions", "First controlled live-vs-take-home comparison;\nscoped instructor-facing flagging tool", LIGHT_RED, RED),
    ]

    n = len(stages)
    bh = 1.35
    top_margin = 0.75
    total_gap = H - top_margin - n * bh
    gap = total_gap / n
    bw = 7.2
    x = (W - bw) / 2

    y = H - top_margin
    ys = []
    for title, desc, fc, ec in stages:
        y -= bh
        ys.append(y)
        box(ax, (x, y), bw, bh, title + "\n" + desc, facecolor=fc, edgecolor=ec,
            fontsize=10.2, textcolor=NAVY)
        y -= gap

    for i in range(n - 1):
        y_start = ys[i]
        y_end = ys[i + 1] + bh
        arrow(ax, (W / 2, y_start), (W / 2, y_end), color=GREY, lw=2.2, mutation_scale=20)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "04_research_roadmap.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_research_gap()
    fig_system_architecture()
    fig_study_design()
    fig_roadmap()
    print("Generated flowcharts in", OUTDIR)
