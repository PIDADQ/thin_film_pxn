# -*- coding: utf-8 -*-
"""Draw the three schematic figures (workflow, physical model, MLP).

  - figures/fig1_workflow.png   overall study workflow
  - figures/fig2_physical.png   multilayer stack + TMM data flow
  - figures/fig3_mlp.png        MLP architecture 4-128-128-64-41
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_BOX = "#e8f0fe"
C_EDGE = "#1a5fb4"
C_ACC = "#fde7e9"
C_ACC_EDGE = "#c0392b"


def box(ax, x, y, w, h, text, fc=C_BOX, ec=C_EDGE, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 linewidth=1.2, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle="-|>", mutation_scale=14, color=color, lw=1.3))


def fig1_workflow():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.3, 4.6)
    ax.axis("off")

    steps = [
        (0.3, 3.3, 2.6, 1.0, "TMM physical\nsimulation", C_BOX, C_EDGE),
        (3.6, 3.3, 2.6, 1.0, "Dataset\ngeneration\n(5000 samples)", C_BOX, C_EDGE),
        (6.9, 3.3, 2.8, 1.0, "MLP surrogate\n4-128-128-64-41", C_ACC, C_ACC_EDGE),
        (6.9, 1.6, 2.8, 1.0, "MLP screening\n(10000 candidates)", C_ACC, C_ACC_EDGE),
        (3.6, 1.6, 2.6, 1.0, "TMM verification\n(Top-10)", C_BOX, C_EDGE),
        (0.3, 1.6, 2.6, 1.0, "Training-size\nexperiment", C_BOX, C_EDGE),
    ]
    for x, y, w, h, t, fc, ec in steps:
        box(ax, x, y, w, h, t, fc, ec, fs=9)

    arrow(ax, 2.9, 3.8, 3.6, 3.8)
    arrow(ax, 6.2, 3.8, 6.9, 3.8)
    arrow(ax, 8.3, 3.3, 8.3, 2.6)
    arrow(ax, 6.9, 2.1, 6.2, 2.1)
    arrow(ax, 3.6, 2.1, 2.9, 2.1)
    arrow(ax, 1.6, 3.3, 1.6, 2.6)

    ax.text(5.0, 4.35, "Overall workflow of the AI4S thin-film study",
            ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_workflow.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def fig2_physical():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # --- left: multilayer stack -------------------------------------------
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    layers = [
        ("Air", 0.90, 1.00, "#ffffff", "n0 = 1.00"),
        ("H  nH = 2.30, d1", 0.74, 0.90, "#5b9bd5", None),
        ("L  nL = 1.45, d2", 0.58, 0.74, "#f4b183", None),
        ("H  nH = 2.30, d3", 0.42, 0.58, "#5b9bd5", None),
        ("L  nL = 1.45, d4", 0.26, 0.42, "#f4b183", None),
        ("Glass", 0.02, 0.26, "#d9d9d9", "nsub = 1.52"),
    ]
    for name, y0, y1, c, sub in layers:
        ax.add_patch(mpatches.Rectangle((0.15, y0), 0.7, y1 - y0,
                     facecolor=c, edgecolor="black", lw=1.0))
        if sub:
            ax.text(0.85, (y0 + y1) / 2, sub, va="center", fontsize=8,
                    ha="left")
        ax.text(0.5, (y0 + y1) / 2, name, ha="center", va="center",
                fontsize=8.5)

    # incident / reflected arrows
    ax.annotate("", xy=(0.03, 0.78), xytext=(0.03, 0.97),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.6))
    ax.annotate("", xy=(0.03, 0.68), xytext=(0.03, 0.86),
                arrowprops=dict(arrowstyle="-|>", color="#1a5fb4", lw=1.6))
    ax.text(0.045, 0.905, "incident", fontsize=8, color="#c0392b",
            va="center", ha="left")
    ax.text(0.045, 0.79, "reflected", fontsize=8, color="#1a5fb4",
            va="center", ha="left")
    ax.set_title("Physical model", fontsize=10, fontweight="bold")

    # --- right: TMM data flow --------------------------------------------
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box(ax, 0.05, 0.72, 0.9, 0.2, "Sample 4 thicknesses\nd = [d1, d2, d3, d4]\n(40-180 nm)", fs=9)
    box(ax, 0.05, 0.40, 0.9, 0.2,
        "TMM: M = M1 M2 M3 M4\nR(λ) = |r(λ)|²", fs=9)
    box(ax, 0.05, 0.08, 0.9, 0.2, "41 reflectance values\n(400-800 nm, 10 nm step)", fs=9)
    arrow(ax, 0.5, 0.72, 0.5, 0.60)
    arrow(ax, 0.5, 0.40, 0.5, 0.28)
    ax.set_title("TMM data generation (thickness → spectrum)",
                 fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_physical.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


def fig3_mlp():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)

    layer_sizes = [4, 128, 128, 64, 41]
    labels = ["Input\n(4)", "Hidden\n(128)", "Hidden\n(128)", "Hidden\n(64)",
              "Output\n(41)"]
    xs = [0.5, 3.0, 5.5, 8.0, 9.9]
    colors = [C_BOX, C_BOX, C_BOX, C_BOX, C_ACC]

    # draw nodes
    for x, size, lab, c in zip(xs, layer_sizes, labels, colors):
        ec = C_ACC_EDGE if c == C_ACC else C_EDGE
        # limit visual node count for the wide hidden layers
        n_show = min(size, 12)
        y_centers = [4.2 * 0.5 + (i - (n_show - 1) / 2) * 0.28
                     for i in range(n_show)]
        for yc in y_centers:
            ax.add_patch(plt.Circle((x, yc), 0.09, facecolor=c, edgecolor=ec,
                                    lw=0.8, zorder=3))
        ax.text(x, 0.25, lab, ha="center", va="center", fontsize=9)
        if size != n_show:
            ax.text(x, 4.0, "...", ha="center", fontsize=10, color="gray")

    # draw connections (subsample for clarity)
    prev = None
    for x, size in zip(xs, layer_sizes):
        n_show = min(size, 12)
        y_centers = [4.2 * 0.5 + (i - (n_show - 1) / 2) * 0.28
                     for i in range(n_show)]
        if prev is not None:
            for y1 in prev[1]:
                for y2 in y_centers:
                    ax.plot([prev[0], x], [y1, y2], color="#b0c4de",
                            lw=0.4, alpha=0.5, zorder=1)
        prev = (x, y_centers)

    ax.text(5.2, 0.0, "MLP surrogate model: 4 - 128 - 128 - 64 - 41 "
            "(ReLU hidden, linear output)",
            ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_mlp.png"), dpi=300,
                bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig1_workflow()
    fig2_physical()
    fig3_mlp()
    print("Schematic figures saved.")
