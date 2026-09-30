"""Compact SCvx computation overview accompanying Algorithm 1.

The drawing is sized for a 6.8-inch manuscript column, with 10-point text.
Algorithm 1 specifies bootstrap acceptance, both stopping tests, and failures;
the diagram shows the shared computation and the two distinct return paths.

Run: python figures/scp_pipeline.py --save
"""
from pathlib import Path
import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build_figure(save=False):
    plt.rcParams.update({"mathtext.fontset": "dejavusans", "mathtext.default": "regular"})
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 6.8), ylim=(0, 3.7), aspect="equal")
    ax.axis("off")
    ink, blue = "#333333", "#006BA4"
    left, right, w, h = 1.65, 5.15, 2.9, 0.72
    high, mid, low = 3.12, 2.02, 0.92

    def box(x, y, lines, fill="#ffffff", edge=ink):
        ax.add_patch(FancyBboxPatch(
            (x-w/2, y-h/2), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
            linewidth=1.05, facecolor=fill, edgecolor=edge, zorder=2))
        offsets = [0] if len(lines) == 1 else ([0.13, -0.13] if len(lines) == 2 else [0.22, 0, -0.22])
        for line, dy in zip(lines, offsets):
            ax.text(x, y+dy, line, fontsize=10, ha="center", va="center", color=ink, zorder=3)

    def arrow(points):
        if len(points) > 2:
            ax.plot(*zip(*points[:-1]), color=ink, linewidth=1.0, zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                     mutation_scale=9, linewidth=1.0, color=ink, shrinkA=0, shrinkB=0, zorder=1))

    def label(x, y, text, align="center"):
        ax.text(x, y, text, fontsize=9.5, color=ink, ha=align, va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8), zorder=4)

    box(left, high, [r"Initialize $\mathbf{x}^{(0)}$, $\Delta_0$; assemble operators",
                     r"$D_N,\ E_{N-1},\ G_N,\ A_{\mathrm{bc}},\ \mathbf{b}_{\mathrm{bc}}$"],
        fill="#edf5fa", edge=blue)
    box(left, mid, ["Subdivide and construct local model", "Linearize avoidance and gravity"], fill="#f3f3f3")
    box(left, low, ["Solve convex QP", r"Trial $\hat{\mathbf{x}}$ and slack $\boldsymbol{\nu}$"], fill="#deedf5", edge=blue)
    box(right, low, ["Evaluate trial", r"Compute $\rho_k$ when required"], fill="#f3f3f3")
    box(right, mid, ["Apply acceptance and stopping rules",
                     r"Update reference and $\Delta_k$ (Algorithm 1)"], fill="#f3f3f3")
    box(right, high, ["Return trajectory"])

    # Setup and local-model solve.
    arrow([(left, high-h/2-.015), (left, mid+h/2+.015)])
    arrow([(left, mid-h/2-.015), (left, low+h/2+.015)])
    arrow([(left+w/2+.015, low), (right-w/2-.015, low)])
    # Apply Algorithm 1 before choosing return, a new model, or a retry.
    arrow([(right, low+h/2+.015), (right, mid-h/2-.015)])
    arrow([(right, mid+h/2+.015), (right, high-h/2-.015)])
    label(right+.12, (mid+high)/2, "converged", "left")
    arrow([(right-w/2-.015, mid), (left+w/2+.015, mid)])
    label((left+right)/2, mid+.17, "next")
    # A rejected trial retains the reference and re-solves the same local model.
    arrow([(right+w/2+.015, mid), (6.72, mid), (6.72, .22),
           (left, .22), (left, low-h/2-.015)])
    label((left+right)/2, .22, r"rejected: shrink $\Delta_k$")

    if save:
        out = Path(__file__).resolve().parent
        for ext in ("pdf", "png"):
            path = out / f"scp_pipeline.{ext}"
            fig.savefig(path, dpi=300, facecolor="white")
            print(f"Saved {path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    build_figure(save="--save" in sys.argv)
