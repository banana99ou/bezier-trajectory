"""Figure 3: acceptance of the subproblem solution and the trust-region update.

Algorithm 1 delegates exactly this decision to the figure, so the two never
restate each other. Every arrow points right or down; the iteration loop itself
lives in Algorithm 1, not here. Korean text uses the manuscript's body font
(NanumBarunGothic, from the export header) and math uses Computer Modern, the
closest mathtext match to the body's Latin Modern.

Run: python figures/scp_pipeline.py --save
"""
from pathlib import Path
import sys

import matplotlib
if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

BODY_FONT = Path.home() / "Library/Fonts/NanumBarunGothic.ttf"
INK = "#333333"


def build_figure(save=False):
    font_manager.fontManager.addfont(str(BODY_FONT))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(BODY_FONT)).get_name(),
        "mathtext.fontset": "cm",
    })
    W, H = 6.8, 2.25
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, W), ylim=(0, H), aspect="equal")
    ax.axis("off")

    spine, leaf = 1.72, 0.42          # y of the decision row and of the outcome row
    dw, dh = 1.24, 0.62               # diamond width / height
    bw, bh = 1.5, 0.5                 # outcome box width / height

    def diamond(x, text):
        ax.add_patch(Polygon([(x - dw/2, spine), (x, spine + dh/2), (x + dw/2, spine), (x, spine - dh/2)],
                             closed=True, facecolor="white", edgecolor=INK, linewidth=1.0, zorder=2))
        ax.text(x, spine, text, fontsize=9.5, ha="center", va="center", color=INK, zorder=3)

    def box(x, y, lines, w=bw):
        ax.add_patch(FancyBboxPatch((x - w/2, y - bh/2), w, bh,
                                    boxstyle="round,pad=0.012,rounding_size=0.05",
                                    facecolor="white", edgecolor=INK, linewidth=1.0, zorder=2))
        for line, dy in zip(lines, [0.1, -0.11] if len(lines) == 2 else [0]):
            ax.text(x, y + dy, line, fontsize=9.5, ha="center", va="center", color=INK, zorder=3)

    def arrow(p, q, label=None, at=None):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                     linewidth=1.0, color=INK, shrinkA=0, shrinkB=0, zorder=1))
        if label:
            ax.text(*at, label, fontsize=8.5, ha="left", va="center", color=INK, zorder=3)

    gap = 0.32
    x1 = 0.3 + gap + dw/2
    x2, x3 = x1 + dw + gap, x1 + 2*(dw + gap)
    ew = 1.48                         # the outcome box at the end of the decision row
    xe = x3 + dw/2 + gap + ew/2
    diamond(x1, r"$A_{\mathrm{bc}}\mathbf{x}^{(k)} = \mathbf{b}_{\mathrm{bc}}$")
    diamond(x2, r"$\rho_k > 0.1$")
    diamond(x3, r"$\rho_k > 0.9$")
    box(xe, spine, [r"$\mathbf{x}^{(k+1)} \leftarrow \hat{\mathbf{x}}$",
                    r"$\Delta_{k+1} \leftarrow \min(2\Delta_k, 4\Delta_0)$"], w=ew)
    box(x1, leaf, [r"$\mathbf{x}^{(k+1)} \leftarrow \hat{\mathbf{x}}$", r"$\Delta_{k+1} \leftarrow \Delta_k$"], w=1.3)
    box(x2, leaf, [r"$\mathbf{x}^{(k+1)} \leftarrow \mathbf{x}^{(k)}$", r"$\Delta_{k+1} \leftarrow \Delta_k/2$"], w=1.3)
    box(x3, leaf, [r"$\mathbf{x}^{(k+1)} \leftarrow \hat{\mathbf{x}}$", r"$\Delta_{k+1} \leftarrow \Delta_k$"], w=1.3)

    ax.text(0.16, spine, r"$\hat{\mathbf{x}}$", fontsize=10, ha="center", va="center", color=INK)
    arrow((0.3, spine), (x1 - dw/2, spine))
    for a, b in ((x1, x2), (x2, x3)):
        arrow((a + dw/2, spine), (b - dw/2, spine), "예", (a + dw/2 + 0.05, spine + 0.1))
    arrow((x3 + dw/2, spine), (xe - ew/2 - 0.012, spine), "예", (x3 + dw/2 + 0.05, spine + 0.1))
    for x in (x1, x2, x3):
        arrow((x, spine - dh/2), (x, leaf + bh/2 + 0.012), "아니오", (x + 0.06, (spine - dh/2 + leaf + bh/2) / 2))

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
