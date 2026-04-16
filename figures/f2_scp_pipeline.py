"""
F2. SCP pipeline in control-point space.

Single-column vertical flowchart showing the successive convexification
loop.  Color-coded regions distinguish pieces built once (reusable
operators) from pieces rebuilt at each outer iteration (supporting
half-spaces, gravity linearization, IRLS weights).

Usage:
    python figures/f2_scp_pipeline.py          # show interactively
    python figures/f2_scp_pipeline.py --save   # save to figures/f2_scp_pipeline.{pdf,png}
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

BOX_W = 5.2
BOX_H = 0.70
X_C = 0.0

# Y positions (top to bottom, center of each element)
Y = dict(
    init=11.2,
    assemble=9.6,
    subdivide=7.5,
    build=6.1,
    solve=4.7,
    update=3.3,
    converge=2.2,
    ret=0.0,
)

# Diamond half-extents (wider than tall for readability)
DIA_WX = 1.45
DIA_WY = 0.65

# Background-region y-bounds (bottom, top)
BLUE_Y = (Y["assemble"] - BOX_H * 0.65 - 0.25, Y["assemble"] + BOX_H * 0.65 + 0.25)
LOOP_Y = (Y["converge"] - DIA_WY - 0.20, Y["subdivide"] + BOX_H / 2 + 0.30)

# Return box sits below the loop region — keep it outside
assert Y["ret"] < LOOP_Y[0], "Return box must be below the loop region"

LOOP_X_RIGHT = BOX_W / 2 + 2.0  # x-coord of the loop-back path

# ---------------------------------------------------------------------------
# Color palette (consistent with F1)
# ---------------------------------------------------------------------------

PAL = dict(
    blue_bg="#ddeaf6",    blue_box="#c5ddf0",   blue_ec="#2980b9",  blue_tx="#1a5276",
    loop_bg="#fef0de",    loop_box="#fde4c4",   loop_ec="#e67e22",  loop_tx="#935116",
    qp_box="#d5f5e3",     qp_ec="#1abc9c",      qp_tx="#0e6655",
    neut_box="#f2f3f4",   neut_ec="#7f8c8d",
    ret_box="#d5f5e3",    ret_ec="#27ae60",     ret_tx="#1e8449",
    text="#2c3e50",       arrow="#2c3e50",
    yes="#27ae60",        no="#c0392b",
)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rounded_box(ax, cx, cy, w, h, fc, ec, lw=1.6, pad=0.12):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        fc=fc, ec=ec, lw=lw, zorder=3,
    )
    ax.add_patch(box)
    return box


def _text(ax, cx, cy, s, fs=10, color=None, bold=False, **kw):
    ax.text(
        cx, cy, s,
        ha="center", va="center",
        fontsize=fs, color=color or PAL["text"],
        fontweight="bold" if bold else "normal",
        zorder=5, **kw,
    )


def _arrow(ax, xy0, xy1, color=None, lw=1.4, ms=13):
    a = FancyArrowPatch(
        xy0, xy1,
        arrowstyle="-|>", color=color or PAL["arrow"],
        lw=lw, mutation_scale=ms, zorder=4,
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(a)


def _diamond(ax, cx, cy, wx, wy, fc, ec):
    verts = [(cx, cy + wy), (cx + wx, cy), (cx, cy - wy), (cx - wx, cy)]
    d = Polygon(verts, closed=True, fc=fc, ec=ec, lw=1.6, zorder=3)
    ax.add_patch(d)


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_f2(save=False):
    fig, ax = plt.subplots(figsize=(7.0, 11.0), constrained_layout=True)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-4.8, LOOP_X_RIGHT + 1.6)
    ax.set_ylim(-1.0, 12.2)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Background regions
    # ------------------------------------------------------------------

    # Blue: built once
    bw = BOX_W + 1.2
    _rounded_box(ax, X_C, (BLUE_Y[0] + BLUE_Y[1]) / 2,
                 bw, BLUE_Y[1] - BLUE_Y[0],
                 fc=PAL["blue_bg"], ec=PAL["blue_ec"],
                 lw=1.0, pad=0.20)
    ax.text(X_C - bw / 2 + 0.12, BLUE_Y[1] - 0.08,
            "Built once", fontsize=8, color=PAL["blue_tx"],
            fontweight="bold", va="top", ha="left", zorder=1,
            fontstyle="italic")

    # Orange: SCP outer loop
    lw_reg = BOX_W + 3.4
    lh = LOOP_Y[1] - LOOP_Y[0]
    loop_rect = FancyBboxPatch(
        (X_C - BOX_W / 2 - 0.6, LOOP_Y[0]),
        lw_reg, lh,
        boxstyle="round,pad=0.20",
        fc=PAL["loop_bg"], ec=PAL["loop_ec"],
        lw=1.0, zorder=0,
    )
    ax.add_patch(loop_rect)
    ax.text(X_C - BOX_W / 2 - 0.35, LOOP_Y[1] - 0.08,
            "Rebuilt each SCP iteration",
            fontsize=8, color=PAL["loop_tx"],
            fontweight="bold", va="top", ha="left", zorder=1,
            fontstyle="italic")

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------

    bh = BOX_H
    bh_tall = BOX_H * 1.15

    # 1 ── Initialize
    _rounded_box(ax, X_C, Y["init"], BOX_W, bh,
                 PAL["neut_box"], PAL["neut_ec"])
    _text(ax, X_C, Y["init"],
          r"Initialize control polygon $\mathbf{P}^{(0)}$   (straight line)",
          fs=10)

    # 2 ── Assemble reusable operators
    _rounded_box(ax, X_C, Y["assemble"], BOX_W, bh_tall,
                 PAL["blue_box"], PAL["blue_ec"])
    _text(ax, X_C, Y["assemble"] + 0.15,
          "Assemble reusable operators", fs=10,
          color=PAL["blue_tx"], bold=True)
    _text(ax, X_C, Y["assemble"] - 0.18,
          r"$D_N,\; E_M,\; G_N,\; \widetilde{G}_N,\; A_{\mathrm{bc}},\; b_{\mathrm{bc}}$",
          fs=9, color=PAL["blue_tx"])

    # 3 ── Subdivide
    _rounded_box(ax, X_C, Y["subdivide"], BOX_W, bh,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["subdivide"],
          r"Subdivide:  $P^{(s)} = S^{(s)}\, P^{(k)}$",
          fs=10, color=PAL["loop_tx"])

    # 4 ── Build constraints / linearize
    _rounded_box(ax, X_C, Y["build"], BOX_W, bh_tall,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["build"] + 0.15,
          r"Build supporting half-spaces $H^{(s)}$",
          fs=10, color=PAL["loop_tx"])
    _text(ax, X_C, Y["build"] - 0.18,
          "Linearize gravity  ·  update IRLS weights",
          fs=9, color=PAL["loop_tx"])

    # 5 ── Solve convex QP  (highlighted)
    _rounded_box(ax, X_C, Y["solve"], BOX_W, bh,
                 PAL["qp_box"], PAL["qp_ec"], lw=2.0)
    _text(ax, X_C, Y["solve"],
          r"Solve convex QP  $\;\longrightarrow\; P^{(k+1)}$",
          fs=11, color=PAL["qp_tx"], bold=True)

    # 6 ── Update / step clipping
    _rounded_box(ax, X_C, Y["update"], BOX_W, bh,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["update"],
          "Step clipping  ·  proximal update",
          fs=10, color=PAL["loop_tx"])

    # 7 ── Convergence diamond
    _diamond(ax, X_C, Y["converge"], DIA_WX, DIA_WY,
             PAL["neut_box"], PAL["arrow"])
    _text(ax, X_C, Y["converge"] + 0.12,
          r"$\|\mathbf{P}^{(k+1)} - \mathbf{P}^{(k)}\|_F$",
          fs=8, color=PAL["text"])
    _text(ax, X_C, Y["converge"] - 0.20,
          r"$< \mathrm{tol}$ ?", fs=9, color=PAL["text"])

    # 8 ── Return
    _rounded_box(ax, X_C, Y["ret"], BOX_W * 0.6, bh,
                 PAL["ret_box"], PAL["ret_ec"], lw=2.0)
    _text(ax, X_C, Y["ret"],
          r"Return $\mathbf{P}^*$", fs=11,
          color=PAL["ret_tx"], bold=True)

    # ------------------------------------------------------------------
    # Vertical arrows (consecutive boxes)
    # ------------------------------------------------------------------

    def _box_bot(key, h=bh):
        return (X_C, Y[key] - h / 2)

    def _box_top(key, h=bh):
        return (X_C, Y[key] + h / 2)

    _arrow(ax, _box_bot("init"),       _box_top("assemble", bh_tall))
    _arrow(ax, _box_bot("assemble", bh_tall), _box_top("subdivide"))
    _arrow(ax, _box_bot("subdivide"),  _box_top("build", bh_tall))
    _arrow(ax, _box_bot("build", bh_tall), _box_top("solve"))
    _arrow(ax, _box_bot("solve"),      _box_top("update"))
    _arrow(ax, _box_bot("update"),     (X_C, Y["converge"] + DIA_WY))

    # Yes → Return
    _arrow(ax, (X_C, Y["converge"] - DIA_WY),
           _box_top("ret"), color=PAL["yes"])
    ax.text(X_C + 0.20, Y["converge"] - DIA_WY - 0.15,
            "yes", fontsize=9, color=PAL["yes"], fontweight="bold",
            va="top", ha="left")

    # No → loop back to Subdivide
    no_start = (X_C + DIA_WX, Y["converge"])
    _arrow(ax, no_start, (LOOP_X_RIGHT, Y["converge"]),
           color=PAL["no"], lw=1.3)
    _arrow(ax, (LOOP_X_RIGHT, Y["converge"]),
           (LOOP_X_RIGHT, Y["subdivide"]),
           color=PAL["no"], lw=1.3)
    _arrow(ax, (LOOP_X_RIGHT, Y["subdivide"]),
           (X_C + BOX_W / 2, Y["subdivide"]),
           color=PAL["no"], lw=1.3)

    ax.text(X_C + DIA_WX + 0.15, Y["converge"] + 0.15,
            "no", fontsize=9, color=PAL["no"], fontweight="bold",
            va="bottom", ha="left")

    # Iteration label on the loop-back path
    ax.text(LOOP_X_RIGHT + 0.15,
            (Y["converge"] + Y["subdivide"]) / 2,
            r"$k \leftarrow k+1$",
            fontsize=9, color=PAL["no"],
            rotation=90, ha="left", va="center")

    # ------------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------------
    if save:
        out = Path(__file__).resolve().parent
        for ext in ("pdf", "png"):
            p = out / f"f2_scp_pipeline.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved {p}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    build_f2("--save" in sys.argv)
