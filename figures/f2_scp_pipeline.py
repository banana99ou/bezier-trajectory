"""
F2. SCvx iteration in control-point space.

Single-column vertical flowchart of the canonical SCvx loop of §3.3
(Algorithm 1).  Operators that are assembled once are separated from the
pieces rebuilt at every iteration (supporting half-spaces, gravity
linearization).  The trial point returned by the convex QP is accepted or
rejected by the penalized-merit ratio test, so the chart carries two
loop-backs: a rejected trial is re-solved at the same reference point with
a smaller trust region, while an accepted trial advances the iteration.

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

BOX_W = 5.8
BOX_H = 0.70
X_C = 0.0

# Y positions (top to bottom, center of each element)
Y = dict(
    init=13.7,
    assemble=11.9,
    subdivide=9.3,
    build=7.9,
    solve=6.4,
    merit=5.0,
    ratio=3.5,      # diamond: rho_k > eta ?
    accept=2.1,
    converge=0.6,   # diamond: converged ?
    ret=-1.3,
)

# Diamond half-extents (wider than tall for readability)
DIA_WX = 1.70
DIA_WY = 0.72

# Background-region y-bounds (bottom, top).
# The top margins leave room for the italic region labels, which are drawn
# inside the top-left corner and must clear the first box of each region.
BLUE_Y = (Y["assemble"] - BOX_H * 0.65 - 0.25, Y["assemble"] + BOX_H * 0.65 + 0.60)
LOOP_Y = (Y["converge"] - DIA_WY - 0.25, Y["subdivide"] + BOX_H * 0.65 + 0.95)

# Return box sits below the loop region — keep it outside
assert Y["ret"] < LOOP_Y[0], "Return box must be below the loop region"

# Two loop-back paths: inner = rejected trial, outer = next iteration
LOOP_X_INNER = BOX_W / 2 + 1.15
LOOP_X_OUTER = BOX_W / 2 + 2.55

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

def _rounded_box(ax, cx, cy, w, h, fc, ec, lw=1.6, pad=0.12, zorder=3):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
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
    fig, ax = plt.subplots(figsize=(7.6, 13.7), constrained_layout=True)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-4.6, LOOP_X_OUTER + 1.7)
    ax.set_ylim(-2.4, 14.7)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Background regions
    # ------------------------------------------------------------------

    # Blue: assembled once
    bw = BOX_W + 1.2
    _rounded_box(ax, X_C, (BLUE_Y[0] + BLUE_Y[1]) / 2,
                 bw, BLUE_Y[1] - BLUE_Y[0],
                 fc=PAL["blue_bg"], ec=PAL["blue_ec"],
                 lw=1.0, pad=0.20, zorder=0)
    ax.text(X_C - bw / 2 + 0.12, BLUE_Y[1] - 0.08,
            "Assembled once", fontsize=8, color=PAL["blue_tx"],
            fontweight="bold", va="top", ha="left", zorder=1,
            fontstyle="italic")

    # Orange: one SCvx iteration
    left = X_C - BOX_W / 2 - 0.6
    lw_reg = (LOOP_X_OUTER + 0.55) - left
    lh = LOOP_Y[1] - LOOP_Y[0]
    loop_rect = FancyBboxPatch(
        (left, LOOP_Y[0]), lw_reg, lh,
        boxstyle="round,pad=0.20",
        fc=PAL["loop_bg"], ec=PAL["loop_ec"],
        lw=1.0, zorder=0,
    )
    ax.add_patch(loop_rect)
    ax.text(left + 0.25, LOOP_Y[1] - 0.08,
            "One SCvx iteration",
            fontsize=8, color=PAL["loop_tx"],
            fontweight="bold", va="top", ha="left", zorder=1,
            fontstyle="italic")

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------

    bh = BOX_H
    bh_tall = BOX_H * 1.15

    # 1 ── Initialize
    _rounded_box(ax, X_C, Y["init"], BOX_W, bh_tall,
                 PAL["neut_box"], PAL["neut_ec"])
    _text(ax, X_C, Y["init"] + 0.15,
          r"Initialize control polygon $\mathbf{P}^{(0)}$   (straight line)",
          fs=10)
    _text(ax, X_C, Y["init"] - 0.18,
          r"trust region  $r_0$", fs=9)

    # 2 ── Assemble reusable operators
    _rounded_box(ax, X_C, Y["assemble"], BOX_W, bh_tall,
                 PAL["blue_box"], PAL["blue_ec"])
    _text(ax, X_C, Y["assemble"] + 0.15,
          "Assemble reusable operators", fs=10,
          color=PAL["blue_tx"], bold=True)
    _text(ax, X_C, Y["assemble"] - 0.18,
          r"$D_N,\; E_M,\; G_N,\; \widetilde{G}_N,\; A_{\mathrm{bc}},\; b_{\mathrm{bc}}$",
          fs=9, color=PAL["blue_tx"])

    # 3 ── Rebuild supporting half-spaces
    _rounded_box(ax, X_C, Y["subdivide"], BOX_W, bh_tall,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["subdivide"] + 0.15,
          r"Subdivide:  $P^{(s)} = S^{(s)}\, P^{(k)}$",
          fs=10, color=PAL["loop_tx"])
    _text(ax, X_C, Y["subdivide"] - 0.18,
          r"rebuild supporting half-spaces  $\mathcal{H}^{(s)}$",
          fs=9, color=PAL["loop_tx"])

    # 4 ── Relinearize gravity
    _rounded_box(ax, X_C, Y["build"], BOX_W, bh_tall,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["build"] + 0.15,
          "Relinearize gravity at representative points",
          fs=10, color=PAL["loop_tx"])
    _text(ax, X_C, Y["build"] - 0.18,
          r"$\longrightarrow\; H^{(k)},\; \mathbf{f}^{(k)}$",
          fs=9, color=PAL["loop_tx"])

    # 5 ── Solve convex QP  (highlighted)
    _rounded_box(ax, X_C, Y["solve"], BOX_W, bh_tall,
                 PAL["qp_box"], PAL["qp_ec"], lw=2.0)
    _text(ax, X_C, Y["solve"] + 0.15,
          r"Solve convex QP   (trust region $r_k$)",
          fs=11, color=PAL["qp_tx"], bold=True)
    _text(ax, X_C, Y["solve"] - 0.18,
          r"$\longrightarrow\;$ trial $\hat{\mathbf{x}}$,   slack $\mathbf{s}$",
          fs=9, color=PAL["qp_tx"])

    # 6 ── Merit evaluation / ratio
    _rounded_box(ax, X_C, Y["merit"], BOX_W, bh_tall,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["merit"] + 0.15,
          "Evaluate penalized merit function",
          fs=10, color=PAL["loop_tx"])
    _text(ax, X_C, Y["merit"] - 0.18,
          r"$\rho_k = \Delta_{\mathrm{actual}} \,/\, \Delta_{\mathrm{predicted}}$",
          fs=9, color=PAL["loop_tx"])

    # 7 ── Ratio-test diamond
    _diamond(ax, X_C, Y["ratio"], DIA_WX, DIA_WY,
             PAL["neut_box"], PAL["arrow"])
    _text(ax, X_C, Y["ratio"], r"$\rho_k > \eta$ ?",
          fs=10, color=PAL["text"])

    # 7b ── Accept
    _rounded_box(ax, X_C, Y["accept"], BOX_W, bh_tall,
                 PAL["loop_box"], PAL["loop_ec"])
    _text(ax, X_C, Y["accept"] + 0.15,
          r"Take $\hat{\mathbf{x}}$ as the new reference point",
          fs=10, color=PAL["loop_tx"])
    _text(ax, X_C, Y["accept"] - 0.18,
          r"enlarge $r$ if $\rho_k \approx 1$",
          fs=9, color=PAL["loop_tx"])

    # 8 ── Convergence diamond
    _diamond(ax, X_C, Y["converge"], DIA_WX, DIA_WY,
             PAL["neut_box"], PAL["arrow"])
    _text(ax, X_C, Y["converge"] + 0.16,
          r"$|\Delta\phi| / |\phi| < \mathrm{tol}$  or  $r = r_{\min}$",
          fs=7.5, color=PAL["text"])
    _text(ax, X_C, Y["converge"] - 0.20,
          r"and  $h = 0$ ?", fs=8.5, color=PAL["text"])

    # 9 ── Return
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

    _arrow(ax, _box_bot("init", bh_tall),      _box_top("assemble", bh_tall))
    _arrow(ax, _box_bot("assemble", bh_tall),  _box_top("subdivide", bh_tall))
    _arrow(ax, _box_bot("subdivide", bh_tall), _box_top("build", bh_tall))
    _arrow(ax, _box_bot("build", bh_tall),     _box_top("solve", bh_tall))
    _arrow(ax, _box_bot("solve", bh_tall),     _box_top("merit", bh_tall))
    _arrow(ax, _box_bot("merit", bh_tall),     (X_C, Y["ratio"] + DIA_WY))

    # ratio yes → accept
    _arrow(ax, (X_C, Y["ratio"] - DIA_WY), _box_top("accept", bh_tall),
           color=PAL["yes"])
    ax.text(X_C + 0.18, Y["ratio"] - DIA_WY - 0.10,
            "yes", fontsize=9, color=PAL["yes"], fontweight="bold",
            va="top", ha="left")

    # accept → convergence
    _arrow(ax, _box_bot("accept", bh_tall), (X_C, Y["converge"] + DIA_WY))

    # convergence yes → Return
    _arrow(ax, (X_C, Y["converge"] - DIA_WY), _box_top("ret"),
           color=PAL["yes"])
    ax.text(X_C + 0.18, Y["converge"] - DIA_WY - 0.10,
            "yes", fontsize=9, color=PAL["yes"], fontweight="bold",
            va="top", ha="left")

    # ------------------------------------------------------------------
    # Loop-back 1 (inner): rejected trial → re-solve at same reference
    # ------------------------------------------------------------------
    _arrow(ax, (X_C + DIA_WX, Y["ratio"]), (LOOP_X_INNER, Y["ratio"]),
           color=PAL["no"], lw=1.3)
    _arrow(ax, (LOOP_X_INNER, Y["ratio"]), (LOOP_X_INNER, Y["solve"]),
           color=PAL["no"], lw=1.3)
    _arrow(ax, (LOOP_X_INNER, Y["solve"]), (X_C + BOX_W / 2, Y["solve"]),
           color=PAL["no"], lw=1.3)
    ax.text(X_C + DIA_WX + 0.12, Y["ratio"] + 0.14,
            "no", fontsize=9, color=PAL["no"], fontweight="bold",
            va="bottom", ha="left")
    ax.text(LOOP_X_INNER + 0.12, (Y["ratio"] + Y["solve"]) / 2,
            "reject $\\hat{\\mathbf{x}}$,  shrink $r$",
            fontsize=8, color=PAL["no"],
            rotation=90, ha="left", va="center")

    # ------------------------------------------------------------------
    # Loop-back 2 (outer): not converged → next iteration
    # ------------------------------------------------------------------
    _arrow(ax, (X_C + DIA_WX, Y["converge"]), (LOOP_X_OUTER, Y["converge"]),
           color=PAL["no"], lw=1.3)
    _arrow(ax, (LOOP_X_OUTER, Y["converge"]), (LOOP_X_OUTER, Y["subdivide"]),
           color=PAL["no"], lw=1.3)
    _arrow(ax, (LOOP_X_OUTER, Y["subdivide"]), (X_C + BOX_W / 2, Y["subdivide"]),
           color=PAL["no"], lw=1.3)
    ax.text(X_C + DIA_WX + 0.12, Y["converge"] + 0.14,
            "no", fontsize=9, color=PAL["no"], fontweight="bold",
            va="bottom", ha="left")
    ax.text(LOOP_X_OUTER + 0.12, (Y["converge"] + Y["subdivide"]) / 2,
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
    build_f2(save="--save" in sys.argv)
