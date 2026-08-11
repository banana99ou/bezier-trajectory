"""
SCvx iteration in control-point space.

Single-column vertical flowchart of the canonical SCvx loop of §3.3
(Algorithm 1).  Operators that are assembled once are separated from the
pieces rebuilt at every iteration (supporting half-spaces, gravity
linearization).  The trial point returned by the convex QP is accepted or
rejected by the penalized-merit ratio test, so the chart carries two
loop-backs: a rejected trial is re-solved at the same reference point with
a smaller trust region, while an accepted trial advances the iteration.

Layout rules (keep these when editing):
  * consecutive elements are separated by GAP inside a region, and by the
    tighter GAP_CROSS where a background-region edge falls between them:
    the region outline and its label add visual bulk, so a smaller box gap
    is what actually reads as even;
  * font size is fixed per element class (FS_BOX, FS_SUB, FS_DIA, FS_EDGE);
  * arrows start and end exactly on the drawn edge of a box or diamond,
    with the head outside the target, never underneath it.

Usage:
    python figures/scp_pipeline.py          # show interactively
    python figures/scp_pipeline.py --save   # save to figures/scp_pipeline.{pdf,png}
"""

import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from pathlib import Path

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

X_C = 0.0
BOX_W = 6.0
BOX_H = 0.86          # identical for every rectangular box
PAD = 0.12            # FancyBboxPatch inflates the drawn shape by this much
BOX_HH = BOX_H / 2 + PAD          # half-height of the *drawn* box
BOX_HW = BOX_W / 2 + PAD          # half-width  of the *drawn* box

DIA_WX = 2.30
DIA_WY = 0.80

GAP = 0.75            # gap between two elements inside the same region
GAP_CROSS = 0.55      # gap across a region edge, which supplies its own bulk

# Element order, top to bottom: (key, shape, enclosing background region)
SEQUENCE = [
    ("init",      "rect",    None),
    ("assemble",  "rect",    "setup"),
    ("subdivide", "rect",    "loop"),
    ("build",     "rect",    "loop"),
    ("solve",     "rect",    "loop"),
    ("merit",     "rect",    "loop"),
    ("ratio",     "diamond", "loop"),
    ("accept",    "rect",    "loop"),
    ("converge",  "diamond", "loop"),
    ("ret",       "rect",    None),
]


def _half_height(shape):
    return BOX_HH if shape == "rect" else DIA_WY


def _gap(region_a, region_b):
    """Tighter gap wherever a region edge already fills the space."""
    return GAP if region_a == region_b else GAP_CROSS


def _stack_positions():
    """Centre y of each element, stacked top to bottom."""
    ys = {}
    y = 0.0
    for i, (key, shape, region) in enumerate(SEQUENCE):
        ys[key] = y
        if i + 1 < len(SEQUENCE):
            _, next_shape, next_region = SEQUENCE[i + 1]
            y -= (_half_height(shape) + _gap(region, next_region)
                  + _half_height(next_shape))
    return ys


Y = _stack_positions()
SHAPE = {key: shape for key, shape, _ in SEQUENCE}

# Two loop-back paths: inner = rejected trial, outer = next iteration
LOOP_X_INNER = BOX_HW + 1.05
LOOP_X_OUTER = LOOP_X_INNER + 1.30

# Background-region bounds. The top margin holds the italic region label.
# These are the *drawn* bounds: _region() subtracts its own corner padding so
# the rounded rectangle lands exactly here and regions cannot overlap.
REGION_PAD = 0.18
REGION_TOP_PAD = 0.34
REGION_BOT_PAD = 0.10
BLUE_Y = (Y["assemble"] - BOX_HH - REGION_BOT_PAD,
          Y["assemble"] + BOX_HH + REGION_TOP_PAD)
LOOP_Y = (Y["converge"] - DIA_WY - REGION_BOT_PAD,
          Y["subdivide"] + BOX_HH + REGION_TOP_PAD)

assert Y["ret"] + BOX_HH < LOOP_Y[0], "Return box must sit below the loop region"
assert LOOP_Y[1] < BLUE_Y[0], "Loop region must sit below the blue region"

# ---------------------------------------------------------------------------
# Type scale — one size per class of element
# ---------------------------------------------------------------------------

FS_BOX = 10.0     # primary line inside a box
FS_SUB = 8.5      # secondary line inside a box
FS_DIA = 8.5      # text inside a decision diamond
FS_EDGE = 8.5     # branch labels and loop annotations
FS_REGION = 8.5   # italic region labels

# ---------------------------------------------------------------------------
# Palette — Tableau Color Blind 10, blue + neutral grey subset.
#   light blue = assembled once,  grey = the iteration,  blue = the convex QP.
# Published hex values are used unchanged for borders, accents and labels;
# fills are the same colour lightened toward white, so no new hue is invented.
# Source: https://gist.github.com/AndiH/c957b4d769e628f506bd
# ---------------------------------------------------------------------------

def _lighten(h, t):
    c = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * t) for v in c)


def _darken(h, t):
    c = [int(h[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(v * (1 - t)) for v in c)


C_INK = "#333333"      # CB10 dark grey
C_SETUP = "#5F9ED1"    # CB10 light blue
C_ITER = "#ABABAB"     # CB10 light grey
C_ACCENT = "#006BA4"   # CB10 blue
C_LOOP = "#898989"     # CB10 mid grey

PAL = dict(
    white="#ffffff",
    ink=C_INK,
    setup_bg=_lighten(C_SETUP, 0.88), setup_bg_ec=_lighten(C_SETUP, 0.55),
    setup_box=_lighten(C_SETUP, 0.80), setup_ec=C_SETUP,
    setup_tx=_darken(C_SETUP, 0.30),
    iter_bg=_lighten(C_ITER, 0.88), iter_bg_ec=_lighten(C_ITER, 0.50),
    iter_box=_lighten(C_ITER, 0.84), iter_ec=C_ITER,
    iter_tx=_darken(C_ITER, 0.45),
    qp_box=_lighten(C_ACCENT, 0.78), qp_ec=C_ACCENT,
    qp_tx=_darken(C_ACCENT, 0.32),
    loop=C_LOOP,
)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _rounded_box(ax, cx, cy, w, h, fc, ec, lw=1.4, pad=PAD, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
    ))


def _diamond(ax, cx, cy, wx, wy, fc, ec):
    verts = [(cx, cy + wy), (cx + wx, cy), (cx, cy - wy), (cx - wx, cy)]
    ax.add_patch(Polygon(verts, closed=True, fc=fc, ec=ec, lw=1.4, zorder=3))


def _text(ax, cx, cy, s, fs, color=None, **kw):
    ax.text(cx, cy, s, ha="center", va="center",
            fontsize=fs, color=color or PAL["ink"], zorder=5, **kw)


def _arrow(ax, xy0, xy1, color=None, lw=1.3, ls="-"):
    """Arrow whose tail and head land exactly on the given points."""
    ax.add_patch(FancyArrowPatch(
        xy0, xy1,
        arrowstyle="-|>", color=color or PAL["ink"],
        lw=lw, mutation_scale=11, zorder=4,
        shrinkA=0, shrinkB=0, linestyle=ls,
        connectionstyle="arc3,rad=0",
    ))


def _elbow(ax, pts, color=None, lw=1.3, ls="-"):
    """Polyline with a single arrowhead, on the final segment only."""
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=color or PAL["ink"], lw=lw, zorder=4, ls=ls,
                solid_capstyle="round", solid_joinstyle="round")
    _arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls)


# --- exact edge anchors ----------------------------------------------------

def top(key):
    return (X_C, Y[key] + _half_height(SHAPE[key]))


def bottom(key):
    return (X_C, Y[key] - _half_height(SHAPE[key]))


def right(key):
    return (X_C + (BOX_HW if SHAPE[key] == "rect" else DIA_WX), Y[key])


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(save=False):
    # Match mathtext to the sans body text so one font reads throughout.
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"

    fig, ax = plt.subplots(figsize=(7.5, 14.0), constrained_layout=True)
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    ax.set_xlim(-3.95, LOOP_X_OUTER + 1.10)
    ax.set_ylim(Y["ret"] - BOX_HH - 0.45, Y["init"] + BOX_HH + 0.45)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Background regions
    # ------------------------------------------------------------------
    region_left = X_C - BOX_HW - REGION_TOP_PAD

    def _region(y_bounds, x_right, fc, ec, tx, label):
        # shrink by REGION_PAD so the drawn outline matches y_bounds exactly
        _rounded_box(ax, (region_left + x_right) / 2,
                     (y_bounds[0] + y_bounds[1]) / 2,
                     (x_right - region_left) - 2 * REGION_PAD,
                     (y_bounds[1] - y_bounds[0]) - 2 * REGION_PAD,
                     fc=fc, ec=ec, lw=0.9, pad=REGION_PAD, zorder=0)
        ax.text(region_left + 0.30, y_bounds[1] - 0.10, label,
                fontsize=FS_REGION, color=tx, va="top", ha="left",
                zorder=1, fontstyle="italic")

    _region(BLUE_Y, X_C + BOX_HW + REGION_TOP_PAD,
            PAL["setup_bg"], PAL["setup_bg_ec"], PAL["setup_tx"], "Assembled once")
    _region(LOOP_Y, LOOP_X_OUTER + 0.50,
            PAL["iter_bg"], PAL["iter_bg_ec"], PAL["iter_tx"], "One SCvx iteration")

    # ------------------------------------------------------------------
    # Boxes
    # ------------------------------------------------------------------

    def _box(key, primary, secondary=None, fc=None, ec=None, tx=None, lw=1.4):
        fc = fc or PAL["white"]
        ec = ec or PAL["ink"]
        tx = tx or PAL["ink"]
        _rounded_box(ax, X_C, Y[key], BOX_W, BOX_H, fc, ec, lw=lw)
        if secondary is None:
            _text(ax, X_C, Y[key], primary, FS_BOX, tx)
        else:
            _text(ax, X_C, Y[key] + 0.17, primary, FS_BOX, tx)
            _text(ax, X_C, Y[key] - 0.19, secondary, FS_SUB, tx)

    _box("init",
         r"Initialize control polygon $\mathbf{P}^{(0)}$  (straight line)",
         r"trust region  $r_0$")

    _box("assemble",
         "Assemble reusable operators",
         r"$D_N,\; E_M,\; G_N,\; \widetilde{G}_N,\; A_{\mathrm{bc}},\; b_{\mathrm{bc}}$",
         fc=PAL["setup_box"], ec=PAL["setup_ec"], tx=PAL["setup_tx"])

    _box("subdivide",
         r"Subdivide:  $P^{(s)} = S^{(s)} P^{(k)}$",
         r"rebuild supporting half-spaces  $\mathcal{H}^{(s)}$",
         fc=PAL["iter_box"], ec=PAL["iter_ec"])

    _box("build",
         "Relinearize gravity at representative points",
         r"$\longrightarrow\; H^{(k)},\; \mathbf{f}^{(k)}$",
         fc=PAL["iter_box"], ec=PAL["iter_ec"])

    # the one accented element
    _box("solve",
         r"Solve convex QP  (trust region $r_k$)",
         r"$\longrightarrow\;$ trial $\hat{\mathbf{x}}$,  slack $\mathbf{s}$",
         fc=PAL["qp_box"], ec=PAL["qp_ec"], tx=PAL["qp_tx"], lw=2.0)

    _box("merit",
         "Evaluate penalized merit function",
         r"$\rho_k = \Delta_{\mathrm{actual}} \,/\, \Delta_{\mathrm{predicted}}$",
         fc=PAL["iter_box"], ec=PAL["iter_ec"])

    _box("accept",
         r"Take $\hat{\mathbf{x}}$ as the new reference point",
         r"enlarge $r$ if $\rho_k \approx 1$",
         fc=PAL["iter_box"], ec=PAL["iter_ec"])

    _box("ret", r"Return $\mathbf{P}^{*}$")

    # Diamonds
    _diamond(ax, X_C, Y["ratio"], DIA_WX, DIA_WY, PAL["white"], PAL["ink"])
    _text(ax, X_C, Y["ratio"], r"$\rho_k > \eta$ ?", FS_DIA)

    _diamond(ax, X_C, Y["converge"], DIA_WX, DIA_WY, PAL["white"], PAL["ink"])
    _text(ax, X_C, Y["converge"] + 0.17,
          r"$|\Delta\phi| / |\phi| < \mathrm{tol}$   or   $r = r_{\min}$", FS_DIA)
    _text(ax, X_C, Y["converge"] - 0.19, r"and  $h = 0$ ?", FS_DIA)

    # ------------------------------------------------------------------
    # Forward arrows — every endpoint sits on a drawn edge
    # ------------------------------------------------------------------
    for a, b in [("init", "assemble"), ("assemble", "subdivide"),
                 ("subdivide", "build"), ("build", "solve"),
                 ("solve", "merit"), ("merit", "ratio"),
                 ("ratio", "accept"), ("accept", "converge"),
                 ("converge", "ret")]:
        _arrow(ax, bottom(a), top(b))

    # ------------------------------------------------------------------
    # Branch labels — placed beside the arrow, inside the gap
    # ------------------------------------------------------------------
    def _yes_label(src, dst):
        y_mid = (bottom(src)[1] + top(dst)[1]) / 2
        ax.text(X_C + 0.16, y_mid, "yes", fontsize=FS_EDGE, color=PAL["ink"],
                ha="left", va="center", zorder=6)

    def _no_label(src):
        ax.text(right(src)[0] + 0.16, Y[src] + 0.13, "no",
                fontsize=FS_EDGE, color=_darken(C_LOOP, 0.25),
                ha="left", va="bottom", zorder=6)

    _yes_label("ratio", "accept")
    _yes_label("converge", "ret")
    _no_label("ratio")
    _no_label("converge")

    # ------------------------------------------------------------------
    # Loop-back 1 (inner): rejected trial → re-solve at the same reference
    # ------------------------------------------------------------------
    _elbow(ax, [right("ratio"),
                (LOOP_X_INNER, Y["ratio"]),
                (LOOP_X_INNER, Y["solve"]),
                right("solve")], color=PAL["loop"], lw=1.2)
    ax.text(LOOP_X_INNER + 0.14, (Y["ratio"] + Y["solve"]) / 2,
            r"reject $\hat{\mathbf{x}}$,  shrink $r$",
            fontsize=FS_EDGE, color=_darken(C_LOOP, 0.25),
            rotation=90, ha="left", va="center")

    # ------------------------------------------------------------------
    # Loop-back 2 (outer): not converged → next iteration
    # ------------------------------------------------------------------
    _elbow(ax, [right("converge"),
                (LOOP_X_OUTER, Y["converge"]),
                (LOOP_X_OUTER, Y["subdivide"]),
                right("subdivide")], color=PAL["loop"], lw=1.2)
    ax.text(LOOP_X_OUTER + 0.14, (Y["converge"] + Y["subdivide"]) / 2,
            r"$k \leftarrow k+1$",
            fontsize=FS_EDGE, color=_darken(C_LOOP, 0.25),
            rotation=90, ha="left", va="center")

    # ------------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------------
    if save:
        out = Path(__file__).resolve().parent
        for ext in ("pdf", "png"):
            p = out / f"scp_pipeline.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved {p}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    build_figure(save="--save" in sys.argv)
