"""
F7. De Casteljau subdivision of the control points.

Two-panel 2D concept figure illustrating how the De Casteljau subdivision
matrix S^(s) splits the whole control polygon P into per-segment control
points P^(s) = S^(s) P, and how the per-segment convex hulls enclose the
curve more tightly than the single global hull (the convex-hull property
that the §3.1 supporting half-space relies on).

  (a) Before subdivision: control points P, one loose convex hull
  (b) After subdivision:   per-segment control points P^(s)=S^(s)P, tight hulls

Usage:
    python figures/f7_control_subdivision.py          # show interactively
    python figures/f7_control_subdivision.py --save    # save .pdf and .png
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Real repo subdivision matrices: P^(s) = S^(s) P
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbital_docking.de_casteljau import segment_matrices_equal_params


# ---------------------------------------------------------------------------
# Bernstein basis & Bézier evaluation (degree-generic)
# ---------------------------------------------------------------------------

def bernstein(n, i, t):
    return comb(n, i) * t**i * (1 - t)**(n - i)


def bezier_eval(P, t_arr):
    """Evaluate a degree-N Bézier curve at an array of parameter values."""
    N = len(P) - 1
    t_arr = np.asarray(t_arr)
    pts = np.zeros((len(t_arr), P.shape[1]))
    for i in range(N + 1):
        pts += np.outer(bernstein(N, i, t_arr), P[i])
    return pts


# ---------------------------------------------------------------------------
# Palette (matched to repo conventions: f1_koz_linearization.py, visualization.py)
# ---------------------------------------------------------------------------
CURVE          = "#2c3e50"
CP_LINE        = "#34495e"
CP_MARK        = "#2980b9"
GLOBAL_HULL_FC = "#d5dbdb"
GLOBAL_HULL_EC = "#7f8c8d"
SEG3           = ["#E74C3C", "#3498DB", "#F39C12"]   # repo per-segment palette
CENTROID       = "#8e44ad"

# Degree-5 teaching curve (concept figure; hand-tuned so the hull-tightening
# is visually obvious).
P_GLOBAL = np.array([
    [-5.0, -1.5],
    [-3.2,  4.0],
    [-0.8,  4.6],
    [ 1.6,  3.6],
    [ 3.6, -0.2],
    [ 5.0, -2.4],
])
N_SEG = 3

TITLE_A = "Before subdivision:  $P$\nsingle convex hull loosely encloses the curve"
TITLE_B = ("After subdivision:  $P^{(s)}=S^{(s)}P$\n"
           "per-segment hulls enclose the curve more tightly")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_hull(ax, pts, fc, ec, alpha=0.18, ls="-", lw=1.3, zorder=1):
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]
    ax.add_patch(plt.Polygon(verts, fc=fc, ec=ec, lw=lw, ls=ls,
                             alpha=alpha, zorder=zorder))


def style_axes(ax, title, tag):
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.set_title(title, fontsize=11, pad=8, color="#333333")
    ax.text(0.02, 0.97, tag, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", color="#333333")


def seg_control_points(P, n_seg):
    """Return list of P^(s) = S^(s) P using the repo subdivision matrices."""
    S_list = segment_matrices_equal_params(len(P) - 1, n_seg)
    return [S @ P for S in S_list]


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------

def build_f7(save=False):
    Qs = seg_control_points(P_GLOBAL, N_SEG)
    t_dense = np.linspace(0, 1, 400)
    curve = bezier_eval(P_GLOBAL, t_dense)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)

    xlim = (P_GLOBAL[:, 0].min() - 1.0, P_GLOBAL[:, 0].max() + 1.0)
    ylim = (min(P_GLOBAL[:, 1].min(), curve[:, 1].min()) - 1.0,
            max(P_GLOBAL[:, 1].max(), curve[:, 1].max()) + 1.0)

    # -----------------------------------------------------------------------
    # (a) Before subdivision: whole control polygon + one loose hull
    # -----------------------------------------------------------------------
    ax = axes[0]
    style_axes(ax, TITLE_A, "(a)")
    draw_hull(ax, P_GLOBAL, GLOBAL_HULL_FC, GLOBAL_HULL_EC, alpha=0.5, ls="-")
    ax.plot(curve[:, 0], curve[:, 1], color=CURVE, lw=2.2, zorder=4)
    ax.plot(P_GLOBAL[:, 0], P_GLOBAL[:, 1], "--s", color=CP_LINE,
            mfc=CP_MARK, mec=CP_MARK, ms=7, lw=1.2, zorder=5)
    for i, p in enumerate(P_GLOBAL):
        ax.annotate(rf"$P_{{{i}}}$", xy=p, xytext=(p[0] + 0.18, p[1] + 0.22),
                    fontsize=10, color=CP_MARK, zorder=10)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # -----------------------------------------------------------------------
    # (b) After subdivision: per-segment control polygons + tight hulls
    # -----------------------------------------------------------------------
    ax = axes[1]
    style_axes(ax, TITLE_B, "(b)")
    # faded global hull for reference
    draw_hull(ax, P_GLOBAL, GLOBAL_HULL_FC, GLOBAL_HULL_EC, alpha=0.25, ls=":")
    for s, Q in enumerate(Qs):
        c = SEG3[s % len(SEG3)]
        draw_hull(ax, Q, c, c, alpha=0.20, zorder=2)
        ax.plot(Q[:, 0], Q[:, 1], "--o", color=c, mfc=c, mec=c,
                ms=4.5, lw=1.1, zorder=5)
        cen = Q.mean(axis=0)
        ax.plot(*cen, "D", color=CENTROID, ms=7, zorder=6)
        ax.annotate(rf"$c^{{({s+1})}}$", xy=cen,
                    xytext=(cen[0] + 0.12, cen[1] + 0.28),
                    fontsize=9.5, color=CENTROID, fontweight="bold", zorder=10)
    ax.plot(curve[:, 0], curve[:, 1], color=CURVE, lw=2.2, zorder=4)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # -----------------------------------------------------------------------
    if save:
        out_dir = Path(__file__).resolve().parent
        for ext in ("pdf", "png"):
            out_path = out_dir / f"f7_control_subdivision.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
            print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    build_f7(save="--save" in sys.argv)
