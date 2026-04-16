"""
F1. Representative KOZ linearization on one sub-arc.

Three-panel 2D figure illustrating the supporting half-space construction
for conservative spherical-KOZ avoidance.

  (a) Whole curve with a violating sub-arc highlighted
  (b) Geometric construction: centroid, normal, supporting half-space
  (c) Corrected sub-arc with all control points on the safe side

Usage:
    python figures/f1_koz_linearization.py          # show interactively
    python figures/f1_koz_linearization.py --save    # save to figures/f1_koz_linearization.pdf
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Bernstein basis & Bézier evaluation (degree-generic)
# ---------------------------------------------------------------------------

def bernstein(n, i, t):
    from math import comb
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
# De Casteljau subdivision (dimension-agnostic)
# ---------------------------------------------------------------------------

def de_casteljau_split(P, tau):
    """Split a Bézier curve at parameter tau. Returns (left_P, right_P)."""
    N = len(P) - 1
    left = [P[0].copy()]
    right = [P[-1].copy()]
    W = P.copy().astype(float)
    for _ in range(N):
        W = (1 - tau) * W[:-1] + tau * W[1:]
        left.append(W[0].copy())
        right.append(W[-1].copy())
    return np.array(left), np.array(right[::-1])


def subdivide_equal(P, n_seg):
    """Equal-parameter subdivision into n_seg sub-arcs."""
    segments = []
    remainder = P.copy().astype(float)
    for k in range(n_seg, 1, -1):
        tau = 1.0 / k
        left, remainder = de_casteljau_split(remainder, tau)
        segments.append(left)
    segments.append(remainder)
    return segments


# ---------------------------------------------------------------------------
# KOZ geometry helpers
# ---------------------------------------------------------------------------

def segment_violates(seg_P, center, radius, n_samples=200):
    ts = np.linspace(0, 1, n_samples)
    pts = bezier_eval(seg_P, ts)
    dists = np.linalg.norm(pts - center, axis=1)
    return np.any(dists < radius - 1e-9)


def compute_support(center, radius, seg_P):
    """Centroid-based supporting half-space construction."""
    centroid = seg_P.mean(axis=0)
    vec = centroid - center
    nrm = np.linalg.norm(vec)
    if nrm < 1e-12:
        n_hat = np.array([1.0, 0.0])
    else:
        n_hat = vec / nrm
    support_pt = center + radius * n_hat
    return centroid, n_hat, support_pt


def push_to_safe_side(seg_P, center, radius, n_hat, support_pt, margin=0.15):
    """Push control points to the safe side of the half-space."""
    corrected = seg_P.copy()
    for i in range(len(corrected)):
        signed_dist = np.dot(corrected[i] - support_pt, n_hat)
        if signed_dist < margin:
            corrected[i] += (margin - signed_dist) * n_hat
    return corrected


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_koz(ax, center, radius, label=True):
    circle = plt.Circle(center, radius, fc="#ffcccc", ec="#cc0000",
                        lw=1.8, alpha=0.35, zorder=1)
    ax.add_patch(circle)
    outline = plt.Circle(center, radius, fc="none", ec="#cc0000",
                         lw=1.8, zorder=2)
    ax.add_patch(outline)
    if label:
        ax.annotate(r"$\mathcal{K}$", xy=center, fontsize=14,
                    ha="center", va="center", color="#990000", zorder=3)


def draw_halfspace_line(ax, support_pt, n_hat, length=6.0, **kwargs):
    tangent = np.array([-n_hat[1], n_hat[0]])
    p1 = support_pt - tangent * length
    p2 = support_pt + tangent * length
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)


def draw_safe_region(ax, support_pt, n_hat, length=6.0, depth=4.0):
    """Shade the safe side of the half-space."""
    tangent = np.array([-n_hat[1], n_hat[0]])
    corners = np.array([
        support_pt - tangent * length,
        support_pt + tangent * length,
        support_pt + tangent * length + n_hat * depth,
        support_pt - tangent * length + n_hat * depth,
    ])
    poly = plt.Polygon(corners, fc="#d4edda", ec="none", alpha=0.35, zorder=0)
    ax.add_patch(poly)


def label_ctrl_points(ax, pts, fmt=r"$q_{%d}^{(s)}$", color="black",
                      fontsize=9, offsets=None):
    """Label control points with per-point offset tuning."""
    default_offset = (0.2, 0.25)
    for k, p in enumerate(pts):
        if offsets and k < len(offsets):
            dx, dy = offsets[k]
        else:
            dx, dy = default_offset
        ax.annotate(fmt % k, xy=p, xytext=(p[0] + dx, p[1] + dy),
                    fontsize=fontsize, color=color, zorder=10,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))


# ---------------------------------------------------------------------------
# Synthetic example geometry
# ---------------------------------------------------------------------------

# KOZ
C_KOZ = np.array([0.0, 0.0])
R_KOZ = 3.5

# Degree-5 Bézier curve that dips into the KOZ (hand-tuned for visual clarity).
# The middle control points are spread enough that the zoomed panels are readable.
P_GLOBAL = np.array([
    [-6.5,  5.0],
    [-3.5,  2.5],
    [-0.8,  1.0],
    [ 1.2,  1.2],
    [ 3.8,  2.8],
    [ 6.5,  5.0],
])

N_SEG = 5  # number of sub-arcs


# ---------------------------------------------------------------------------
# Build the figure
# ---------------------------------------------------------------------------

def build_f1(save=False):
    segments = subdivide_equal(P_GLOBAL, N_SEG)

    # Find the most-violating sub-arc
    viol_idx = None
    for i, seg in enumerate(segments):
        if segment_violates(seg, C_KOZ, R_KOZ):
            viol_idx = i
            break
    if viol_idx is None:
        raise RuntimeError("No violating sub-arc found; adjust P_GLOBAL or R_KOZ.")

    seg_viol = segments[viol_idx]
    centroid, n_hat, support_pt = compute_support(C_KOZ, R_KOZ, seg_viol)
    seg_fixed = push_to_safe_side(seg_viol, C_KOZ, R_KOZ, n_hat, support_pt,
                                   margin=0.25)

    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), constrained_layout=True)
    panel_labels = ["(a)", "(b)", "(c)"]
    panel_titles = [
        "Curve with violating sub-arc",
        "Supporting half-space construction",
        "Corrected sub-arc",
    ]

    t_dense = np.linspace(0, 1, 500)

    # --- common axis styling --------------------------------------------------
    for i, (ax, plabel, ptitle) in enumerate(zip(axes, panel_labels, panel_titles)):
        ax.set_aspect("equal")
        ax.set_facecolor("white")
        if i == 0:
            ax.tick_params(labelsize=8, colors="#888888")
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
        ax.set_title(ptitle, fontsize=11, pad=8, color="#333333")
        ax.text(0.02, 0.97, plabel, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", color="#333333")

    # =======================================================================
    # Panel (a): whole curve, sub-arcs, violating one highlighted
    # =======================================================================
    ax = axes[0]
    draw_koz(ax, C_KOZ, R_KOZ)

    # Radius annotation
    r_end = C_KOZ + np.array([R_KOZ, 0])
    ax.annotate("", xy=r_end, xytext=C_KOZ,
                arrowprops=dict(arrowstyle="<->", color="#990000", lw=1.2))
    ax.text(C_KOZ[0] + R_KOZ * 0.5, C_KOZ[1] - 0.35, r"$r_e$",
            fontsize=11, color="#990000", ha="center")

    # KOZ center
    ax.plot(*C_KOZ, "x", color="#990000", ms=6, mew=1.5, zorder=4)
    ax.annotate(r"$c_{\mathrm{KOZ}}$", xy=C_KOZ,
                xytext=(C_KOZ[0] + 0.3, C_KOZ[1] - 0.6),
                fontsize=10, color="#990000",
                arrowprops=dict(arrowstyle="-", color="#990000", lw=0.6))

    # Draw each sub-arc
    for i, seg in enumerate(segments):
        pts = bezier_eval(seg, t_dense)
        color = "#e67e22" if i == viol_idx else "#555555"
        lw = 2.4 if i == viol_idx else 1.4
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=3)
        # sub-arc junction tick
        ax.plot(*seg[0], "o", color="#333333", ms=3.5, zorder=5)
    ax.plot(*segments[-1][-1], "o", color="#333333", ms=3.5, zorder=5)

    # Violating sub-arc control polygon
    ax.plot(seg_viol[:, 0], seg_viol[:, 1], "--", color="#e67e22",
            lw=1.2, marker="o", ms=4, zorder=4, alpha=0.7)

    # Global start/end markers
    ax.plot(*P_GLOBAL[0], "s", color="#2980b9", ms=7, zorder=6)
    ax.plot(*P_GLOBAL[-1], "s", color="#2980b9", ms=7, zorder=6)

    ax.set_xlim(-7.0, 7.5)
    ax.set_ylim(-5.0, 6.5)

    # =======================================================================
    # Panel (b): zoomed construction on the violating sub-arc
    # =======================================================================
    ax = axes[1]

    # Compute zoom bounds from the sub-arc control polygon + corrected points
    all_pts_b = np.vstack([seg_viol, seg_fixed, [centroid], [support_pt]])
    mid = all_pts_b.mean(axis=0)
    extent = np.max(np.abs(all_pts_b - mid), axis=0)
    pad = max(extent) * 1.0 + 0.8

    draw_koz(ax, C_KOZ, R_KOZ, label=False)

    # Safe region shading
    draw_safe_region(ax, support_pt, n_hat, length=pad * 2, depth=pad * 2)

    # Half-space boundary line
    draw_halfspace_line(ax, support_pt, n_hat, length=pad * 1.8,
                        color="#2c3e50", lw=1.8, ls="-", zorder=4)
    # Label H^{(s)} on the upper portion of the half-space line
    tangent = np.array([-n_hat[1], n_hat[0]])
    h_label_pos = support_pt + tangent * (pad * 0.65)
    ax.annotate(r"$H^{(s)}$", xy=h_label_pos, fontsize=12,
                color="#2c3e50", ha="center", va="bottom",
                fontweight="bold", zorder=10)

    # Sub-arc curve
    pts_viol = bezier_eval(seg_viol, t_dense)
    ax.plot(pts_viol[:, 0], pts_viol[:, 1], color="#e67e22", lw=2.2, zorder=3)

    # Control polygon
    ax.plot(seg_viol[:, 0], seg_viol[:, 1], "--", color="#e67e22",
            lw=1.4, marker="o", ms=5, zorder=5)
    # Per-point offsets to avoid overlaps
    n_cp = len(seg_viol)
    cp_offsets_b = []
    for k in range(n_cp):
        if k < n_cp // 2:
            cp_offsets_b.append((-0.55, 0.25))
        elif k == n_cp // 2:
            cp_offsets_b.append((0.15, -0.45))
        else:
            cp_offsets_b.append((0.25, 0.25))
    label_ctrl_points(ax, seg_viol, color="#b45309", fontsize=9,
                      offsets=cp_offsets_b)

    # Centroid
    ax.plot(*centroid, "D", color="#8e44ad", ms=7, zorder=6)
    ax.annotate(r"$c^{(s)}$", xy=centroid,
                xytext=(centroid[0] + 0.25, centroid[1] + 0.35),
                fontsize=11, color="#8e44ad", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#8e44ad", lw=0.7),
                zorder=10)

    # Normal arrow from KOZ center through centroid
    arrow_end = support_pt + n_hat * 0.6
    ax.annotate("", xy=arrow_end, xytext=C_KOZ,
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=1.5, mutation_scale=12),
                zorder=6)
    n_label_pos = support_pt + n_hat * 0.7
    ax.annotate(r"$\hat{n}^{(s)}$", xy=n_label_pos, fontsize=11,
                color="#2c3e50", ha="left", va="bottom", zorder=10)

    # Support point on sphere surface
    ax.plot(*support_pt, "o", color="#2c3e50", ms=5, zorder=7)

    # KOZ center
    ax.plot(*C_KOZ, "x", color="#990000", ms=6, mew=1.5, zorder=4)

    ax.set_xlim(mid[0] - pad * 1.3, mid[0] + pad * 1.3)
    ax.set_ylim(mid[1] - pad * 1.3, mid[1] + pad * 1.3)

    # =======================================================================
    # Panel (c): corrected sub-arc, control points on safe side
    # =======================================================================
    ax = axes[2]

    draw_koz(ax, C_KOZ, R_KOZ, label=False)
    draw_safe_region(ax, support_pt, n_hat, length=pad * 2, depth=pad * 2)
    draw_halfspace_line(ax, support_pt, n_hat, length=pad * 1.8,
                        color="#2c3e50", lw=1.8, ls="-", zorder=4)

    # Original (faded)
    ax.plot(pts_viol[:, 0], pts_viol[:, 1], color="#e67e22", lw=1.2,
            alpha=0.3, zorder=2)
    ax.plot(seg_viol[:, 0], seg_viol[:, 1], "--", color="#e67e22",
            lw=0.8, marker="o", ms=3, alpha=0.3, zorder=2)

    # Corrected curve
    pts_fixed = bezier_eval(seg_fixed, t_dense)
    ax.plot(pts_fixed[:, 0], pts_fixed[:, 1], color="#2980b9", lw=2.4, zorder=3)

    # Corrected control polygon
    ax.plot(seg_fixed[:, 0], seg_fixed[:, 1], "--", color="#2980b9",
            lw=1.4, marker="o", ms=5, zorder=5)
    n_cp = len(seg_fixed)
    cp_offsets_c = []
    for k in range(n_cp):
        if k == 0:
            cp_offsets_c.append((-0.6, 0.45))
        elif k == 1:
            cp_offsets_c.append((-0.6, -0.45))
        elif k == n_cp // 2:
            cp_offsets_c.append((0.15, 0.50))
        elif k == n_cp - 2:
            cp_offsets_c.append((0.35, 0.45))
        elif k == n_cp - 1:
            cp_offsets_c.append((0.35, -0.35))
        else:
            cp_offsets_c.append((0.30, 0.35))
    label_ctrl_points(ax, seg_fixed, color="#1a5276", fontsize=9,
                      offsets=cp_offsets_c)

    # Convex hull of corrected control polygon (light fill)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(seg_fixed)
    hull_verts = seg_fixed[hull.vertices]
    hull_poly = plt.Polygon(hull_verts, fc="#aed6f1", ec="#2980b9",
                            lw=1.2, alpha=0.35, ls=":", zorder=1)
    ax.add_patch(hull_poly)

    # Normal arrow
    arrow_end = support_pt + n_hat * 0.6
    ax.annotate("", xy=arrow_end, xytext=C_KOZ,
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50",
                                lw=1.5, mutation_scale=12),
                zorder=6)

    ax.plot(*C_KOZ, "x", color="#990000", ms=6, mew=1.5, zorder=4)
    ax.plot(*support_pt, "o", color="#2c3e50", ms=5, zorder=7)

    ax.set_xlim(mid[0] - pad * 1.3, mid[0] + pad * 1.3)
    ax.set_ylim(mid[1] - pad * 1.3, mid[1] + pad * 1.3)

    # -----------------------------------------------------------------------
    # Save or show
    # -----------------------------------------------------------------------
    if save:
        out_dir = Path(__file__).resolve().parent
        for ext in ("pdf", "png"):
            out_path = out_dir / f"f1_koz_linearization.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight",
                        facecolor="white")
            print(f"Saved {out_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    save = "--save" in sys.argv
    build_f1(save=save)
