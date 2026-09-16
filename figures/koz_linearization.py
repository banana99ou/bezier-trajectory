"""
Supporting half-space construction on one sub-arc (concept figure for 3.1).

  (a) Whole curve, sub-arc junctions, and the sub-arc that violates the KOZ
  (b) Construction on that sub-arc: centroid c^(s) -> outward normal n^(s)
      -> supporting half-space H^(s), tangent to the sphere at distance r_e
  (c) Result of imposing the constraint: every q_m^(s) inside H^(s), and the
      corrected curve clear of the KOZ

The geometry is synthetic 2D, chosen so the sphere's curvature is visible at the
zoom level of a single sub-arc. It cannot come from the real solver: that code
path requires dim == 3, and at the real scale (r_e = 6471 km, one sub-arc chord
~725 km at n_seg=16) the KOZ boundary departs from a straight line by 1.4% of
the chord -- the half-space and the sphere would draw as the same line, which is
exactly the distinction this figure exists to show. The results figures are
where real solver output belongs.

The *constraint* is the real one. The corrected control points are not pushed by
hand; they solve

    min_{P'} ||P' - P||_F^2
    s.t.     n^(s) . (S^(s) P')_m >= n^(s) . c_KOZ + r_e   for all s, m

over the global control points, with S^(s) taken from the repo's own De
Casteljau subdivision matrices and the normals frozen at the reference curve P.
That is one SCvx iteration of the constraint in 3.1, so the corrected curve is
continuous by construction and the endpoints stay put.

Usage:
    python figures/koz_linearization.py           # show interactively
    python figures/koz_linearization.py --save    # write .pdf and .png
    python figures/koz_linearization.py --diag    # print geometry diagnostics
"""

import sys
from math import comb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize

# Real repo subdivision matrices: P^(s) = S^(s) P
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbital_docking.de_casteljau import segment_matrices_equal_params


# ---------------------------------------------------------------------------
# Palette -- Tableau Colorblind 10, same set adopted for scp_pipeline.py.
#   orange  = the keep-out zone (hazard)
#   grey    = the reference curve, i.e. "before"
#   blue    = the corrected curve, i.e. "after"
#   ink     = the construction itself
# ---------------------------------------------------------------------------
C_INK        = "#333333"
C_KOZ_EDGE   = "#C85200"
C_KOZ_FILL   = "#FFBC79"
C_VIOL       = "#FF800E"   # violating sub-arc, before correction
C_REF        = "#898989"   # reference curve away from the violating sub-arc
C_GHOST      = "#CFCFCF"   # reference curve, ghosted into the zoomed panels
C_FIX        = "#006BA4"   # corrected sub-arc
C_FIX_GHOST  = "#A2C8EC"   # corrected curve either side of the sub-arc
C_SAFE       = "#A2C8EC"   # safe side of the half-space

FS_TITLE, FS_PANEL, FS_SYM, FS_SYM_SM = 11.5, 12.5, 12.5, 11.0


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
C_KOZ = np.array([0.0, 0.0])
R_KOZ = 3.0

# Degree-5 reference curve. Tuned so that a single interior sub-arc dips well
# inside the KOZ: its centroid then sits clearly off the sphere surface, which
# keeps c^(s), the support point and the tangent line from collapsing onto one
# another in panel (b).
P_GLOBAL = np.array([
    [-6.5,  6.5],
    [-4.2, -0.5],
    [-1.4,  1.4],
    [ 1.8,  2.1],
    [ 4.4, -0.3],
    [ 6.5,  6.8],
])
N_SEG = 8


def bernstein(n, i, t):
    return comb(n, i) * t**i * (1 - t)**(n - i)


def bezier_eval(P, t_arr):
    """Evaluate a degree-N Bezier curve at an array of parameter values."""
    N = len(P) - 1
    t_arr = np.asarray(t_arr)
    pts = np.zeros((len(t_arr), P.shape[1]))
    for i in range(N + 1):
        pts += np.outer(bernstein(N, i, t_arr), P[i])
    return pts


def outward_normal(centroid, center):
    """n^(s) = (c^(s) - c_KOZ) / ||c^(s) - c_KOZ||, undefined when they coincide
    (assumption 4 of the proposition)."""
    v = centroid - center
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        raise ValueError("c^(s) coincides with c_KOZ; normal undefined.")
    return v / nv


def supporting_halfspaces(P, S_list, center, radius):
    """Per-sub-arc supporting half-spaces built from the reference curve P.

    Returns a list of (n_hat, offset) with the half-space being
    {r : n_hat . r >= offset}, offset = n_hat . c_KOZ + r_e.
    """
    out = []
    for S in S_list:
        n_hat = outward_normal((S @ P).mean(axis=0), center)
        out.append((n_hat, float(n_hat @ center + radius)))
    return out


def solve_halfspace_projection(P, S_list, half_spaces):
    """One SCvx iteration of the 3.1 constraint, as a projection.

        min ||P' - P||_F^2   s.t.  n^(s) . (S^(s) P')_m >= n^(s) . c_KOZ + r_e

    Linear in P', so the corrected curve stays a single Bezier curve and the
    endpoints are held at the boundary conditions.
    """
    Np1, dim = P.shape

    def obj(z):
        return float(np.sum((z.reshape(Np1, dim) - P) ** 2))

    def obj_jac(z):
        return (2.0 * (z.reshape(Np1, dim) - P)).ravel()

    cons = []
    for S, (n_hat, off) in zip(S_list, half_spaces):
        # row m of A is d/dP' of  n . (S P')_m , flattened row-major
        A = np.kron(S, n_hat.reshape(1, -1))
        cons.append(dict(type="ineq",
                         fun=lambda z, A=A, off=off: A @ z - off,
                         jac=lambda z, A=A: A))
    for idx in (0, Np1 - 1):                      # boundary conditions
        E = np.zeros((dim, Np1 * dim))
        for d in range(dim):
            E[d, idx * dim + d] = 1.0
        cons.append(dict(type="eq",
                         fun=lambda z, E=E, p=P[idx].copy(): E @ z - p,
                         jac=lambda z, E=E: E))

    res = minimize(obj, P.ravel().copy(), jac=obj_jac, constraints=cons,
                   method="SLSQP", options=dict(maxiter=500, ftol=1e-12))
    if not res.success:
        raise RuntimeError(f"projection failed: {res.message}")
    return res.x.reshape(Np1, dim)


def build_geometry():
    """Reference curve, its sub-arcs, the violating one, and the correction."""
    S_list = segment_matrices_equal_params(len(P_GLOBAL) - 1, N_SEG)
    segs = [S @ P_GLOBAL for S in S_list]
    half_spaces = supporting_halfspaces(P_GLOBAL, S_list, C_KOZ, R_KOZ)

    # The sub-arc that goes deepest into the KOZ is the one worth drawing.
    t = np.linspace(0, 1, 400)
    depth = [R_KOZ - np.linalg.norm(bezier_eval(q, t) - C_KOZ, axis=1).min()
             for q in segs]
    viol = int(np.argmax(depth))
    if depth[viol] <= 0:
        raise RuntimeError("No sub-arc violates the KOZ; retune P_GLOBAL.")

    P_fix = solve_halfspace_projection(P_GLOBAL, S_list, half_spaces)
    segs_fix = [S @ P_fix for S in S_list]

    n_hat, off = half_spaces[viol]
    for m, q in enumerate(segs_fix[viol]):
        assert n_hat @ q - off > -1e-7, f"corrected q_{m} outside H^(s)"

    return dict(S_list=S_list, segs=segs, segs_fix=segs_fix, viol=viol,
                P_fix=P_fix, n_hat=n_hat, offset=off, depth=depth,
                centroid=segs[viol].mean(axis=0),
                support_pt=C_KOZ + R_KOZ * n_hat)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_koz(ax, label_at=None):
    ax.add_patch(plt.Circle(C_KOZ, R_KOZ, fc=C_KOZ_FILL, ec="none",
                            alpha=0.20, zorder=1))
    ax.add_patch(plt.Circle(C_KOZ, R_KOZ, fc="none", ec=C_KOZ_EDGE, lw=1.9,
                            zorder=2))
    if label_at is not None:
        ax.text(*label_at, r"$\mathcal{K}$", fontsize=FS_SYM + 2,
                ha="center", va="center", color=C_KOZ_EDGE, zorder=3)


def draw_halfspace(ax, support_pt, n_hat, reach, faint=False):
    """Boundary line of H^(s) plus shading on the safe side."""
    tang = np.array([-n_hat[1], n_hat[0]])
    a, b = support_pt - tang * reach, support_pt + tang * reach
    ax.add_patch(plt.Polygon([a, b, b + n_hat * reach, a + n_hat * reach],
                             fc=C_SAFE, ec="none",
                             alpha=0.14 if faint else 0.22, zorder=0))
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C_INK,
            lw=1.2 if faint else 1.8, alpha=0.45 if faint else 1.0, zorder=4)


def style_axes(ax, title, tag):
    # Square axes box in every panel: the equal aspect is met by widening the
    # data window, not by shrinking the box, so (a) matches (b) and (c).
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.set_title(title, fontsize=FS_TITLE, pad=8, color=C_INK)
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="top", color=C_INK)


def label_ctrl_polygon(ax, pts, anchor, offset, color):
    """Name the control polygon once. Labelling every q_m adds no information
    and crowds the region where the correction is largest."""
    p = pts[anchor]
    ax.annotate(r"$\mathbf{q}^{(s)}_m$", xy=p,
                xytext=(p[0] + offset[0], p[1] + offset[1]),
                fontsize=FS_SYM_SM, color=color, ha="center", va="center",
                zorder=10, arrowprops=dict(arrowstyle="-", color=color, lw=0.7))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

# Repository plotting defaults make all math upright. Match the manuscript's
# LaTeX convention locally: italic scalar/index symbols, explicit bold vectors,
# upright descriptive KOZ subscripts, and calligraphic sets.
@plt.rc_context({"mathtext.fontset": "cm", "mathtext.default": "it"})
def build_figure(save=False):
    g = build_geometry()
    segs, segs_fix, viol = g["segs"], g["segs_fix"], g["viol"]
    n_hat, centroid, support_pt = g["n_hat"], g["centroid"], g["support_pt"]
    tang = np.array([-n_hat[1], n_hat[0]])

    t = np.linspace(0, 1, 400)
    curve_ref = bezier_eval(P_GLOBAL, t)
    curve_fix = bezier_eval(g["P_fix"], t)
    arc_ref, arc_fix = bezier_eval(segs[viol], t), bezier_eval(segs_fix[viol], t)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4), constrained_layout=True)

    # Window shared by (b) and (c). It has to hold the whole sphere, not just the
    # sub-arc: c_KOZ is where the normal is measured from, so cropping it out is
    # what turns the r_e dimension into a line running off the panel.
    focus = np.vstack([arc_ref, arc_fix, segs[viol], segs_fix[viol],
                       [C_KOZ], [centroid], [support_pt]])
    lo, hi = focus.min(axis=0), focus.max(axis=0)
    mid = (lo + hi) / 2.0
    pad = float(np.max(hi - lo)) / 2.0 * 1.32

    def ghost(ax, pts, color, lw=1.6, zorder=2.5, alpha=1.0):
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round")

    # ===================================================================== (a)
    ax = axes[0]
    style_axes(ax, "Curve with violating sub-arc", "(a)")
    draw_koz(ax, label_at=C_KOZ + np.array([-1.45, -1.75]))

    # r_e down a clear diagonal, started off-centre so nothing stacks on c_KOZ
    u = np.array([np.cos(np.deg2rad(-52.0)), np.sin(np.deg2rad(-52.0))])
    ax.add_patch(FancyArrowPatch(C_KOZ + u * 0.42, C_KOZ + u * R_KOZ,
                                 arrowstyle="-|>", color=C_KOZ_EDGE, lw=1.2,
                                 mutation_scale=10, shrinkA=0, shrinkB=0,
                                 zorder=5))
    rl = C_KOZ + u * (R_KOZ * 0.62) + np.array([0.62, 0.30])
    ax.text(*rl, r"$R_{\mathrm{KOZ}}$", fontsize=FS_SYM_SM, color=C_KOZ_EDGE,
            ha="center", va="center", zorder=6)

    ax.plot(*C_KOZ, "o", color=C_KOZ_EDGE, ms=4.0, zorder=6)
    ax.text(C_KOZ[0] - 0.28, C_KOZ[1] + 0.42, r"$\mathbf{c}_{\mathrm{KOZ}}$",
            fontsize=FS_SYM_SM, color=C_KOZ_EDGE, ha="right", va="bottom",
            zorder=6)

    for i, q in enumerate(segs):
        pts = bezier_eval(q, t)
        hot = i == viol
        ax.plot(pts[:, 0], pts[:, 1], color=C_VIOL if hot else C_REF,
                lw=2.8 if hot else 1.5, zorder=4 if hot else 3,
                solid_capstyle="round")
        ax.plot(*q[0], "o", color=C_INK, ms=3.6, zorder=5)
    ax.plot(*segs[-1][-1], "o", color=C_INK, ms=3.6, zorder=5)

    ax.annotate("sub-arc $s$", xy=arc_ref[len(arc_ref) // 2],
                xytext=(arc_ref[len(arc_ref) // 2][0] - 0.2, 4.55),
                fontsize=FS_SYM_SM, color=C_VIOL, ha="center", va="center",
                zorder=10, arrowprops=dict(arrowstyle="-", color=C_VIOL,
                                           lw=0.7))

    ax.plot(*P_GLOBAL[0], "s", color=C_INK, ms=6.5, zorder=6)
    ax.plot(*P_GLOBAL[-1], "s", color=C_INK, ms=6.5, zorder=6)
    ax.set_xlim(-6.9, 6.9)
    ax.set_ylim(-3.6, 7.6)

    # ===================================================================== (b)
    ax = axes[1]
    style_axes(ax, "Supporting half-space construction", "(b)")
    draw_koz(ax)
    draw_halfspace(ax, support_pt, n_hat, pad * 2.2)
    ghost(ax, curve_ref, C_GHOST)

    # n^(s) is by definition the c_KOZ -> c^(s) direction, so the whole
    # construction is collinear. r_e is therefore drawn as an offset dimension
    # line with extension ticks, not stacked on top of the normal.
    ax.add_patch(FancyArrowPatch(C_KOZ, support_pt, arrowstyle="<|-|>",
                                 color=C_KOZ_EDGE, lw=1.1, mutation_scale=8,
                                 shrinkA=0, shrinkB=0, zorder=3))
    rl = C_KOZ + n_hat * 0.55 - tang * 0.45
    ax.text(*rl, r"$R_{\mathrm{KOZ}}$", fontsize=FS_SYM_SM, color=C_KOZ_EDGE,
            ha="center", va="center", zorder=6)
    ax.plot(*C_KOZ, "o", color=C_KOZ_EDGE, ms=4.0, zorder=6)
    ax.text(C_KOZ[0] + 0.18, C_KOZ[1] + 0.16, r"$\mathbf{c}_{\mathrm{KOZ}}$",
            fontsize=FS_SYM_SM, color=C_KOZ_EDGE, ha="left", va="bottom",
            zorder=6)

    ghost(ax, arc_ref, C_VIOL, lw=2.8, zorder=4)
    ax.plot(segs[viol][:, 0], segs[viol][:, 1], "--o", color=C_VIOL, lw=1.3,
            ms=5, zorder=5)
    label_ctrl_polygon(ax, segs[viol], anchor=0, offset=(-0.55, -0.85),
                       color=C_VIOL)

    ax.plot(*centroid, "D", color=C_INK, ms=7, zorder=7)
    ax.annotate(r"$\mathbf{c}^{(s)}$", xy=centroid,
                xytext=tuple(centroid + tang * 0.90),
                fontsize=FS_SYM_SM, color=C_INK, ha="center", va="center",
                zorder=10, arrowprops=dict(arrowstyle="-", color=C_INK, lw=0.7))

    # the normal, running from c^(s) out through the point it touches
    reach = float(n_hat @ (support_pt - centroid)) + 0.40
    ax.add_patch(FancyArrowPatch(centroid, centroid + n_hat * reach,
                                 arrowstyle="-|>", color=C_INK, lw=2.0,
                                 mutation_scale=13, shrinkA=0, shrinkB=0,
                                 zorder=8))
    nl = centroid + n_hat * (reach * 0.55) - tang * 0.52
    ax.text(*nl, r"$\mathbf{n}^{(s)}$", fontsize=FS_SYM_SM, color=C_INK,
            ha="center", va="center", zorder=10)

    ax.plot(*support_pt, "o", color=C_INK, ms=5.5, zorder=8)
    hl = support_pt - tang * (pad * 0.60) + n_hat * 0.52
    ax.text(*hl, r"$\mathcal{H}^{(s)}$", fontsize=FS_SYM, color=C_INK,
            ha="center", va="center", fontweight="bold", zorder=10)

    ax.set_xlim(mid[0] - pad, mid[0] + pad)
    ax.set_ylim(mid[1] - pad, mid[1] + pad)

    # ===================================================================== (c)
    ax = axes[2]
    style_axes(ax, "Corrected sub-arc", "(c)")
    draw_koz(ax)
    draw_halfspace(ax, support_pt, n_hat, pad * 2.2, faint=True)

    ghost(ax, curve_ref, C_GHOST)                 # before
    ghost(ax, arc_ref, C_VIOL, lw=2.4, alpha=0.60, zorder=2.6)
    ghost(ax, curve_fix, C_FIX_GHOST, lw=2.6, zorder=2.8)   # after, full width

    # No convex hull is drawn here. Projecting onto a half-space straightens the
    # arc, so the corrected control points are near-collinear: the hull comes out
    # 0.39% of the chord thick, which is sub-pixel at 300 dpi. The convex-hull
    # argument behind Proposition 1 is carried by control_subdivision.py, where
    # the geometry is not degenerate.
    ghost(ax, arc_fix, C_FIX, lw=2.8, zorder=4)
    ax.plot(segs_fix[viol][:, 0], segs_fix[viol][:, 1], "--o", color=C_FIX,
            lw=1.3, ms=5, zorder=5)
    label_ctrl_polygon(ax, segs_fix[viol], anchor=0, offset=(-0.55, -0.85),
                       color=C_FIX)

    ax.plot(*support_pt, "o", color=C_INK, ms=5.5, zorder=8)
    hl = support_pt - tang * (pad * 0.60) + n_hat * 0.52
    ax.text(*hl, r"$\mathcal{H}^{(s)}$", fontsize=FS_SYM, color=C_INK,
            ha="center", va="center", fontweight="bold", alpha=0.55, zorder=10)

    ax.set_xlim(mid[0] - pad, mid[0] + pad)
    ax.set_ylim(mid[1] - pad, mid[1] + pad)

    # -----------------------------------------------------------------------
    if save:
        out_dir = Path(__file__).resolve().parent
        for ext in ("pdf", "png"):
            out_path = out_dir / f"koz_linearization.{ext}"
            fig.savefig(out_path, dpi=300, bbox_inches="tight",
                        facecolor="white")
            print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close(fig)


def diagnostics():
    g = build_geometry()
    viol, n_hat = g["viol"], g["n_hat"]
    c, sp = g["centroid"], g["support_pt"]
    t = np.linspace(0, 1, 2000)
    d_ref = np.linalg.norm(bezier_eval(P_GLOBAL, t) - C_KOZ, axis=1).min()
    d_fix = np.linalg.norm(bezier_eval(g["P_fix"], t) - C_KOZ, axis=1).min()
    print(f"violating sub-arc      : {viol} of {N_SEG}")
    print(f"penetration depth      : {g['depth'][viol]:+.3f}  (r_e = {R_KOZ})")
    print(f"|c^(s) - c_KOZ|        : {np.linalg.norm(c - C_KOZ):.3f}"
          f"   -> {R_KOZ - np.linalg.norm(c - C_KOZ):.3f} inside the sphere")
    print(f"|c^(s) - support point|: {np.linalg.norm(c - sp):.3f}"
          f"   (separation in panel b)")
    print(f"normal direction       : {np.rad2deg(np.arctan2(*n_hat[::-1])):+.1f} deg")
    print(f"min |curve - c_KOZ|    : before {d_ref:.3f}   after {d_fix:.3f}"
          f"   (r_e = {R_KOZ})")
    print(f"control-point shift    : "
          f"{np.round(np.linalg.norm(g['P_fix'] - P_GLOBAL, axis=1), 3)}")
    print(f"endpoints fixed        : "
          f"{np.allclose(g['P_fix'][[0, -1]], P_GLOBAL[[0, -1]], atol=1e-9)}")
    print(f"corrected curve C-inf  : single Bezier curve, no splice")


if __name__ == "__main__":
    if "--diag" in sys.argv:
        diagnostics()
    else:
        build_figure(save="--save" in sys.argv)
