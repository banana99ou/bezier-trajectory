#!/usr/bin/env python3
"""The poster's own figures, in the style of the author's clipwall figures.

Five PDFs, all vector, written next to poster.tex:

  fig_lift.pdf       (a) a sphere moving at constant velocity becomes a tilted
                     static tube once time is a coordinate; (b) the loiter
                     scenario's shadow spot at the corridor altitude over one
                     lap, with the two RETURNED curves from the sidecar threaded
                     through it. (b) is drawn from scenario_loiter() and
                     figures/paper1/occlusion_figure.json; (a) is a schematic.
                     Full poster width.
  fig_clip.pdf       one subdivided segment of a full trajectory, the clipping
                     ball, the clipped KOZ volume L and the wall built on it;
                     r = max(d, E + delta) with d, E, r marked. Schematic.
  fig_halfspace.pdf  a hairpin tube that wraps the centroid: the band cut at the
                     distance maximum, one wall per approach segment (orange,
                     purple), the mint slab both walls allow. Schematic.
  fig_iterate.pdf    three SCP iterates, ball and wall rebuilt at each. Schematic.
  fig_runs.pdf       the two returned curves in (x, t) on the corridor axis with
                     the shadow band, from the sidecar. Every number printed on
                     it is read from the sidecar, never typed. This one keeps the
                     paper figures' own palette, since it sits beside Fig. 2.

Style, from fig_clipwall.py (recovered 2026-09-03): white ground, equal aspect,
faint grid, keep-out zone a flat grey tube with a thin dark-slate centreline,
clipping ball a dashed red circle, the clipped KOZ volume the only saturated
fill (orange for wall 1, purple for wall 2), the wall the only thick line in its
piece's colour, the allowed region flat mint, the region a wall excludes that is
not keep-out left white so the conservatism is visible. Segment control points
are small blue dots on a short nearly straight blue line, the centroid a dark
star, the foot a black square, the nearest KOZ point an open circle, the normal a
short arrow from it in the wall's colour. A symbol legend sits below the axes.

Authored at HALF the printed width -- a column on the A0 poster is about 14.8
in, the column figures are 7.4 in -- so a 12 pt label prints at 24 pt and a 1 pt
stroke at 2 pt.

    python3 paper/ksas_2026_fall/poster/make_poster_figures.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from math import comb

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
SIDECAR = REPO / "figures" / "paper1" / "occlusion_figure.json"
sys.path.insert(0, str(REPO))

# clipwall's palette
SLATE, TUBE, BALL = "#212f3d", "#c8ced3", "#c0392b"
ORANGE, PURPLE, MINT = "#e67e22", "#7d3c98", "#a9dfbf"
CTRL = "#2874a6"
PAPER = "#FFFFFF"   # the poster ground; the figures share it
# the paper figures' palette, for the one figure drawn from the measured run
GREY, BLUE, DARK, SHADOW = "#8a8a8a", "#2255bb", "#404040", "#bbbbbb"
W = 7.4  # authored width in inches: half the printed column


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 12, "legend.fontsize": 10,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "axes.edgecolor": SLATE, "text.color": SLATE, "axes.labelcolor": SLATE,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "pdf.fonttype": 42,
    })
    return plt


# ----------------------------------------------------------------------------- geometry
def bezier(cp, n=400):
    cp = np.asarray(cp, float)
    s = np.linspace(0.0, 1.0, n)[:, None]
    deg = len(cp) - 1
    out = np.zeros((n, cp.shape[1]))
    for i, q in enumerate(cp):
        out += comb(deg, i) * (1 - s) ** (deg - i) * s ** i * q
    return out


def de_casteljau_sub(cp, s0, s1):
    """Control points of the piece of a Bezier curve on [s0, s1] (two splits)."""
    def split(pts, s):
        pts = [np.asarray(p, float) for p in pts]
        left, right = [pts[0]], [pts[-1]]
        while len(pts) > 1:
            pts = [(1 - s) * a + s * b for a, b in zip(pts[:-1], pts[1:])]
            left.append(pts[0]); right.append(pts[-1])
        return np.asarray(left), np.asarray(right[::-1])
    _, right = split(cp, s0)
    left, _ = split(right, (s1 - s0) / (1 - s0))
    return left


def tube_polygon(centre, radius, n_cap=40):
    """Outline of the union of discs along a polyline: offset curves plus round
    caps (clipwall's tube_poly). Valid while radius < curvature radius."""
    c = np.asarray(centre, float)
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.column_stack([-d[:, 1], d[:, 0]])
    a = np.linspace(-np.pi / 2, np.pi / 2, n_cap)
    cap_end = c[-1] + radius * (np.cos(a)[:, None] * nrm[-1] + np.sin(a)[:, None] * d[-1])
    cap_start = c[0] + radius * (np.cos(a)[:, None] * (-nrm[0]) + np.sin(a)[:, None] * (-d[0]))
    return np.vstack([c + radius * nrm, cap_end, (c - radius * nrm)[::-1], cap_start])


def grid(lo, hi, n=420):
    gx = np.linspace(lo[0], hi[0], n); gy = np.linspace(lo[1], hi[1], n)
    return np.meshgrid(gx, gy)


def dist_to_polyline(P, line):
    """Distance from each point of P (..., 2) to a polyline (M, 2), by segments."""
    A, B = line[:-1], line[1:]
    AB = B - A
    L2 = np.einsum("ij,ij->i", AB, AB)
    flat = P.reshape(-1, 2)
    best = np.full(len(flat), np.inf)
    for a, ab, l2 in zip(A, AB, L2):
        u = np.clip(((flat - a) @ ab) / l2, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(flat - (a + u[:, None] * ab), axis=1))
    return best.reshape(P.shape[:-1])


def wall_for(c, arm, r_m, r, GX, GY):
    """One wall, clipwall's construction: y* the point of L = KOZ ∩ B(c, r)
    nearest c sets n; b = max n·z over L sets where the wall sits."""
    P = np.stack([GX, GY], axis=-1)
    inL = (dist_to_polyline(P, arm) <= r_m) & (np.hypot(GX - c[0], GY - c[1]) <= r)
    pts = P[inL]
    i_near = int(np.argmin(np.linalg.norm(pts - c, axis=1)))
    y_star = pts[i_near]
    n = (c - y_star) / np.linalg.norm(c - y_star)
    b = float(np.max(pts @ n))
    d_all = np.linalg.norm(arm - c, axis=1)
    foot = arm[int(np.argmin(d_all))]
    return n, b, y_star, foot, inL


def wall_line(ax, n, b, colour, lo, hi, lw=3.0):
    xs = np.linspace(lo[0], hi[0], 60)
    if abs(n[1]) > 1e-9:
        ax.plot(xs, (b - n[0] * xs) / n[1], color=colour, lw=lw, zorder=8, solid_capstyle="butt")
    else:
        ax.axvline(b / n[0], color=colour, lw=lw, zorder=8)


def frame(ax, lo, hi, xlabel="x  (space)", ylabel="t  (time)"):
    ax.set_aspect("equal"); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_xlabel(xlabel, labelpad=3); ax.set_ylabel(ylabel, labelpad=3)


def symbol_legend(ax, handles):
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.03, 1.0), ncol=1,
              frameon=False, handlelength=1.8, labelspacing=0.75, borderaxespad=0.0)


# ----------------------------------------------------------------------------- figures
def tag(ax, xy, text, colour=SLATE, size=12.5, ha="center", va="center", halo=3.2):
    """A label placed ON the geometry, with a white halo. This is what replaces
    the legend: nothing in these figures should need a lookup."""
    import matplotlib.patheffects as pe
    ax.text(xy[0], xy[1], text, fontsize=size, color=colour, ha=ha, va=va, zorder=20,
            path_effects=[pe.withStroke(linewidth=halo, foreground=PAPER)])


def measure(ax, a, b, text, colour=SLATE, off=(0.0, 0.0), size=13):
    ax.annotate("", xy=b, xytext=a, zorder=9,
                arrowprops=dict(arrowstyle="<->", color=colour, lw=1.2, shrinkA=3, shrinkB=3))
    m = 0.5 * (np.asarray(a) + np.asarray(b)) + np.asarray(off)
    tag(ax, m, text, colour, size)


def fig_clip(plt):
    """One segment of a trajectory, the ball around its centroid, the clipped
    part of the keep-out tube, and the one wall built on it. No legend: every
    object is labelled where it sits."""
    from matplotlib.patches import Circle, Polygon
    fig, ax = plt.subplots(figsize=(W, 3.5))
    lo, hi = np.array([-12.0, -8.0]), np.array([12.0, 4.2])
    # the keep-out tube: one dip toward the trajectory, so it is NOT convex,
    # and fat enough to read as a volume rather than a line
    xs = np.linspace(-13.0, 13.0, 400)
    centre = np.column_stack([xs, 2.7 - 3.0 * np.exp(-((xs - 2.0) / 5.5) ** 2)])
    r_m = 1.95
    # the whole trajectory, and ONE subdivided segment of it
    cp = np.array([[-11.5, -5.2], [-8.0, -4.4], [-4.0, -3.5], [0.0, -3.2],
                   [4.0, -3.4], [8.0, -2.6], [11.5, -0.9]])
    curve = bezier(cp, 600)
    seg_cp = de_casteljau_sub(cp, 0.28, 0.60)
    seg_arc = bezier(seg_cp, 200)
    c = seg_cp.mean(axis=0)
    dists = np.linalg.norm(seg_cp - c, axis=1)
    E, i_far = float(np.max(dists)), int(np.argmax(dists))
    delta = 0.6
    d_all = np.linalg.norm(centre - c, axis=1)
    p, d = centre[int(np.argmin(d_all))], float(np.min(d_all))
    r = max(d, E + delta)
    GX, GY = grid(lo, hi)
    n, b, y_star, foot, _ = wall_for(c, centre, r_m, r, GX, GY)

    ax.contourf(GX, GY, (GX * n[0] + GY * n[1] >= b).astype(float), [0.5, 1.5],
                colors=[MINT], alpha=0.42, zorder=0)
    ax.add_patch(Polygon(tube_polygon(centre, r_m), closed=True, fc=TUBE, ec="none", zorder=1))
    L = Polygon(tube_polygon(centre, r_m), closed=True, fc=ORANGE, ec="none", zorder=3)
    ax.add_patch(L); L.set_clip_path(Circle(c, r, transform=ax.transData))
    wall_line(ax, n, b, ORANGE, lo, hi, lw=1.4)          # the half-space is global,
    wall_line(ax, n, b, ORANGE, c - r * 1.15, c + r * 1.15, lw=3.6)  # but it binds here
    ax.add_patch(Circle(c, r, fc="none", ec=BALL, lw=2.2, ls=(0, (6, 4)), zorder=12))
    # the trajectory, the segment ON it, then the segment's own control polygon
    ax.plot(curve[:, 0], curve[:, 1], color=CTRL, lw=1.8, alpha=0.40, zorder=4)
    ax.plot(seg_arc[:, 0], seg_arc[:, 1], color=CTRL, lw=4.6, solid_capstyle="round", zorder=13)
    ax.plot(seg_cp[:, 0], seg_cp[:, 1], "o--", color=CTRL, ms=6.5, lw=1.1, alpha=0.9, zorder=13)
    ax.plot(*c, "*", color=SLATE, ms=15, zorder=14)

    measure(ax, c, p, "d", SLATE, off=(0.85, 0.2))
    eoff = np.array([0.0, -1.0])          # E as a dimension line below the segment
    ax.plot([c[0], c[0]], [c[1], c[1] + eoff[1]], color=SLATE, lw=0.6, alpha=0.6, zorder=8)
    ax.plot([seg_cp[i_far][0]] * 2, [seg_cp[i_far][1], seg_cp[i_far][1] + eoff[1]],
            color=SLATE, lw=0.6, alpha=0.6, zorder=8)
    measure(ax, c + eoff, seg_cp[i_far] + eoff, "E", SLATE, off=(0.0, -0.75))
    aR = np.deg2rad(-150)
    measure(ax, c, c + r * np.array([np.cos(aR), np.sin(aR)]), "r", BALL, off=(-0.3, 0.75))
    tag(ax, c + [0.35, 0.75], "c", SLATE, 13.5, ha="left")
    tag(ax, (-9.4, -4.7), "trajectory", CTRL, 12.5)
    tag(ax, (-3.4, -6.9), "one segment, with its control points", CTRL, 12.5)
    tag(ax, (8.8, 3.0), "keep-out tube", SLATE, 13)
    tag(ax, (3.4, -1.2), "L = tube ∩ ball", "#7a3d0a", 12.5)
    tag(ax, (-7.4, 3.6), "wall  n·Q ≥ b", "#a4530f", 13)
    tag(ax, (8.0, -6.9), "allowed by the wall", "#3d7a55", 12.5)
    tag(ax, (-6.6, -2.4), "ball  B(c, r)", BALL, 12.5)
    frame(ax, lo, hi)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.075, top=0.985)
    return fig


def fig_halfspace(plt):
    """A tube that wraps the centroid. Cut at the distance maximum, one wall per
    approach segment, and what both walls allow is the gap between them."""
    from matplotlib.patches import Circle, Polygon
    fig, ax = plt.subplots(figsize=(W, 4.0))
    lo, hi = np.array([-5.2, 0.8]), np.array([10.8, 8.9])
    hair = np.array([[10.4, 1.7], [-4.8, 3.3], [-4.8, 6.9], [10.4, 8.5]])
    centre = bezier(hair, 700)
    r_m = 0.72
    # a real piece of a trajectory, sitting in the mouth of the hairpin
    scp = np.array([[3.4, 4.92], [4.2, 5.04], [5.0, 5.13], [5.8, 5.18]])
    seg_arc = bezier(scp, 200)
    c = scp.mean(axis=0)
    E = float(np.max(np.linalg.norm(scp - c, axis=1)))
    d = float(np.min(np.linalg.norm(centre - c, axis=1)))
    r = max(d, E + 0.3)
    dist = np.linalg.norm(centre - c, axis=1)
    interior_max = np.flatnonzero((dist[1:-1] > dist[:-2]) & (dist[1:-1] >= dist[2:])) + 1
    i_cut = int(interior_max[0])
    arms = [centre[: i_cut + 1], centre[i_cut:]]
    GX, GY = grid(lo, hi)
    walls = [wall_for(c, arm, r_m, r, GX, GY) for arm in arms]
    free = np.ones_like(GX, dtype=bool)
    for n, b, *_ in walls:
        free &= GX * n[0] + GY * n[1] >= b
    ax.contourf(GX, GY, free.astype(float), [0.5, 1.5], colors=[MINT], alpha=0.45, zorder=0)
    ax.add_patch(Polygon(tube_polygon(centre, r_m), closed=True, fc=TUBE, ec="none", zorder=1))
    for (n, b, y_star, foot, _), arm, colour in zip(walls, arms, (ORANGE, PURPLE)):
        L = Polygon(tube_polygon(arm, r_m), closed=True, fc=colour, ec="none", zorder=3)
        ax.add_patch(L); L.set_clip_path(Circle(c, r, transform=ax.transData))
        wall_line(ax, n, b, colour, lo, hi, lw=1.4)
        wall_line(ax, n, b, colour, c - r * 1.7, c + r * 1.7, lw=3.6)
        ax.annotate("", xy=y_star + n * 0.8, xytext=y_star, zorder=11,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2, mutation_scale=14))
    ax.add_patch(Circle(c, r, fc="none", ec=BALL, lw=2.2, ls=(0, (6, 4)), zorder=12))
    ax.plot(seg_arc[:, 0], seg_arc[:, 1], color=CTRL, lw=4.4, solid_capstyle="round", zorder=13)
    ax.plot(scp[:, 0], scp[:, 1], "o--", color=CTRL, ms=6.0, lw=1.0, alpha=0.9, zorder=13)
    ax.plot(*c, "*", color=SLATE, ms=15, zorder=14)
    cut = centre[i_cut]
    ax.plot(*cut, "x", color=SLATE, ms=14, mew=2.8, zorder=15)

    (n1, b1, ys1, _, _), (n2, b2, ys2, _, _) = walls
    tag(ax, c + [0.05, -0.42], "c", SLATE, 13.5)
    tag(ax, (4.6, 5.78), "one segment", CTRL, 12.5)
    tag(ax, cut + [0.30, -1.30], "distance maximum:\ncut the centreline here", SLATE, 12, ha="left")
    tag(ax, ys1 + n1 * 0.55 + [0.62, 0.0], "n₁", ORANGE, 13.5)
    tag(ax, ys2 + n2 * 0.55 + [0.62, 0.0], "n₂", PURPLE, 13.5)
    tag(ax, (6.4, 2.10), "approach segment 1", "#a4530f", 12.5)
    tag(ax, (6.4, 8.10), "approach segment 2", "#5b2c72", 12.5)
    tag(ax, (10.2, (b1 - n1[0] * 10.2) / n1[1] - 0.36), "wall 1", "#a4530f", 13, ha="right")
    tag(ax, (10.2, (b2 - n2[0] * 10.2) / n2[1] + 0.36), "wall 2", "#5b2c72", 13, ha="right")
    tag(ax, (10.5, 5.95), "the gap both walls allow", "#3d7a55", 12.5, ha="right")
    frame(ax, lo, hi)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.07, top=0.985)
    return fig


def fig_iterate(plt):
    from matplotlib.patches import Circle, Polygon
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.9))
    s = np.linspace(0, 1, 300)
    centre = np.column_stack([2.4 - 2.2 * np.sin(np.pi * s) ** 2, -3.4 + 6.8 * s])
    r_m = 0.75
    delta = 0.3
    lo, hi = np.array([-4.2, -3.2]), np.array([3.2, 3.2])
    GX, GY = grid(lo, hi, 300)
    # the segment approaches the tube over the iterates: d shrinks, so r = max(d, E+δ) shrinks
    for k, (ax, cx, cy) in enumerate(zip(axes, (-2.5, -1.75, -1.25), (-1.4, -0.2, 0.9))):
        seg = np.array([[cx - 0.45, cy - 0.12], [cx - 0.15, cy - 0.04], [cx + 0.15, cy + 0.04], [cx + 0.45, cy + 0.12]])
        c = seg.mean(axis=0)
        E = float(np.max(np.linalg.norm(seg - c, axis=1)))
        d = float(np.min(np.linalg.norm(centre - c, axis=1)))
        r = max(d, E + delta)
        n, b, y_star, foot, _ = wall_for(c, centre, r_m, r, GX, GY)
        free = (GX * n[0] + GY * n[1] >= b).astype(float)
        ax.contourf(GX, GY, free, [0.5, 1.5], colors=[MINT], alpha=0.55, zorder=0)
        ax.add_patch(Polygon(tube_polygon(centre, r_m), closed=True, fc=TUBE, ec="none", zorder=1))
        ax.plot(centre[:, 0], centre[:, 1], color=SLATE, lw=0.8, alpha=0.6, zorder=2)
        L = Polygon(tube_polygon(centre, r_m), closed=True, fc=ORANGE, ec="none", zorder=3)
        ax.add_patch(L); L.set_clip_path(Circle(c, r, transform=ax.transData))
        wall_line(ax, n, b, ORANGE, lo, hi, lw=2.6)
        ax.add_patch(Circle(c, r, fc="none", ec=BALL, lw=1.8, ls="--", zorder=12))
        ax.plot(seg[:, 0], seg[:, 1], "o-", color=CTRL, ms=4, lw=1.2, zorder=13)
        ax.plot(*c, "*", color=SLATE, ms=12, zorder=14)
        ax.plot(*foot, "s", color="k", ms=6, zorder=10)
        ax.plot(*y_star, "o", color="k", mfc="w", mew=1.4, ms=7, zorder=10)
        ax.annotate("", xy=y_star + n * 0.7, xytext=y_star,
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8, mutation_scale=11), zorder=11)
        ax.set_title("iteration k" + ("" if k == 0 else f" + {k}"), fontsize=12, pad=4)
        frame(ax, lo, hi, xlabel="", ylabel="")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.9, wspace=0.06)
    return fig


def tube_surface(centre, radius, n_theta=36, ref=(0.0, 0.0, 1.0)):
    c = np.asarray(centre, float)
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.asarray(ref, float)
    e1 = np.cross(t, ref)
    bad = np.linalg.norm(e1, axis=1) < 1e-9
    e1[bad] = np.cross(t[bad], [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(t, e1)
    th = np.linspace(0.0, 2 * np.pi, n_theta)
    X = c[:, None, :] + radius * (np.cos(th)[None, :, None] * e1[:, None, :]
                                  + np.sin(th)[None, :, None] * e2[:, None, :])
    return X[..., 0], X[..., 1], X[..., 2]


def sphere(centre, radius, n=28):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, np.pi, n // 2 + 1))
    return (centre[0] + radius * np.cos(u) * np.sin(v),
            centre[1] + radius * np.sin(u) * np.sin(v),
            centre[2] + radius * np.cos(v))


def loiter_shadow_track(sc):
    """Centre of the body's shadow spot at the corridor altitude over the lap,
    and the spot radius: similar triangles from the station, as in the scenario."""
    obs = sc["obstacles"]
    z_c = float(sc["start"][2])
    st = np.asarray(sc["stations"][0], float)
    times, xs, ys = [], [], []
    for o in obs:
        pts = bezier(np.asarray(o["control_points"], float), 120)
        k = (z_c - st[2]) / (pts[:, 2] - st[2])
        times.append(pts[:, 3]); xs.append(st[0] + k * (pts[:, 0] - st[0])); ys.append(st[1] + k * (pts[:, 1] - st[1]))
    body_r = float(obs[0]["radius"])
    body_z = float(np.asarray(obs[0]["control_points"], float)[0, 2])
    spot_r = body_r * (z_c - st[2]) / (body_z - st[2])
    t, x, y = map(np.concatenate, (times, xs, ys))
    order = np.argsort(t)
    return t[order], x[order], y[order], spot_r


def style_3d(ax):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"]["color"] = "#e6e6e6"
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(labelsize=10, pad=-1, colors=SLATE)


def label3(ax, p, text, dx, dy, colour, ha="left", size=11):
    from mpl_toolkits.mplot3d import proj3d
    x2, y2, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
    ax.annotate(text, (x2, y2), xytext=(dx, dy), textcoords="offset points",
                fontsize=size, color=colour, ha=ha, va="center")


def fig_lift(plt, sc, side):
    """(a) alone, at column width: a sphere moving in space (the discs on the
    t = 0 floor) is ONE static tube once time is a coordinate."""
    fig = plt.figure(figsize=(W, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    r, v, T = 6.0, 2.0, 30.0
    ts = np.array([0.0, 10.0, 20.0, 30.0])
    n = 80
    line = np.column_stack([v * np.linspace(0, T, n), np.zeros(n), np.linspace(0, T, n)])
    th = np.linspace(0, 2 * np.pi, 80)
    # the floor: what you see in SPACE -- one sphere, at four instants
    for t in ts:
        ax.plot(v * t + r * np.cos(th), r * np.sin(th), np.zeros_like(th),
                color="#9aa3ab", lw=1.4, zorder=1)
        ax.plot([v * t, v * t], [0, 0], [0, t], color=SLATE, lw=0.7, ls=":", alpha=0.55, zorder=1)
    ax.quiver(v * ts[0], 0, 0, v * T * 1.02, 0, 0, color=SLATE, lw=1.2, alpha=0.8,
              arrow_length_ratio=0.07, zorder=2)
    # the lift: one static tube
    X, Y, Z = tube_surface(line, r, n_theta=44, ref=(0.0, 1.0, 0.0))
    ax.plot_surface(X, Y, Z, color=TUBE, alpha=0.62, linewidth=0, shade=True, zorder=3)
    for t in ts:                              # the same spheres, now inside the tube
        xs, ys, zs = sphere(np.array([v * t, 0.0, t]), r)
        ax.plot_surface(xs, ys, zs, color="#8f99a3", alpha=0.55, linewidth=0, shade=True, zorder=4)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color=SLATE, lw=1.2, alpha=0.85, zorder=5)
    ax.view_init(elev=20, azim=-64)
    ax.set_xlim(-8, 72), ax.set_ylim(-14, 14), ax.set_zlim(0, 34)
    ax.set_box_aspect((80, 28, 36), zoom=1.16)
    style_3d(ax)
    ax.set_xticks([0, 30, 60]), ax.set_yticks([-10, 10]), ax.set_zticks([0, 15, 30])
    ax.set_xlabel("x [m]", labelpad=6), ax.set_ylabel("y [m]", labelpad=6), ax.set_zlabel("t [s]", labelpad=2)
    # notes in FIGURE coordinates: the axes is oversized, so ax.text2D would clip
    ax.set_position([-0.02, 0.015, 1.02, 0.955])
    fig.text(0.50, 0.975, "lifted KOZ: one static tube", ha="center", va="top",
             fontsize=13.5, color=SLATE)
    fig.text(0.985, 0.975, "1 s = 1 m", ha="right", va="top", fontsize=13.5, color=SLATE)
    fig.text(0.015, 0.87, "the same sphere in space,\nseen at four instants",
             ha="left", va="top", fontsize=12, color="#6b7783")
    return fig


def fig_scene(plt, sc, side):
    """(b) alone, at column width: the loiter scenario in the lifted space. The
    corridor is a slab through time, the shadow spot's tube bends across it, and
    the two returned runs are threaded inside the corridor."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(W, 3.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    t, xs, ys, spot_r = loiter_shadow_track(sc)
    t_max = 52.0
    keep = t <= t_max
    helix = np.column_stack([xs[keep], ys[keep], t[keep]])
    (xlo, xhi), (ylo, yhi), _ = sc["coord_bounds"]
    # the corridor: a slab that runs through the whole horizon, not a floor band
    for y in (ylo, yhi):
        wall = Poly3DCollection([[(xlo, y, 0.0), (xhi, y, 0.0), (xhi, y, t_max), (xlo, y, t_max)]],
                                facecolor=CTRL, alpha=0.10, edgecolor=CTRL, lw=0.5)
        wall.set_zorder(1); ax.add_collection3d(wall)
    floor = Poly3DCollection([[(xlo, ylo, 0.0), (xhi, ylo, 0.0), (xhi, yhi, 0.0), (xlo, yhi, 0.0)]],
                             facecolor=CTRL, alpha=0.16, edgecolor="none")
    floor.set_zorder(1); ax.add_collection3d(floor)
    ax.plot(xs, ys, np.zeros_like(xs), color=SLATE, lw=0.7, ls="--", alpha=0.45, zorder=1)
    X, Y, Z = tube_surface(helix, spot_r, n_theta=44)
    ax.plot_surface(X, Y, Z, color=TUBE, alpha=0.60, linewidth=0, shade=True, zorder=3)
    cp = side["control_points"]
    cb = bezier(np.asarray(cp["baseline"])[:, [0, 1, 3]], 300)
    cc = bezier(np.asarray(cp["constrained"])[:, [0, 1, 3]], 300)
    ax.plot(cb[:, 0], cb[:, 1], cb[:, 2], color=DARK, lw=1.8, ls=(0, (5, 3)), zorder=6)
    ax.plot(cc[:, 0], cc[:, 1], cc[:, 2], color=BLUE, lw=2.6, zorder=7)
    lo, hi = side["baseline"]["los_loss_interval"]
    lost = (cb[:, 2] >= lo) & (cb[:, 2] <= hi)
    ax.plot(cb[lost, 0], cb[lost, 1], cb[lost, 2], color=RED_PAPER, lw=5.0, zorder=8)
    ax.view_init(elev=17, azim=112)
    ax.set_xlim(100, -100), ax.set_ylim(-45, 60), ax.set_zlim(0, t_max)
    ax.set_box_aspect((200, 105, t_max * 1.35), zoom=1.12)
    style_3d(ax)
    ax.set_xticks([-100, 0, 100]), ax.set_yticks([-30, 30]), ax.set_zticks([0, 20, 40])
    ax.set_xlabel("x [m]", labelpad=-2), ax.set_ylabel("y [m]", labelpad=-4), ax.set_zlabel("t [s]", labelpad=-4)
    i_top = int(np.argmax(helix[:, 2]))
    label3(ax, helix[i_top], "shadow spot's tube: bent", 0, 16, SLATE, ha="center", size=12)
    label3(ax, (-60, ylo, t_max * 0.5), "corridor", -8, 4, CTRL, ha="right", size=12)
    i_b = int(np.argmin(np.abs(cb[:, 0] - 55.0)))
    i_c = int(np.argmin(np.abs(cc[:, 0] - 20.0)))
    label3(ax, cb[i_b], f"baseline, {side['baseline']['arrival_time']:.1f} s", 0, -16, DARK, ha="center", size=12)
    label3(ax, cc[i_c], f"proposed, {side['constrained']['arrival_time']:.1f} s", 0, 15, BLUE, ha="center", size=12)
    label3(ax, cb[lost].mean(axis=0), f"link lost, {lo:.2f}–{hi:.2f} s", 12, -20, RED_PAPER, ha="left", size=12)
    ax.set_position([-0.04, -0.10, 1.06, 1.16])
    return fig


def fig_runs(plt, sc, side):
    fig, ax = plt.subplots(figsize=(W, 3.7))
    t, xs, ys, spot_r = loiter_shadow_track(sc)
    y_axis = float(sc["start"][1])
    gap = spot_r ** 2 - (ys - y_axis) ** 2
    ok = gap > 0
    w = np.sqrt(np.where(ok, gap, 0.0))
    ax.fill_betweenx(t, xs - w, xs + w, where=ok, fc=SHADOW, ec="none", alpha=0.8, zorder=1)
    ax.plot(xs[ok], t[ok], color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=2)
    cp = side["control_points"]
    cb = bezier(np.asarray(cp["baseline"])[:, [0, 3]], 400)
    cc = bezier(np.asarray(cp["constrained"])[:, [0, 3]], 400)
    ax.plot(cb[:, 0], cb[:, 1], color=DARK, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.plot(cc[:, 0], cc[:, 1], color=BLUE, lw=1.8, zorder=4)
    x_start, x_end = float(sc["start"][0]), float(sc["end"][0])
    base_arr, con_arr = side["baseline"]["arrival_time"], side["constrained"]["arrival_time"]
    ax.text(x_end + 3, cb[-1, 1], f"baseline, {base_arr:.1f} s", fontsize=11, color=DARK, ha="left", va="center")
    ax.text(x_end + 3, cc[-1, 1], f"proposed, {con_arr:.1f} s", fontsize=11, color=BLUE, ha="left", va="center")
    lo, hi = side["baseline"]["los_loss_interval"]
    lost = (cb[:, 1] >= lo) & (cb[:, 1] <= hi)
    ax.plot(cb[lost, 0], cb[lost, 1], color=RED_PAPER, lw=4.0, solid_capstyle="butt", zorder=5)
    ax.text(float(cb[lost, 0].mean()) + 14, float(cb[lost, 1].mean()) + 3.5,
            f"link lost, {lo:.1f}–{hi:.1f} s", fontsize=11, color=RED_PAPER, ha="left", va="center")
    bound = side["axis_arrival_bound"]
    ax.plot([x_end - 34, x_end], [bound, bound], color=GREY, lw=0.8, ls=":", zorder=2)
    ax.text(x_end - 36, bound, f"axis bound, {bound:.1f} s", fontsize=10, color=GREY, ha="right", va="center")
    ax.text(float(xs[ok].mean()) + 16, float(t[ok].mean()) - 4, "shadow on the\ncorridor axis", fontsize=11, color=DARK, ha="left", va="center")
    ax.set_xlim(x_start, x_end + 64), ax.set_ylim(0, con_arr + 6)
    ax.set_xlabel("x along corridor [m]", labelpad=2), ax.set_ylabel("t [s]", labelpad=2)
    ax.tick_params(colors=SLATE)
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.16, top=0.97)
    return fig


RED_PAPER = "#c04040"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", type=pathlib.Path, default=SIDECAR,
                    help="the measured run the data figures are drawn from")
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--only", nargs="*", default=None,
                    help="draw only these figures (fig_lift fig_clip fig_halfspace fig_iterate fig_runs)")
    ap.add_argument("--png", action="store_true",
                    help="also write a 110 dpi PNG of each figure, for a quick look (not git-ignored)")
    args = ap.parse_args(argv)
    plt = setup()
    from spacetime_bezier.scenarios import scenario_loiter
    sc = scenario_loiter()
    side = json.loads(args.sidecar.read_text())
    if side["scenario"] != sc["name"]:
        sys.exit(f"sidecar scenario {side['scenario']!r} is not {sc['name']!r}; the poster figures would lie")
    makers = {"fig_lift": lambda: fig_lift(plt, sc, side), "fig_scene": lambda: fig_scene(plt, sc, side),
              "fig_clip": lambda: fig_clip(plt), "fig_halfspace": lambda: fig_halfspace(plt),
              "fig_iterate": lambda: fig_iterate(plt), "fig_runs": lambda: fig_runs(plt, sc, side)}
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in (args.only or list(makers)):
        fig = makers[name]()
        out = out_dir / f"{name}.pdf"
        fig.savefig(out)
        if args.png:
            fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        shown = out.relative_to(REPO) if out.is_relative_to(REPO) else out
        print(f"wrote {shown}")


if __name__ == "__main__":
    main()
