#!/usr/bin/env python3
"""Concept figure for the KSAS paper: the mission scene and its space-time lift.

A schematic, and captioned as one. Every length in the left panel comes from
`scenario_loiter()` -- station, orbit, body radius, corridor band -- and the
shadow band in the right panel is the body's true umbra on the corridor axis,
computed from the same numbers. The two vehicle curves and the clip ball on
the right are DRAWN to illustrate the construction, not solved: nothing here
is a solver result. `make_paper_figure.py` owns those.

    python3 tools/make_concept_figure.py            # figures/paper1/concept_figure.{png,pdf}
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AZ, EL = np.deg2rad(32.0), np.deg2rad(24.0)


def proj(pts):
    """Orthographic view: azimuth AZ about z, elevation EL above the ground."""
    pts = np.atleast_2d(np.asarray(pts, float))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xr = x * np.cos(AZ) - y * np.sin(AZ)
    yr = x * np.sin(AZ) + y * np.cos(AZ)
    return np.column_stack([xr, yr * np.sin(EL) + z * np.cos(EL)])


def hull(points):
    """Monotone-chain convex hull of 2-D points, counter-clockwise."""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) < 3:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def ring(centre, radius, axis, n=90):
    """Circle of `radius` about `centre`, in the plane normal to `axis` (3-D)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    b = np.cross(a, [1.0, 0.0, 0.0])
    if np.linalg.norm(b) < 1e-9:
        b = np.cross(a, [0.0, 1.0, 0.0])
    b /= np.linalg.norm(b)
    c = np.cross(a, b)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.asarray(centre, float) + radius * (np.outer(np.cos(th), b) + np.outer(np.sin(th), c))


def bezier(cp, n=400):
    cp = np.asarray(cp, float)
    s = np.linspace(0.0, 1.0, n)[:, None]
    p = np.zeros((n, cp.shape[1]))
    deg = len(cp) - 1
    from math import comb
    for i, q in enumerate(cp):
        p += comb(deg, i) * (1 - s) ** (deg - i) * s ** i * q
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO / "figures" / "paper1")
    ap.add_argument("--panel-b", choices=("solver", "schematic", "none"), default="solver",
                    help="right panel: the two RETURNED curves from the sidecar in (x, t) "
                         "(default), the hand-drawn construction sketch, or no right panel")
    args = ap.parse_args()

    from spacetime_bezier.scenarios import scenario_loiter
    sc = scenario_loiter()
    station = np.asarray(sc["stations"][0], float)
    obs = sc["obstacles"][0]
    cps = np.asarray(obs["control_points"], float)
    body_r = float(obs["radius"])
    body_z = float(cps[0, 2])
    orbit_r = float(np.hypot(cps[0, 0], cps[0, 1]))
    phase = float(np.arctan2(cps[0, 1], cps[0, 0]))
    T = float(sc["T"])
    (_, _), (ylo, yhi), (zlo, zhi) = sc["coord_bounds"]
    y_axis, z_axis = float(sc["start"][1]), float(sc["start"][2])
    x_start, x_end = float(sc["start"][0]), float(sc["end"][0])
    scale = z_axis / body_z                      # shadow magnification at the corridor
    spot_r = body_r * scale

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    plt.rcParams.update({"font.size": 6, "axes.labelsize": 6,
                         "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                         "axes.linewidth": 0.5, "lines.linewidth": 0.8,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    GREY, BLUE, RED, DARK = "#8a8a8a", "#2255bb", "#c04040", "#404040"
    if args.panel_b == "none":
        fig = plt.figure(figsize=(3.15, 1.2))
        ax1, ax2 = fig.add_subplot(1, 1, 1), None
    else:
        fig = plt.figure(figsize=(3.15, 1.6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

    # ---- left: the scene, body at the instant its shadow sits on the corridor
    body = np.array([0.0, orbit_r, body_z])
    spot = np.array([0.0, y_axis, z_axis])
    x_lo, x_hi = -70.0, 70.0                    # the corridor drawn in part

    ground = proj([[-95, -55, 0], [95, -55, 0], [95, 75, 0], [-95, 75, 0]])
    ax1.add_patch(Polygon(ground, closed=True, fc="#f2f2f2", ec="none", zorder=0))

    # cone: apex at the station, tangent to the body, drawn to just above the corridor
    axis = body - station
    dist = np.linalg.norm(axis)
    alpha = np.arcsin(body_r / dist)
    a = axis / dist
    b = np.cross(a, [1.0, 0.0, 0.0]); b /= np.linalg.norm(b)
    c = np.cross(a, b)
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    gens = a * np.cos(alpha) + (np.outer(np.cos(th), b) + np.outer(np.sin(th), c)) * np.sin(alpha)
    z_top = zhi + 4.0
    top_ring = station + gens * (z_top / gens[:, 2])[:, None]
    cone = hull(np.vstack([proj(station), proj(top_ring)]))
    ax1.add_patch(Polygon(cone, closed=True, fc="#bbbbbb", ec=GREY, lw=0.4, alpha=0.45, zorder=1))

    # orbit and body
    orb = proj(ring([0, 0, body_z], orbit_r, [0, 0, 1], 200))
    ax1.plot(np.r_[orb[:, 0], orb[0, 0]], np.r_[orb[:, 1], orb[0, 1]], ls=(0, (2, 1.5)), color=GREY, lw=0.6, zorder=2)
    pb = proj(body)[0]
    ax1.add_patch(Circle(pb, body_r, fc=RED, ec="none", zorder=4))
    vel = np.array([-1.0, 0.0, 0.0]) * 14.0     # counter-clockwise at (0, r)
    ax1.annotate("", xy=proj(body + vel)[0], xytext=pb,
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.6, mutation_scale=5), zorder=4)

    # shadow spot on the corridor altitude
    sp = proj(ring(spot, spot_r, [0, 0, 1], 80))
    ax1.add_patch(Polygon(sp, closed=True, fc=DARK, ec="none", alpha=0.55, zorder=3))

    # corridor slab, drawn in part
    box = lambda x, y, z: proj([x, y, z])[0]
    top = np.array([box(x_lo, ylo, zhi), box(x_hi, ylo, zhi), box(x_hi, yhi, zhi), box(x_lo, yhi, zhi)])
    ax1.add_patch(Polygon(top, closed=True, fc=BLUE, ec="none", alpha=0.18, zorder=2))
    for (xa, ya, za), (xb, yb, zb) in [
        ((x_lo, ylo, zlo), (x_hi, ylo, zlo)), ((x_lo, ylo, zhi), (x_hi, ylo, zhi)),
        ((x_lo, yhi, zlo), (x_hi, yhi, zlo)), ((x_lo, yhi, zhi), (x_hi, yhi, zhi)),
        ((x_lo, ylo, zlo), (x_lo, ylo, zhi)), ((x_lo, yhi, zlo), (x_lo, yhi, zhi)),
        ((x_lo, ylo, zlo), (x_lo, yhi, zlo)), ((x_lo, ylo, zhi), (x_lo, yhi, zhi)),
        ((x_hi, ylo, zlo), (x_hi, ylo, zhi)), ((x_hi, ylo, zhi), (x_hi, yhi, zhi)),
    ]:
        pa, pb2 = box(xa, ya, za), box(xb, yb, zb)
        ax1.plot([pa[0], pb2[0]], [pa[1], pb2[1]], color=BLUE, lw=0.4, alpha=0.7, zorder=2)
    ax_line = proj([[x_lo, y_axis, z_axis], [x_hi, y_axis, z_axis]])
    ax1.plot(ax_line[:, 0], ax_line[:, 1], color=BLUE, lw=0.5, ls=(0, (3, 2)), zorder=3)

    # vehicle, its heading and the sight line to the station
    veh = np.array([-45.0, y_axis, z_axis])
    pv = proj(veh)[0]
    ax1.plot(*pv, marker="o", ms=2.5, color=BLUE, zorder=5)
    ax1.annotate("", xy=proj(veh + [16, 0, 0])[0], xytext=pv,
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.6, mutation_scale=5), zorder=5)
    ps = proj(station)[0]
    ax1.plot([ps[0], pv[0]], [ps[1], pv[1]], color=DARK, lw=0.5, ls=(0, (1, 1.5)), zorder=3)
    ax1.plot([ps[0], ps[0]], [ps[1], ps[1] + 5], color=DARK, lw=0.8, zorder=5)
    ax1.plot(*ps, marker="^", ms=3, color=DARK, zorder=5)

    L = dict(fontsize=5, color=DARK, zorder=6)
    ax1.text(ps[0] + 4, ps[1] - 4, "ground station", ha="left", va="top", **L)
    ax1.text(pb[0] + 9, pb[1] - 2, "loitering body", ha="left", va="center", color=RED, fontsize=5, zorder=6)
    spc = proj(spot)[0]
    ax1.text(spc[0] + 10, spc[1] - 1, "shadow", ha="left", va="center", **L)
    ax1.text(pv[0] - 3, pv[1] + 5, "vehicle", ha="right", va="bottom", color=BLUE, fontsize=5, zorder=6)
    ax1.text(top[1][0], top[1][1] - 9, "corridor", ha="right", va="top", color=BLUE, fontsize=5, zorder=6)
    mid = 0.5 * (ps + pv)
    ax1.text(mid[0] - 3, mid[1], "sight line", ha="right", va="center", fontsize=4.5, color=DARK, zorder=6)
    ax1.set_aspect("equal")
    ax1.set_xlim(-100, 62)
    ax1.set_ylim(-14, 96)
    ax1.axis("off")
    if ax2 is not None:
        ax1.set_title("(a) scene", fontsize=6, pad=1)

    # ---- right: the lift, x along the corridor against time
    if ax2 is None:
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        return save(fig, args.out_dir)
    t = np.linspace(0.0, T, 1601)
    theta = phase + 2 * np.pi * t / T
    xs, ys = orbit_r * np.cos(theta) * scale, orbit_r * np.sin(theta) * scale
    gap = spot_r ** 2 - (ys - y_axis) ** 2
    ok = gap > 0
    w = np.sqrt(np.where(ok, gap, 0.0))
    ax2.fill_betweenx(t, xs - w, xs + w, where=ok, fc="#bbbbbb", ec="none", alpha=0.7, zorder=1)
    ax2.plot(xs[ok], t[ok], color=GREY, lw=0.4, ls=(0, (2, 1.5)), zorder=2)

    if args.panel_b == "solver":
        # The two RETURNED control polygons of the measured run, read from the
        # figure sidecar -- a picture of the run the tables report, not a re-solve.
        import json
        side = json.loads((REPO / "figures" / "paper1" / "occlusion_figure.json").read_text())
        cp = side["control_points"]
        cb, cc = bezier(np.asarray(cp["baseline"])[:, [0, 3]]), bezier(np.asarray(cp["constrained"])[:, [0, 3]])
        ax2.plot(cb[:, 0], cb[:, 1], color=DARK, lw=0.7, ls=(0, (3, 2)), zorder=3)
        ax2.plot(cc[:, 0], cc[:, 1], color=BLUE, lw=0.9, zorder=4)
        i_mid = int(np.argmax(ok))
        ax2.annotate("", xy=(xs[ok][-1], t[ok][-1]), xytext=(xs[i_mid], t[i_mid]),
                     arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.5, mutation_scale=5), zorder=2)
        ax2.text(float(xs[ok].mean()) + 9, float(t[ok].mean()) - 1, "shadow", fontsize=5, color=DARK,
                 ha="left", va="top", zorder=6)
        ax2.text(x_end + 2, cb[-1, 1], f"{side['baseline']['arrival_time']:.1f} s", fontsize=5,
                 color=DARK, ha="left", va="center", zorder=6)
        ax2.text(x_end + 2, cc[-1, 1], f"{side['constrained']['arrival_time']:.1f} s", fontsize=5,
                 color=BLUE, ha="left", va="center", zorder=6)
        ax2.text(-96, 30, "proposed", fontsize=5, color=BLUE, ha="left", va="bottom", zorder=6)
        ax2.text(-96, 24, "baseline", fontsize=5, color=DARK, ha="left", va="top", zorder=6)
        ax2.set_xlim(x_start, x_end + 22)
        ax2.set_ylim(0, 50)
        ax2.set_xlabel("x along corridor [m]", labelpad=1)
        ax2.set_ylabel("t [s]", labelpad=1)
        ax2.set_title("(b) the two runs in (x, t)", fontsize=6, pad=1)
        fig.subplots_adjust(left=0.0, right=0.985, bottom=0.17, top=0.92, wspace=0.05)
        return save(fig, args.out_dir)

    v_max = 5.0
    ax2.plot([x_start, x_end], [0, (x_end - x_start) / v_max], color=DARK, lw=0.7, ls=(0, (3, 2)), zorder=3)
    prop = bezier([[x_start, 0.0], [-60.0, 12.0], [-30.0, 20.5], [x_end, 46.6]])
    ax2.plot(prop[:, 0], prop[:, 1], color=BLUE, lw=0.9, zorder=4)

    # the local construction, drawn around one point of the proposed curve
    i_c = int(np.argmin(np.abs(prop[:, 0] + 30.0)))
    c2 = prop[i_c]
    r_clip = 9.0
    cen = np.column_stack([xs, t])[ok]
    p_near = cen[np.argmin(np.linalg.norm(cen - c2, axis=1))]
    n = (c2 - p_near) / np.linalg.norm(c2 - p_near)
    gx, gt = np.meshgrid(np.linspace(-45, 0, 300), np.linspace(6, 32, 300))
    # (grid covers the ball wherever c lands; the mask does the clipping)
    inside_ball = (gx - c2[0]) ** 2 + (gt - c2[1]) ** 2 <= r_clip ** 2
    xs_g = np.interp(gt, t, xs); ys_g = np.interp(gt, t, ys)
    gap_g = spot_r ** 2 - (ys_g - y_axis) ** 2
    inside_blob = (gap_g > 0) & (np.abs(gx - xs_g) <= np.sqrt(np.clip(gap_g, 0, None)))
    Lmask = inside_ball & inside_blob
    b_sup = float(np.max(n[0] * gx[Lmask] + n[1] * gt[Lmask])) if Lmask.any() else float("nan")
    ax2.scatter(gx[Lmask][::7], gt[Lmask][::7], s=0.15, color=RED, zorder=5, linewidths=0)
    ax2.add_patch(Circle(c2, r_clip, fc="none", ec=RED, lw=0.5, ls=(0, (2, 1.5)), zorder=5))
    ax2.plot(*c2, marker="o", ms=2, color=RED, zorder=6)
    foot = c2 - (np.dot(n, c2) - b_sup) * n
    tdir = np.array([-n[1], n[0]])
    seg = np.vstack([foot - 1.4 * r_clip * tdir, foot + 1.4 * r_clip * tdir])
    ax2.plot(seg[:, 0], seg[:, 1], color=RED, lw=0.7, zorder=5)
    ax2.annotate("", xy=foot + 4.5 * n, xytext=foot,
                 arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.6, mutation_scale=5), zorder=6)

    ax2.text(c2[0] + 1.2, c2[1] + 0.6, "c", fontsize=5, color=RED, ha="left", va="bottom", zorder=6)
    ax2.text(foot[0] + 5.0 * n[0] + 1.0, foot[1] + 5.0 * n[1], "n", fontsize=5, color=RED, ha="left", va="center", zorder=6)
    ax2.text(c2[0] - r_clip - 0.5, c2[1] + r_clip * 0.5, "B(c, r)", fontsize=4.5, color=RED, ha="right", va="center", zorder=6)
    ax2.text(seg[1][0] + 0.5, seg[1][1], "n·Q ≥ b", fontsize=4.5, color=RED, ha="left", va="center", zorder=6)
    ax2.text(-9, 11.5, "shadow KOZ", fontsize=5, color=DARK, ha="left", va="center", zorder=6)
    ax2.text(-20.5, 15.0, "L", fontsize=5, color=RED, ha="left", va="top", zorder=6)
    ax2.text(-46, 10.0, "baseline", fontsize=5, color=DARK, ha="left", va="top", rotation=11, zorder=6)
    ax2.text(-46, 14.6, "proposed", fontsize=5, color=BLUE, ha="left", va="bottom", rotation=17, zorder=6)
    ax2.set_aspect("equal")               # 1 s is 1 m in the lifted space
    ax2.set_xlim(-48, 12)
    ax2.set_ylim(0, 46)
    ax2.set_xlabel("x along corridor [m]", labelpad=1)
    ax2.set_ylabel("t [s]", labelpad=1)
    ax2.set_title("(b) space-time lift", fontsize=6, pad=1)

    fig.subplots_adjust(left=0.0, right=0.985, bottom=0.17, top=0.92, wspace=0.05)
    return save(fig, args.out_dir)


def save(fig, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / "concept_figure.png", out_dir / "concept_figure.pdf"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    print(f"wrote {png} and {pdf}  (figure {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in)")


if __name__ == "__main__":
    main()
