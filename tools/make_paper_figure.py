#!/usr/bin/env python3
"""The paper's occlusion figure: one figure, two panels, from two runs.

Left panel: the SCENE, in plan view, frozen at the instant the baseline's
sight line is most blocked -- station, the body on its lap, the shadow spot it
throws onto the corridor's altitude, the corridor, and where each run's vehicle
is at that instant with its sight line drawn. Everything in it comes from the
scenario parameters and the two returned curves; nothing from the solver's
rows. Right panel: line-of-sight margin against time for both runs, computed
by `compute_los_margin` -- the independent check, sampling the true geometry.
The two panels share the instant: it is marked on the time axis.

The scenario is a parameter (default: loiter, the paper's demo since
2026-08-26). Free arrival is exposed because it IS the demo's claim -- timing
as a decision variable -- and B10 requires it to be priced: --free-arrival
demands --time-weight > 0 and --v-max, or the solver refuses.

Honesty gates, in order:
  1. The constrained run must be FIGURE-GRADE (item B7). A run standing on
     slack, uncertified, unconverged, or penetrating aborts this script --
     no figure exists for it.
  2. The baseline (occlusion rows off) must actually LOSE line of sight for a
     measurable interval. A baseline that stays visible means the constraint
     was decoration and the figure proves nothing (PAPER_1: "the baseline that
     must be able to fail"); the script aborts rather than draw it.

Output: figures/paper1/occlusion_figure.pdf (vector, for the manuscript) and
.png (for quick looks), plus a .json sidecar recording every parameter and
measured number the figure rests on -- the figure is reproducible from the
sidecar alone.

Usage: python3 tools/make_paper_figure.py [--scenario loiter] [--N 8] [--seg 8]
       [--free-arrival --time-weight 1.0 --v-max 5.0] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np


def solve_pair(scenario: str, N: int, n_seg: int,
               free_arrival: bool, time_weight: float, v_max,
               sound_clip: bool = False, trust_radius=None):
    from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_elastic_weight
    from spacetime_bezier.optimize import optimize_spacetime

    fn, _ = SCENARIO_MAP[scenario]
    sc = fn()
    weight = scenario_elastic_weight(scenario)
    common = dict(
        N=N, dim=len(sc["start"]), p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=n_seg, max_iter=200, tol=1e-6,
        # The trust radius is a length, so it comes from the scene when the
        # scene declares one -- `loiter` is 200 m across and the 0.5 default
        # leaves its constrained run uncertifiable.
        # The clip floor is E + delta*sqrt(d+1): delta is what the paper pays
        # in arrival time. An explicit value wins over the scene's (measured
        # 2026-08-30: the scene's 5.0 costs ~9 s of delay at 8 segments).
        scp_trust_radius=float(trust_radius if trust_radius is not None
                               else sc.get("trust_radius", 0.5)),
        min_dt=0.1, elastic_weight=weight,
        # The scenario's own workspace band; optimize_spacetime takes explicit
        # arguments, so the key must be forwarded by hand here.
        coord_bounds=sc.get("coord_bounds"),
        free_arrival_time=free_arrival, time_weight=time_weight, v_max=v_max,
        # PAPER_1 statement (8): floor the clip ball at the segment radius plus
        # the trust-box reach, so every wall covers the whole keep-out zone and
        # the certificate is sound BY CONSTRUCTION. Off, the certificate covers
        # only the clipped pieces wherever `koz_unsound_clips` is nonzero.
        sound_clip=sound_clip,
        verbose=False, init_curve=sc.get("init_curve"),
    )
    import time
    t0 = time.perf_counter()
    P_con, info_con = optimize_spacetime(stations=sc["stations"], **common)
    info_con["solve_seconds"] = time.perf_counter() - t0
    # The baseline differs by ONE thing: the occlusion rows. Same arrival
    # freedom, same band, same weights -- or the comparison is not a comparison.
    P_base, info_base = optimize_spacetime(stations=None, **common)
    return sc, weight, (P_con, info_con), (P_base, info_base)


def occluder_centres(obstacle: dict, times: "np.ndarray") -> "np.ndarray":
    """Spatial centre of an obstacle at each time, either authoring form.

    Canonical obstacles carry lifted control points with the window intrinsic;
    legacy ones carry pos0/vel. Both are the scenario's own parameters, so the
    drawing stays independent of any solver output.
    """
    from spacetime_bezier.geometry import obstacle_positions_at

    if "control_points" in obstacle:
        cps = np.asarray(obstacle["control_points"], float)
        return obstacle_positions_at(cps, times)
    return (np.asarray(obstacle["pos0"], float)[None, :]
            + np.asarray(obstacle["vel"], float)[None, :] * times[:, None])


def obstacle_window(obstacle: dict, t_lo: float, t_hi: float) -> tuple:
    if "control_points" in obstacle:
        cps = np.asarray(obstacle["control_points"], float)
        return max(float(cps[0, -1]), t_lo), min(float(cps[-1, -1]), t_hi)
    return (max(float(obstacle.get("t_start", 0.0)), t_lo),
            min(float(obstacle.get("t_end", t_hi)), t_hi))


def axis_arrival_bound(sc: dict, station, v_max: float, dim: int,
                       dx: float = 0.25, dt: float = 0.05) -> float:
    """Earliest arrival of a point that moves along the corridor AXIS at most
    v_max and is never in the blocked set -- forward reachability on an (x, t)
    grid, the blocked set from `los_margin_at` (the true sight geometry, not
    the solver's rows). This is the number the returned arrival is compared
    with: the gap above it is what the formulation's conservatism costs, and
    it is measured on the same scene the figure draws. NaN if v_max is None
    or the end is never reached inside the scene's horizon.
    """
    from spacetime_bezier.geometry import los_margin_at
    if v_max is None:
        return float("nan")
    start_sp = np.asarray(sc["start"], float)[:dim - 1]
    end_sp = np.asarray(sc["end"], float)[:dim - 1]
    length = float(np.linalg.norm(end_sp - start_sp))
    n_x = int(round(length / dx)) + 1
    line = start_sp[None, :] + np.linspace(0.0, 1.0, n_x)[:, None] * (end_sp - start_sp)[None, :]
    cell = length / (n_x - 1)
    step = int(np.ceil(v_max * dt / cell))
    reach = np.zeros(n_x, dtype=bool)
    reach[0] = True
    n_t = int(float(sc["T"]) / dt) + 1
    for k in range(1, n_t):
        t = k * dt
        free = ~(los_margin_at(line, np.full(n_x, t), station, sc["obstacles"]) < 0.0)
        grown = reach.copy()
        for s in range(1, step + 1):
            grown[s:] |= reach[:-s]
        reach = grown & free
        if not reach.any():
            return float("nan")
        if reach[-1]:
            return float(t)
    return float("nan")


def figure_grade_or_die(info: dict, clearance: float, label: str):
    """Gate 1: the item-B7 predicate itself, never a reimplementation.

    An earlier version of this function rewrote the conditions inline with
    `if x > tol`, which INVERTS the gate's NaN polarity: the real gate is
    written `if not (x <= tol)` precisely so that a missing or NaN input
    (stale extension, no accepted step) FAILS. The rewrite drew in exactly
    those cases. Adversarial review 2026-08-20, finding 1. So: map the info
    keys to row keys with NaN defaults -- absent evidence must refuse -- and
    call `figure_grade_failures`, which also inherits any condition the gate
    grows later, provided its row key is forwarded here.
    """
    from spacetime_bezier.optimize import figure_grade_failures

    row = {
        "converged": bool(info.get("converged", 0.0)),
        "stop_label": str(info.get("stop_label", "unknown")),
        "certificate_violation": float(info.get("koz_violation_reference", np.nan)),
        # This scenario always has a station, so a missing occlusion key means
        # a stale extension, not "no occlusion" -- default NaN, which fails.
        "occlusion_violation": float(info.get("occlusion_violation_reference", np.nan)),
        "total_slack": float(info.get("total_koz_slack_returned", np.nan)),
        "min_clearance": float(clearance),
    }
    for passthrough in ("speed_cap_violation", "occlusion_planes_dropped"):
        if passthrough in info:
            row[passthrough] = float(info[passthrough])
    reasons = figure_grade_failures(row)
    if reasons:
        sys.exit(f"REFUSING to draw: {label} is not figure-grade: {'; '.join(reasons)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="loiter")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--free-arrival", action="store_true")
    ap.add_argument("--time-weight", type=float, default=0.0)
    ap.add_argument("--v-max", type=float, default=None)
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO / "figures" / "paper1")
    ap.add_argument("--sound-clip", action="store_true",
                    help="floor the clip radius at the reach (PAPER_1 statement 8)")
    ap.add_argument("--trust-radius", type=float, default=None,
                    help="override the scene's trust radius (a length, in metres)")
    args = ap.parse_args()

    from spacetime_bezier.geometry import (
        bezier_curve, compute_min_clearance, compute_los_margin, los_margin_at)

    sc, weight, (P_con, info_con), (P_base, info_base) = solve_pair(
        args.scenario, args.N, args.seg, args.free_arrival, args.time_weight, args.v_max,
        sound_clip=args.sound_clip, trust_radius=args.trust_radius)
    station = sc["stations"][0]
    dim = len(sc["start"])

    clear_con = float(compute_min_clearance(P_con, sc["obstacles"], dim=dim, n_eval=20001))
    figure_grade_or_die(info_con, clear_con, "constrained run")

    # Gate 1b: the certificate must speak for the WHOLE keep-out zone. This is
    # PAPER_1 statement (7) counted at the returned iterate: a nonzero count
    # means some wall was built against a clipped piece the next iterate could
    # leave, and the figure would rest on a certificate about less than the
    # obstacle. Measured on `loiter` 2026-08-30: 24 such pairs at the default,
    # 0 with --sound-clip, both runs "figure-grade" by gate 1 alone. NaN (an
    # extension predating the key) refuses too -- absent evidence is not zero.
    unsound = float(info_con.get("koz_unsound_clips", np.nan))
    if not unsound <= 0.0:
        sys.exit(f"REFUSING to draw: {unsound:.0f} (segment, obstacle) pair(s) have an "
                 "unsound clip, so the certificate covers only the clipped pieces; "
                 "pass --sound-clip")

    t_con, m_con = compute_los_margin(P_con, station, sc["obstacles"], dim=dim, n_eval=2001)
    t_base, m_base = compute_los_margin(P_base, station, sc["obstacles"], dim=dim, n_eval=2001)

    bound = axis_arrival_bound(sc, station, args.v_max, dim)

    # Gate 2: the baseline must fail, measurably.
    if float(np.min(m_base)) >= 0.0:
        sys.exit("REFUSING to draw: the baseline keeps line of sight everywhere -- "
                 "the occlusion rows were decoration in this configuration.")
    lost = t_base[m_base < 0.0]
    loss_interval = (float(lost.min()), float(lost.max()))

    pts_con = bezier_curve(np.asarray(P_con, float), num_pts=600)
    pts_base = bezier_curve(np.asarray(P_base, float), num_pts=600)

    # The schedule GRAFT, the measurement behind "timing is the decision
    # variable": the constrained run's spatial path flown on the baseline's
    # schedule (same curve parameter, baseline's time coordinate). If line of
    # sight survives that, the retiming was decoration and the path did the
    # work. Lateral deviation says how far the path itself moved off the
    # corridor axis. Both through `los_margin_at`/plain arithmetic on the two
    # solved curves -- no solver output beyond the control points.
    fine_con = bezier_curve(np.asarray(P_con, float), num_pts=4001)
    fine_base = bezier_curve(np.asarray(P_base, float), num_pts=4001)
    graft_min_los = float(np.min(los_margin_at(
        fine_con[:, :dim - 1], fine_base[:, -1], station, sc["obstacles"])))
    axis_y = float(sc["start"][1])
    lateral_dev_con = float(np.max(np.abs(fine_con[:, 1] - axis_y)))
    occluder_top = max(
        float(np.max(np.asarray(o["control_points"], float)[:, 2])) + float(o["radius"])
        if "control_points" in o else float(o["pos0"][2]) + float(o["r"])
        for o in sc["obstacles"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Drawn AT the width it is printed: one KSAS column is 3.15 in, and a
    # figure authored at 9 in and shrunk to fit had 2.5 pt legends on paper.
    # No legends at all at this size -- every element is labelled where it is.
    plt.rcParams.update({"font.size": 6, "axes.labelsize": 6,
                         "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                         "axes.linewidth": 0.5, "lines.linewidth": 1.0,
                         "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2})
    from matplotlib.patches import Circle
    fig = plt.figure(figsize=(3.4, 1.55))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    GREY, BLUE, RED, GREEN = "#8a8a8a", "#2255bb", "#c04040", "#117733"
    LAB = dict(fontsize=5, ha="left", va="center")

    # The instant the panels share: where the baseline's sight line is most
    # blocked. A scene frozen at any other time shows nothing being avoided.
    i_star = int(np.argmin(m_base))
    t_star = float(t_base[i_star])

    def at_time(pts, t):
        """Spatial position of a returned curve at time t (time is monotone)."""
        return np.array([np.interp(t, pts[:, -1], pts[:, k]) for k in range(dim - 1)])

    # The body at t*: the piece of the chain whose window holds t*.
    body_c, body_r = None, None
    for o in sc["obstacles"]:
        t0, t1 = obstacle_window(o, -np.inf, np.inf)
        if t0 - 1e-9 <= t_star <= t1 + 1e-9:
            body_c = occluder_centres(o, np.array([t_star]))[0]
            body_r = float(o.get("radius", o.get("r", 0.0)))
            o_act = o
            break
    if body_c is None:
        sys.exit("no obstacle piece is active at the blocking instant")
    st = np.asarray(station, float)
    z_c = float(sc["start"][2])
    # The shadow spot on the corridor's altitude: the body's disc projected
    # from the station as a point source. Similar triangles, nothing else.
    k = (z_c - st[2]) / (body_c[2] - st[2])
    spot_c = st[:2] + k * (body_c[:2] - st[:2])
    spot_r = k * body_r
    v_base, v_con = at_time(pts_base, t_star), at_time(pts_con, t_star)

    # LEFT: the scene in three dimensions at t*. The lap and its shadow ring
    # are traced from the pieces' own control points over their windows, not
    # assumed circular; the body's direction of travel is the finite difference
    # of the active piece; the sight lines are the station-to-vehicle segments.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    ax1.remove()
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.computed_zorder = False   # painter's order below is the draw order
    lap = np.vstack([occluder_centres(o, np.linspace(*obstacle_window(o, -np.inf, np.inf), 60))
                     for o in sc["obstacles"]])
    ring = st[:2] + k * (lap[:, :2] - st[:2])
    bounds = sc.get("coord_bounds")
    ylo, yhi = (float(bounds[1][0]), float(bounds[1][1])) if bounds is not None else (axis_y - 5, axis_y + 5)
    span = max(np.abs(ring).max(), np.abs(lap[:, :2]).max()) + 8
    x_lo, x_hi = min(-span, v_con[0] - 10), max(span, spot_c[0] + spot_r) + 10
    y_lo, y_hi = -span, max(yhi + 8, span)
    # corridor band and the shared spatial path, at the corridor altitude
    band = Poly3DCollection(
        [[(x_lo, ylo, z_c), (x_hi, ylo, z_c), (x_hi, yhi, z_c), (x_lo, yhi, z_c)]],
        facecolor="#cfcfcf", alpha=0.45, edgecolor="#a0a0a0", lw=0.3)
    band.set_zorder(1)
    ax1.add_collection3d(band)
    keep = (pts_con[:, 0] >= x_lo) & (pts_con[:, 0] <= x_hi)
    ax1.plot(pts_con[keep, 0], pts_con[keep, 1], pts_con[keep, 2], color=BLUE, lw=0.7,
             alpha=0.6, zorder=2)
    # the lap at its altitude, with three arrows along it for the direction of travel
    ax1.plot(lap[:, 0], lap[:, 1], lap[:, 2], color=RED, lw=0.5, ls="--", alpha=0.7, zorder=2)
    for frac in (0.15, 0.48, 0.81):
        i = int(frac * (len(lap) - 1))
        tng = lap[min(i + 1, len(lap) - 1)] - lap[max(i - 1, 0)]
        tng = tng / (np.linalg.norm(tng) + 1e-12) * 9.0
        ax1.quiver(lap[i, 0], lap[i, 1], lap[i, 2], tng[0], tng[1], tng[2],
                   color=RED, lw=0.7, alpha=0.85, arrow_length_ratio=0.9, zorder=3)
    # the shadow CONE: apex at the station, through the body, extended past the
    # corridor altitude so it reads as a cone and not a disc. Radius grows in
    # proportion to the distance from the station (the body radius at the body).
    u = (body_c - st) / np.linalg.norm(body_c - st)
    e1 = np.cross(u, [0.0, 0.0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    s_body = float(np.linalg.norm(body_c - st))
    s_top = (z_c + 24.0 - st[2]) / u[2]
    ss = np.linspace(s_body, s_top, 12)
    S_, TH = np.meshgrid(ss, np.linspace(0.0, 2.0 * np.pi, 40))
    rho = body_r * S_ / s_body
    cone = (st[None, None, :] + S_[..., None] * u[None, None, :]
            + rho[..., None] * (np.cos(TH)[..., None] * e1[None, None, :]
                                + np.sin(TH)[..., None] * e2[None, None, :]))
    ax1.plot_surface(cone[..., 0], cone[..., 1], cone[..., 2], color=RED, alpha=0.18,
                     linewidth=0, antialiased=True, shade=False, zorder=3)
    for a in (0.0, np.pi):   # two generatrices so the cone has an outline
        gen = st[None, :] + ss[:, None] * u[None, :] + (body_r * ss / s_body)[:, None] * (
            np.cos(a) * e1 + np.sin(a) * e2)[None, :]
        ax1.plot(gen[:, 0], gen[:, 1], gen[:, 2], color=RED, lw=0.4, alpha=0.6, zorder=3)
    cone_top = st + s_top * u
    # the body at t*
    ax1.scatter([body_c[0]], [body_c[1]], [body_c[2]], s=36, facecolor=RED, alpha=0.6,
                edgecolor=RED, lw=0.6, depthshade=False, zorder=4)
    # the shadow spot where the cone crosses the corridor altitude
    th = np.linspace(0.0, 2.0 * np.pi, 48)
    spot = Poly3DCollection([[(spot_c[0] + spot_r * np.cos(a), spot_c[1] + spot_r * np.sin(a), z_c)
                              for a in th]], facecolor=RED, alpha=0.35, edgecolor=RED, lw=0.4)
    spot.set_zorder(4)
    ax1.add_collection3d(spot)
    # station on the ground; sight lines to both vehicles at t*, faint so the
    # scene stays readable: the blocked one dashed, with a cross where it meets
    # the body, the kept one solid. Arrowheads are short quivers at the tips.
    SIGHT = "#6aa84f"
    ax1.scatter([st[0]], [st[1]], [st[2]], marker="^", s=28, color=GREEN, depthshade=False, zorder=6)
    for v_pos, ls in ((v_base, "--"), (v_con, "-")):
        d = v_pos - st
        ax1.plot([st[0], v_pos[0]], [st[1], v_pos[1]], [st[2], v_pos[2]],
                 color=SIGHT, lw=0.6, ls=ls, alpha=0.45, zorder=5)
        tip = st + 0.86 * d
        ax1.quiver(tip[0], tip[1], tip[2], 0.14 * d[0], 0.14 * d[1], 0.14 * d[2],
                   color=SIGHT, lw=0.6, alpha=0.45, arrow_length_ratio=0.6, zorder=5)
    # where the blocked line meets the body: the segment point nearest the centre
    d = v_base - st
    s_hit = float(np.clip(np.dot(body_c - st, d) / np.dot(d, d), 0.0, 1.0))
    hit = st + s_hit * d
    ax1.scatter([hit[0]], [hit[1]], [hit[2]], marker="x", s=30, color="#5a0000", lw=1.1,
                depthshade=False, zorder=6)
    ax1.scatter([v_base[0]], [v_base[1]], [v_base[2]], s=14, facecolor="white",
                edgecolor=GREY, lw=0.8, depthshade=False, zorder=7)
    ax1.scatter([v_con[0]], [v_con[1]], [v_con[2]], s=14, color=BLUE, depthshade=False, zorder=7)

    # view: from the side the vehicle comes from, slightly above the corridor
    ax1.view_init(elev=24, azim=-58)
    ax1.set_xlim(x_lo, x_hi), ax1.set_ylim(y_lo, y_hi), ax1.set_zlim(0.0, z_c + 10)
    ax1.set_box_aspect((x_hi - x_lo, y_hi - y_lo, 0.9 * (z_c + 10)), zoom=0.92)
    for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#dddddd")
        axis._axinfo["grid"]["color"] = "#eeeeee"
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax1.set_xticks([-50, 0, 50]), ax1.set_yticks([-25, 25]), ax1.set_zticks([0, 50])
    ax1.tick_params(labelsize=5, pad=-2)
    ax1.set_xlabel("x [m]", labelpad=-6), ax1.set_ylabel("y [m]", labelpad=-6)
    ax1.set_zlabel("z [m]", labelpad=-11)

    # Labels placed in SCREEN space: project each anchor with the view above,
    # then offset in points, so they cannot land on each other or on the data.
    from mpl_toolkits.mplot3d import proj3d
    def label(p, text, dx, dy, colour, ha="left"):
        x2, y2, _ = proj3d.proj_transform(p[0], p[1], p[2], ax1.get_proj())
        ax1.annotate(text, (x2, y2), xytext=(dx, dy), textcoords="offset points",
                     fontsize=5, color=colour, ha=ha, va="center")
    label(st, "station", 5, -5, GREEN)
    label(body_c, "body", 7, -3, RED)
    label(cone_top, "shadow", 6, 3, RED)
    label(v_base, "off", -4, 8, GREY, ha="right")
    label(v_con, "on", -3, 9, BLUE, ha="right")
    label(st + 0.42 * (v_con - st), "sight line", -6, -3, SIGHT, ha="right")
    label(np.array([x_hi, yhi, z_c]), "corridor", -2, 8, "#555555", ha="right")
    ax1.text2D(0.02, 0.96, f"t = {t_star:.1f} s", transform=ax1.transAxes,
               fontsize=5.5, ha="left", va="top")

    # RIGHT: the proof panel. Labels at the curve ends, no legend.
    ax2.axhline(0.0, color="#666666", lw=0.5)
    ax2.axvspan(*loss_interval, color=RED, alpha=0.12, lw=0)
    ax2.axvline(t_star, color="#444444", lw=0.5, ls=":")
    ax2.plot(t_base, m_base, color=GREY, ls="--", lw=1.0)
    ax2.plot(t_con, m_con, color=BLUE, lw=1.2)
    top = float(max(np.max(m_base[np.isfinite(m_base)]), np.max(m_con[np.isfinite(m_con)])))
    ax2.set_ylim(top=top + 9)
    ax2.text(float(t_base[-1]), float(m_base[-1]) + 2, f"off, {t_base[-1]:.1f} s",
             color=GREY, fontsize=5, ha="right", va="bottom")
    ax2.text(float(t_con[-1]) - 0.5, float(m_con[-1]) - 3, f"on, {t_con[-1]:.1f} s",
             color=BLUE, fontsize=5, ha="right", va="top")
    ax2.set_xlabel("time [s]"), ax2.set_ylabel("sight margin [m]")
    ax2.set_xlim(0.0, float(max(t_base[-1], t_con[-1])) * 1.02)

    fig.subplots_adjust(left=0.0, right=0.985, bottom=0.2, top=0.97, wspace=0.42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = args.out_dir / "occlusion_figure.pdf", args.out_dir / "occlusion_figure.png"
    fig.savefig(pdf), fig.savefig(png, dpi=600)

    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, cwd=REPO).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"],
                      capture_output=True, text=True, cwd=REPO).stdout.strip():
        git += "+dirty"
    sidecar = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git, "scenario": args.scenario,
        "N": args.N, "n_seg": args.seg, "elastic_weight": weight,
        "free_arrival": bool(args.free_arrival),
        "time_weight": float(args.time_weight),
        "v_max": None if args.v_max is None else float(args.v_max),
        # The RESOLVED radius the solve used, not the scene's key: the override
        # is what the paper run is drawn with.
        "trust_radius": float(args.trust_radius if args.trust_radius is not None
                              else sc.get("trust_radius", 0.5)),
        # The clip-soundness record, both halves: whether the reach floor was
        # requested, and how many (segment, obstacle) pairs at the returned
        # iterate fell short of it anyway. A certificate is a statement about
        # what it was built against; with `unsound_clips` 0 that is the whole
        # keep-out zone, otherwise only the clipped pieces. NaN if the
        # extension predates the key, which fails any reader that compares it.
        "sound_clip": bool(args.sound_clip),
        "constrained": {
            "min_clearance": clear_con,
            "min_los_margin": float(np.min(m_con)),
            "max_z": float(np.max(pts_con[:, 2])),
            "koz_certificate": float(info_con.get("koz_violation_reference", np.nan)),
            "occlusion_certificate": float(info_con.get("occlusion_violation_reference", np.nan)),
            "iterations": int(info_con.get("iterations", -1)),
            "arrival_time": float(info_con.get("arrival_time", np.nan)),
            "unsound_clips": float(info_con.get("koz_unsound_clips", np.nan)),
            "lateral_deviation": lateral_dev_con,
            "graft_min_los_margin": graft_min_los,
            # Wall-clock of the constrained solve alone, on `machine` below. A
            # cost number is only honest with the hardware beside it.
            "solve_seconds": float(info_con.get("solve_seconds", np.nan)),
            # Against the true-shadow, axis-only reachability bound below:
            # what the formulation's conservatism costs on this scene.
            "arrival_gap_to_axis_bound": float(info_con.get("arrival_time", np.nan)) - bound,
        },
        "axis_arrival_bound": bound,
        "snapshot_time": t_star,
        "baseline": {
            "min_los_margin": float(np.min(m_base)),
            "los_loss_interval": loss_interval,
            "max_z": float(np.max(pts_base[:, 2])),
            "arrival_time": float(info_base.get("arrival_time", np.nan)),
        },
        "occluder_top": occluder_top, "station": list(map(float, station)),
        "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or None,
    }
    (args.out_dir / "occlusion_figure.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {pdf}\nwrote {png}")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
