#!/usr/bin/env python3
"""The paper's occlusion figure: one figure, two panels, from two runs.

Left panel: the scenario's trajectory in three spatial dimensions, with the
station and the occluder drawn from the same scenario parameters the solver
consumed (canonical control-point obstacles are sampled along their own lifted
centreline). Right panel: line-of-sight margin against time for both runs,
computed by `compute_los_margin` -- the independent check, sampling the true
geometry, never the solver's rows.

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
               sound_clip: bool = False):
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
        scp_trust_radius=float(sc.get("trust_radius", 0.5)), min_dt=0.1,
        elastic_weight=weight,
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
    args = ap.parse_args()

    from spacetime_bezier.geometry import (
        bezier_curve, compute_min_clearance, compute_los_margin, los_margin_at)

    sc, weight, (P_con, info_con), (P_base, info_base) = solve_pair(
        args.scenario, args.N, args.seg, args.free_arrival, args.time_weight, args.v_max,
        sound_clip=args.sound_clip)
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
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.2})
    fig = plt.figure(figsize=(3.4, 1.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # LEFT: the space-time panel, and it is the figure's whole argument. The
    # blocked set is not an obstacle in space that a path steers around -- it is
    # a REGION IN (position, time), and the constrained run leaves it along the
    # time axis while flying the same line through space. A spatial view cannot
    # show that: once the two runs share a path they plot on top of each other.
    #
    # The region is a property of the SCENE, not of either run: it is the set of
    # (station-to-vehicle) pairs on the corridor centreline whose sight line the
    # body blocks at that instant, evaluated on a grid neither solve visited,
    # through `los_margin_at` -- the same function that draws the panel beside
    # it, so the shading and the curves cannot disagree.
    start_sp = np.asarray(sc["start"], float)[:dim - 1]
    end_sp = np.asarray(sc["end"], float)[:dim - 1]
    t_top = max(float(pts_con[-1, -1]), float(pts_base[-1, -1]))
    us = np.linspace(0.0, 1.0, 241)
    ts = np.linspace(0.0, t_top, 241)
    UU, TT = np.meshgrid(us, ts)
    line = start_sp[None, :] + UU.reshape(-1, 1) * (end_sp - start_sp)[None, :]
    grid = los_margin_at(line, TT.reshape(-1), station, sc["obstacles"])
    grid = np.where(np.isfinite(grid), grid, np.nan).reshape(UU.shape)
    axis_x = start_sp[0] + us * (end_sp[0] - start_sp[0])
    ax1.contourf(axis_x, ts, grid, levels=[-1e18, 0.0], colors=["#e8b4b4"])
    ax1.contour(axis_x, ts, grid, levels=[0.0], colors=["#c04040"], linewidths=0.6)
    ax1.plot(pts_base[:, 0], pts_base[:, -1], color="#999999", ls="--", lw=1.0,
             label="occlusion off")
    ax1.plot(pts_con[:, 0], pts_con[:, -1], color="#2255bb", lw=1.4,
             label="occlusion on")
    ax1.set_xlabel("x [m]"), ax1.set_ylabel("time [s]")
    for pts, colour, dx in ((pts_base, "#777777", 3), (pts_con, "#2255bb", 3)):
        ax1.annotate(f"{pts[-1, -1]:.1f} s", (pts[-1, 0], pts[-1, -1]),
                     xytext=(-3, 3), textcoords="offset points", ha="right",
                     fontsize=5.5, color=colour)
    ax1.set_xlim(float(axis_x.min()), float(axis_x.max()))
    ax1.set_ylim(0.0, t_top)
    # The shaded set gets a legend entry of its own: a reader who takes it for a
    # spatial obstacle has read the panel backwards.
    from matplotlib.patches import Patch
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Patch(facecolor="#e8b4b4", edgecolor="#c04040", lw=0.6))
    labels.append("line of sight blocked (corridor axis)")
    ax1.legend(handles, labels, loc="upper left", fontsize=5, frameon=False,
               handlelength=1.6, borderaxespad=0.2)

    # Right: the proof panel.
    ax2.axhline(0.0, color="#666666", lw=0.6)
    ax2.plot(t_base, m_base, color="#999999", ls="--", lw=1.0, label="occlusion off")
    ax2.plot(t_con, m_con, color="#2255bb", lw=1.4, label="occlusion on")
    # The graft, drawn: the constrained run's path on the baseline's schedule.
    # This is the curve that has to dip below zero for the retiming to be the
    # thing that saved the link; if it stayed positive the path did the work.
    graft_margins = los_margin_at(fine_con[:, :dim - 1], fine_base[:, -1],
                                  station, sc["obstacles"])
    ax2.plot(fine_base[:, -1], graft_margins, color="#2255bb", lw=1.0, ls=":",
             label="constrained path, baseline schedule")
    ax2.axvspan(*loss_interval, color="#c04040", alpha=0.08)
    ax2.set_xlabel("time [s]"), ax2.set_ylabel("line-of-sight margin [m]")
    # Lower right is empty by construction: both runs end high and the loss
    # interval sits in the first half. "best" put the box on the dashed tail.
    ax2.legend(fontsize=5, frameon=False, handlelength=1.6, borderaxespad=0.2,
               loc="lower right")

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = args.out_dir / "occlusion_figure.pdf", args.out_dir / "occlusion_figure.png"
    fig.savefig(pdf), fig.savefig(png, dpi=400)

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
        "trust_radius": float(sc.get("trust_radius", 0.5)),
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
        },
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
