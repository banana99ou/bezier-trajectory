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
               free_arrival: bool, time_weight: float, v_max):
    from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_elastic_weight
    from spacetime_bezier.optimize import optimize_spacetime

    fn, _ = SCENARIO_MAP[scenario]
    sc = fn()
    weight = scenario_elastic_weight(scenario)
    common = dict(
        N=N, dim=len(sc["start"]), p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=n_seg, max_iter=200, tol=1e-6,
        scp_trust_radius=0.5, min_dt=0.1, elastic_weight=weight,
        # The scenario's own workspace band; optimize_spacetime takes explicit
        # arguments, so the key must be forwarded by hand here.
        coord_bounds=sc.get("coord_bounds"),
        free_arrival_time=free_arrival, time_weight=time_weight, v_max=v_max,
        verbose=False, init_curve=sc.get("init_curve"),
    )
    P_con, info_con = optimize_spacetime(stations=sc["stations"], **common)
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
    args = ap.parse_args()

    from spacetime_bezier.geometry import bezier_curve, compute_min_clearance, compute_los_margin

    sc, weight, (P_con, info_con), (P_base, info_base) = solve_pair(
        args.scenario, args.N, args.seg, args.free_arrival, args.time_weight, args.v_max)
    station = sc["stations"][0]
    dim = len(sc["start"])

    clear_con = float(compute_min_clearance(P_con, sc["obstacles"], dim=dim, n_eval=20001))
    figure_grade_or_die(info_con, clear_con, "constrained run")

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
    occluder_top = max(
        float(np.max(np.asarray(o["control_points"], float)[:, 2])) + float(o["radius"])
        if "control_points" in o else float(o["pos0"][2]) + float(o["r"])
        for o in sc["obstacles"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.0, 3.4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Left: spatial view. Occluder sampled at a few times within its own
    # window -- from the scenario parameters, not from any solver output.
    t_lo, t_hi = float(P_con[0][-1]), float(P_con[-1][-1])
    for o in sc["obstacles"]:
        t0, t1 = obstacle_window(o, t_lo, t_hi)
        if t1 < t0:
            continue
        radius = float(o.get("radius", o.get("r", 0.0)))
        for t in np.linspace(t0, t1, 5):
            c = occluder_centres(o, np.array([t]))[0]
            th = np.linspace(0, 2 * np.pi, 24)
            ax3.plot(c[0] + radius * np.cos(th), c[1] + radius * np.sin(th),
                     np.full_like(th, c[2]), color="#c04040", alpha=0.25, lw=0.7)
    ax3.plot(pts_base[:, 0], pts_base[:, 1], pts_base[:, 2],
             color="#999999", lw=1.2, ls="--", label="occlusion off")
    ax3.plot(pts_con[:, 0], pts_con[:, 1], pts_con[:, 2],
             color="#2255bb", lw=2.0, label="occlusion on")
    ax3.scatter(*station, marker="^", s=60, color="#117733", label="station")
    ax3.set_xlabel("x"), ax3.set_ylabel("y"), ax3.set_zlabel("z")
    ax3.legend(loc="upper left", fontsize=7)

    # Right: the proof panel.
    ax2.axhline(0.0, color="#666666", lw=0.8)
    ax2.plot(t_base, m_base, color="#999999", ls="--", lw=1.4, label="occlusion off")
    ax2.plot(t_con, m_con, color="#2255bb", lw=1.8, label="occlusion on")
    ax2.axvspan(*loss_interval, color="#c04040", alpha=0.08)
    ax2.set_xlabel("time"), ax2.set_ylabel("line-of-sight margin")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = args.out_dir / "occlusion_figure.pdf", args.out_dir / "occlusion_figure.png"
    fig.savefig(pdf), fig.savefig(png, dpi=160)

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
        "constrained": {
            "min_clearance": clear_con,
            "min_los_margin": float(np.min(m_con)),
            "max_z": float(np.max(pts_con[:, 2])),
            "koz_certificate": float(info_con.get("koz_violation_reference", np.nan)),
            "occlusion_certificate": float(info_con.get("occlusion_violation_reference", np.nan)),
            "iterations": int(info_con.get("iterations", -1)),
            "arrival_time": float(info_con.get("arrival_time", np.nan)),
        },
        "baseline": {
            "min_los_margin": float(np.min(m_base)),
            "los_loss_interval": loss_interval,
            "max_z": float(np.max(pts_base[:, 2])),
            "arrival_time": float(info_base.get("arrival_time", np.nan)),
        },
        "occluder_top": occluder_top, "station": list(map(float, station)),
    }
    (args.out_dir / "occlusion_figure.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {pdf}\nwrote {png}")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
