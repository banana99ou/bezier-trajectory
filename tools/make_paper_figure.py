#!/usr/bin/env python3
"""The paper's occlusion figure: one figure, two panels, from two runs.

Left panel: the station_fence trajectory in three spatial dimensions -- the
climb over the moving fence, with the station and the fence pieces drawn from
the same scenario parameters the solver consumed. Right panel: line-of-sight
margin against time for both runs, computed by `compute_los_margin` -- the
independent check, sampling the true geometry, never the solver's rows.

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

Usage: python3 tools/make_paper_figure.py [--N 8] [--seg 8] [--out-dir DIR]
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


def solve_pair(N: int, n_seg: int):
    from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_elastic_weight
    from spacetime_bezier.optimize import optimize_spacetime

    fn, _ = SCENARIO_MAP["station_fence"]
    sc = fn()
    weight = scenario_elastic_weight("station_fence")
    common = dict(
        N=N, dim=len(sc["start"]), p_start=sc["start"], p_end=sc["end"],
        obstacles=sc["obstacles"], n_seg=n_seg, max_iter=200, tol=1e-6,
        scp_trust_radius=0.5, min_dt=0.1, elastic_weight=weight,
        verbose=False, init_curve=sc.get("init_curve"),
    )
    P_con, info_con = optimize_spacetime(stations=sc["stations"], **common)
    P_base, info_base = optimize_spacetime(stations=None, **common)
    return sc, weight, (P_con, info_con), (P_base, info_base)


def figure_grade_or_die(info: dict, clearance: float, label: str):
    """Gate 1. Mirrors the B7 conditions; refuses rather than renders."""
    reasons = []
    if not bool(info.get("converged", 0.0)):
        reasons.append("did not converge")
    if float(info.get("koz_violation_reference", np.nan)) > 1e-6:
        reasons.append("keep-out certificate violated")
    if float(info.get("occlusion_violation_reference", 0.0)) > 1e-6:
        reasons.append("occlusion certificate violated")
    if float(info.get("total_koz_slack_returned", np.nan)) > 1e-8:
        reasons.append("standing on slack")
    if not clearance > 0.0:
        reasons.append("penetrates")
    if reasons:
        sys.exit(f"REFUSING to draw: {label} is not figure-grade: {'; '.join(reasons)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seg", type=int, default=8)
    ap.add_argument("--out-dir", type=pathlib.Path, default=REPO / "figures" / "paper1")
    args = ap.parse_args()

    from spacetime_bezier.geometry import bezier_curve, compute_min_clearance, compute_los_margin

    sc, weight, (P_con, info_con), (P_base, info_base) = solve_pair(args.N, args.seg)
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
    fence_top = max(float(o["pos0"][2]) + float(o["r"]) for o in sc["obstacles"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.0, 3.4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Left: spatial view. Fence pieces sampled at a few times within their own
    # windows -- from the scenario parameters, not from any solver output.
    for o in sc["obstacles"]:
        t0 = max(float(o.get("t_start", 0.0)), float(P_con[0][-1]))
        t1 = min(float(o.get("t_end", 10.0)), float(P_con[-1][-1]))
        for t in np.linspace(t0, t1, 5):
            c = np.asarray(o["pos0"], float) + np.asarray(o["vel"], float) * t
            th = np.linspace(0, 2 * np.pi, 24)
            ax3.plot(c[0] + o["r"] * np.cos(th), c[1] + o["r"] * np.sin(th),
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
    sidecar = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "git": git, "N": args.N, "n_seg": args.seg, "elastic_weight": weight,
        "constrained": {
            "min_clearance": clear_con,
            "min_los_margin": float(np.min(m_con)),
            "max_z": float(np.max(pts_con[:, 2])),
            "koz_certificate": float(info_con.get("koz_violation_reference", np.nan)),
            "occlusion_certificate": float(info_con.get("occlusion_violation_reference", np.nan)),
            "iterations": int(info_con.get("iterations", -1)),
        },
        "baseline": {
            "min_los_margin": float(np.min(m_base)),
            "los_loss_interval": loss_interval,
            "max_z": float(np.max(pts_base[:, 2])),
        },
        "fence_top": fence_top, "station": list(map(float, station)),
    }
    (args.out_dir / "occlusion_figure.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {pdf}\nwrote {png}")
    print(json.dumps(sidecar, indent=2))


if __name__ == "__main__":
    main()
