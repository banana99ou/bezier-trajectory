#!/usr/bin/env python3
"""
Build F4: the optimized trajectory for every geometry in table 2, one 3D scene.

Table 2 varies the transfer GEOMETRY, so its figure has to as well -- the
previous F4 showed three Bezier degrees on one geometry, which belongs with
table 4 in section 5.3.

The five geometries share the same keep-out sphere and the same Earth, so they
are drawn overlaid rather than in panels: what the figure is for is showing how
differently the same method routes around one obstacle, and panels would put the
obstacle in five places.

Every trajectory is solved here, cache off, at the same N and n_seg the table
uses -- the figure and the table come from one source rather than from a cache
that may predate either.

Run:  .venv/bin/python tools/build_scenario_trajectories.py [--no-open]
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go

from tools.verify import harness_common as H
from tools.build_tables import T3_SCENARIOS, T3_DEGREE, T3_NSEG
from orbital_docking.visualization import (
    _plotly_earth_trace,
    _plotly_wire_sphere_traces,
    _default_plotly_scene,
)

OUT = ROOT / "figures" / "scenario_trajectories.html"
SAMPLES = 400

# One hue per geometry, ordered as table 2 orders them.
COLORS = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C", "#9B59B6"]


def main(open_browser=True):
    fig = go.Figure()
    fig.add_trace(_plotly_earth_trace(opacity=0.55))

    taus = np.linspace(0.0, 1.0, SAMPLES)
    span = 0.0
    rows = []

    for (name, label), color in zip(T3_SCENARIOS, COLORS):
        sc = H.make_scenario(name, N=T3_DEGREE)
        P, info = H.run_rust(sc, n_seg=T3_NSEG)
        pts = H.positions(P, taus)
        r = np.linalg.norm(pts, axis=1)
        k = int(np.argmin(r))
        margin = float(info["min_radius"]) - sc["r_e"]
        binds = 0 < k < len(taus) - 1
        span = max(span, float(np.abs(pts).max()))

        # The KOZ radius is the same for every geometry, so a single wire sphere
        # is drawn once, after this loop.
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines", name=label, legendgroup=name,
            line={"color": color, "width": 5},
            hovertemplate=(f"{label}<br>tau=%{{customdata:.3f}}"
                           "<br>r=%{text:.1f} km<extra></extra>"),
            customdata=taus, text=r,
        ))

        # Endpoints. Circle = departure, diamond = arrival.
        for pt, sym, tag in ((pts[0], "circle", "출발"), (pts[-1], "diamond", "도착")):
            fig.add_trace(go.Scatter3d(
                x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
                name=label, legendgroup=name, showlegend=False,
                marker={"size": 5, "color": color, "symbol": sym},
                hovertemplate=f"{label} {tag}<extra></extra>",
            ))

        # The point of closest approach, but ONLY where the KOZ actually binds.
        # On phase70 the minimum sits at tau=0, so marking it would suggest the
        # constraint shaped a trajectory it never touched.
        if binds:
            fig.add_trace(go.Scatter3d(
                x=[pts[k, 0]], y=[pts[k, 1]], z=[pts[k, 2]], mode="markers",
                name=label, legendgroup=name, showlegend=False,
                marker={"size": 6, "color": color, "symbol": "x"},
                hovertemplate=(f"{label} 최근접<br>여유 {margin:.2f} km"
                               f"<br>tau={taus[k]:.3f}<extra></extra>"),
            ))

        rows.append((label, margin, binds, taus[k], int(info["iterations"]),
                     float(info["mean_control_accel_ms2"])))

    sc0 = H.make_scenario(T3_SCENARIOS[0][0], N=T3_DEGREE)
    for t in _plotly_wire_sphere_traces(sc0["r_e"], "#C0392B", "KOZ"):
        fig.add_trace(t)

    fig.update_layout(
        title=(f"표 3의 다섯 전이 기하 (N={T3_DEGREE}, n_seg={T3_NSEG}) · "
               f"KOZ 반지름 {sc0['r_e']:.0f} km · × 표시는 최근접점"),
        template="plotly_white",
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    fig.update_scenes(**_default_plotly_scene(span * 1.05))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT), include_plotlyjs="cdn")

    print(f"{'geometry':<22} {'margin km':>10} {'KOZ binds':>10} "
          f"{'tau*':>7} {'iters':>6} {'cost m/s^2':>11}")
    for label, margin, binds, tau, iters, cost in rows:
        print(f"{label:<22} {margin:>10.2f} {str(binds):>10} "
              f"{tau:>7.4f} {iters:>6} {cost:>11.3f}")
    print(f"\nwrote {OUT}")

    if open_browser:
        webbrowser.open(OUT.as_uri())
    return OUT


if __name__ == "__main__":
    main(open_browser="--no-open" not in sys.argv)
