"""
Interactive 3D viz (Plotly HTML, pan/zoom/rotate) of frozen vs unfrozen
trajectories at n_seg=16. Saves an HTML file that opens in a browser.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from orbital_docking import constants
from orbital_docking.bezier import BezierCurve
from orbital_docking.optimization import (
    optimize_orbital_docking,
    generate_initial_control_points,
)
from tools.probe_frozen_jacobian import (
    eci_from_circular,
    EARTH_RADIUS_KM,
    PROGRESS_ALT,
    ISS_ALT,
    INC,
    RAAN,
    ISS_U,
    PROGRESS_LAG,
    KOZ_RADIUS,
    TRANSFER_TIME,
    N_DEG,
)


N_SEG = 16
MAX_ITER = 10000
TOL = 1e-12
OUT_PATH = Path("artifacts/probe_frozen_jacobian/trajectories_n_seg16.html")


def make_p_init():
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PROGRESS_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, P_start, P_end, v0, v1


def run(freeze, freeze_after_iter, P_init, v0, v1):
    t0 = time.time()
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=N_SEG,
        r_e=KOZ_RADIUS,
        max_iter=MAX_ITER,
        tol=TOL,
        v0=v0,
        v1=v1,
        sample_count=100,
        objective_mode="dv",
        scp_prox_weight=1e-6,
        scp_trust_radius=2000.0,
        transfer_time=TRANSFER_TIME,
        freeze_gravity_jacobian=freeze,
        freeze_after_iter=freeze_after_iter,
        verbose=False,
        use_cache=False,
        ignore_existing_cache=True,
    )
    return P_opt, info, time.time() - t0


def sample_curve(P_opt, n=400):
    curve = BezierCurve(P_opt)
    taus = np.linspace(0.0, 1.0, n)
    pts = np.array([curve.point(t) for t in taus])
    return pts


def make_sphere_mesh(radius, n_u=40, n_v=20):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v)
    x = radius * np.cos(U) * np.sin(V)
    y = radius * np.sin(U) * np.sin(V)
    z = radius * np.cos(V)
    return x, y, z


def main():
    import plotly.graph_objects as go

    P_init, P_start, P_end, v0, v1 = make_p_init()

    print(f"Running unfrozen at n_seg={N_SEG}, max_iter={MAX_ITER}...")
    P_u, info_u, wall_u = run(False, 1, P_init, v0, v1)
    print(
        f"  dv={info_u['dv_proxy_m_s']:.2f} m/s, min_r={info_u['min_radius']:.3f} km, "
        f"final_delta={info_u.get('final_delta_norm', 0.0):.3e}, wall={wall_u:.1f}s"
    )

    print(f"Running frozen   at iter-1 freeze, n_seg={N_SEG}, max_iter={MAX_ITER}...")
    P_f, info_f, wall_f = run(True, 1, P_init, v0, v1)
    print(
        f"  dv={info_f['dv_proxy_m_s']:.2f} m/s, min_r={info_f['min_radius']:.3f} km, "
        f"final_delta={info_f.get('final_delta_norm', 0.0):.3e}, wall={wall_f:.1f}s"
    )

    pts_u = sample_curve(P_u)
    pts_f = sample_curve(P_f)
    diff_p = float(np.linalg.norm(P_f - P_u))
    print(f"||P_frozen - P_unfrozen|| = {diff_p:.2f} km")

    # Build figure
    fig = go.Figure()

    # Earth
    ex, ey, ez = make_sphere_mesh(EARTH_RADIUS_KM)
    fig.add_trace(
        go.Surface(
            x=ex, y=ey, z=ez,
            colorscale=[[0, "#88c"], [1, "#88c"]],
            showscale=False, opacity=0.45,
            name="Earth", showlegend=True,
            hoverinfo="skip",
        )
    )

    # KOZ shell
    kx, ky, kz = make_sphere_mesh(KOZ_RADIUS, n_u=30, n_v=18)
    fig.add_trace(
        go.Surface(
            x=kx, y=ky, z=kz,
            colorscale=[[0, "#e44"], [1, "#e44"]],
            showscale=False, opacity=0.12,
            name="KOZ shell", showlegend=True,
            hoverinfo="skip",
        )
    )

    # Unfrozen trajectory
    fig.add_trace(
        go.Scatter3d(
            x=pts_u[:, 0], y=pts_u[:, 1], z=pts_u[:, 2],
            mode="lines",
            line=dict(color="royalblue", width=6),
            name=f"unfrozen (dv={info_u['dv_proxy_m_s']:.0f} m/s)",
        )
    )
    # Unfrozen control polygon
    fig.add_trace(
        go.Scatter3d(
            x=P_u[:, 0], y=P_u[:, 1], z=P_u[:, 2],
            mode="lines+markers",
            line=dict(color="royalblue", width=2, dash="dot"),
            marker=dict(size=4, color="royalblue", symbol="circle"),
            name="unfrozen ctrl polygon",
            opacity=0.7,
        )
    )

    # Frozen trajectory
    fig.add_trace(
        go.Scatter3d(
            x=pts_f[:, 0], y=pts_f[:, 1], z=pts_f[:, 2],
            mode="lines",
            line=dict(color="crimson", width=6),
            name=f"frozen iter-1 (dv={info_f['dv_proxy_m_s']:.0f} m/s)",
        )
    )
    # Frozen control polygon
    fig.add_trace(
        go.Scatter3d(
            x=P_f[:, 0], y=P_f[:, 1], z=P_f[:, 2],
            mode="lines+markers",
            line=dict(color="crimson", width=2, dash="dot"),
            marker=dict(size=4, color="crimson", symbol="circle"),
            name="frozen ctrl polygon",
            opacity=0.7,
        )
    )

    # Endpoints
    fig.add_trace(
        go.Scatter3d(
            x=[P_start[0]], y=[P_start[1]], z=[P_start[2]],
            mode="markers+text",
            marker=dict(size=8, color="black"),
            text=["start"], textposition="top center",
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[P_end[0]], y=[P_end[1]], z=[P_end[2]],
            mode="markers+text",
            marker=dict(size=8, color="black", symbol="diamond"),
            text=["end"], textposition="top center",
            name="end",
        )
    )

    fig.update_layout(
        title=(
            f"Frozen vs unfrozen gravity-Jacobian — n_seg={N_SEG}, "
            f"max_iter={MAX_ITER}, tol={TOL:g}<br>"
            f"<sub>Δdv = {info_f['dv_proxy_m_s']-info_u['dv_proxy_m_s']:+.0f} m/s "
            f"({(info_f['dv_proxy_m_s']/info_u['dv_proxy_m_s']-1)*100:+.1f}%) · "
            f"||P_frozen − P_unfrozen|| = {diff_p:.0f} km</sub>"
        ),
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
        height=800,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_PATH), include_plotlyjs="cdn")
    print(f"Wrote: {OUT_PATH}")
    print(f"Open in browser: file://{OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
