"""
Interactive viz dashboard for the follow-up tests.

Reads `artifacts/probe_frozen_jacobian/followups_runs.json` for the metric
tables, and re-runs a small set of selected configs (at lower max_iter for
speed) to capture trajectory geometry for the 3D plots.

Generates four HTML files:
  - followups_warmstart_dv.html       — dv vs freeze_after_iter (line+marker)
  - followups_phaselag_dv.html        — dv vs phase lag, frozen vs unfrozen (bar)
  - followups_warmstart_3d.html       — 3D trajectories for warm-start sweep
  - followups_phaselag_3d.html        — 3D trajectories per lag, frozen vs unfrozen
"""

from __future__ import annotations

import json
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
    KOZ_RADIUS,
    TRANSFER_TIME,
    N_DEG,
)


N_SEG = 16
TOL = 1e-12
MAX_ITER_VIZ = 2000  # lower cap for viz re-runs; trajectory shape is settled by ~1000-2000

OUT_DIR = Path("artifacts/probe_frozen_jacobian")
RUNS_JSON = OUT_DIR / "followups_runs.json"


def make_p_init(progress_lag_deg: float):
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - progress_lag_deg)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, P_start, P_end, v0, v1


def run(
    *, P_init, v0, v1,
    objective_mode="dv",
    freeze_gravity_jacobian=False,
    freeze_after_iter=1,
    max_iter=MAX_ITER_VIZ,
):
    t0 = time.time()
    P_opt, info = optimize_orbital_docking(
        P_init, n_seg=N_SEG, r_e=KOZ_RADIUS, max_iter=max_iter, tol=TOL,
        v0=v0, v1=v1, sample_count=100,
        objective_mode=objective_mode,
        scp_prox_weight=1e-6, scp_trust_radius=2000.0,
        transfer_time=TRANSFER_TIME,
        freeze_gravity_jacobian=freeze_gravity_jacobian,
        freeze_after_iter=freeze_after_iter,
        verbose=False, use_cache=False, ignore_existing_cache=True,
    )
    return P_opt, info, time.time() - t0


def sample_curve(P_opt, n=400):
    curve = BezierCurve(P_opt)
    taus = np.linspace(0, 1, n)
    pts = np.array([curve.point(t) for t in taus])
    return taus, pts


def make_sphere_mesh(radius, n_u=40, n_v=20):
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v)
    return (radius * np.cos(U) * np.sin(V),
            radius * np.sin(U) * np.sin(V),
            radius * np.cos(V))


def add_earth_and_koz(fig):
    import plotly.graph_objects as go
    ex, ey, ez = make_sphere_mesh(EARTH_RADIUS_KM)
    fig.add_trace(go.Surface(
        x=ex, y=ey, z=ez,
        colorscale=[[0, "#88c"], [1, "#88c"]],
        showscale=False, opacity=0.45,
        name="Earth", showlegend=True, hoverinfo="skip",
    ))
    kx, ky, kz = make_sphere_mesh(KOZ_RADIUS, n_u=30, n_v=18)
    fig.add_trace(go.Surface(
        x=kx, y=ky, z=kz,
        colorscale=[[0, "#e44"], [1, "#e44"]],
        showscale=False, opacity=0.10,
        name="KOZ shell", showlegend=True, hoverinfo="skip",
    ))


# ---- Chart 1: dv vs freeze_after_iter ----

def chart_warmstart_dv(records, out_path: Path):
    import plotly.graph_objects as go

    ws = [r for r in records if r["test"] == "warm_start"]
    unfrozen = [r for r in ws if not r["freeze"]]
    frozen = sorted([r for r in ws if r["freeze"]],
                    key=lambda r: r["freeze_after_iter"])
    if not unfrozen or not frozen:
        return
    baseline_dv = unfrozen[0]["dv_proxy_m_s"]
    fai = [r["freeze_after_iter"] for r in frozen]
    dv = [r["dv_proxy_m_s"] for r in frozen]
    pdist = [r.get("p_dist_to_unfrozen", 0.0) for r in frozen]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fai, y=dv, mode="lines+markers",
        marker=dict(size=10), line=dict(width=2),
        name="frozen (warm-start)",
        text=[f"freeze_after={k}<br>dv={v:.0f}<br>‖ΔP‖={p:.0f} km"
              for k, v, p in zip(fai, dv, pdist)],
        hoverinfo="text",
    ))
    fig.add_hline(
        y=baseline_dv, line_dash="dash", line_color="green",
        annotation_text=f"unfrozen baseline = {baseline_dv:.0f} m/s",
        annotation_position="top right",
    )
    fig.update_layout(
        title=(
            "Test 3 — Warm-start freeze: Δv vs freeze_after_iter<br>"
            "<sub>n_seg=16, dv objective, 120° lag, max_iter=10000</sub>"
        ),
        xaxis=dict(title="freeze_after_iter (K)", type="log"),
        yaxis=dict(title="dv_proxy (m/s)"),
        height=550,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Wrote: {out_path}")


# ---- Chart 2: dv vs phase lag (bar) ----

def chart_phaselag_dv(records, out_path: Path):
    import plotly.graph_objects as go

    pl = [r for r in records if r["test"] == "phase_lag"]
    by_lag = {}
    for r in pl:
        by_lag.setdefault(r["phase_lag_deg"], {})[r["freeze"]] = r
    lags = sorted(by_lag.keys())
    unfrozen_dv = [by_lag[l].get(False, {}).get("dv_proxy_m_s", float("nan")) for l in lags]
    frozen_dv = [by_lag[l].get(True, {}).get("dv_proxy_m_s", float("nan")) for l in lags]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags, y=unfrozen_dv, name="unfrozen",
        marker_color="royalblue",
    ))
    fig.add_trace(go.Bar(
        x=lags, y=frozen_dv, name="frozen (iter-1)",
        marker_color="crimson",
    ))
    fig.update_layout(
        title=(
            "Test 1 — Phase-lag sweep: Δv comparison<br>"
            "<sub>n_seg=16, dv objective, max_iter=10000</sub>"
        ),
        xaxis_title="Phase lag (deg)",
        yaxis_title="dv_proxy (m/s)",
        barmode="group",
        height=550,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Wrote: {out_path}")


# ---- Chart 3: 3D trajectories — warm-start sweep ----

def chart_warmstart_3d(out_path: Path):
    import plotly.graph_objects as go

    print("  Re-running warm-start configs at max_iter=2000 for trajectory viz...")
    P_init, P_start, P_end, v0, v1 = make_p_init(120.0)

    P_u, info_u, _ = run(P_init=P_init, v0=v0, v1=v1, freeze_gravity_jacobian=False)
    print(f"    unfrozen: dv={info_u['dv_proxy_m_s']:.0f} m/s")
    _, pts_u = sample_curve(P_u)

    fai_list = [1, 5, 20, 100, 500]
    frozen_runs = []
    for fai in fai_list:
        P_f, info_f, _ = run(
            P_init=P_init, v0=v0, v1=v1,
            freeze_gravity_jacobian=True, freeze_after_iter=fai,
        )
        _, pts_f = sample_curve(P_f)
        diff_p = float(np.linalg.norm(P_f - P_u))
        frozen_runs.append((fai, P_f, info_f, pts_f, diff_p))
        print(
            f"    freeze_after={fai}: dv={info_f['dv_proxy_m_s']:.0f} m/s "
            f"(Δ={info_f['dv_proxy_m_s']-info_u['dv_proxy_m_s']:+.0f}), "
            f"‖ΔP‖={diff_p:.0f} km"
        )

    fig = go.Figure()
    add_earth_and_koz(fig)
    fig.add_trace(go.Scatter3d(
        x=pts_u[:, 0], y=pts_u[:, 1], z=pts_u[:, 2],
        mode="lines", line=dict(color="green", width=8),
        name=f"unfrozen (dv={info_u['dv_proxy_m_s']:.0f})",
    ))
    palette = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    for (fai, P_f, info_f, pts_f, diff_p), color in zip(frozen_runs, palette):
        fig.add_trace(go.Scatter3d(
            x=pts_f[:, 0], y=pts_f[:, 1], z=pts_f[:, 2],
            mode="lines", line=dict(color=color, width=5),
            name=(
                f"freeze_after={fai} (dv={info_f['dv_proxy_m_s']:.0f}, "
                f"‖ΔP‖={diff_p:.0f} km)"
            ),
        ))
    fig.add_trace(go.Scatter3d(
        x=[P_start[0]], y=[P_start[1]], z=[P_start[2]],
        mode="markers+text", marker=dict(size=8, color="black"),
        text=["start"], textposition="top center", name="start",
    ))
    fig.add_trace(go.Scatter3d(
        x=[P_end[0]], y=[P_end[1]], z=[P_end[2]],
        mode="markers+text", marker=dict(size=8, color="black", symbol="diamond"),
        text=["end"], textposition="top center", name="end",
    ))
    fig.update_layout(
        title=(
            "Test 3 — Warm-start freeze: 3D trajectories<br>"
            f"<sub>n_seg=16, dv obj, 120° lag, max_iter={MAX_ITER_VIZ}. "
            "Toggle traces in the legend.</sub>"
        ),
        scene=dict(
            xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
        height=750,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Wrote: {out_path}")


# ---- Chart 4: 3D trajectories per phase lag ----

def chart_phaselag_3d(out_path: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print(f"  Re-running phase-lag configs at max_iter={MAX_ITER_VIZ} for trajectory viz...")
    lags = [60.0, 120.0, 180.0, 240.0]
    panels = []
    for lag in lags:
        P_init, P_start, P_end, v0, v1 = make_p_init(lag)
        P_u, info_u, _ = run(P_init=P_init, v0=v0, v1=v1, freeze_gravity_jacobian=False)
        P_f, info_f, _ = run(P_init=P_init, v0=v0, v1=v1, freeze_gravity_jacobian=True, freeze_after_iter=1)
        _, pts_u = sample_curve(P_u)
        _, pts_f = sample_curve(P_f)
        diff_p = float(np.linalg.norm(P_f - P_u))
        panels.append({
            "lag": lag, "P_start": P_start, "P_end": P_end,
            "pts_u": pts_u, "pts_f": pts_f,
            "info_u": info_u, "info_f": info_f, "diff_p": diff_p,
        })
        print(
            f"    lag={lag:.0f}°: unfrozen dv={info_u['dv_proxy_m_s']:.0f}, "
            f"frozen dv={info_f['dv_proxy_m_s']:.0f} (Δ={info_f['dv_proxy_m_s']-info_u['dv_proxy_m_s']:+.0f}), "
            f"‖ΔP‖={diff_p:.0f} km"
        )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"lag={p['lag']:.0f}°  Δdv={p['info_f']['dv_proxy_m_s']-p['info_u']['dv_proxy_m_s']:+.0f} m/s, "
            f"‖ΔP‖={p['diff_p']:.0f} km"
            for p in panels
        ],
    )
    for i, p in enumerate(panels):
        row = i // 2 + 1
        col = i % 2 + 1
        ex, ey, ez = make_sphere_mesh(EARTH_RADIUS_KM, n_u=30, n_v=15)
        kx, ky, kz = make_sphere_mesh(KOZ_RADIUS, n_u=30, n_v=15)
        fig.add_trace(go.Surface(
            x=ex, y=ey, z=ez,
            colorscale=[[0, "#88c"], [1, "#88c"]],
            showscale=False, opacity=0.4,
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Surface(
            x=kx, y=ky, z=kz,
            colorscale=[[0, "#e44"], [1, "#e44"]],
            showscale=False, opacity=0.08,
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=p["pts_u"][:, 0], y=p["pts_u"][:, 1], z=p["pts_u"][:, 2],
            mode="lines", line=dict(color="royalblue", width=6),
            name="unfrozen", showlegend=(i == 0),
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=p["pts_f"][:, 0], y=p["pts_f"][:, 1], z=p["pts_f"][:, 2],
            mode="lines", line=dict(color="crimson", width=6),
            name="frozen iter-1", showlegend=(i == 0),
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=[p["P_start"][0]], y=[p["P_start"][1]], z=[p["P_start"][2]],
            mode="markers", marker=dict(size=6, color="black"),
            name="start", showlegend=(i == 0),
        ), row=row, col=col)
        fig.add_trace(go.Scatter3d(
            x=[p["P_end"][0]], y=[p["P_end"][1]], z=[p["P_end"][2]],
            mode="markers", marker=dict(size=6, color="black", symbol="diamond"),
            name="end", showlegend=(i == 0),
        ), row=row, col=col)

    fig.update_layout(
        title=(
            "Test 1 — Phase-lag sweep: 3D trajectories (frozen vs unfrozen)<br>"
            f"<sub>n_seg=16, dv obj, max_iter={MAX_ITER_VIZ}. "
            "Each panel pans/rotates independently.</sub>"
        ),
        height=900,
        margin=dict(l=0, r=0, t=80, b=0),
    )
    # Force aspect mode "data" for each scene
    for k in ("scene", "scene2", "scene3", "scene4"):
        fig.update_layout(**{k: dict(aspectmode="data")})

    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"  Wrote: {out_path}")


def main():
    if not RUNS_JSON.exists():
        raise SystemExit(
            f"{RUNS_JSON} not found — run probe_frozen_followups.py first."
        )
    records = json.loads(RUNS_JSON.read_text())

    print("Generating chart: warm-start dv vs freeze_after_iter")
    chart_warmstart_dv(records, OUT_DIR / "followups_warmstart_dv.html")

    print("Generating chart: phase-lag dv comparison")
    chart_phaselag_dv(records, OUT_DIR / "followups_phaselag_dv.html")

    print("Generating chart: warm-start 3D trajectories")
    chart_warmstart_3d(OUT_DIR / "followups_warmstart_3d.html")

    print("Generating chart: phase-lag 3D trajectories")
    chart_phaselag_3d(OUT_DIR / "followups_phaselag_3d.html")

    print()
    print("All viz files in artifacts/probe_frozen_jacobian/:")
    for name in [
        "followups_warmstart_dv.html",
        "followups_phaselag_dv.html",
        "followups_warmstart_3d.html",
        "followups_phaselag_3d.html",
    ]:
        print(f"  file://{(OUT_DIR / name).resolve()}")


if __name__ == "__main__":
    main()
