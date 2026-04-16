#!/usr/bin/env python3
"""Visualize Bézier optimizer initial guess and result for DCM experiment cases.

Usage:
    .venv/bin/python tools/dcm_visualize_case.py --case-id 2
    .venv/bin/python tools/dcm_visualize_case.py --case-id 126
    .venv/bin/python tools/dcm_visualize_case.py              # all short-transfer cases
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
OTA_ROOT = REPO_ROOT / "orbit-transfer-analysis"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OTA_ROOT / "src"))

from orbital_docking.bezier import BezierCurve
from orbital_docking.optimization import optimize_orbital_docking
from orbital_docking.visualization import (
    _plotly_earth_trace,
    _plotly_sphere_surface,
    _plotly_wire_sphere_traces,
    _sample_segment_paths,
)
from orbit_transfer.astrodynamics.kepler import kepler_propagate
from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.dynamics.two_body import gravity_acceleration
from orbit_transfer.optimizer.two_pass import TwoPassOptimizer
from orbit_transfer.types import TransferConfig

DEFAULT_DB = OTA_ROOT / "data" / "trajectories.duckdb"
OUT_DIR = REPO_ROOT / "results" / "dcm_visualize"


def load_config(db_path: Path, case_id: int) -> TransferConfig:
    con = duckdb.connect(str(db_path), read_only=True)
    row = con.execute(
        "SELECT h0, delta_a, delta_i, T_normed, COALESCE(e0,0), COALESCE(ef,0) "
        "FROM trajectories WHERE id = ?", [case_id]
    ).fetchone()
    con.close()
    if row is None:
        raise ValueError(f"Case {case_id} not found")
    return TransferConfig(h0=row[0], delta_a=row[1], delta_i=row[2],
                          T_max_normed=row[3], e0=row[4], ef=row[5])


def sample_orbit(r0, v0, T, n=500):
    pts = np.zeros((n, 3))
    for k in range(n):
        rk, _ = kepler_propagate(r0, v0, k / (n - 1) * T, MU_EARTH)
        pts[k] = rk
    return pts


def run_bezier(cfg, degree=6, n_seg=16):
    T = cfg.T_max
    r_e = R_E + cfg.h_min
    r0, v0 = oe_to_rv((cfg.a0, cfg.e0, cfg.i0, 0, 0, 0.0), MU_EARTH)
    rf, vf = oe_to_rv((cfg.af, cfg.ef, cfg.if_, 0, 0, np.pi), MU_EARTH)

    N = degree
    P_init = np.zeros((N + 1, 3))
    for j in range(N + 1):
        rj, _ = kepler_propagate(r0, v0, j / N * T, MU_EARTH)
        P_init[j] = rj

    P_opt, info = optimize_orbital_docking(
        P_init, n_seg=n_seg, r_e=r_e, v0=v0, v1=vf,
        transfer_time=T, verbose=False, use_cache=True,
    )
    return P_init, P_opt, info, r0, v0, rf, vf


def run_dcm_baseline(cfg):
    opt = TwoPassOptimizer(cfg)
    return opt.solve()


def run_dcm_proposed(cfg, P_opt, v0, vf, T):
    curve = BezierCurve(P_opt)
    n_ws = 61
    tau_ws = np.linspace(0, 1, n_ws)
    x_ws = np.zeros((6, n_ws))
    u_ws = np.zeros((3, n_ws))
    for k, tk in enumerate(tau_ws):
        r_k = curve.point(tk)
        v_k = curve.velocity(tk) / T
        a_k = curve.acceleration(tk) / (T * T)
        g_k = gravity_acceleration(r_k, MU_EARTH)
        x_ws[:3, k] = r_k
        x_ws[3:, k] = v_k
        u_ws[:, k] = a_k - g_k
    t_ws = tau_ws * T
    opt = TwoPassOptimizer(cfg)
    return opt.solve(x_init=x_ws, u_init=u_ws, t_init=t_ws), x_ws, u_ws, t_ws


def visualize_case(case_id: int, cfg: TransferConfig):
    T = cfg.T_max
    r_e = R_E + cfg.h_min
    degree = 6
    n_seg = 16

    tag = f"case_{case_id:03d}"
    print(f"\n{'='*60}")
    print(f"Case {case_id}: h0={cfg.h0} da={cfg.delta_a:.1f} di={cfg.delta_i:.2f}° "
          f"T_normed={cfg.T_max_normed:.3f} ({T:.0f}s)")
    print(f"{'='*60}")

    # Run everything
    P_init, P_opt, bez_info, r0, v0, rf, vf = run_bezier(cfg, degree, n_seg)
    res_base = run_dcm_baseline(cfg)
    (res_prop, x_ws, u_ws, t_ws) = run_dcm_proposed(cfg, P_opt, v0, vf, T)

    print(f"Bézier: feasible={bez_info.get('feasible')} iter={bez_info.get('iterations')} "
          f"min_r={bez_info.get('min_radius', 0):.0f}")
    print(f"Baseline DCM: conv={res_base.converged} cost={res_base.cost:.6e}")
    print(f"Proposed DCM: conv={res_prop.converged} cost={res_prop.cost:.6e}")

    # Sample curves
    tau = np.linspace(0, 1, 1000)
    curve_init = BezierCurve(P_init)
    curve_opt = BezierCurve(P_opt)
    pts_init = np.array([curve_init.point(t) for t in tau])
    pts_opt = np.array([curve_opt.point(t) for t in tau])

    # Reference orbits (one full period)
    dep_orbit = sample_orbit(r0, v0, cfg.T0)
    arr_orbit = sample_orbit(rf, vf, cfg.T0)

    # ── Figure 1: 3D — init guess, optimized, and both DCM results ──
    all_nearby = np.vstack([P_init, P_opt, pts_init, pts_opt,
                            dep_orbit, arr_orbit,
                            res_base.x[:3].T, res_prop.x[:3].T])
    view_radius = np.max(np.abs(all_nearby)) * 1.08

    fig = go.Figure()

    # Earth + KOZ
    fig.add_trace(_plotly_earth_trace(opacity=0.55))
    fig.add_trace(_plotly_sphere_surface(
        r_e, color="#E74C3C", opacity=0.08, name="KOZ",
        resolution_u=40, resolution_v=24, showlegend=False))
    for wt in _plotly_wire_sphere_traces(r_e, color="#C0392B", name="KOZ",
                                          line_width=2, alpha=0.6):
        fig.add_trace(wt)

    # Reference orbits
    fig.add_trace(go.Scatter3d(
        x=dep_orbit[:, 0], y=dep_orbit[:, 1], z=dep_orbit[:, 2],
        mode='lines', line=dict(color='green', width=2, dash='dot'),
        name='Departure orbit', opacity=0.4))
    fig.add_trace(go.Scatter3d(
        x=arr_orbit[:, 0], y=arr_orbit[:, 1], z=arr_orbit[:, 2],
        mode='lines', line=dict(color='orange', width=2, dash='dot'),
        name='Arrival orbit', opacity=0.4))

    # Initial control polygon + curve
    fig.add_trace(go.Scatter3d(
        x=P_init[:, 0], y=P_init[:, 1], z=P_init[:, 2],
        mode='lines+markers',
        line=dict(color='gray', width=2, dash='dash'),
        marker=dict(size=4, color='gray'),
        name='P_init (ctrl pts)',
        hovertemplate='P%{pointNumber}<br>‖r‖=%{customdata:.0f} km<extra></extra>',
        customdata=np.linalg.norm(P_init, axis=1)))
    fig.add_trace(go.Scatter3d(
        x=pts_init[:, 0], y=pts_init[:, 1], z=pts_init[:, 2],
        mode='lines', line=dict(color='gray', width=3),
        name='Init Bézier curve', opacity=0.5))

    # Optimized control polygon
    fig.add_trace(go.Scatter3d(
        x=P_opt[:, 0], y=P_opt[:, 1], z=P_opt[:, 2],
        mode='lines+markers',
        line=dict(color='purple', width=2, dash='dash'),
        marker=dict(size=5, color='purple'),
        name='P_opt (ctrl pts)',
        hovertemplate='P%{pointNumber}<br>‖r‖=%{customdata:.0f} km<extra></extra>',
        customdata=np.linalg.norm(P_opt, axis=1)))

    # Optimized Bézier curve (colored segments)
    segments = _sample_segment_paths(P_opt, n_seg)
    for si, seg in enumerate(segments):
        fig.add_trace(go.Scatter3d(
            x=seg["points"][:, 0], y=seg["points"][:, 1], z=seg["points"][:, 2],
            mode='lines', line=dict(color=seg["color"], width=5),
            name='Opt Bézier' if si == 0 else None,
            showlegend=(si == 0), legendgroup='opt_bez'))

    # Bézier warm-start trajectory (what DCM actually receives)
    fig.add_trace(go.Scatter3d(
        x=x_ws[0], y=x_ws[1], z=x_ws[2],
        mode='lines+markers',
        line=dict(color='magenta', width=3, dash='dash'),
        marker=dict(size=2, color='magenta'),
        name='Warm-start (sampled)'))

    # Baseline DCM result
    fig.add_trace(go.Scatter3d(
        x=res_base.x[0], y=res_base.x[1], z=res_base.x[2],
        mode='lines', line=dict(color='blue', width=4),
        name=f'Baseline DCM (cost={res_base.cost:.2e})'))

    # Proposed DCM result
    fig.add_trace(go.Scatter3d(
        x=res_prop.x[0], y=res_prop.x[1], z=res_prop.x[2],
        mode='lines', line=dict(color='red', width=4),
        name=f'Proposed DCM (cost={res_prop.cost:.2e})'))

    # Start/end
    fig.add_trace(go.Scatter3d(
        x=[r0[0]], y=[r0[1]], z=[r0[2]],
        mode='markers', marker=dict(size=8, color='green', symbol='diamond'),
        name='Start (ν=0)'))
    fig.add_trace(go.Scatter3d(
        x=[rf[0]], y=[rf[1]], z=[rf[2]],
        mode='markers', marker=dict(size=8, color='orange', symbol='diamond'),
        name='End (ν=π)'))

    axis_range = [-view_radius, view_radius]
    fig.update_layout(
        title=f"Case {case_id}: h0={cfg.h0} Δa={cfg.delta_a:.1f} Δi={cfg.delta_i:.2f}° "
              f"T={cfg.T_max_normed:.3f} orbits  N={degree}",
        scene=dict(
            xaxis=dict(title='X [km]', range=axis_range, showbackground=False),
            yaxis=dict(title='Y [km]', range=axis_range, showbackground=False),
            zaxis=dict(title='Z [km]', range=axis_range, showbackground=False),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.45, y=-1.45, z=0.72))),
        width=1100, height=900)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_DIR / f"{tag}_3d.html"))
    print(f"  3D: {OUT_DIR / f'{tag}_3d.html'}")

    # ── Figure 2: Radius vs τ ──
    radii_init = np.linalg.norm(pts_init, axis=1)
    radii_opt = np.linalg.norm(pts_opt, axis=1)

    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=tau, y=radii_init,
        mode='lines', line=dict(color='gray', width=2, dash='dash'),
        name='Init Bézier'))
    fig_r.add_trace(go.Scatter(x=tau, y=radii_opt,
        mode='lines', line=dict(color='purple', width=2),
        name='Opt Bézier'))
    fig_r.add_trace(go.Scatter(
        x=np.linspace(0, 1, len(P_init)), y=np.linalg.norm(P_init, axis=1),
        mode='markers', marker=dict(size=8, color='gray', symbol='square'),
        name='P_init radii'))
    fig_r.add_trace(go.Scatter(
        x=np.linspace(0, 1, len(P_opt)), y=np.linalg.norm(P_opt, axis=1),
        mode='markers', marker=dict(size=8, color='purple', symbol='square'),
        name='P_opt radii'))
    fig_r.add_hline(y=r_e, line_dash='dash', line_color='red',
                     annotation_text=f'KOZ = {r_e:.0f} km')
    fig_r.add_hline(y=cfg.a0, line_dash='dot', line_color='green',
                     annotation_text=f'dep = {cfg.a0:.0f} km')
    fig_r.add_hline(y=cfg.af, line_dash='dot', line_color='orange',
                     annotation_text=f'arr = {cfg.af:.0f} km')
    fig_r.update_layout(
        title=f'Radius vs τ — Case {case_id}',
        xaxis_title='τ', yaxis_title='‖r‖ [km]', height=500, width=900)
    fig_r.write_html(str(OUT_DIR / f"{tag}_radius.html"))
    print(f"  Radius: {OUT_DIR / f'{tag}_radius.html'}")

    # ── Figure 3: Thrust profiles side-by-side ──
    fig_t = make_subplots(rows=3, cols=1, shared_xaxes=False,
        subplot_titles=['Bézier warm-start', 'Baseline DCM', 'Proposed DCM'],
        vertical_spacing=0.08)

    u_ws_mag = np.linalg.norm(u_ws, axis=0)
    u_base_mag = np.linalg.norm(res_base.u, axis=0)
    u_prop_mag = np.linalg.norm(res_prop.u, axis=0)

    fig_t.add_trace(go.Scatter(x=t_ws, y=u_ws_mag * 1e6,
        mode='lines', line=dict(color='magenta'), name='Warm-start'), row=1, col=1)
    fig_t.add_trace(go.Scatter(x=res_base.t, y=u_base_mag * 1e6,
        mode='lines', line=dict(color='blue'), name='Baseline'), row=2, col=1)
    fig_t.add_trace(go.Scatter(x=res_prop.t, y=u_prop_mag * 1e6,
        mode='lines', line=dict(color='red'), name='Proposed'), row=3, col=1)

    fig_t.update_yaxes(title_text='‖u‖ [mm/s²]')
    fig_t.update_xaxes(title_text='Time [s]', row=3, col=1)
    fig_t.update_layout(title=f'Thrust Profiles — Case {case_id}',
                         height=700, width=900, showlegend=False)
    fig_t.write_html(str(OUT_DIR / f"{tag}_thrust.html"))
    print(f"  Thrust: {OUT_DIR / f'{tag}_thrust.html'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-id", type=int, nargs='*', default=None,
                   help="Case ID(s). Default: all short-transfer cases.")
    p.add_argument("--db", type=str, default=str(DEFAULT_DB))
    args = p.parse_args()

    if args.case_id:
        case_ids = args.case_id
    else:
        con = duckdb.connect(str(args.db), read_only=True)
        rows = con.execute(
            "SELECT id FROM trajectories "
            "WHERE converged = TRUE AND method = 'collocation' "
            "AND COALESCE(e0,0) <= 0.01 AND COALESCE(ef,0) <= 0.01 "
            "AND T_normed <= 0.5 ORDER BY id"
        ).fetchall()
        con.close()
        case_ids = [r[0] for r in rows]

    db_path = Path(args.db)
    for cid in case_ids:
        cfg = load_config(db_path, cid)
        visualize_case(cid, cfg)

    print(f"\nAll plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
