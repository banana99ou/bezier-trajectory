"""
Visualize frozen vs unfrozen converged trajectories at n_seg=16, max_iter=10000.

Three panels:
  A) 3D trajectory (Earth surface + KOZ shell + both curves + control polygons)
  B) Radial distance r(tau) vs tau, with KOZ line
  C) Control acceleration magnitude |a_u|(tau) — what dv_proxy integrates
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
OUT_PATH = Path("artifacts/probe_frozen_jacobian/trajectories_n_seg16.png")


def make_p_init():
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PROGRESS_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, P_start, P_end, v0, v1


def run(freeze, P_init, v0, v1):
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
        verbose=False,
        use_cache=False,
        ignore_existing_cache=True,
    )
    return P_opt, info, time.time() - t0


def sample_curve(P_opt, n=400):
    curve = BezierCurve(P_opt)
    taus = np.linspace(0.0, 1.0, n)
    pts = np.array([curve.point(t) for t in taus])
    accels_param = np.array([curve.acceleration(t) for t in taus])  # d²P/dτ²
    return taus, pts, accels_param


def control_accel_ms2(P_opt, T, n=400):
    """|a_u|(tau) = |a_geom - a_grav| in m/s^2."""
    from orbital_docking.constants import EARTH_MU_SCALED, EARTH_RADIUS_KM, EARTH_J2
    from orbital_docking.optimization import _accel_total

    taus, pts, a_param = sample_curve(P_opt, n=n)
    a_geom_kms2 = a_param / (T**2)
    a_grav_kms2 = np.array(
        [_accel_total(p, EARTH_MU_SCALED, EARTH_RADIUS_KM, EARTH_J2) for p in pts]
    )
    a_u_ms2 = np.linalg.norm(a_geom_kms2 - a_grav_kms2, axis=1) * 1e3
    return taus, pts, a_u_ms2


def plot_earth_sphere(ax, radius, color, alpha):
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 18j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=True)


def main():
    P_init, P_start, P_end, v0, v1 = make_p_init()

    print(f"Running unfrozen at n_seg={N_SEG}, max_iter={MAX_ITER}...")
    P_u, info_u, wall_u = run(False, P_init, v0, v1)
    print(
        f"  dv={info_u['dv_proxy_m_s']:.2f} m/s, min_r={info_u['min_radius']:.3f} km, "
        f"final_delta={info_u.get('final_delta_norm', 0.0):.3e}, wall={wall_u:.1f}s"
    )

    print(f"Running frozen   at n_seg={N_SEG}, max_iter={MAX_ITER}...")
    P_f, info_f, wall_f = run(True, P_init, v0, v1)
    print(
        f"  dv={info_f['dv_proxy_m_s']:.2f} m/s, min_r={info_f['min_radius']:.3f} km, "
        f"final_delta={info_f.get('final_delta_norm', 0.0):.3e}, wall={wall_f:.1f}s"
    )

    diff_p = float(np.linalg.norm(P_f - P_u))
    print(f"||P_frozen - P_unfrozen|| = {diff_p:.2f} km")

    taus_u, pts_u, au_u = control_accel_ms2(P_u, TRANSFER_TIME)
    taus_f, pts_f, au_f = control_accel_ms2(P_f, TRANSFER_TIME)
    r_u = np.linalg.norm(pts_u, axis=1)
    r_f = np.linalg.norm(pts_f, axis=1)

    # ---- Plot ----
    fig = plt.figure(figsize=(14, 10))

    # Panel A: 3D
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_earth_sphere(ax1, EARTH_RADIUS_KM, color="lightblue", alpha=0.25)
    plot_earth_sphere(ax1, KOZ_RADIUS, color="red", alpha=0.08)
    ax1.plot(pts_u[:, 0], pts_u[:, 1], pts_u[:, 2], color="C0", lw=2.0,
             label=f"unfrozen (dv={info_u['dv_proxy_m_s']:.0f} m/s)")
    ax1.plot(pts_f[:, 0], pts_f[:, 1], pts_f[:, 2], color="C3", lw=2.0,
             label=f"frozen   (dv={info_f['dv_proxy_m_s']:.0f} m/s)")
    ax1.scatter(*P_u.T, color="C0", s=14, alpha=0.7, label="ctrl pts (unfrozen)")
    ax1.scatter(*P_f.T, color="C3", s=14, alpha=0.7, label="ctrl pts (frozen)")
    ax1.scatter(*P_start, color="black", s=60, marker="o", label="start")
    ax1.scatter(*P_end, color="black", s=60, marker="*", label="end")
    ax1.set_xlabel("X (km)"); ax1.set_ylabel("Y (km)"); ax1.set_zlabel("Z (km)")
    ax1.set_title(f"3D trajectory — n_seg={N_SEG}, max_iter={MAX_ITER}")
    ax1.legend(loc="upper left", fontsize=8)
    # Tighten view
    ax1.set_box_aspect([1, 1, 0.6])

    # Panel B: top-down XY projection
    ax2 = fig.add_subplot(2, 2, 2)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.fill(EARTH_RADIUS_KM * np.cos(theta), EARTH_RADIUS_KM * np.sin(theta),
             color="lightblue", alpha=0.4, label="Earth")
    ax2.plot(KOZ_RADIUS * np.cos(theta), KOZ_RADIUS * np.sin(theta),
             color="red", linestyle="--", lw=1.0, alpha=0.6, label="KOZ shell")
    ax2.plot(pts_u[:, 0], pts_u[:, 1], color="C0", lw=2.0,
             label=f"unfrozen")
    ax2.plot(pts_f[:, 0], pts_f[:, 1], color="C3", lw=2.0,
             label=f"frozen")
    ax2.plot(P_u[:, 0], P_u[:, 1], "o-", color="C0", alpha=0.4, ms=5, lw=0.7)
    ax2.plot(P_f[:, 0], P_f[:, 1], "o-", color="C3", alpha=0.4, ms=5, lw=0.7)
    ax2.scatter(P_start[0], P_start[1], color="black", s=60, marker="o", zorder=5)
    ax2.scatter(P_end[0], P_end[1], color="black", s=60, marker="*", zorder=5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (km)"); ax2.set_ylabel("Y (km)")
    ax2.set_title(f"XY projection — ‖P_frozen − P_unfrozen‖ = {diff_p:.0f} km")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel C: r(tau)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(taus_u, r_u, color="C0", lw=2.0,
             label=f"unfrozen  (min_r={r_u.min():.2f} km)")
    ax3.plot(taus_f, r_f, color="C3", lw=2.0,
             label=f"frozen    (min_r={r_f.min():.2f} km)")
    ax3.axhline(KOZ_RADIUS, color="red", linestyle="--", lw=1.0, alpha=0.7,
                label=f"KOZ = {KOZ_RADIUS} km")
    ax3.axhline(EARTH_RADIUS_KM, color="gray", linestyle=":", lw=0.8, alpha=0.7,
                label=f"Earth = {EARTH_RADIUS_KM} km")
    ax3.set_xlabel("τ"); ax3.set_ylabel("|r| (km)")
    ax3.set_title("Radial distance along trajectory")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel D: control accel magnitude
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(taus_u, au_u, color="C0", lw=2.0, label="unfrozen")
    ax4.plot(taus_f, au_f, color="C3", lw=2.0, label="frozen")
    ax4.set_xlabel("τ"); ax4.set_ylabel("|a_u| (m/s²)")
    ax4.set_title("Control acceleration magnitude (integrand of dv proxy)")
    ax4.set_yscale("log")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Frozen vs unfrozen gravity-Jacobian solutions  ·  n_seg={N_SEG}, "
        f"max_iter={MAX_ITER}, tol={TOL:g}\n"
        f"dv penalty: {info_f['dv_proxy_m_s'] - info_u['dv_proxy_m_s']:+.0f} m/s "
        f"({(info_f['dv_proxy_m_s']/info_u['dv_proxy_m_s']-1)*100:+.1f}%)",
        fontsize=12,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
