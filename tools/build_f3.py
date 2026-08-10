#!/usr/bin/env python3
"""
Build F3: Representative orbital transfer trajectories (3-panel figure).

Loads cached optimization results for N=6,7,8 at n_seg=16 and composes
a publication-ready 1×3 panel figure.

Usage:
    python tools/build_f3.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from orbital_docking import constants, generate_initial_control_points
from orbital_docking.cache import load_from_cache
from orbital_docking.visualization import (
    add_earth_sphere,
    add_wire_sphere,
    plot_segments_gradient,
    set_axes_equal_around,
    set_isometric,
    beautify_3d_axes,
    EARTH_RADIUS_KM,
    PROGRESS_MARKER, PROGRESS_LABEL,
    ISS_MARKER, ISS_LABEL,
)

# ── Orbital geometry (must match Orbital_Docking_Optimizer.py) ──────────

INCLINATION_DEG = 51.64
RAAN_DEG = 0.0
ISS_U_DEG = 45.0
PROGRESS_LAG_DEG = 120.0


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _eci(radius, inc_deg, raan_deg, u_deg):
    mu = constants.EARTH_MU_SCALED
    inc, raan, u = np.deg2rad(inc_deg), np.deg2rad(raan_deg), np.deg2rad(u_deg)
    r = np.array([radius * np.cos(u), radius * np.sin(u), 0.0])
    v = np.sqrt(mu / radius) * np.array([-np.sin(u), np.cos(u), 0.0])
    q = _rotz(raan) @ _rotx(inc)
    return q @ r, q @ v


CACHE_FILES = {
    6: ROOT / "cache" / "opt_6e47e495_nseg16.pkl",
    7: ROOT / "cache" / "opt_2fe97d12_nseg16.pkl",
    8: ROOT / "cache" / "opt_7df04265_nseg16.pkl",
}


def load_result(N):
    """Load a cached optimization result by degree."""
    path = CACHE_FILES[N]
    result = load_from_cache(path)
    if result is None:
        print(f"  ✗ cache miss for N={N} ({path})")
        return None
    P_opt, info = result
    print(f"  ✓ N={N}: {path.name}  "
          f"(effort={info['cost_true_energy'] * info['T_transfer_s'] * 1e6:.4g} m2/s3)")
    return result


def main():
    progress_r = EARTH_RADIUS_KM + constants.PROGRESS_START_ALTITUDE_KM
    iss_r = EARTH_RADIUS_KM + constants.ISS_TARGET_ALTITUDE_KM
    P_start, v0 = _eci(progress_r, INCLINATION_DEG, RAAN_DEG,
                        ISS_U_DEG - PROGRESS_LAG_DEG)
    P_end, v1 = _eci(iss_r, INCLINATION_DEG, RAAN_DEG, ISS_U_DEG)

    n_seg = 16
    orders = [6, 7, 8]
    panels = []

    for N in orders:
        P_init = generate_initial_control_points(N, P_start, P_end)
        result = load_result(N)
        if result is None:
            sys.exit(1)
        P_opt, info = result
        panels.append((N, P_init, P_opt, info))

    # ── Build figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    axes = []
    labels = ["(A)", "(B)", "(C)"]

    # Compute shared view radius — zoom to trajectory region
    all_pts = []
    for N, P_init, P_opt, _ in panels:
        all_pts.append(P_init)
        all_pts.append(P_opt)
    all_pts = np.vstack(all_pts)
    max_extent = np.linalg.norm(all_pts, axis=1).max()
    view_radius = max(max_extent * 0.55, EARTH_RADIUS_KM * 0.75)

    for i, (N, P_init, P_opt, info) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        axes.append(ax)

        # Earth + KOZ
        add_earth_sphere(ax, radius=EARTH_RADIUS_KM, color="blue", alpha=1)
        add_wire_sphere(ax, radius=constants.KOZ_RADIUS, color="red",
                        alpha=0.2, resolution=15)

        # Trajectory
        plot_segments_gradient(ax, P_opt, n_seg, lw=3.0)

        # Endpoints
        p0, p1 = P_init[0], P_init[-1]
        ax.scatter(*p0, color="green", s=140, marker=PROGRESS_MARKER,
                   label=PROGRESS_LABEL, zorder=10)
        ax.scatter(*p1, color="orange", s=140, marker=ISS_MARKER,
                   label=ISS_LABEL, zorder=10)

        # Velocity arrows
        base_r = float(np.linalg.norm(p0))
        vmax = max(np.linalg.norm(v0), np.linalg.norm(v1))
        scale = base_r / vmax if vmax > 1e-9 else 0.0
        if scale > 0:
            dv0 = np.asarray(v0) * scale
            ax.quiver(*p0, *dv0, color="cyan", linewidth=1.5,
                      arrow_length_ratio=0.08, label="v₀")
            dv1 = np.asarray(v1) * scale
            ax.quiver(*p1, *dv1, color="magenta", linewidth=1.5,
                      arrow_length_ratio=0.08, label="v₁")

        # Styling
        set_axes_equal_around(ax, center=(0, 0, 0), radius=view_radius, pad=0.05)
        set_isometric(ax, elev=20, azim=-95)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)

        ax.set_xlabel("x (km)", fontsize=8, labelpad=2)
        ax.set_ylabel("y (km)", fontsize=8, labelpad=2)
        ax.set_zlabel("z (km)", fontsize=8, labelpad=2)

        # Title
        feasible = bool(info.get("feasible", False))
        # No `or` fallback: it fired on 0.0 as well as on a missing key and
        # silently substituted a ~1e-8 cost for a ~6400 m/s quantity.
        J = info.get("cost_true_energy")
        T = info.get("T_transfer_s")
        effort = J * T * 1e6 if (J is not None and T is not None) else None
        effort_str = f"{effort:,.4g}" if effort is not None else "n/a"
        ax.set_title(
            f"{labels[i]}  N = {N}, {n_seg} segments\n"
            f"$\\int\\|u\\|^2 dt$ = {effort_str} m$^2$/s$^3$",
            fontsize=11, pad=12,
        )

        # Legend (first panel only)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90,
                        wspace=0.08)

    out = ROOT / "figures" / "f3_representative_settings.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    print(f"\nSaved → {out}  ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
