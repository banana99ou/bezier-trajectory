#!/usr/bin/env python3
"""
Build F4: the optimized trajectory for every geometry in table 2, one 3D scene.

Table 2 varies the transfer GEOMETRY, so its figure has to as well. The previous
version of this script drew three Bezier degrees on one geometry, which belongs
with table 4 in section 5.3, and it read them out of hard-coded cache files that
predate the solver fixes.

Every trajectory here is solved fresh with the cache off, at the N and n_seg
table 2 uses, through the same harness the table is built from -- so the figure
and the table cannot disagree.

The five geometries share one keep-out sphere, so they are overlaid rather than
panelled: the point of the figure is how differently the same method routes
around one obstacle, and panels would draw the obstacle five times.

Run:  .venv/bin/python tools/build_representative_trajectories.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.verify import harness_common as H
from tools.build_tables import T2_SCENARIOS, T2_DEGREE, T2_NSEG
from orbital_docking.visualization import (
    set_axes_equal_around,
    beautify_3d_axes,
    EARTH_RADIUS_KM,
)

OUT = ROOT / "figures" / "representative_trajectories.png"
SAMPLES = 400

# One hue per geometry, ordered as table 2 orders them -- the same assignment
# tools/build_scenario_trajectories.py uses for the interactive version.
COLORS = ["#2ECC71", "#3498DB", "#F39C12", "#E74C3C", "#9B59B6"]


def _orbit_frame(sc):
    """Basis of the DEPARTURE orbit plane: e1 toward the start point, e3 along
    the orbit normal, e2 completing it. Drawing in this frame puts the four
    coplanar transfers flat in e1-e2, so the plane-change case is the only one
    that leaves it -- which is the whole point of including it."""
    e1 = sc["P_start"] / np.linalg.norm(sc["P_start"])
    e3 = np.cross(sc["P_start"], sc["v0"])
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3])


def _sphere(ax, radius, color, alpha, lw=0.0, wire=False, n=48):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    if wire:
        ax.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=lw,
                          rstride=2, cstride=2, zorder=1)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0,
                        shade=True, zorder=0)


def main():
    taus = np.linspace(0.0, 1.0, SAMPLES)

    # One solve per geometry, cache off, through the harness the table uses.
    solved = []
    r_koz = None
    for (name, label), color in zip(T2_SCENARIOS, COLORS):
        sc = H.make_scenario(name, N=T2_DEGREE)
        P, info = H.run_rust(sc, n_seg=T2_NSEG)
        pts = H.positions(P, taus)
        k = int(np.argmin(np.linalg.norm(pts, axis=1)))
        margin = float(info["min_radius"]) - sc["r_e"]
        # A closest approach at either end is the departure/arrival orbit's own
        # altitude, not clearance the method produced -- do not mark it.
        binds = 0 < k < len(taus) - 1
        r_koz = float(sc["r_e"])
        solved.append((label, color, sc, pts, k, margin, binds, info))

    basis = _orbit_frame(solved[0][2])
    span = max(float(np.abs(pts @ basis.T).max()) for *_, pts, _, _, _, _ in
               [(a, b, c, d, e, f, g, h) for a, b, c, d, e, f, g, h in solved])

    fig = plt.figure(figsize=(14, 6.4))

    # (A) looking down the departure-orbit normal: the routes around the sphere.
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _sphere(ax, EARTH_RADIUS_KM, "#AEC6CF", 0.95)
    _sphere(ax, r_koz, "#C0392B", 0.30, lw=0.4, wire=True)
    for label, color, sc, pts, k, margin, binds, info in solved:
        q = pts @ basis.T
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color=color, lw=2.4,
                label=f"{label} · 여유 {margin:.2f} km", zorder=5)
        ax.scatter(*q[0], color=color, s=42, marker="o",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.scatter(*q[-1], color=color, s=54, marker="^",
                   edgecolors="black", linewidths=0.5, zorder=6)
        if binds:
            ax.scatter(*q[k], color=color, s=80, marker="x",
                       linewidths=2.2, zorder=7)
    set_axes_equal_around(ax, center=(0, 0, 0), radius=span * 1.02, pad=0.02)
    ax.view_init(elev=80, azim=-90)
    beautify_3d_axes(ax, show_ticks=True, show_grid=True)
    ax.set_xlabel("출발점 방향 (km)", fontsize=9, labelpad=4)
    ax.set_ylabel("궤도면 내 수직 방향 (km)", fontsize=9, labelpad=4)
    ax.set_title("(A) 출발 궤도면 위쪽에서 본 전이 궤적", fontsize=11, pad=4)

    # (B) the out-of-plane component, which (A) cannot show: the four coplanar
    # geometries stay at zero and only the plane change leaves the plane.
    ax2 = fig.add_subplot(1, 2, 2)
    for label, color, sc, pts, k, margin, binds, info in solved:
        clear_km = np.linalg.norm(pts, axis=1) - r_koz
        ax2.plot(taus, clear_km, color=color, lw=2.2, label=label)
        ax2.scatter(taus[k], clear_km[k], color=color, zorder=6,
                    s=80 if binds else 54,
                    marker="x" if binds else "o",
                    linewidths=2.2 if binds else 0.5,
                    edgecolors="none" if binds else "black")
    ax2.set_yscale("log")
    ax2.axhline(0.0, color="#C0392B", lw=1.0, ls="--", zorder=0)
    ax2.set_xlabel(r"곡선 매개변수 $\tau$", fontsize=9)
    ax2.set_ylabel("KOZ 표면으로부터의 거리 (km)", fontsize=9)
    ax2.set_title("(B) KOZ 표면으로부터의 여유", fontsize=11, pad=4)
    ax2.set_yticks([20, 50, 100, 200, 400])
    ax2.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.grid(alpha=0.25)
    ax2.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    handles, labels_ = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"표 2의 다섯 전이 기하 (N = {T2_DEGREE}, $n_{{seg}}$ = {T2_NSEG}) · "
        f"KOZ 반지름 {r_koz:.0f} km · ○ 출발 · △ 도착 · × 최근접점",
        fontsize=12, y=0.97)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.16, wspace=0.12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)

    print(f"{'geometry':<22} {'margin km':>10} {'KOZ binds':>10} "
          f"{'tau*':>7} {'iters':>6} {'cost m/s^2':>11}")
    for label, color, sc, pts, k, margin, binds, info in solved:
        print(f"{label:<22} {margin:>10.2f} {str(binds):>10} "
              f"{taus[k]:>7.4f} {int(info['iterations']):>6} "
              f"{float(info['mean_control_accel_ms2']):>11.3f}")
    print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
