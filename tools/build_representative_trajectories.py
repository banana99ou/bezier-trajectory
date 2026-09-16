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
from tools.build_tables import T2_SCENARIOS, T2_DEGREE, T2_NSEG, PAPER_R0
from orbital_docking.visualization import (
    set_axes_equal_around,
    beautify_3d_axes,
    EARTH_RADIUS_KM,
)

OUT = ROOT / "figures" / "representative_trajectories.png"

# Figure text is English; the Korean scenario labels in T2_SCENARIOS feed the tables.
EN_LABEL = {
    "phase70": "central angle 70 deg",
    "phase120": "central angle 120 deg (baseline)",
    "phase135": "central angle 135 deg",
    "phase170": "central angle 170 deg",
    "planechange": "plane change",
}
SAMPLES = 400

# One hue per geometry, ordered as table 2 orders them -- the same assignment
# tools/build_scenario_trajectories.py uses for the interactive version.
COLORS = ["#178344", "#2471A3", "#A9660A", "#C0392B", "#76448A"]
# Keep scenarios identifiable in grayscale in both panels and the shared legend.
LINESTYLES = {
    EN_LABEL["phase70"]: "-",
    EN_LABEL["phase120"]: (0, (6, 2.5)),
    EN_LABEL["phase135"]: (0, (1, 2)),
    EN_LABEL["phase170"]: (0, (6, 2, 1, 2)),
    EN_LABEL["planechange"]: (0, (6, 2, 1, 2, 1, 2)),
}


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
    for (name, _label_ko), color in zip(T2_SCENARIOS, COLORS):
        label = EN_LABEL[name]
        sc = H.make_scenario(name, N=T2_DEGREE)
        P, info = H.run_rust(sc, n_seg=T2_NSEG,
                             scp_trust_radius=PAPER_R0.get(name, sc["r0"]))
        pts = H.positions(P, taus)
        k = int(np.argmin(np.linalg.norm(pts, axis=1)))
        margin = float(info["min_radius"]) - sc["r_e"]
        # Endpoint clearance is set by the boundary positions; reserve the
        # closest-approach cross for an interior point.
        binds = 0 < k < len(taus) - 1
        r_koz = float(sc["r_e"])
        solved.append((label, color, sc, pts, k, margin, binds, info))

    basis = _orbit_frame(solved[0][2])
    span = max(float(np.abs(pts @ basis.T).max()) for *_, pts, _, _, _, _ in
               [(a, b, c, d, e, f, g, h) for a, b, c, d, e, f, g, h in solved])

    fig = plt.figure(figsize=(14, 6.4))

    # (A) looking down the departure-orbit normal: the routes around the sphere.
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # mplot3d ignores the zorder we pass to 3D artists and sorts by each
    # artist's AVERAGE depth instead. A trajectory is one artist, so it gets an
    # all-or-nothing verdict against the sphere's quads and can vanish behind a
    # body it never enters -- every curve here sits at 6616 km or more, outside
    # the 6371 km surface. Turning the automatic sort off makes zorder mean what
    # it says.
    ax.computed_zorder = False
    _sphere(ax, EARTH_RADIUS_KM, "#AEC6CF", 0.95)
    _sphere(ax, r_koz, "#C0392B", 0.30, lw=0.4, wire=True)
    for label, color, sc, pts, k, margin, binds, info in solved:
        q = pts @ basis.T
        ax.plot(q[:, 0], q[:, 1], q[:, 2], color=color, lw=2.4,
                linestyle=LINESTYLES[label],
                label=f"{label} · {margin:.2f} km", zorder=5)
        ax.scatter(*q[0], color=color, s=42, marker="o",
                   edgecolors="black", linewidths=0.5, zorder=6)
        ax.scatter(*q[-1], color=color, s=54, marker="^",
                   edgecolors="black", linewidths=0.5, zorder=6)
        if binds:
            ax.scatter(*q[k], color=color, s=80, marker="x",
                       linewidths=2.2, zorder=7)
    set_axes_equal_around(ax, center=(0, 0, 0), radius=span * 1.02, pad=0.02)
    ax.view_init(elev=80, azim=0)
    beautify_3d_axes(ax, show_ticks=True, show_grid=True)
    ax.set_xlabel("Departure direction (km)", fontsize=9, labelpad=4)
    ax.set_ylabel("In-plane normal direction (km)", fontsize=9, labelpad=4)
    ax.set_title("(A) Trajectories viewed from above the departure orbit plane", fontsize=11, pad=4)

    # (B) distance to the obstacle surface along each trajectory.
    ax2 = fig.add_subplot(1, 2, 2)
    for label, color, sc, pts, k, margin, binds, info in solved:
        clear_km = np.linalg.norm(pts, axis=1) - r_koz
        ax2.plot(taus, clear_km, color=color, lw=2.2,
                 linestyle=LINESTYLES[label], label=label)
        ax2.scatter(taus[k], clear_km[k], color=color, zorder=6,
                    s=80 if binds else 54,
                    marker="x" if binds else "o",
                    linewidths=2.2 if binds else 0.5,
                    edgecolors="none" if binds else "black")
    ax2.set_yscale("log")
    ax2.axhline(0.0, color="#C0392B", lw=1.0, ls="--", zorder=0)
    ax2.set_xlabel(r"Curve parameter $\tau$", fontsize=9)
    ax2.set_ylabel("Clearance (km)", fontsize=9)
    ax2.set_title("(B) Clearance from the obstacle surface", fontsize=11, pad=4)
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
               title="Minimum clearance", title_fontsize=9,
               frameon=False, handlelength=4.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Five transfer scenarios (N = {T2_DEGREE}, $n_{{seg}}$ = {T2_NSEG}) · "
        f"KOZ radius {r_koz:.0f} km · ○ departure · △ arrival · × closest approach",
        fontsize=12, y=0.97)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.88, bottom=0.16, wspace=0.12)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)

    print(f"{'geometry':<22} {'clearance km':>12} {'KOZ binds':>10} "
          f"{'tau*':>7} {'iters':>6} {'cost m/s^2':>11}")
    for label, color, sc, pts, k, margin, binds, info in solved:
        print(f"{label:<22} {margin:>12.2f} {str(binds):>10} "
              f"{taus[k]:>7.4f} {int(info['iterations']):>6} "
              f"{float(info['mean_control_accel_ms2']):>11.3f}")
    print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
