"""
Space-Time Bezier trajectory optimization conceptual demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spacetime_bezier.geometry import MovingObstacle, bezier_curve
from spacetime_bezier.scenarios import scenario_original


def build_scene():
    scenario = scenario_original()
    obstacles = [MovingObstacle.from_dict(obs) for obs in scenario["obstacles"]]
    control_points = np.array(
        [
            [0.5, 1.0, 0.0],
            [1.5, 4.5, 2.5],
            [5.0, 7.0, 5.0],
            [7.5, 4.0, 7.5],
            [8.5, 8.5, 10.0],
        ],
        dtype=float,
    )
    return obstacles, control_points, float(scenario["T"])


# ── Plotting ──────────────────────────────────────────────────────

def draw_obstacle_tube(ax, obs, t_range, color, alpha=0.15):
    """Draw a slanted cylinder (obstacle world-tube) in (x, y, t)."""
    rings = obs.tube_mesh(t_range, n_circ=24, n_t=40)
    # draw surface strips
    for i in range(len(rings) - 1):
        for j in range(len(rings[i]) - 1):
            poly = [rings[i][j], rings[i][j+1],
                    rings[i+1][j+1], rings[i+1][j]]
            ax.add_collection3d(Poly3DCollection(
                [poly], color=color, alpha=alpha, edgecolor='none'))
    # draw top and bottom circles
    for ring in [rings[0], rings[-1]]:
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                color=color, alpha=0.4, lw=0.8)


def draw_snapshot_plane(ax, obstacles, t_snap, xy_lim, alpha=0.06):
    """Draw a faint horizontal plane at t=t_snap with obstacle circles."""
    # plane
    xs = np.array([xy_lim[0], xy_lim[1], xy_lim[1], xy_lim[0]])
    ys = np.array([xy_lim[0], xy_lim[0], xy_lim[1], xy_lim[1]])
    zs = np.full(4, t_snap)
    verts = [list(zip(xs, ys, zs))]
    ax.add_collection3d(Poly3DCollection(
        verts, color='gray', alpha=alpha, edgecolor='gray', linewidth=0.3))
    # obstacle cross-sections
    theta = np.linspace(0, 2 * np.pi, 60)
    colors = ['#e74c3c', '#2980b9', '#27ae60']
    for obs, c in zip(obstacles, colors):
        cx, cy = obs.position(t_snap)
        ax.plot(cx + obs.radius * np.cos(theta),
                cy + obs.radius * np.sin(theta),
                np.full_like(theta, t_snap),
                color=c, alpha=0.5, lw=1.0)


def main():
    obstacles, ctrl_pts, T = build_scene()
    curve = bezier_curve(ctrl_pts, num_pts=300)

    # ── Figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7))

    # --- Left panel: 2D snapshots (top-down view) ---
    ax1 = fig.add_subplot(121)
    ax1.set_title("2D snapshots at different times", fontsize=12, fontweight='bold')
    snap_times = [0, 2.5, 5.0, 7.5, 10.0]
    colors_obs = ['#e74c3c', '#2980b9', '#27ae60']
    alphas = np.linspace(0.15, 0.6, len(snap_times))
    theta = np.linspace(0, 2 * np.pi, 80)

    for k, t_s in enumerate(snap_times):
        for obs, c in zip(obstacles, colors_obs):
            cx, cy = obs.position(t_s)
            ax1.fill(cx + obs.radius * np.cos(theta),
                     cy + obs.radius * np.sin(theta),
                     color=c, alpha=alphas[k] * 0.4)
            ax1.plot(cx + obs.radius * np.cos(theta),
                     cy + obs.radius * np.sin(theta),
                     color=c, alpha=alphas[k], lw=0.8)
        # small time label
        ax1.text(9.2, 0.3 + k * 0.7, f"t={t_s:.0f}", fontsize=7,
                 alpha=alphas[k], color='gray')

    # project trajectory onto xy
    ax1.plot(curve[:, 0], curve[:, 1], 'k-', lw=2, label='trajectory (xy)')
    ax1.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'ks--', ms=5, lw=1,
             alpha=0.5, label='control pts (xy)')
    ax1.plot(curve[0, 0], curve[0, 1], 'go', ms=8, zorder=5, label='start')
    ax1.plot(curve[-1, 0], curve[-1, 1], 'r^', ms=8, zorder=5, label='end')

    # arrows showing obstacle motion
    for obs, c in zip(obstacles, colors_obs):
        ax1.annotate('', xy=obs.pos0 + obs.velocity * 3,
                     xytext=obs.pos0,
                     arrowprops=dict(arrowstyle='->', color=c, lw=1.5))

    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 11)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # --- Right panel: 3D space-time view ---
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Space-time (x, y, t) — the key insight", fontsize=12, fontweight='bold')

    # obstacle tubes
    for obs, c in zip(obstacles, colors_obs):
        draw_obstacle_tube(ax2, obs, [0, T], color=c, alpha=0.12)

    # snapshot planes
    for t_s in [0, 5.0, 10.0]:
        draw_snapshot_plane(ax2, obstacles, t_s, [-1, 11], alpha=0.04)

    # trajectory curve in (x, y, t)
    ax2.plot(curve[:, 0], curve[:, 1], curve[:, 2],
             'k-', lw=2.5, label='Bézier trajectory', zorder=10)

    # control points and control polygon
    ax2.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], ctrl_pts[:, 2],
             'ks--', ms=6, lw=1, alpha=0.6, label='control points')

    # convex hull hint — shade control polygon
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(ctrl_pts)
        for simplex in hull.simplices:
            tri = ctrl_pts[simplex]
            ax2.add_collection3d(Poly3DCollection(
                [tri], color='gold', alpha=0.06, edgecolor='goldenrod',
                linewidth=0.5))
    except Exception:
        pass

    # start / end markers
    ax2.scatter(*ctrl_pts[0], color='green', s=80, zorder=11,
                label='start', depthshade=False)
    ax2.scatter(*ctrl_pts[-1], color='red', s=80, zorder=11, marker='^',
                label='end', depthshade=False)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('t (time)')
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-1, 11)
    ax2.set_zlim(0, T)
    ax2.legend(fontsize=7, loc='upper left')
    ax2.view_init(elev=22, azim=-55)

    plt.tight_layout()
    import os
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/spacetime_bezier_concept.png', dpi=180,
                bbox_inches='tight', facecolor='white')
    print("Saved: figures/spacetime_bezier_concept.png")


if __name__ == '__main__':
    main()
