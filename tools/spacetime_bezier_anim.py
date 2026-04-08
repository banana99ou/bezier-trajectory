"""
Animated space-time Bezier concept demo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spacetime_bezier.geometry import MovingObstacle, bezier_curve
from spacetime_bezier.scenarios import scenario_original


SCENE = scenario_original()
T_TOTAL = float(SCENE["T"])
OBSTACLES = [MovingObstacle.from_dict(obs) for obs in SCENE["obstacles"]]
CTRL_PTS = np.array(
    [
        [0.5, 1.0, 0.0],
        [1.5, 4.5, 2.5],
        [5.0, 7.0, 5.0],
        [7.5, 4.0, 7.5],
        [8.5, 8.5, 10.0],
    ],
    dtype=float,
)
CURVE = bezier_curve(CTRL_PTS, 300)
N_FRAMES = 120
THETA = np.linspace(0.0, 2.0 * np.pi, 60)


def make_tube_polys(obs, t_max, n_circ=20, n_t=30):
    """Pre-build tube polygons up to t_max."""
    ts = np.linspace(0, t_max, max(n_t, 2))
    rings = []
    for t in ts:
        cx, cy = obs.position(t)
        ring = np.column_stack([
            cx + obs.radius * np.cos(THETA[:n_circ+1]),
            cy + obs.radius * np.sin(THETA[:n_circ+1]),
            np.full(n_circ+1, t),
        ])
        rings.append(ring)
    polys = []
    for i in range(len(rings) - 1):
        for j in range(len(rings[i]) - 1):
            polys.append([rings[i][j], rings[i][j+1],
                          rings[i+1][j+1], rings[i+1][j]])
    return polys


# ── Animation ─────────────────────────────────────────────────────

def animate():
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    fig.subplots_adjust(wspace=0.25)

    # -- Static 3D setup --
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('t (time)')
    ax2.set_xlim(-1, 11); ax2.set_ylim(-1, 11); ax2.set_zlim(0, T_TOTAL)
    ax2.view_init(elev=22, azim=-55)

    # pre-draw full ghost trajectory in 3D
    ax2.plot(CURVE[:, 0], CURVE[:, 1], CURVE[:, 2],
             color='gray', lw=1, alpha=0.25, zorder=1)
    # control polygon
    ax2.plot(CTRL_PTS[:, 0], CTRL_PTS[:, 1], CTRL_PTS[:, 2],
             'ks--', ms=4, lw=0.8, alpha=0.3)

    # persistent 3D artists
    trail_3d, = ax2.plot([], [], [], 'k-', lw=2.5, zorder=10)
    dot_3d = ax2.scatter([], [], [], color='black', s=60, zorder=11,
                         depthshade=False)
    ax2.scatter(*CTRL_PTS[0], color='green', s=70, zorder=11,
                depthshade=False, label='start')
    ax2.scatter(*CTRL_PTS[-1], color='red', s=70, zorder=11, marker='^',
                depthshade=False, label='end')

    # store tube collections so we can update them
    tube_collections = []

    def init():
        ax1.set_xlim(-1, 11)
        ax1.set_ylim(-1, 11)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x'); ax1.set_ylabel('y')
        return []

    def update(frame):
        frac = frame / (N_FRAMES - 1)
        t_now = frac * T_TOTAL

        # find curve index closest to current time
        idx = np.searchsorted(CURVE[:, 2], t_now)
        idx = min(idx, len(CURVE) - 1)

        # ── Left panel: 2D ────────────────────────────────────
        ax1.cla()
        ax1.set_xlim(-1, 11); ax1.set_ylim(-1, 11)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x'); ax1.set_ylabel('y')
        ax1.set_title(f'2D view — t = {t_now:.1f}', fontsize=12, fontweight='bold')

        # obstacles at current time
        for obs in OBSTACLES:
            cx, cy = obs.position(t_now)
            ax1.fill(cx + obs.radius * np.cos(THETA),
                     cy + obs.radius * np.sin(THETA),
                     color=obs.color, alpha=0.3)
            ax1.plot(cx + obs.radius * np.cos(THETA),
                     cy + obs.radius * np.sin(THETA),
                     color=obs.color, lw=1.5)
            # velocity arrow
            ax1.annotate('', xy=[cx + obs.velocity[0]*1.5, cy + obs.velocity[1]*1.5],
                         xytext=[cx, cy],
                         arrowprops=dict(arrowstyle='->', color=obs.color, lw=1.2))

        # ghost of full trajectory
        ax1.plot(CURVE[:, 0], CURVE[:, 1], color='gray', lw=1, alpha=0.3)
        # traced trajectory so far
        ax1.plot(CURVE[:idx+1, 0], CURVE[:idx+1, 1], 'k-', lw=2.5)
        # current position
        ax1.plot(CURVE[idx, 0], CURVE[idx, 1], 'ko', ms=8, zorder=10)
        # start/end
        ax1.plot(CURVE[0, 0], CURVE[0, 1], 'go', ms=8, zorder=5)
        ax1.plot(CURVE[-1, 0], CURVE[-1, 1], 'r^', ms=8, zorder=5)

        # ── Right panel: 3D ───────────────────────────────────
        ax2.set_title('Space-time (x, y, t)', fontsize=12, fontweight='bold')

        # remove old tube collections
        for coll in tube_collections:
            try:
                coll.remove()
            except Exception:
                pass
        tube_collections.clear()

        # draw tubes up to current time
        for obs in OBSTACLES:
            if t_now > 0.1:
                polys = make_tube_polys(obs, t_now, n_circ=16, n_t=20)
                coll = Poly3DCollection(polys, color=obs.color,
                                        alpha=0.10, edgecolor='none')
                ax2.add_collection3d(coll)
                tube_collections.append(coll)

                # top ring
                cx, cy = obs.position(t_now)
                ring_x = cx + obs.radius * np.cos(THETA)
                ring_y = cy + obs.radius * np.sin(THETA)
                line, = ax2.plot(ring_x, ring_y, np.full_like(THETA, t_now),
                                 color=obs.color, alpha=0.6, lw=1)
                tube_collections.append(line)

        # update trajectory trail in 3D
        trail_3d.set_data(CURVE[:idx+1, 0], CURVE[:idx+1, 1])
        trail_3d.set_3d_properties(CURVE[:idx+1, 2])

        # current point
        dot_3d._offsets3d = ([CURVE[idx, 0]], [CURVE[idx, 1]], [CURVE[idx, 2]])

        # time-sweep plane (horizontal line markers)
        # just draw faint horizontal lines at current t
        return []

    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                         init_func=init, blit=False, interval=50)

    os.makedirs('figures', exist_ok=True)

    # Try MP4 first, fall back to GIF
    try:
        writer = FFMpegWriter(fps=24, bitrate=2000)
        anim.save('figures/spacetime_bezier_anim.mp4', writer=writer,
                  dpi=140)
        print("Saved: figures/spacetime_bezier_anim.mp4")
    except Exception as e:
        print(f"MP4 failed ({e}), saving as GIF...")
        writer = PillowWriter(fps=20)
        anim.save('figures/spacetime_bezier_anim.gif', writer=writer,
                  dpi=100)
        print("Saved: figures/spacetime_bezier_anim.gif")

    plt.close()


if __name__ == '__main__':
    animate()
