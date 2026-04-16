"""3D 궤적 시각화."""

import numpy as np
import matplotlib.pyplot as plt

from ..constants import R_E
from ..collocation.interpolation import dense_output


def plot_trajectory_3d(result, ax=None, show_earth=True, dense=True,
                       n_points=300):
    """3D 궤적 플롯.

    Args:
        result: TrajectoryResult
        ax: matplotlib 3D Axes (None이면 새로 생성)
        show_earth: 지구 와이어프레임 표시 여부
        dense: True이면 cubic spline 보간으로 부드러운 궤적 출력
        n_points: dense 보간 시 출력 점 수
    Returns:
        fig, ax
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    if dense and len(result.t) >= 4:
        pb = None
        if hasattr(result, 'solver_stats') and result.solver_stats:
            pb = result.solver_stats.get('phase_boundaries')
        t_d, x_d, _ = dense_output(result.t, result.x, result.u, n_points,
                                   phase_boundaries=pb)
        r = x_d[:3]
    else:
        r = result.x[:3]  # (3, N)

    ax.plot(r[0], r[1], r[2], 'b-', linewidth=0.8)

    # 출발/도착 마커는 원본 데이터 사용
    r_orig = result.x[:3]
    ax.plot([r_orig[0, 0]], [r_orig[1, 0]], [r_orig[2, 0]], 'go',
            markersize=8, label='Start')
    ax.plot([r_orig[0, -1]], [r_orig[1, -1]], [r_orig[2, -1]], 'rs',
            markersize=8, label='End')

    if show_earth:
        # 지구 표면 (wireframe)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = R_E * np.outer(np.cos(u), np.sin(v))
        y = R_E * np.outer(np.sin(u), np.sin(v))
        z = R_E * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, alpha=0.1, color='cyan')

    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.legend()
    return fig, ax
