"""추력 프로파일 시각화."""

import numpy as np
import matplotlib.pyplot as plt

from ..types import TrajectoryResult
from ..collocation.interpolation import dense_output


def _get_phase_boundaries(result: TrajectoryResult):
    """TrajectoryResult에서 phase 경계 추출."""
    if result.solver_stats and 'phase_boundaries' in result.solver_stats:
        return result.solver_stats['phase_boundaries']
    return None


def plot_thrust_magnitude(result: TrajectoryResult, ax=None, dense=True,
                          n_points=300, **kwargs):
    """추력 크기 ||u(t)|| 플롯.

    Args:
        result: TrajectoryResult
        ax: matplotlib Axes (None이면 새로 생성)
        dense: True이면 cubic spline 보간으로 부드러운 곡선 출력
        n_points: dense 보간 시 출력 점 수
    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    if dense and len(result.t) >= 4:
        pb = _get_phase_boundaries(result)
        t_d, _, u_d = dense_output(result.t, result.x, result.u, n_points,
                                   phase_boundaries=pb)
        u_mag = np.linalg.norm(u_d, axis=0)
        t_hr = t_d / 3600.0
    else:
        u_mag = np.linalg.norm(result.u, axis=0)
        t_hr = result.t / 3600.0

    ax.plot(t_hr, u_mag, **kwargs)
    ax.set_xlabel('Time [hr]')
    ax.set_ylabel(r'$\|\mathbf{u}(t)\|$ [km/s$^2$]')
    ax.set_title(f'Thrust Profile (class={result.profile_class}, peaks={result.n_peaks})')
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_thrust_components(result: TrajectoryResult, ax=None, dense=True,
                           n_points=300):
    """추력 성분별 플롯 (u_x, u_y, u_z)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    if dense and len(result.t) >= 4:
        pb = _get_phase_boundaries(result)
        t_d, _, u_d = dense_output(result.t, result.x, result.u, n_points,
                                   phase_boundaries=pb)
        t_hr = t_d / 3600.0
        u_plot = u_d
    else:
        t_hr = result.t / 3600.0
        u_plot = result.u

    labels = [r'$u_x$', r'$u_y$', r'$u_z$']
    for i, label in enumerate(labels):
        ax.plot(t_hr, u_plot[i], label=label)
    ax.set_xlabel('Time [hr]')
    ax.set_ylabel(r'Thrust [km/s$^2$]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax
