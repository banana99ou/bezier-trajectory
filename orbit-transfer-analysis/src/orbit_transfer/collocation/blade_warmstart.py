"""BLADE 해 → Multi-Phase LGL 콜로케이션 warm-start 생성.

BLADE-SCP 결과를 조밀한 궤적으로 재구성한 뒤,
CubicSpline 보간으로 각 phase의 LGL 노드에 상태/제어 초기값을 생성한다.
기존 interpolate_pass1_to_pass2()와 동일한 출력 형식을 따른다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .lgl_nodes import compute_lgl_nodes


def interpolate_blade_to_lgl(
    t_dense: NDArray,
    x_dense: NDArray,
    u_dense: NDArray,
    phases: list[dict],
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """BLADE dense 궤적을 Multi-Phase LGL 초기값으로 변환.

    Parameters
    ----------
    t_dense : (N,) array
        BLADE 조밀 궤적의 물리 시각 [s].
    x_dense : (6, N) array
        물리 단위 상태 벡터 [km, km/s].
    u_dense : (3, N) array
        물리 단위 제어 벡터 [km/s^2].
    phases : list of dict
        Phase 구조 (blade_phase_structure 또는 determine_phase_structure 출력).
        각 dict: {'t_start', 't_end', 'n_nodes', 'type'}.

    Returns
    -------
    t_phases : list of (N_k,) arrays
        Phase별 물리 시각 [s].
    x_phases : list of (6, N_k) arrays
        Phase별 상태 초기값.
    u_phases : list of (3, N_k) arrays
        Phase별 제어 초기값.
    """
    # CubicSpline 보간기 (전체 dense 궤적 기반)
    # 시간 정렬 + 중복 제거
    sort_idx = np.argsort(t_dense)
    t_s = t_dense[sort_idx]
    x_s = x_dense[:, sort_idx]
    u_s = u_dense[:, sort_idx]

    mask = np.concatenate([[True], np.diff(t_s) > 1e-12])
    t_s = t_s[mask]
    x_s = x_s[:, mask]
    u_s = u_s[:, mask]

    x_interp = CubicSpline(t_s, x_s, axis=1, extrapolate=True)
    u_interp = CubicSpline(t_s, u_s, axis=1, extrapolate=True)

    t_phases: list[NDArray] = []
    x_phases: list[NDArray] = []
    u_phases: list[NDArray] = []

    for phase in phases:
        N_k = phase["n_nodes"]
        tau = compute_lgl_nodes(N_k - 1)  # N nodes -> degree N-1

        t_start = phase["t_start"]
        t_end = phase["t_end"]
        # LGL 노드를 [-1, 1] → [t_start, t_end]로 변환
        t_phys = t_start + (t_end - t_start) / 2.0 * (tau + 1.0)

        t_phases.append(t_phys)
        x_phases.append(x_interp(t_phys))
        u_phases.append(u_interp(t_phys))

    return t_phases, x_phases, u_phases


def blade_to_dense_trajectory(
    p_segments: list[NDArray],
    n: int,
    K: int,
    t_f_phys: float,
    x0_norm: NDArray,
    cu,
    n_dense: int = 30,
) -> tuple[NDArray, NDArray, NDArray]:
    """BLADE 해를 조밀한 물리 단위 궤적으로 재구성.

    Parameters
    ----------
    p_segments : list of (n+1, 3) arrays
        BLADE 세그먼트별 Bernstein 제어점 (정규화 단위).
    n : int
        세그먼트 차수.
    K : int
        세그먼트 수.
    t_f_phys : float
        비행시간 [s].
    x0_norm : (6,) array
        초기 상태 (정규화).
    cu : CanonicalUnits
        정규화 단위.
    n_dense : int
        세그먼트당 출력 점 수.

    Returns
    -------
    t_arr : (N,) array — 물리 시각 [s]
    x_phys : (6, N) array — [km, km/s]
    u_phys : (3, N) array — [km/s^2]
    """
    from bezier_orbit.blade.orbit import _propagate_blade_reference
    from bezier_orbit.bezier.basis import bernstein

    t_f_norm = t_f_phys / cu.TU
    deltas = np.full(K, 1.0 / K)

    # 궤적 재구성
    seg_trajs = _propagate_blade_reference(
        p_segments, n, K, deltas, t_f_norm,
        x0_norm, cu.R_earth_star, n_dense,
    )

    # 세그먼트 연결
    x_norm = np.vstack([seg[:-1] for seg in seg_trajs[:-1]] + [seg_trajs[-1]])
    N = x_norm.shape[0]

    # 추력 프로파일 재구성
    n_per = seg_trajs[0].shape[0]
    u_norm = np.zeros((N, 3))
    idx = 0
    for k in range(K):
        pk = p_segments[k]
        pts = n_per - 1 if k < K - 1 else n_per
        for j in range(pts):
            tau_local = j / (n_per - 1)
            B = bernstein(n, tau_local)
            u_norm[idx] = B @ pk
            idx += 1

    # 물리 단위 변환
    x_phys = np.zeros_like(x_norm)
    x_phys[:, :3] = x_norm[:, :3] * cu.DU
    x_phys[:, 3:] = x_norm[:, 3:] * cu.VU
    u_phys = u_norm * cu.AU

    t_arr = np.linspace(0.0, t_f_phys, N)

    # (N, D) → (D, N)
    return t_arr, x_phys.T, u_phys.T
