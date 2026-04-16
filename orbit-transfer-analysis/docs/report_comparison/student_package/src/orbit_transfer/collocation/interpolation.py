"""보간 모듈: Pass 1 → Pass 2 warm start + dense output."""

import numpy as np
from scipy.interpolate import CubicSpline
from .lgl_nodes import compute_lgl_nodes


def dense_output(t, x, u, n_points=300, phase_boundaries=None):
    """궤적 데이터를 조밀하게 보간하여 부드러운 출력 생성.

    Cubic spline을 사용하여 콜로케이션 노드 값을 조밀한 그리드에 보간한다.
    phase_boundaries가 주어지면 각 phase 구간 내에서 개별 보간 후 결합한다.
    Multi-Phase LGL 결과에서는 반드시 phase_boundaries를 전달해야
    phase 경계 간 오버슈트를 방지할 수 있다.

    Args:
        t: 시간 배열 [s], shape (N,)
        x: 상태 배열, shape (6, N)
        u: 제어 배열, shape (3, N)
        n_points: 출력 점 수 (default 300)
        phase_boundaries: Phase 경계 리스트 [(t_start, t_end), ...]
            None이면 글로벌 보간 (H-S, 단일 phase용)

    Returns:
        t_dense: shape (n_points,)
        x_dense: shape (6, n_points)
        u_dense: shape (3, n_points)
    """
    # 시간 순 정렬 및 중복 제거 (Phase 경계 중첩 대응)
    sort_idx = np.argsort(t)
    t_s = t[sort_idx]
    x_s = x[:, sort_idx]
    u_s = u[:, sort_idx]

    # 엄격히 증가하는 시간만 유지
    mask = np.concatenate([[True], np.diff(t_s) > 1e-12])
    t_s = t_s[mask]
    x_s = x_s[:, mask]
    u_s = u_s[:, mask]

    if len(t_s) < 4:
        return t_s, x_s, u_s

    if phase_boundaries is not None and len(phase_boundaries) > 1:
        return _dense_output_phased(t_s, x_s, u_s, n_points, phase_boundaries)

    # 글로벌 CubicSpline (H-S 균일 노드 또는 단일 phase)
    t_dense = np.linspace(t_s[0], t_s[-1], n_points)
    x_spl = CubicSpline(t_s, x_s, axis=1)
    u_spl = CubicSpline(t_s, u_s, axis=1)

    return t_dense, x_spl(t_dense), u_spl(t_dense)


def _dense_output_phased(t, x, u, n_points, phase_boundaries):
    """Phase별 개별 CubicSpline 보간 후 결합.

    각 phase 내에서만 보간하여 phase 경계 간 오버슈트를 방지한다.
    LGL 노드는 phase 내부에서 잘 조건화되어 있으므로
    phase 내 CubicSpline은 안정적이다.
    """
    T_total = t[-1] - t[0]
    if T_total <= 0:
        return t, x, u

    t_dense_list = []
    x_dense_list = []
    u_dense_list = []

    for i, (t_start, t_end) in enumerate(phase_boundaries):
        dt = t_end - t_start
        # phase 길이에 비례하여 출력 점 수 배분 (최소 10점)
        n_pts = max(10, int(n_points * dt / T_total))

        # 이 phase에 속하는 노드 추출
        eps = (t_end - t_start) * 1e-8
        ph_mask = (t >= t_start - eps) & (t <= t_end + eps)
        t_ph = t[ph_mask]
        x_ph = x[:, ph_mask]
        u_ph = u[:, ph_mask]

        if len(t_ph) < 2:
            continue

        t_d = np.linspace(t_ph[0], t_ph[-1], n_pts)

        if len(t_ph) < 4:
            # 점이 너무 적으면 선형 보간
            x_d = np.array([np.interp(t_d, t_ph, x_ph[j]) for j in range(6)])
            u_d = np.array([np.interp(t_d, t_ph, u_ph[j]) for j in range(3)])
        else:
            x_spl = CubicSpline(t_ph, x_ph, axis=1)
            u_spl = CubicSpline(t_ph, u_ph, axis=1)
            x_d = x_spl(t_d)
            u_d = u_spl(t_d)

        # 경계점 중복 제거 (두 번째 phase부터 첫 점 제외)
        if i > 0 and len(t_dense_list) > 0:
            t_d = t_d[1:]
            x_d = x_d[:, 1:]
            u_d = u_d[:, 1:]

        t_dense_list.append(t_d)
        x_dense_list.append(x_d)
        u_dense_list.append(u_d)

    if not t_dense_list:
        return t, x, u

    return (
        np.concatenate(t_dense_list),
        np.hstack(x_dense_list),
        np.hstack(u_dense_list),
    )


def interpolate_pass1_to_pass2(t_pass1, x_pass1, u_pass1, phases):
    """Pass 1 해를 Pass 2의 Multi-Phase LGL 노드에 보간.

    Args:
        t_pass1: Pass 1 시간 배열 [s], shape (N1,)
        x_pass1: Pass 1 상태 (6, N1)
        u_pass1: Pass 1 제어 (3, N1)
        phases: Phase 구조 리스트 (determine_phase_structure 출력)
            각 dict: {'t_start', 't_end', 'n_nodes', 'type'}

    Returns:
        t_phases: list of arrays, Phase별 물리 시각
        x_phases: list of (6, N_k) arrays
        u_phases: list of (3, N_k) arrays
    """
    t_phases = []
    x_phases = []
    u_phases = []

    # cubic spline interpolation
    x_interp = CubicSpline(t_pass1, x_pass1, axis=1, extrapolate=True)
    u_interp = CubicSpline(t_pass1, u_pass1, axis=1, extrapolate=True)

    for phase in phases:
        N = phase['n_nodes']
        tau = compute_lgl_nodes(N - 1)  # N nodes -> degree N-1

        # 정규화 좌표 -> 물리 시각 변환
        t_start = phase['t_start']
        t_end = phase['t_end']
        t_phys = t_start + (t_end - t_start) / 2.0 * (tau + 1.0)

        t_phases.append(t_phys)
        x_phases.append(x_interp(t_phys))
        u_phases.append(u_interp(t_phys))

    return t_phases, x_phases, u_phases
