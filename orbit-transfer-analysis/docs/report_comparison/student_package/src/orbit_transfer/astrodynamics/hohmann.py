"""호만 전이 delta-v, 비행시간, 궤적 생성."""

import numpy as np


def hohmann_dv(a1, a2, mu):
    """호만 전이의 delta-v 계산 (원궤도 → 원궤도, 경사각 변화 없음).

    Parameters
    ----------
    a1 : float  초기 원궤도 반지름 [km]
    a2 : float  최종 원궤도 반지름 [km]
    mu : float  Gravitational parameter [km³/s²]

    Returns
    -------
    dv1, dv2, dv_total : float  [km/s]
    """
    a_t = (a1 + a2) / 2.0
    v1 = np.sqrt(mu / a1)
    v2 = np.sqrt(mu / a2)
    v_t1 = np.sqrt(mu * (2.0 / a1 - 1.0 / a_t))
    v_t2 = np.sqrt(mu * (2.0 / a2 - 1.0 / a_t))
    dv1 = abs(v_t1 - v1)
    dv2 = abs(v2 - v_t2)
    return dv1, dv2, dv1 + dv2


def hohmann_tof(a1, a2, mu):
    """호만 전이의 비행시간 (전이 궤도 반 주기) [s]."""
    a_t = (a1 + a2) / 2.0
    return np.pi * np.sqrt(a_t ** 3 / mu)


def hohmann_dv_with_inclination(a1, a2, delta_i_rad, mu):
    """경사각 변화를 포함한 호만 전이 최적 delta-v 계산.

    출발 번(burn 1)과 도착 번(burn 2)에 경사각 변화를 최적 분리하여
    총 Δv를 최소화한다.

    Parameters
    ----------
    a1 : float          초기 원궤도 반지름 [km]
    a2 : float          최종 원궤도 반지름 [km]
    delta_i_rad : float 경사각 변화 [rad]
    mu : float          중력 파라미터 [km³/s²]

    Returns
    -------
    dv1_opt : float     출발 번 delta-v [km/s]
    dv2_opt : float     도착 번 delta-v [km/s]
    dv_total_opt : float 총 delta-v [km/s]
    beta_opt : float    출발 번 경사각 분량 [rad]
    """
    from scipy.optimize import minimize_scalar

    if delta_i_rad < 1e-8:
        dv1, dv2, dv_total = hohmann_dv(a1, a2, mu)
        return dv1, dv2, dv_total, 0.0

    a_t = (a1 + a2) / 2.0
    v1 = np.sqrt(mu / a1)
    v2 = np.sqrt(mu / a2)
    v_dep = np.sqrt(mu * (2.0 / a1 - 1.0 / a_t))
    v_arr = np.sqrt(mu * (2.0 / a2 - 1.0 / a_t))

    def _total_dv(beta):
        dv1_ = np.sqrt(v1 ** 2 + v_dep ** 2 - 2.0 * v1 * v_dep * np.cos(beta))
        dv2_ = np.sqrt(v2 ** 2 + v_arr ** 2 - 2.0 * v2 * v_arr * np.cos(delta_i_rad - beta))
        return dv1_ + dv2_

    res = minimize_scalar(_total_dv, bounds=(0.0, delta_i_rad), method="bounded")
    beta_opt = float(res.x)
    dv1_opt = np.sqrt(v1 ** 2 + v_dep ** 2 - 2.0 * v1 * v_dep * np.cos(beta_opt))
    dv2_opt = np.sqrt(v2 ** 2 + v_arr ** 2 - 2.0 * v2 * v_arr * np.cos(delta_i_rad - beta_opt))
    return dv1_opt, dv2_opt, dv1_opt + dv2_opt, beta_opt


def hohmann_trajectory(config, n_points: int = 300):
    """호만 전이 궤적 생성 (경사각 변화 포함).

    ν₀=0 에서 출발 (r1=(a0, 0, 0)), 전이 타원 코스팅, 도착 번(burn 2)에서
    경사각 변화 전부 수행(beta=0 근사).

    Parameters
    ----------
    config : TransferConfig
    n_points : int  출력 점 수 (default 300)

    Returns
    -------
    t : ndarray (n_points,)     시간 [s]
    x : ndarray (6, n_points)   ECI 상태 [km, km/s]
    u : ndarray (3, n_points)   추력 가속도 (코스팅이므로 0) [km/s²]
    impulses : list of dict
        [{'t': float, 'dv_vec': ndarray(3,), 'dv': float}, ...]
    metrics : dict
        dv1, dv2, dv_total, dv_total_opt, beta_opt_deg, tof
    """
    from ..constants import MU_EARTH
    from ..astrodynamics.kepler import kepler_propagate
    from ..astrodynamics.orbital_elements import oe_to_rv

    mu = MU_EARTH
    a0 = config.a0
    af = config.af
    di = float(np.radians(config.delta_i))

    # 출발 위치 (ν₀=0, 적도 원궤도)
    r1, v1_circ = oe_to_rv((a0, 0.0, 0.0, 0.0, 0.0, 0.0), mu)

    # 전이 궤도 파라미터
    a_t = (a0 + af) / 2.0
    tof = np.pi * np.sqrt(a_t ** 3 / mu)
    v_dep_mag = np.sqrt(mu * (2.0 / a0 - 1.0 / a_t))
    v1_circ_mag = float(np.linalg.norm(v1_circ))

    # 출발 속도: 방향 유지, 크기만 변경
    v1_transfer = v1_circ * (v_dep_mag / v1_circ_mag)
    dv1_vec = v1_transfer - v1_circ

    # 도착 위치 (케플러 전파)
    r2_trans, v2_trans = kepler_propagate(r1, v1_transfer, tof, mu)

    # 도착 원궤도 속도 (경사 포함, ν≈π 에서)
    _, v2_final_circ = oe_to_rv((af, 0.0, di, 0.0, 0.0, np.pi), mu)
    dv2_vec = v2_final_circ - v2_trans

    # 최적 분리 Δv (참고값)
    _, _, dv_total_opt, beta_opt = hohmann_dv_with_inclination(a0, af, di, mu)

    # 궤적 생성 (코스팅)
    ts = np.linspace(0.0, tof, n_points)
    x_arr = np.zeros((6, n_points))
    u_arr = np.zeros((3, n_points))
    for k, tk in enumerate(ts):
        rk, vk = kepler_propagate(r1, v1_transfer, tk, mu)
        x_arr[:3, k] = rk
        x_arr[3:, k] = vk

    metrics = {
        "dv1": float(np.linalg.norm(dv1_vec)),
        "dv2": float(np.linalg.norm(dv2_vec)),
        "dv_total": float(np.linalg.norm(dv1_vec) + np.linalg.norm(dv2_vec)),
        "dv_total_opt": float(dv_total_opt),
        "beta_opt_deg": float(np.degrees(beta_opt)),
        "tof": float(tof),
    }
    impulses = [
        {"t": 0.0, "dv_vec": dv1_vec.copy(), "dv": metrics["dv1"]},
        {"t": tof, "dv_vec": dv2_vec.copy(), "dv": metrics["dv2"]},
    ]
    return ts, x_arr, u_arr, impulses, metrics
