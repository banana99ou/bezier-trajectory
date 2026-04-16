"""지수 대기 모델 기반 항력 가속도."""

import numpy as np

from orbit_transfer.constants import (
    R_E,
    CD as CD_DEFAULT,
    AREA_MASS_RATIO as AREA_MASS_DEFAULT,
    OMEGA_EARTH as OMEGA_EARTH_DEFAULT,
    ATMO_PARAMS,
)


def _get_atmo_layer(h):
    """고도에 해당하는 대기 모델 계수를 반환.

    Args:
        h: 고도 [km]

    Returns:
        (rho0, h_ref, H) 튜플

    Raises:
        ValueError: 고도가 모델 범위 밖인 경우
    """
    for h_lower, h_upper, rho0, h_ref, H in ATMO_PARAMS:
        if h_lower <= h < h_upper:
            return rho0, h_ref, H
    # 마지막 구간 상한 포함
    last = ATMO_PARAMS[-1]
    if abs(h - last[1]) < 1e-6:
        return last[2], last[3], last[4]
    raise ValueError(
        f"고도 {h:.1f} km는 대기 모델 범위 "
        f"({ATMO_PARAMS[0][0]:.0f}-{ATMO_PARAMS[-1][1]:.0f} km) 밖입니다."
    )


def exponential_drag(
    r,
    v,
    Cd=CD_DEFAULT,
    area_mass=AREA_MASS_DEFAULT,
    omega_earth=OMEGA_EARTH_DEFAULT,
):
    """지수 대기 모델 기반 항력 가속도 (NumPy 전용).

    Args:
        r: 위치 벡터 [km], shape (3,)
        v: 속도 벡터 [km/s], shape (3,)
        Cd: 항력 계수 (무차원)
        area_mass: 단면적/질량 비 [m^2/kg]
        omega_earth: 지구 자전 각속도 [rad/s]

    Returns:
        a_drag: 항력 가속도 [km/s^2], shape (3,)
    """
    r_norm = np.linalg.norm(r)
    h = r_norm - R_E

    # 대기 모델 계수 조회
    rho0, h_ref, H = _get_atmo_layer(h)

    # 대기 밀도 [kg/m^3]
    rho = rho0 * np.exp(-(h - h_ref) / H)

    # 대기 상대 속도: v_rel = v - omega x r
    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_rel = v - np.cross(omega_vec, r)

    v_rel_norm = np.linalg.norm(v_rel)

    # a_drag = -0.5 * Cd * (A/m) * rho * |v_rel| * v_rel
    # 단위: [m^2/kg]*[kg/m^3]*[km/s]*[km/s] = [km^2/(m*s^2)]
    # km^2/m = km * (km/m) = km * 1000 이므로 *1e-3 하여 km/s^2로 변환
    a_drag = -0.5 * Cd * area_mass * rho * v_rel_norm * v_rel * 1e-3

    return a_drag
