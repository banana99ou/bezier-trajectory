"""Lambert 문제 솔버 (Universal Variable 법).

Curtis, "Orbital Mechanics for Engineering Students" 4th ed., Algorithm 5.2 기반.
scipy.optimize.brentq로 TOF 방정식을 수치 풀이.
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Stumpff 함수 (내부 전용)
# ---------------------------------------------------------------------------

def _c2(psi: float) -> float:
    if psi > 1e-6:
        return (1.0 - np.cos(np.sqrt(psi))) / psi
    elif psi < -1e-6:
        return (np.cosh(np.sqrt(-psi)) - 1.0) / (-psi)
    return 0.5


def _c3(psi: float) -> float:
    if psi > 1e-6:
        sp = np.sqrt(psi)
        return (sp - np.sin(sp)) / (psi * sp)
    elif psi < -1e-6:
        sp = np.sqrt(-psi)
        return (np.sinh(sp) - sp) / ((-psi) * sp)
    return 1.0 / 6.0


# ---------------------------------------------------------------------------
# 공개 인터페이스
# ---------------------------------------------------------------------------

def lambert(
    r1_vec: np.ndarray,
    r2_vec: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Lambert 문제 풀기.

    두 위치벡터와 비행시간이 주어졌을 때 출발/도착 속도벡터를 계산한다.
    단일 혁명(single revolution) 궤도만 지원한다.

    Parameters
    ----------
    r1_vec : ndarray (3,)
        출발 위치 [km]
    r2_vec : ndarray (3,)
        도착 위치 [km]
    tof : float
        비행시간 [s] (양수)
    mu : float
        중력 파라미터 [km³/s²]
    prograde : bool
        True = 순행 전이, False = 역행 전이

    Returns
    -------
    v1 : ndarray (3,)
        출발점에서의 속도 [km/s]
    v2 : ndarray (3,)
        도착점에서의 속도 [km/s]

    Raises
    ------
    ValueError
        해를 찾지 못했거나 기하학적으로 특이한 경우
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    if r1 < 1e-10 or r2 < 1e-10:
        raise ValueError("Lambert: 위치벡터 크기가 0에 가깝습니다.")

    # --- 전이 각도 ---
    cos_dnu = float(np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0))
    cross = np.cross(r1_vec, r2_vec)
    cross_z = float(cross[2])

    if prograde:
        dnu = np.arccos(cos_dnu) if cross_z >= 0.0 else (2.0 * np.pi - np.arccos(cos_dnu))
    else:
        dnu = np.arccos(cos_dnu) if cross_z < 0.0 else (2.0 * np.pi - np.arccos(cos_dnu))

    sin_dnu = np.sin(dnu)

    # 반평행 (180°) 특이점 체크
    if abs(1.0 - cos_dnu) < 1e-9:
        raise ValueError(
            "Lambert: r1, r2가 반평행(180° 전이)으로 전이면이 정의되지 않습니다. "
            "Hohmann 솔버를 사용하거나 출발 true anomaly를 변경하세요."
        )

    # --- A 계수 (Curtis Eq. 5.35) ---
    A = sin_dnu * np.sqrt(r1 * r2 / (1.0 - cos_dnu))

    # --- TOF 방정식 F(z) = 0, z = α χ² ---
    def _y(z: float) -> float:
        c2v = _c2(z)
        if c2v < 1e-12:
            return r1 + r2
        c3v = _c3(z)
        return r1 + r2 + A * (z * c3v - 1.0) / np.sqrt(c2v)

    def _tof_eq(z: float) -> float:
        c2v = _c2(z)
        c3v = _c3(z)
        y = _y(z)
        if y <= 0.0 or c2v < 1e-12:
            return -tof
        chi = np.sqrt(y / c2v)
        return (chi ** 3 * c3v + A * np.sqrt(y)) / np.sqrt(mu) - tof

    # --- 브래킷 탐색 ---
    z_upper = 4.0 * np.pi ** 2 - 1e-4  # 타원 단일 혁명 상한
    z_lower = -4.0 * np.pi ** 2

    # z_lower 조정: y > 0 보장
    for _ in range(400):
        if _y(z_lower) > 0.0:
            break
        z_lower += 0.1
    else:
        raise ValueError("Lambert: y > 0 조건을 만족하는 z_lower를 찾지 못했습니다.")

    f_l = _tof_eq(z_lower)
    f_u = _tof_eq(z_upper)

    # 부호 변환이 없으면 범위를 확장하여 재시도
    if f_l * f_u > 0:
        z_upper_ext = 20.0 * np.pi ** 2
        f_u_ext = _tof_eq(z_upper_ext)
        if f_l * f_u_ext <= 0:
            z_upper = z_upper_ext
            f_u = f_u_ext
        else:
            raise ValueError(
                f"Lambert: tof={tof:.1f}s에 대한 해가 없거나 탐색 범위를 벗어났습니다. "
                f"f_lower={f_l:.4f}, f_upper={f_u:.4f}"
            )

    z_sol = brentq(_tof_eq, z_lower, z_upper, xtol=1e-8, rtol=1e-10, maxiter=300)

    # --- 속도벡터 복원 (Lagrange 계수) ---
    c2v = _c2(z_sol)
    c3v = _c3(z_sol)
    y_sol = _y(z_sol)
    chi = np.sqrt(y_sol / c2v)

    f_lag = 1.0 - y_sol / r1
    g_lag = A * np.sqrt(y_sol / mu)
    g_dot = 1.0 - y_sol / r2

    v1 = (r2_vec - f_lag * r1_vec) / g_lag
    v2 = (g_dot * r2_vec - r1_vec) / g_lag

    return v1, v2
