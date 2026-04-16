"""Keplerian 궤도요소 ↔ Cartesian 상태벡터 변환.

고전 궤도요소 (a, e, i, Ω, ω, ν) ↔ ECI Cartesian (r, v) 양방향 변환.

표기법:
- a   : 반장축 [DU]
- e   : 이심률 [-]
- inc : 궤도경사각 [rad]
- raan: 승교점 적경 Ω [rad]
- aop : 근점인수 ω [rad]
- ta  : 진근점이각 ν [rad]
- mu  : 중력 상수 (정규화 시 1.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def keplerian_to_cartesian(
    a: float,
    e: float,
    inc: float,
    raan: float,
    aop: float,
    ta: float,
    mu: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """Keplerian 궤도요소 → ECI Cartesian (r, v).

    Parameters
    ----------
    a : float
        반장축.
    e : float
        이심률 (0 ≤ e < 1 for ellipse).
    inc : float
        궤도경사각 [rad].
    raan : float
        승교점 적경 Ω [rad].
    aop : float
        근점인수 ω [rad].
    ta : float
        진근점이각 ν [rad].
    mu : float
        중력 상수 (정규화 시 1.0).

    Returns
    -------
    r : ndarray, shape (3,)
        위치 벡터.
    v : ndarray, shape (3,)
        속도 벡터.
    """
    # 반통경
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(ta))

    # 근점좌표계(PQW) 위치·속도
    r_pqw = np.array([r_mag * np.cos(ta), r_mag * np.sin(ta), 0.0])
    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(ta), e + np.cos(ta), 0.0])

    # PQW → ECI 회전행렬
    R = _rotation_pqw_to_eci(inc, raan, aop)

    return R @ r_pqw, R @ v_pqw


def cartesian_to_keplerian(
    r: NDArray,
    v: NDArray,
    mu: float = 1.0,
) -> tuple[float, float, float, float, float, float]:
    """ECI Cartesian (r, v) → Keplerian 궤도요소.

    Returns
    -------
    (a, e, inc, raan, aop, ta) : tuple of floats
        각도는 [rad], 범위 [0, 2π).
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # 각운동량 벡터
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # 노드 벡터 (z축과 h의 외적)
    k_hat = np.array([0.0, 0.0, 1.0])
    n = np.cross(k_hat, h)
    n_mag = np.linalg.norm(n)

    # 이심률 벡터
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # 비역학 에너지 → 반장축
    energy = v_mag**2 / 2.0 - mu / r_mag
    if abs(e - 1.0) < 1e-12:
        # 포물선 (실용적으로 거의 없음)
        a = float("inf")
    else:
        a = -mu / (2.0 * energy)

    # 궤도경사각
    inc = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    # 승교점 적경 Ω
    if n_mag < 1e-12:
        # 적도 궤도: Ω 정의 불가 → 0
        raan = 0.0
    else:
        raan = np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0))
        if n[1] < 0.0:
            raan = 2.0 * np.pi - raan

    # 근점인수 ω
    if n_mag < 1e-12:
        # 적도 궤도
        if e > 1e-12:
            # e_vec의 방향으로부터 ω 계산
            aop = np.arctan2(e_vec[1], e_vec[0])
            if aop < 0:
                aop += 2.0 * np.pi
        else:
            aop = 0.0
    else:
        if e < 1e-12:
            # 원 궤도: ω 정의 불가 → 0
            aop = 0.0
        else:
            cos_aop = np.dot(n, e_vec) / (n_mag * e)
            aop = np.arccos(np.clip(cos_aop, -1.0, 1.0))
            if e_vec[2] < 0.0:
                aop = 2.0 * np.pi - aop

    # 진근점이각 ν
    if e < 1e-12:
        # 원 궤도: 위도인수 u = ω + ν 로 대체
        if n_mag < 1e-12:
            # 적도+원 궤도: 경도로 대체
            ta = np.arctan2(r[1], r[0])
        else:
            cos_u = np.dot(n, r) / (n_mag * r_mag)
            u = np.arccos(np.clip(cos_u, -1.0, 1.0))
            if r[2] < 0.0:
                u = 2.0 * np.pi - u
            ta = u - aop
    else:
        cos_ta = np.dot(e_vec, r) / (e * r_mag)
        ta = np.arccos(np.clip(cos_ta, -1.0, 1.0))
        if np.dot(r, v) < 0.0:
            ta = 2.0 * np.pi - ta

    # 각도 정규화 [0, 2π)
    raan = raan % (2.0 * np.pi)
    aop = aop % (2.0 * np.pi)
    ta = ta % (2.0 * np.pi)

    return float(a), float(e), float(inc), float(raan), float(aop), float(ta)


def state_to_vector(r: NDArray, v: NDArray) -> NDArray:
    """(r, v) → 상태벡터 x = [r; v] ∈ R^6."""
    return np.concatenate([np.asarray(r), np.asarray(v)])


def vector_to_state(x: NDArray) -> tuple[NDArray, NDArray]:
    """상태벡터 x = [r; v] → (r, v)."""
    x = np.asarray(x)
    return x[:3].copy(), x[3:6].copy()


# ── 내부 유틸리티 ───────────────────────────────────────────────
def _rotation_pqw_to_eci(inc: float, raan: float, aop: float) -> NDArray:
    """PQW 좌표계 → ECI 좌표계 회전행렬 R = R3(-Ω)·R1(-i)·R3(-ω)."""
    co, so = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(aop), np.sin(aop)

    R = np.array([
        [co * cw - so * ci * sw, -co * sw - so * ci * cw,  so * si],
        [so * cw + co * ci * sw, -so * sw + co * ci * cw, -co * si],
        [si * sw,                 si * cw,                  ci     ],
    ])
    return R
