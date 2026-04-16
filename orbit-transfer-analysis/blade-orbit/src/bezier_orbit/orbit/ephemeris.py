"""저정밀 태양·달 천체력 (Low-precision analytical ephemeris).

SRP 및 3체 섭동 (Level 2) 계산에 필요한 태양/달의 ECI 위치를 제공한다.
외부 라이브러리(SPICE 등) 의존 없이, Meeus 알고리즘 기반 해석 공식을 사용.

정밀도:
  - 태양: ~0.01° (~1 arcmin), 충분히 SRP 계산에 적합
  - 달:   ~0.1° (~6 arcmin), 3체 섭동 계산에 적합

참고: Jean Meeus, "Astronomical Algorithms" (2nd ed., 1998), Ch. 25, 47.

모든 출력은 ECI J2000 좌표계 [km] 또는 정규화 단위.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..normalize import CanonicalUnits, MU_EARTH


# ── 물리 상수 ─────────────────────────────────────────────────
AU_KM: float = 149_597_870.7  # 1 AU [km]
MU_SUN: float = 1.327_124_4e11  # km³/s²
MU_MOON: float = 4_902.8  # km³/s²
OBLIQUITY_J2000: float = math.radians(23.4393)  # 황도경사각 [rad]


# ── Julian Date 유틸리티 ──────────────────────────────────────
def jd_to_centuries(jd: float) -> float:
    """Julian Date → J2000 기준 Julian centuries (T)."""
    return (jd - 2_451_545.0) / 36_525.0


# ── 태양 위치 (Meeus Ch. 25) ─────────────────────────────────
def sun_position_eci(jd: float) -> NDArray:
    """저정밀 태양 위치 (ECI J2000) [km].

    Meeus 알고리즘: 기하학적 황경 + 장동 보정.
    정밀도 ~0.01° (1991–2100).

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    r_sun : (3,) ECI 좌표 [km].
    """
    T = jd_to_centuries(jd)

    # 태양의 평균 경도 (deg)
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    L0 = L0 % 360.0

    # 태양의 평균 이각 (deg)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    M_rad = math.radians(M % 360.0)

    # 중심차 (equation of center)
    C = (1.914602 - 0.004817 * T) * math.sin(M_rad) \
        + 0.019993 * math.sin(2 * M_rad) \
        + 0.000290 * math.sin(3 * M_rad)

    # 태양의 진황경 (ecliptic longitude)
    lon_sun = math.radians((L0 + C) % 360.0)

    # 태양 거리 (AU)
    e = 0.016708634 - 0.000042037 * T
    R_au = 1.000001018 * (1 - e**2) / (1 + e * math.cos(M_rad))
    R_km = R_au * AU_KM

    # 황도 좌표 → ECI (황도경사각으로 회전, 황위 ≈ 0 가정)
    eps = OBLIQUITY_J2000
    x = R_km * math.cos(lon_sun)
    y = R_km * math.sin(lon_sun) * math.cos(eps)
    z = R_km * math.sin(lon_sun) * math.sin(eps)

    return np.array([x, y, z])


# ── 달 위치 (Meeus Ch. 47, 주요 항) ─────────────────────────
def moon_position_eci(jd: float) -> NDArray:
    """저정밀 달 위치 (ECI J2000) [km].

    Meeus Ch. 47의 주요 6개 항만 사용.
    정밀도 ~0.1° (수천 km 오차, 지심거리 ~384,400 km 대비 ~1%).

    Parameters
    ----------
    jd : float
        Julian Date.

    Returns
    -------
    r_moon : (3,) ECI 좌표 [km].
    """
    T = jd_to_centuries(jd)

    # 기본 각도 (deg)
    # L': 달의 평균 경도
    Lp = 218.3165 + 481267.8813 * T
    # D: 평균 이각 (달-태양)
    D = 297.8502 + 445267.1115 * T
    # M: 태양의 평균 이각
    M = 357.5291 + 35999.0503 * T
    # M': 달의 평균 이각
    Mp = 134.9634 + 477198.8676 * T
    # F: 달의 위도 인수
    F = 93.2721 + 483202.0175 * T

    # 라디안 변환
    Lp_r = math.radians(Lp % 360)
    D_r = math.radians(D % 360)
    M_r = math.radians(M % 360)
    Mp_r = math.radians(Mp % 360)
    F_r = math.radians(F % 360)

    # 황경 섭동 (주요 6항, arcsec → deg)
    dl = (6288774 * math.sin(Mp_r)
          + 1274027 * math.sin(2 * D_r - Mp_r)
          + 658314 * math.sin(2 * D_r)
          + 213618 * math.sin(2 * Mp_r)
          - 185116 * math.sin(M_r)
          - 114332 * math.sin(2 * F_r)) * 1e-6  # 10⁻⁶ deg

    lon_moon = math.radians((Lp + dl) % 360)

    # 황위 섭동 (주요 4항)
    db = (5128122 * math.sin(F_r)
          + 280602 * math.sin(Mp_r + F_r)
          + 277693 * math.sin(Mp_r - F_r)
          + 173237 * math.sin(2 * D_r - F_r)) * 1e-6

    lat_moon = math.radians(db)

    # 거리 (km) — 주요 4항
    dr = (-20905355 * math.cos(Mp_r)
          - 3699111 * math.cos(2 * D_r - Mp_r)
          - 2955968 * math.cos(2 * D_r)
          - 569925 * math.cos(2 * Mp_r)) * 1e-3  # 10⁻³ km → km 보정 제거

    R_km = 385000.56 + dr * 1e-3  # 기본 거리 + 보정 (mm→km)

    # 황도 좌표 → ECI
    eps = OBLIQUITY_J2000
    x_ec = R_km * math.cos(lat_moon) * math.cos(lon_moon)
    y_ec = R_km * math.cos(lat_moon) * math.sin(lon_moon)
    z_ec = R_km * math.sin(lat_moon)

    # 황도 → 적도 (x축 회전)
    x = x_ec
    y = y_ec * math.cos(eps) - z_ec * math.sin(eps)
    z = y_ec * math.sin(eps) + z_ec * math.cos(eps)

    return np.array([x, y, z])


# ── SCP 인터페이스: τ → r*(τ) 변환 함수 팩토리 ───────────────
def make_body_func(
    epoch_jd: float,
    t_f_sec: float,
    body: str,
    cu: CanonicalUnits,
) -> Callable[[float], NDArray]:
    """SCP용 천체 위치 함수 생성.

    반환되는 함수는 τ ∈ [0,1]을 받아 정규화된 천체 위치를 반환한다.

    Parameters
    ----------
    epoch_jd : float
        비행 시작 시각 (Julian Date).
    t_f_sec : float
        비행시간 [초] (물리 단위).
    body : str
        'sun' 또는 'moon'.
    cu : CanonicalUnits
        정규화 단위.

    Returns
    -------
    func : Callable[[float], NDArray]
        τ → r_body*(τ) (정규화 좌표).
    """
    if body == "sun":
        pos_func = sun_position_eci
    elif body == "moon":
        pos_func = moon_position_eci
    else:
        raise ValueError(f"Unknown body: {body!r}. Use 'sun' or 'moon'.")

    DU = cu.DU

    def body_pos(tau: float) -> NDArray:
        t_sec = tau * t_f_sec
        jd = epoch_jd + t_sec / 86400.0
        r_km = pos_func(jd)
        return r_km / DU  # 정규화

    return body_pos


def compute_mu_star(
    body: str,
    cu: CanonicalUnits,
) -> float:
    """천체의 정규화된 중력상수 μ_body* = μ_body / (DU³/TU²).

    Parameters
    ----------
    body : str
        'sun' 또는 'moon'.
    cu : CanonicalUnits
        정규화 단위.

    Returns
    -------
    mu_star : float
        정규화된 중력상수.
    """
    if body == "sun":
        mu_phys = MU_SUN
    elif body == "moon":
        mu_phys = MU_MOON
    else:
        raise ValueError(f"Unknown body: {body!r}")

    # mu* = mu_body / mu_earth (정규화 체계에서 mu_earth* = 1)
    return mu_phys / cu.mu
