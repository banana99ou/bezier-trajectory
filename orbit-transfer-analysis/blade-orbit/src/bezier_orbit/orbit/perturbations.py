"""고차 섭동 모델 (Level 1–2).

이론: docs/reports/002_orbital_dynamics/

Level 0 (J2): dynamics.py에 내장.
Level 1: J3–J6, 대기항력 (본 모듈).
Level 2: 태양복사압(SRP), 3체 섭동 (본 모듈).

모든 입출력은 정규화된 변수 (mu* = 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..normalize import R_EARTH

# ── 물리 상수 ───────────────────────────────────────────────────
# 구면조화 계수 (대역 조화만)
J_COEFFS: dict[int, float] = {
    2: 1.08263e-3,
    3: -2.5324e-6,
    4: -1.6204e-6,
    5: -2.2730e-7,
    6: 5.4060e-7,
}

# 지수 대기 모델 파라미터 (고도 구간별)
# (h_base [km], rho_base [kg/m^3], H_scale [km])
_EXPO_ATM_TABLE: list[tuple[float, float, float]] = [
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.158e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
]

# 태양 복사압
P_SUN: float = 4.56e-6  # N/m^2 at 1 AU
AU_KM: float = 1.496e8  # km

# 3체 중력 상수
MU_SUN: float = 1.327124e11  # km^3/s^2
MU_MOON: float = 4902.8      # km^3/s^2

# 지구 자전 각속도 [rad/s]
OMEGA_EARTH: float = 7.2921159e-5


# ═══════════════════════════════════════════════════════════════
#  Level 1: 고차 비구형 섭동 (J3–J6)
# ═══════════════════════════════════════════════════════════════
def accel_jn(
    r: NDArray,
    n: int,
    *,
    Re_star: float = 1.0,
    Jn: float | None = None,
) -> NDArray:
    """Jn 대역 조화 섭동 가속도 (정규화).

    포텐셜 미분으로 계산:
    U_Jn = -(mu/r) * Jn * (Re/r)^n * P_n(sin φ)
    a_Jn = -∇U_Jn

    Parameters
    ----------
    r : (3,) 위치 벡터 (정규화).
    n : int 차수 (2–6).
    Re_star : 정규화된 지구 반경.
    Jn : Jn 계수. None이면 기본값 사용.
    """
    if Jn is None:
        Jn = J_COEFFS[n]

    x, y, z = r
    r_mag = np.linalg.norm(r)
    r2 = r_mag**2

    # sin(φ) = z/r
    s = z / r_mag
    Re_r = Re_star / r_mag  # (Re*/r*)

    if n == 2:
        # J2 — dynamics.py에도 있지만 통일 인터페이스 제공
        # a_J2 = (3 J2 Re*^2) / (2 r^5) · [...]
        r5 = r_mag**5
        coeff = 1.5 * Jn * Re_star**2 / r5
        s2 = s**2
        return coeff * np.array([
            x * (5.0 * s2 - 1.0),
            y * (5.0 * s2 - 1.0),
            z * (5.0 * s2 - 3.0),
        ])
    elif n == 3:
        # J3: 적도 근방(sin φ ≈ 0)에서 해석적 공식의 1/sin(φ) 특이점 회피.
        # 수치 기울기로 안전하게 계산.
        return _accel_jn_numerical(r, n, Re_star, Jn)
    elif n == 4:
        coeff = Jn * Re_r**4 / (8.0 * r2)
        s2 = s**2
        s4 = s2**2
        return coeff * np.array([
            x * (3.0 - 42.0 * s2 + 63.0 * s4),
            y * (3.0 - 42.0 * s2 + 63.0 * s4),
            z * (15.0 - 70.0 * s2 + 63.0 * s4),
        ])
    else:
        # J5, J6: 수치 기울기 (일반적 구현)
        return _accel_jn_numerical(r, n, Re_star, Jn)


def _accel_jn_numerical(
    r: NDArray, n: int, Re_star: float, Jn: float,
) -> NDArray:
    """수치 기울기로 Jn 가속도 계산 (범용)."""
    eps = 1e-8
    a = np.zeros(3)
    for j in range(3):
        rp = r.copy(); rp[j] += eps
        rm = r.copy(); rm[j] -= eps
        Up = _potential_jn(rp, n, Re_star, Jn)
        Um = _potential_jn(rm, n, Re_star, Jn)
        a[j] = -(Up - Um) / (2.0 * eps)
    return a


def _potential_jn(r: NDArray, n: int, Re_star: float, Jn: float) -> float:
    """Jn 섭동 포텐셜 U_Jn = -(1/r) * Jn * (Re/r)^n * P_n(sin φ)."""
    r_mag = np.linalg.norm(r)
    sin_phi = r[2] / r_mag
    Pn = _legendre_p(n, sin_phi)
    return -(1.0 / r_mag) * Jn * (Re_star / r_mag)**n * Pn


def _legendre_p(n: int, x: float) -> float:
    """르장드르 다항식 P_n(x)."""
    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        P_prev, P_curr = 1.0, x
        for k in range(2, n + 1):
            P_next = ((2 * k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev, P_curr = P_curr, P_next
        return P_curr


def accel_jn_sum(
    r: NDArray,
    *,
    Re_star: float = 1.0,
    max_degree: int = 6,
) -> NDArray:
    """J3부터 max_degree까지의 합산 섭동 가속도."""
    a = np.zeros(3)
    for n in range(3, max_degree + 1):
        if n in J_COEFFS:
            a += accel_jn(r, n, Re_star=Re_star)
    return a


# ═══════════════════════════════════════════════════════════════
#  Level 1: 대기항력
# ═══════════════════════════════════════════════════════════════
def exponential_density(h_km: float) -> float:
    """지수 대기 밀도 모델.

    Parameters
    ----------
    h_km : float
        지심 고도 [km] (물리 단위).

    Returns
    -------
    rho : float
        대기 밀도 [kg/m^3].
    """
    if h_km < 0.0:
        return _EXPO_ATM_TABLE[0][1]
    if h_km > 1000.0:
        h0, rho0, Hs = _EXPO_ATM_TABLE[-1]
        return rho0 * np.exp(-(h_km - h0) / Hs)

    # 해당 고도 구간 찾기
    for i in range(len(_EXPO_ATM_TABLE) - 1, -1, -1):
        if h_km >= _EXPO_ATM_TABLE[i][0]:
            h0, rho0, Hs = _EXPO_ATM_TABLE[i]
            return rho0 * np.exp(-(h_km - h0) / Hs)

    h0, rho0, Hs = _EXPO_ATM_TABLE[0]
    return rho0 * np.exp(-(h_km - h0) / Hs)


def accel_drag(
    r: NDArray,
    v: NDArray,
    *,
    Re_star: float,
    DU: float,
    TU: float,
    Cd_A_over_m: float = 0.01,  # m^2/kg (위성 기본값)
    omega_earth_star: float | None = None,
) -> NDArray:
    """대기항력 가속도 (정규화).

    a_drag = -0.5 * (Cd*A/m) * ρ * v_rel * v_rel_vec

    Parameters
    ----------
    r, v : (3,) 정규화 위치/속도.
    Re_star : 정규화된 지구 반경.
    DU, TU : 정규화 기준량 (물리 단위 밀도 변환용).
    Cd_A_over_m : 항력 계수 × 면적/질량 [m^2/kg].
    omega_earth_star : 정규화된 지구 자전 각속도. None이면 자동 계산.
    """
    # 고도 (물리 단위)
    r_mag = np.linalg.norm(r)
    h_km = (r_mag - Re_star) * DU  # km

    if h_km > 1000.0:
        return np.zeros(3)

    # 대기 밀도 (물리 단위)
    rho = exponential_density(h_km)  # kg/m^3

    # 대기 상대 속도 (지구 자전 고려)
    if omega_earth_star is None:
        omega_earth_star = OMEGA_EARTH * TU

    v_atm = np.array([-omega_earth_star * r[1], omega_earth_star * r[0], 0.0])
    v_rel = v - v_atm
    v_rel_mag = np.linalg.norm(v_rel)

    if v_rel_mag < 1e-15:
        return np.zeros(3)

    # 단위 변환: rho [kg/m^3] → 정규화 단위
    # a_drag [km/s^2] = 0.5 * Cd_A_m * rho * v_rel^2 / (정규화 → km 변환)
    # rho * DU(km→m)^3 = rho * (DU*1e3)^3 ← 하지만 정규화 체계에서 직접 계산
    #
    # 물리 단위: a = 0.5 * (Cd*A/m) * rho * v_rel * v_rel_vec  [m/s^2]
    # → [km/s^2] = a * 1e-3
    # → 정규화: a* = a_km / AU = a_km * TU^2 / DU
    VU = DU / TU  # km/s
    v_rel_phys = v_rel_mag * VU  # km/s → m/s = * 1e3
    v_rel_phys_ms = v_rel_phys * 1e3  # m/s

    # a [m/s^2]
    a_phys = 0.5 * Cd_A_over_m * rho * v_rel_phys_ms  # m/s^2 (scalar)

    # → 정규화 가속도
    AU = DU / TU**2  # km/s^2
    a_star = a_phys * 1e-3 / AU  # 1e-3: m/s^2 → km/s^2

    return -a_star * v_rel / v_rel_mag


# ═══════════════════════════════════════════════════════════════
#  Level 2: 태양복사압 (SRP)
# ═══════════════════════════════════════════════════════════════
def accel_srp(
    r: NDArray,
    r_sun: NDArray,
    *,
    DU: float,
    TU: float,
    Cr_A_over_m: float = 0.01,  # m^2/kg
) -> NDArray:
    """태양복사압 가속도 (정규화).

    a_SRP = -P_sun * Cr * A/m * r_hat_sun

    Parameters
    ----------
    r : (3,) 우주비행체 위치 (정규화).
    r_sun : (3,) 태양 위치 (정규화, ECI).
    DU, TU : 정규화 기준량.
    Cr_A_over_m : 복사압계수 × 면적/질량 [m^2/kg].
    """
    d = r - r_sun  # 태양→위성 벡터
    d_mag = np.linalg.norm(d)
    d_hat = d / d_mag

    # 1 AU 기준 태양복사압 → 실제 거리 보정
    d_phys = d_mag * DU  # km
    P_actual = P_SUN * (AU_KM / d_phys)**2  # N/m^2

    # 물리 가속도 [m/s^2]
    a_phys = Cr_A_over_m * P_actual  # m/s^2

    # 정규화
    AU_unit = DU / TU**2  # km/s^2
    a_star = a_phys * 1e-3 / AU_unit

    return -a_star * d_hat


# ═══════════════════════════════════════════════════════════════
#  Level 2: 3체 섭동
# ═══════════════════════════════════════════════════════════════
def accel_third_body(
    r: NDArray,
    r_body: NDArray,
    *,
    mu_body_star: float,
) -> NDArray:
    """3체 섭동 가속도 (정규화).

    a_3b = mu_body * ( (r_body - r) / |r_body - r|^3  -  r_body / |r_body|^3 )

    Parameters
    ----------
    r : (3,) 우주비행체 위치 (정규화).
    r_body : (3,) 제3체 위치 (정규화, 지구 중심).
    mu_body_star : 제3체 정규화 중력 상수.
    """
    d = r_body - r
    d_mag = np.linalg.norm(d)
    rb_mag = np.linalg.norm(r_body)

    return mu_body_star * (d / d_mag**3 - r_body / rb_mag**3)


# ═══════════════════════════════════════════════════════════════
#  통합 섭동 계산기
# ═══════════════════════════════════════════════════════════════
def compute_perturbations(
    r: NDArray,
    v: NDArray,
    *,
    level: int = 0,
    Re_star: float = 1.0,
    DU: float = R_EARTH,
    TU: float = 1.0,
    max_jn_degree: int = 6,
    Cd_A_over_m: float = 0.01,
    Cr_A_over_m: float = 0.01,
    r_sun: NDArray | None = None,
    r_moon: NDArray | None = None,
    mu_sun_star: float = 0.0,
    mu_moon_star: float = 0.0,
) -> NDArray:
    """레벨별 섭동 합산.

    Level 0: 없음 (J2는 dynamics.py에서 처리)
    Level 1: J3–J6 + 대기항력
    Level 2: Level 1 + SRP + 3체
    """
    a = np.zeros(3)

    if level >= 1:
        a += accel_jn_sum(r, Re_star=Re_star, max_degree=max_jn_degree)
        a += accel_drag(r, v, Re_star=Re_star, DU=DU, TU=TU,
                        Cd_A_over_m=Cd_A_over_m)

    if level >= 2:
        if r_sun is not None:
            a += accel_srp(r, r_sun, DU=DU, TU=TU, Cr_A_over_m=Cr_A_over_m)
            a += accel_third_body(r, r_sun, mu_body_star=mu_sun_star)
        if r_moon is not None:
            a += accel_third_body(r, r_moon, mu_body_star=mu_moon_star)

    return a
