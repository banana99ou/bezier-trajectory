"""적응형 궤도역학 정규화 (Canonical Units).

이론: docs/reports/001_nondimensionalization/

초기 궤도 반장축 a0를 기준으로 DU/TU/VU를 정의하여
모든 궤도 변수를 O(1) 스케일로 유지한다.
"""

from __future__ import annotations

import dataclasses
import math

# ── 물리 상수 (km, s 단위계) ──────────────────────────────────
MU_EARTH: float = 398_600.4418  # km^3/s^2
R_EARTH: float = 6_378.137  # km
J2_EARTH: float = 1.08263e-3

import numpy as np
from numpy.typing import NDArray


@dataclasses.dataclass(frozen=True, slots=True)
class CanonicalUnits:
    """정규화 기준량.

    Parameters
    ----------
    a0 : float
        기준 거리 (보통 초기 궤도 반장축), km.
    mu : float
        중력 상수, km^3/s^2.
    """

    a0: float
    mu: float = MU_EARTH

    # ── 유도량 (cached properties) ──────────────────────────────
    @property
    def DU(self) -> float:
        """거리 단위 [km]."""
        return self.a0

    @property
    def TU(self) -> float:
        """시간 단위 [s]."""
        return math.sqrt(self.a0**3 / self.mu)

    @property
    def VU(self) -> float:
        """속도 단위 [km/s]."""
        return math.sqrt(self.mu / self.a0)

    @property
    def AU(self) -> float:
        """가속도 단위 [km/s^2]."""
        return self.mu / self.a0**2

    # ── 물리 → 정규화 ──────────────────────────────────────────
    def nondim_pos(self, r: NDArray) -> NDArray:
        """위치 벡터 정규화: r* = r / DU."""
        return np.asarray(r) / self.DU

    def nondim_vel(self, v: NDArray) -> NDArray:
        """속도 벡터 정규화: v* = v / VU."""
        return np.asarray(v) / self.VU

    def nondim_time(self, t: float | NDArray) -> float | NDArray:
        """시간 정규화: t* = t / TU."""
        return np.asarray(t) / self.TU

    def nondim_accel(self, a: NDArray) -> NDArray:
        """가속도 벡터 정규화: a* = a / AU."""
        return np.asarray(a) / self.AU

    def nondim_state(self, x: NDArray) -> NDArray:
        """상태벡터 [r; v] 정규화. x shape: (6,) or (6, ...)."""
        x = np.asarray(x, dtype=float).copy()
        x[:3] = x[:3] / self.DU
        x[3:6] = x[3:6] / self.VU
        return x

    # ── 정규화 → 물리 ──────────────────────────────────────────
    def dim_pos(self, r_star: NDArray) -> NDArray:
        """위치 역변환: r = r* · DU."""
        return np.asarray(r_star) * self.DU

    def dim_vel(self, v_star: NDArray) -> NDArray:
        """속도 역변환: v = v* · VU."""
        return np.asarray(v_star) * self.VU

    def dim_time(self, t_star: float | NDArray) -> float | NDArray:
        """시간 역변환: t = t* · TU."""
        return np.asarray(t_star) * self.TU

    def dim_accel(self, a_star: NDArray) -> NDArray:
        """가속도 역변환: a = a* · AU."""
        return np.asarray(a_star) * self.AU

    def dim_state(self, x_star: NDArray) -> NDArray:
        """상태벡터 역변환."""
        x = np.asarray(x_star, dtype=float).copy()
        x[:3] = x[:3] * self.DU
        x[3:6] = x[3:6] * self.VU
        return x

    # ── 정규화된 물리 상수 ─────────────────────────────────────
    @property
    def mu_star(self) -> float:
        """정규화된 중력 상수 (항상 1.0)."""
        return 1.0

    @property
    def R_earth_star(self) -> float:
        """정규화된 지구 반경."""
        return R_EARTH / self.DU

    # ── 유틸리티 ───────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"CanonicalUnits(a0={self.a0:.3f} km, "
            f"DU={self.DU:.3f} km, "
            f"TU={self.TU:.3f} s, "
            f"VU={self.VU:.6f} km/s)"
        )


def from_orbit(a0: float, mu: float = MU_EARTH) -> CanonicalUnits:
    """초기 궤도 반장축으로부터 정규화 단위 생성.

    Parameters
    ----------
    a0 : float
        초기 궤도 반장축 [km].
    mu : float
        중력 상수 [km^3/s^2].
    """
    if a0 <= 0:
        raise ValueError(f"a0 must be positive, got {a0}")
    return CanonicalUnits(a0=a0, mu=mu)


def standard_earth() -> CanonicalUnits:
    """표준 지구 정규화 (DU = R_earth)."""
    return CanonicalUnits(a0=R_EARTH, mu=MU_EARTH)
