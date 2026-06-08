"""공유 데이터 클래스 정의."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TransferConfig:
    """궤도전이 문제 구성 파라미터.

    Attributes:
        h0: 초기 궤도 고도 [km]
        delta_a: 장반경 변화 [km]
        delta_i: 경사각 변화 [deg]
        T_max_normed: 전이시간 상한 / T0 (초기 궤도 주기 대비)
        e0: 초기 이심률
        ef: 최종 이심률
        u_max: 추력 가속도 상한 [km/s^2]
        h_min: 최소 허용 고도 [km]
    """

    h0: float
    delta_a: float
    delta_i: float
    T_max_normed: float
    e0: float = 0.0
    ef: float = 0.0
    u_max: float = 0.01
    h_min: float = 150.0

    @property
    def a0(self) -> float:
        """초기 장반경 [km]."""
        from .constants import R_E
        return R_E + self.h0

    @property
    def af(self) -> float:
        """최종 장반경 [km]."""
        return self.a0 + self.delta_a

    @property
    def i0(self) -> float:
        """초기 경사각 [rad]."""
        return 0.0

    @property
    def if_(self) -> float:
        """최종 경사각 [rad]."""
        return np.radians(self.delta_i)

    @property
    def T0(self) -> float:
        """초기 궤도 주기 [s]."""
        from .constants import MU_EARTH
        return 2.0 * np.pi * np.sqrt(self.a0**3 / MU_EARTH)

    @property
    def T_max(self) -> float:
        """전이시간 상한 [s]."""
        return self.T_max_normed * self.T0

    @property
    def T_min(self) -> float:
        """전이시간 하한 [s]."""
        from .config import T_MIN_FACTOR
        return T_MIN_FACTOR * self.T0


@dataclass
class TrajectoryResult:
    """궤적 최적화 결과.

    Attributes:
        converged: 수렴 여부
        cost: 최적 비용 (L2 적분)
        t: 시간 배열 [s], shape (N,)
        x: 상태 배열 [km, km/s], shape (6, N)
        u: 제어 배열 [km/s^2], shape (3, N)
        nu0: 출발 true anomaly [rad]
        nuf: 도착 true anomaly [rad]
        n_peaks: 피크 개수
        profile_class: 분류 (0=unimodal, 1=bimodal, 2=multimodal)
        pass1_cost: Pass 1 비용 (있는 경우)
        solver_stats: 솔버 통계 정보
    """

    converged: bool
    cost: float
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    nu0: float
    nuf: float
    n_peaks: int
    profile_class: int
    T_f: float = 0.0
    pass1_cost: float | None = None
    solver_stats: dict | None = field(default_factory=dict)
