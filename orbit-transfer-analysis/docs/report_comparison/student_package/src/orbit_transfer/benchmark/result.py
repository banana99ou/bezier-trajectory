"""BenchmarkResult 데이터클래스."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class BenchmarkResult:
    """단일 궤도전이 기법의 실행 결과.

    Attributes
    ----------
    method : str
        기법 이름 ('hohmann', 'lambert', 'collocation', 또는 사용자 정의)
    converged : bool
        풀이 성공 여부
    t : ndarray (N,)
        시간 배열 [s]
    x : ndarray (6, N)
        ECI 상태벡터 [km, km/s]
    u : ndarray (3, N)
        추력 가속도 [km/s²]  — 임펄스 기법은 코스팅 구간에서 0
    is_impulsive : bool
        True = 임펄스 기법 (Hohmann, Lambert), False = 연속 추력
    impulses : list of dict
        임펄스 기법일 때 각 기동 정보.
        각 원소: {'t': float, 'dv_vec': ndarray(3,), 'dv': float}
    metrics : dict
        산출된 비교 지표 (compute_metrics() 결과가 채워짐)
        기본 키:
          - dv_total     [km/s]  총 Δv (임펄스 합산 또는 ∫||u||dt)
          - cost_l1      [km/s]  ∫||u||dt
          - cost_l2      [km²/s³] ∫||u||²dt  (연속 추력에서만 의미)
          - tof          [s]     비행시간
          - tof_norm     [-]     tof / T₀
          - n_peaks      int     피크 수 (연속 추력 분류 결과)
          - profile_class int    0=unimodal, 1=bimodal, 2=multimodal
    extra : dict
        사용자가 자유롭게 추가할 수 있는 임의 메타데이터
    """

    method: str
    converged: bool
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    is_impulsive: bool = True
    impulses: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 편의 속성
    # ------------------------------------------------------------------

    @property
    def tof(self) -> float:
        """비행시간 [s]."""
        if len(self.t) >= 2:
            return float(self.t[-1] - self.t[0])
        return self.metrics.get("tof", 0.0)

    @property
    def dv_total(self) -> float:
        """총 Δv [km/s]."""
        return float(self.metrics.get("dv_total", np.nan))

    @property
    def cost_l2(self) -> float:
        """L2 에너지 비용 ∫||u||²dt [km²/s³]."""
        return float(self.metrics.get("cost_l2", np.nan))
