"""비교 지표 계산 모듈.

새 지표를 추가하려면 compute_metrics()의 반환 dict에 키를 추가하면 된다.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.integrate import trapezoid

if TYPE_CHECKING:
    from .result import BenchmarkResult
    from ..types import TransferConfig


def compute_metrics(result: "BenchmarkResult", config: "TransferConfig") -> dict:
    """BenchmarkResult에서 비교 지표를 산출한다.

    Parameters
    ----------
    result : BenchmarkResult
    config : TransferConfig

    Returns
    -------
    dict with keys:
        dv_total     [km/s]     총 Δv
        cost_l1      [km/s]     ∫||u||dt  (연속: 등가 Δv)
        cost_l2      [km²/s³]  ∫||u||²dt
        tof          [s]        비행시간
        tof_norm     [-]        tof / T₀ (초기 궤도 주기 대비)
        n_peaks      int        피크 수 (연속 추력만; 임펄스는 impulse 수)
        profile_class int       0/1/2  (연속만; 임펄스는 None)
        dv1          [km/s]     첫 번째 기동 크기 (임펄스만)
        dv2          [km/s]     두 번째 기동 크기 (임펄스만, 해당되면)
        dv_total_opt [km/s]     최적 경사 분리 Δv (Hohmann만; 그 외 NaN)
    """
    m: dict = {}

    t = result.t
    u = result.u

    # ── 비행시간 ────────────────────────────────────────────────────────
    tof = float(t[-1] - t[0]) if len(t) >= 2 else result.metrics.get("tof", 0.0)
    m["tof"] = tof
    m["tof_norm"] = tof / config.T0 if config.T0 > 0 else np.nan

    # ── 추력 크기 배열 ──────────────────────────────────────────────────
    u_mag = np.linalg.norm(u, axis=0)  # shape (N,)

    # ── L1 비용: ∫||u||dt (등가 Δv) ────────────────────────────────────
    if len(t) >= 2:
        cost_l1 = float(trapezoid(u_mag, t))
    else:
        cost_l1 = 0.0
    m["cost_l1"] = cost_l1

    # ── L2 비용: ∫||u||²dt ─────────────────────────────────────────────
    if len(t) >= 2:
        cost_l2 = float(trapezoid(u_mag ** 2, t))
    else:
        cost_l2 = 0.0
    m["cost_l2"] = cost_l2 if not result.is_impulsive else np.nan

    # ── 총 Δv ──────────────────────────────────────────────────────────
    if result.is_impulsive and result.impulses:
        dv_total = float(sum(imp["dv"] for imp in result.impulses))
    else:
        dv_total = cost_l1  # 연속 추력의 등가 Δv
    m["dv_total"] = dv_total

    # ── 개별 기동 Δv (임펄스 기법) ──────────────────────────────────────
    if result.impulses:
        for idx, imp in enumerate(result.impulses):
            m[f"dv{idx + 1}"] = float(imp["dv"])
    else:
        m["dv1"] = np.nan
        m["dv2"] = np.nan

    # ── 최적 경사 분리 Δv (Hohmann 솔버가 extra에 저장해 둔 값) ─────────
    m["dv_total_opt"] = result.extra.get("dv_total_opt", np.nan)

    # ── 피크 수 / 분류 ──────────────────────────────────────────────────
    if result.is_impulsive:
        m["n_peaks"] = len(result.impulses)
        m["profile_class"] = None
    else:
        # collocation 결과: extra에 저장된 값 참조
        m["n_peaks"] = result.extra.get("n_peaks", np.nan)
        m["profile_class"] = result.extra.get("profile_class", None)

    return m


def add_metric(result: "BenchmarkResult", key: str, value) -> None:
    """BenchmarkResult.metrics에 사용자 정의 지표를 추가한다.

    Parameters
    ----------
    result : BenchmarkResult
    key : str       지표 이름
    value : any     지표 값
    """
    result.metrics[key] = value
