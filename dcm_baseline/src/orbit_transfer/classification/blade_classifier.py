"""BLADE 세그먼트 구조 기반 피크/코스팅 분류.

BLADE-SCP의 세그먼트별 제어점 norm을 이용하여 peak/coast를 직접 식별한다.
기존 topological persistence 기반 사후 피크 탐지(peak_detection.py)와 달리,
최적화 결과의 구조적 정보에서 분류를 추출하므로 임계값 민감도가 낮다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from orbit_transfer.config import LGL_NODES_PEAK, LGL_NODES_COAST, MIN_PHASE_FRACTION
from orbit_transfer.classification.classifier import (
    classify_profile,
    _merge_short_phases,
    _enforce_continuity,
)


def blade_classify_segments(
    p_segments: list[NDArray],
    threshold: float = 0.05,
) -> tuple[list[str], int]:
    """세그먼트별 추력 norm → peak/coast 분류.

    Parameters
    ----------
    p_segments : list of (n+1, 3) arrays
        BLADE 세그먼트별 Bernstein 제어점.
    threshold : float
        coast 판정 임계값 (최대 norm 대비 비율).
        ℓ₁ 정규화 사용 시 0.01까지 낮출 수 있다.

    Returns
    -------
    seg_types : list of str
        각 세그먼트의 'peak' 또는 'coast'.
    n_peaks : int
        연속 peak 세그먼트 그룹의 수.
    """
    norms = [float(np.max(np.linalg.norm(pk, axis=1))) for pk in p_segments]
    max_norm = max(norms) if norms else 0.0

    if max_norm < 1e-15:
        return ["coast"] * len(p_segments), 0

    seg_types = [
        "coast" if nm < threshold * max_norm else "peak"
        for nm in norms
    ]
    n_peaks = _count_peak_groups(seg_types)
    return seg_types, n_peaks


def _count_peak_groups(seg_types: list[str]) -> int:
    """연속 peak 세그먼트를 하나의 피크 그룹으로 병합하여 개수를 센다."""
    count = 0
    in_peak = False
    for st in seg_types:
        if st == "peak":
            if not in_peak:
                count += 1
                in_peak = True
        else:
            in_peak = False
    return count


def blade_phase_structure(
    seg_types: list[str],
    K: int,
    t_f: float,
    n_nodes_peak: int = LGL_NODES_PEAK,
    n_nodes_coast: int = LGL_NODES_COAST,
) -> list[dict]:
    """BLADE 세그먼트 분류 → Multi-Phase LGL 구조 매핑.

    연속된 동일 유형(peak/coast) 세그먼트를 하나의 phase로 묶고,
    각 phase의 시간 경계와 LGL 노드 수를 결정한다.

    Parameters
    ----------
    seg_types : list of str
        각 세그먼트의 'peak' 또는 'coast' (blade_classify_segments 출력).
    K : int
        세그먼트 수.
    t_f : float
        전이 시간 [s].
    n_nodes_peak : int
        peak phase의 LGL 노드 수.
    n_nodes_coast : int
        coast phase의 LGL 노드 수.

    Returns
    -------
    phases : list of dict
        각 dict: {'t_start', 't_end', 'n_nodes', 'type'}.
        determine_phase_structure()와 동일한 형식.
    """
    if K == 0 or not seg_types:
        return [{
            "t_start": 0.0, "t_end": t_f,
            "n_nodes": n_nodes_peak, "type": "peak",
        }]

    dt_seg = t_f / K  # 균등 세그먼트 길이

    # 연속된 동일 유형 세그먼트를 하나의 phase로 병합
    phases: list[dict] = []
    group_start = 0
    current_type = seg_types[0]

    for k in range(1, K):
        if seg_types[k] != current_type:
            phases.append({
                "t_start": group_start * dt_seg,
                "t_end": k * dt_seg,
                "n_nodes": n_nodes_peak if current_type == "peak" else n_nodes_coast,
                "type": current_type,
            })
            group_start = k
            current_type = seg_types[k]

    # 마지막 그룹
    phases.append({
        "t_start": group_start * dt_seg,
        "t_end": t_f,
        "n_nodes": n_nodes_peak if current_type == "peak" else n_nodes_coast,
        "type": current_type,
    })

    # 짧은 phase 병합 + 연속성 강제 (기존 로직 재사용)
    phases = _merge_short_phases(phases, t_f)
    phases = _enforce_continuity(phases, t_f)

    return phases
