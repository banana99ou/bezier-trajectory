"""추력 프로파일 분류 및 Multi-Phase 구조 결정 모듈."""

import numpy as np

from orbit_transfer.config import (
    LGL_NODES_PEAK,
    LGL_NODES_COAST,
    MIN_PHASE_FRACTION,
)


def classify_profile(n_peaks):
    """피크 개수로 프로파일 유형을 분류한다.

    Args:
        n_peaks: 피크 개수

    Returns:
        profile_class: 0 (unimodal), 1 (bimodal), 2 (multimodal)
    """
    if n_peaks <= 1:
        return 0
    elif n_peaks == 2:
        return 1
    else:
        return 2


def determine_phase_structure(peak_times, peak_widths, T):
    """피크 구조에 따라 Multi-Phase 구조를 결정한다.

    Args:
        peak_times: 피크 시각 [s], shape (n_peaks,)
        peak_widths: 피크 FWHM [s], shape (n_peaks,)
        T: 전이 시간 [s]

    Returns:
        phases: list of dict, 각 dict:
            {
                't_start': float,  # Phase 시작 시각 [s]
                't_end': float,    # Phase 종료 시각 [s]
                'n_nodes': int,    # LGL 노드 수
                'type': str,       # 'peak' or 'coast'
            }

    Algorithm:
        - Unimodal (1 peak): 단일 phase, N=LGL_NODES_PEAK
        - Bimodal (2 peaks): 3 phases (peak-coast-peak)
        - Multimodal (N peaks): 2N-1 phases (alternating peak-coast)
        - Phase 길이가 MIN_PHASE_FRACTION * T 미만이면 인접 phase와 병합
    """
    peak_times = np.asarray(peak_times, dtype=float)
    peak_widths = np.asarray(peak_widths, dtype=float)
    n_peaks = len(peak_times)

    if n_peaks == 0:
        # 피크 없음: 전체 구간을 단일 coast phase로
        return [
            {
                "t_start": 0.0,
                "t_end": T,
                "n_nodes": LGL_NODES_COAST,
                "type": "coast",
            }
        ]

    if n_peaks == 1:
        # Unimodal: 단일 phase
        return [
            {
                "t_start": 0.0,
                "t_end": T,
                "n_nodes": LGL_NODES_PEAK,
                "type": "peak",
            }
        ]

    # Bimodal / Multimodal: 2N-1 phases (alternating peak-coast)
    phases = []

    for k in range(n_peaks):
        t_pk = peak_times[k]
        w_pk = peak_widths[k]

        # Peak phase 경계
        if k == 0:
            pk_start = 0.0
        else:
            # 이전 피크와 현재 피크 사이의 경계
            pk_start = peak_times[k] - peak_widths[k]

        if k == n_peaks - 1:
            pk_end = T
        else:
            pk_end = t_pk + w_pk

        # 클리핑
        pk_start = max(pk_start, 0.0)
        pk_end = min(pk_end, T)

        # Coast phase (현재 피크와 다음 피크 사이)
        if k < n_peaks - 1:
            coast_start = pk_end
            coast_end = peak_times[k + 1] - peak_widths[k + 1]
            coast_end = max(coast_end, coast_start)
            coast_end = min(coast_end, T)

            phases.append(
                {
                    "t_start": pk_start,
                    "t_end": pk_end,
                    "n_nodes": LGL_NODES_PEAK,
                    "type": "peak",
                }
            )
            phases.append(
                {
                    "t_start": coast_start,
                    "t_end": coast_end,
                    "n_nodes": LGL_NODES_COAST,
                    "type": "coast",
                }
            )
        else:
            # 마지막 피크
            phases.append(
                {
                    "t_start": pk_start,
                    "t_end": pk_end,
                    "n_nodes": LGL_NODES_PEAK,
                    "type": "peak",
                }
            )

    # Phase 병합: 길이가 MIN_PHASE_FRACTION * T 미만인 phase 제거
    phases = _merge_short_phases(phases, T)

    # Phase 연속성 강제: 인접 phase 경계 일치 + 영-길이 제거
    phases = _enforce_continuity(phases, T)

    return phases


def _enforce_continuity(phases, T):
    """Phase 간 연속성을 강제하고 퇴화 phase를 제거한다.

    1. 영-길이(dt ≤ 0) phase 제거
    2. 인접 phase 경계 일치 강제 (gap/overlap 제거)
    3. 첫 phase 시작 = 0, 마지막 phase 종료 = T
    """
    EPS = 1e-10 * T  # 시간 정밀도 임계값

    # 1. 영-길이 phase 제거
    phases = [p for p in phases if p["t_end"] - p["t_start"] > EPS]

    if len(phases) == 0:
        return [{
            "t_start": 0.0, "t_end": T,
            "n_nodes": LGL_NODES_PEAK, "type": "peak",
        }]

    # 2. 인접 phase 경계 일치 강제
    for i in range(len(phases) - 1):
        # 다음 phase의 시작을 현재 phase의 끝에 맞춤
        phases[i + 1]["t_start"] = phases[i]["t_end"]

    # 3. 전체 구간 커버 강제
    phases[0]["t_start"] = 0.0
    phases[-1]["t_end"] = T

    # 4. 최종 검증: 역전된 phase가 있으면 다시 제거
    valid = [p for p in phases if p["t_end"] > p["t_start"] + EPS]
    if len(valid) == 0:
        return [{
            "t_start": 0.0, "t_end": T,
            "n_nodes": LGL_NODES_PEAK, "type": "peak",
        }]

    return valid


def _merge_short_phases(phases, T):
    """짧은 phase를 인접 phase와 병합한다.

    MIN_PHASE_FRACTION * T 미만인 phase를 인접한 phase에 흡수시킨다.
    """
    min_length = MIN_PHASE_FRACTION * T

    merged = True
    while merged:
        merged = False
        new_phases = []
        i = 0
        while i < len(phases):
            phase = phases[i]
            phase_length = phase["t_end"] - phase["t_start"]

            if phase_length < min_length and len(phases) > 1:
                # 짧은 phase를 인접 phase에 병합
                merged = True
                if i > 0 and len(new_phases) > 0:
                    # 이전 phase에 병합
                    new_phases[-1]["t_end"] = phase["t_end"]
                elif i < len(phases) - 1:
                    # 다음 phase에 병합
                    phases[i + 1]["t_start"] = phase["t_start"]
                else:
                    new_phases.append(phase)
            else:
                new_phases.append(phase)
            i += 1

        phases = new_phases

    return phases
