"""추력 크기 프로파일의 피크 탐지 모듈.

위상적 지속성(Topological Persistence) 기반 알고리즘을 사용한다.
각 피크의 "persistence" (= birth_value - death_value) 를 계산하여
유의미한 피크만 자동으로 필터링한다.

참고: https://www.sthu.org/blog/13-perstopology-peakdetection/index.html
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import peak_widths

from orbit_transfer.config import (
    PEAK_PERSISTENCE_RATIO,
    PEAK_INTERP_POINTS,
)


# ================================================================
# 위상적 지속성 (0차 persistent homology)
# ================================================================
class _Peak:
    """위상적 지속성 피크 객체."""
    __slots__ = ('born', 'left', 'right', 'died')

    def __init__(self, born_idx):
        self.born = self.left = self.right = born_idx
        self.died = None

    def persistence(self, seq):
        """피크의 지속성 값. 전역 최대의 경우 inf."""
        if self.died is None:
            return float('inf')
        return seq[self.born] - seq[self.died]


def _persistent_homology(seq):
    """0차 위상적 지속성을 계산한다.

    값이 큰 순서대로 스캔하며, 인접한 "섬"을 추적한다.
    두 섬이 합병될 때 낮은 피크가 "사망"하고, 그 persistence는
    (birth_value - death_value)로 정의된다.

    시간 복잡도: O(n log n) (정렬 지배)

    Args:
        seq: 1D 시계열 배열

    Returns:
        peaks: _Peak 객체 리스트 (persistence 내림차순)
    """
    n = len(seq)
    peaks = []
    idx_to_peak = [None] * n
    indices = sorted(range(n), key=lambda i: seq[i], reverse=True)

    for idx in indices:
        lft = idx_to_peak[idx - 1] if idx > 0 and idx_to_peak[idx - 1] is not None else None
        rgt = idx_to_peak[idx + 1] if idx < n - 1 and idx_to_peak[idx + 1] is not None else None

        if lft is None and rgt is None:
            # 새 섬 탄생
            peaks.append(_Peak(idx))
            idx_to_peak[idx] = len(peaks) - 1
        elif lft is not None and rgt is None:
            peaks[lft].right += 1
            idx_to_peak[idx] = lft
        elif lft is None and rgt is not None:
            peaks[rgt].left -= 1
            idx_to_peak[idx] = rgt
        else:
            # 두 섬 합병: 낮은 피크 사망
            if seq[peaks[lft].born] > seq[peaks[rgt].born]:
                peaks[rgt].died = idx
                peaks[lft].right = peaks[rgt].right
                idx_to_peak[peaks[lft].right] = idx_to_peak[idx] = lft
            else:
                peaks[lft].died = idx
                peaks[rgt].left = peaks[lft].left
                idx_to_peak[peaks[rgt].left] = idx_to_peak[idx] = rgt

    return sorted(peaks, key=lambda p: p.persistence(seq), reverse=True)


# ================================================================
# 메인 피크 탐지 함수
# ================================================================
def detect_peaks(t, u_mag, T, phase_boundaries=None):
    """추력 크기 프로파일에서 피크를 탐지한다.

    위상적 지속성(Topological Persistence) 알고리즘을 사용하여
    각 피크의 유의성을 자동으로 정량화한다. 경계 피크도 자연스럽게
    처리되므로 별도 경계 탐지 로직이 필요 없다.

    Args:
        t: 시간 배열 [s], shape (N,)
        u_mag: 추력 크기 ||u(t)|| [km/s^2], shape (N,)
        T: 전이 시간 [s]
        phase_boundaries: Multi-Phase 경계 [(t_start, t_end), ...]
            None이면 글로벌 CubicSpline 보간 사용

    Returns:
        n_peaks: 피크 개수
        peak_times: 피크 시각 [s], shape (n_peaks,)
        peak_widths_fwhm: 피크 반치폭(FWHM) [s], shape (n_peaks,)

    Algorithm:
        1. 시간 정렬 및 중복 제거
        2. 보간 (phase-aware 또는 글로벌 CubicSpline)
        3. 위상적 지속성 계산
        4. persistence > threshold 인 피크만 선택
        5. 피크 반치폭(FWHM) 추정
    """
    t = np.asarray(t, dtype=float)
    u_mag = np.asarray(u_mag, dtype=float)

    if len(t) < 2:
        return 0, np.array([]), np.array([])

    # Step 1: 시간 정렬 및 중복 제거
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    u_sorted = u_mag[sort_idx]
    _, uniq_idx = np.unique(t_sorted, return_index=True)
    t_uniq = t_sorted[uniq_idx]
    u_uniq = u_sorted[uniq_idx]

    # Step 2: 보간
    t_work, u_work = _interpolate_for_detection(
        t_uniq, u_uniq, PEAK_INTERP_POINTS, phase_boundaries
    )

    u_max = np.max(u_work)
    if u_max == 0.0:
        return 0, np.array([]), np.array([])

    # 상수 신호 (변동 없음) → 피크 없음
    u_range = u_max - np.min(u_work)
    if u_range < 1e-12 * u_max:
        return 0, np.array([]), np.array([])

    # Step 3: 위상적 지속성 계산
    peaks = _persistent_homology(u_work)

    if not peaks:
        return 0, np.array([]), np.array([])

    # Step 4: 유의미한 피크 필터링
    threshold = PEAK_PERSISTENCE_RATIO * u_max
    significant = [p for p in peaks if p.persistence(u_work) >= threshold]

    n_peaks = len(significant)
    if n_peaks == 0:
        return 0, np.array([]), np.array([])

    # 시간순 정렬
    peak_indices = np.array([p.born for p in significant])
    time_order = np.argsort(t_work[peak_indices])
    peak_indices = peak_indices[time_order]
    peak_times = t_work[peak_indices]

    # Step 5: 피크 반치폭 추정
    peak_widths_fwhm = estimate_peak_widths(t_work, u_work, peak_indices)

    return n_peaks, peak_times, peak_widths_fwhm


def _interpolate_for_detection(t, u, n_interp, phase_boundaries=None):
    """피크 탐지를 위한 보간.

    phase_boundaries가 주어지면 phase별 개별 보간 후 결합하여
    cross-phase 보간 아티팩트를 방지한다.

    Args:
        t: 정렬+중복제거된 시간 배열
        u: 대응 추력 크기 배열
        n_interp: 목표 보간 포인트 수
        phase_boundaries: Phase 경계 리스트 (선택)

    Returns:
        t_work, u_work: 보간된 시간/추력 배열
    """
    if len(t) < 4 or len(t) >= n_interp:
        return t.copy(), u.copy()

    if phase_boundaries is not None and len(phase_boundaries) > 1:
        return _interpolate_phased(t, u, n_interp, phase_boundaries)

    # 글로벌 CubicSpline
    cs = CubicSpline(t, u)
    t_work = np.linspace(t[0], t[-1], n_interp)
    u_work = np.maximum(cs(t_work), 0.0)
    return t_work, u_work


def _interpolate_phased(t, u, n_interp, phase_boundaries):
    """Phase별 개별 CubicSpline 보간."""
    T_total = t[-1] - t[0]
    if T_total <= 0:
        return t.copy(), u.copy()

    t_list, u_list = [], []

    for i, (t_start, t_end) in enumerate(phase_boundaries):
        dt = t_end - t_start
        n_pts = max(10, int(n_interp * dt / T_total))

        eps = (t_end - t_start) * 1e-8
        mask = (t >= t_start - eps) & (t <= t_end + eps)
        t_ph, u_ph = t[mask], u[mask]

        if len(t_ph) < 2:
            continue

        if len(t_ph) < 4:
            t_d = np.linspace(t_ph[0], t_ph[-1], n_pts)
            u_d = np.interp(t_d, t_ph, u_ph)
        else:
            t_d = np.linspace(t_ph[0], t_ph[-1], n_pts)
            cs = CubicSpline(t_ph, u_ph)
            u_d = np.maximum(cs(t_d), 0.0)

        if i > 0 and len(t_list) > 0:
            t_d = t_d[1:]
            u_d = u_d[1:]

        t_list.append(t_d)
        u_list.append(u_d)

    if not t_list:
        return t.copy(), u.copy()

    return np.concatenate(t_list), np.concatenate(u_list)


def estimate_peak_widths(t, u_mag, peak_indices):
    """피크 반치폭(FWHM)을 추정한다.

    Args:
        t: 시간 배열 [s], shape (N,)
        u_mag: 추력 크기 배열, shape (N,)
        peak_indices: 피크 인덱스 배열

    Returns:
        widths_time: 피크 반치폭 [s], shape (n_peaks,)
    """
    if len(peak_indices) == 0:
        return np.array([])

    dt_mean = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 1.0

    N = len(u_mag)
    interior_mask = (peak_indices > 0) & (peak_indices < N - 1)
    interior_peaks = peak_indices[interior_mask]

    widths_time = np.zeros(len(peak_indices))

    # 내부 피크: scipy peak_widths 사용
    if len(interior_peaks) > 0:
        try:
            widths_samples, _, _, _ = peak_widths(u_mag, interior_peaks, rel_height=0.5)
            widths_time[interior_mask] = widths_samples * dt_mean
        except Exception:
            # peak_widths 실패 시 (prominence=0 등) 기본값 사용
            widths_time[interior_mask] = dt_mean * 10

    # 경계 피크: half-max 지점까지 거리의 2배로 추정
    boundary_indices = np.where(~interior_mask)[0]
    for bi in boundary_indices:
        pidx = peak_indices[bi]
        half_max = u_mag[pidx] / 2.0
        if pidx == 0:
            below = np.where(u_mag[pidx:] < half_max)[0]
            hw = below[0] if len(below) > 0 else N - 1 - pidx
        else:
            below = np.where(u_mag[:pidx + 1][::-1] < half_max)[0]
            hw = below[0] if len(below) > 0 else pidx
        widths_time[bi] = 2.0 * hw * dt_mean

    return widths_time
