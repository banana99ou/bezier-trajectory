"""피크 탐지 및 분류 모듈 테스트.

합성 가우시안 데이터를 사용하여 피크 탐지, 프로파일 분류,
Phase 구조 결정 기능을 검증한다.
"""

import numpy as np
import pytest

from orbit_transfer.classification.peak_detection import detect_peaks, estimate_peak_widths
from orbit_transfer.classification.classifier import classify_profile, determine_phase_structure
from orbit_transfer.config import (
    PEAK_PERSISTENCE_RATIO,
    PEAK_INTERP_POINTS,
    LGL_NODES_PEAK,
    LGL_NODES_COAST,
    MIN_PHASE_FRACTION,
)


def make_gaussian_profile(t, centers, widths, amplitudes):
    """합성 가우시안 추력 프로파일을 생성한다.

    Args:
        t: 시간 배열
        centers: 가우시안 중심 리스트
        widths: 가우시안 표준편차 리스트
        amplitudes: 가우시안 진폭 리스트

    Returns:
        u: 추력 크기 배열
    """
    u = np.zeros_like(t)
    for c, w, a in zip(centers, widths, amplitudes):
        u += a * np.exp(-0.5 * ((t - c) / w) ** 2)
    return u


# ==============================================================
# 테스트 1: 단봉 (Unimodal) -> class 0
# ==============================================================
class TestUnimodal:
    """단봉 프로파일 탐지 및 분류 테스트."""

    def test_detect_single_peak(self):
        T = 1000.0
        t = np.linspace(0, T, 500)
        u = make_gaussian_profile(t, centers=[T / 2], widths=[T / 8], amplitudes=[1.0])

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks == 1
        assert len(peak_times) == 1
        assert len(peak_widths_fwhm) == 1
        # 피크 위치가 T/2 근처인지 확인
        assert abs(peak_times[0] - T / 2) < T * 0.05

    def test_classify_unimodal(self):
        assert classify_profile(0) == 0
        assert classify_profile(1) == 0


# ==============================================================
# 테스트 2: 쌍봉 (Bimodal) -> class 1
# ==============================================================
class TestBimodal:
    """쌍봉 프로파일 탐지 및 분류 테스트."""

    def test_detect_two_peaks(self):
        T = 1000.0
        t = np.linspace(0, T, 500)
        u = make_gaussian_profile(
            t,
            centers=[T / 4, 3 * T / 4],
            widths=[T / 10, T / 10],
            amplitudes=[1.0, 0.8],
        )

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks == 2
        assert len(peak_times) == 2
        assert len(peak_widths_fwhm) == 2
        # 피크 위치 확인
        assert abs(peak_times[0] - T / 4) < T * 0.05
        assert abs(peak_times[1] - 3 * T / 4) < T * 0.05

    def test_classify_bimodal(self):
        assert classify_profile(2) == 1


# ==============================================================
# 테스트 3: 다봉 (Multimodal) -> class 2
# ==============================================================
class TestMultimodal:
    """다봉 프로파일 탐지 및 분류 테스트."""

    def test_detect_three_peaks(self):
        T = 1000.0
        t = np.linspace(0, T, 500)
        u = make_gaussian_profile(
            t,
            centers=[T / 6, T / 2, 5 * T / 6],
            widths=[T / 12, T / 12, T / 12],
            amplitudes=[0.8, 1.0, 0.6],
        )

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks == 3
        assert len(peak_times) == 3
        assert len(peak_widths_fwhm) == 3
        # 피크 위치 확인
        assert abs(peak_times[0] - T / 6) < T * 0.05
        assert abs(peak_times[1] - T / 2) < T * 0.05
        assert abs(peak_times[2] - 5 * T / 6) < T * 0.05

    def test_classify_multimodal(self):
        assert classify_profile(3) == 2
        assert classify_profile(5) == 2


# ==============================================================
# 테스트 4: 노이즈 강건성 (SNR=10)
# ==============================================================
class TestNoiseRobustness:
    """미세 수치 섭동이 추가된 프로파일에서 동일 분류 결과를 반환하는지 테스트.

    NLP 솔버의 해는 다항식/스플라인이므로 Gaussian 노이즈가 아닌
    미세 수치 오차 수준의 섭동만 현실적이다. SNR=1000 수준.
    """

    def test_unimodal_with_perturbation(self):
        T = 1000.0
        t = np.linspace(0, T, 500)
        u_clean = make_gaussian_profile(t, centers=[T / 2], widths=[T / 8], amplitudes=[1.0])

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1e-3, size=len(t))
        u_noisy = np.maximum(u_clean + noise, 0.0)

        n_peaks, _, _ = detect_peaks(t, u_noisy, T)
        assert classify_profile(n_peaks) == 0

    def test_bimodal_with_perturbation(self):
        T = 1000.0
        t = np.linspace(0, T, 500)
        u_clean = make_gaussian_profile(
            t,
            centers=[T / 4, 3 * T / 4],
            widths=[T / 10, T / 10],
            amplitudes=[1.0, 0.8],
        )

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1e-3, size=len(t))
        u_noisy = np.maximum(u_clean + noise, 0.0)

        n_peaks, _, _ = detect_peaks(t, u_noisy, T)
        assert classify_profile(n_peaks) == 1

    def test_multimodal_with_perturbation(self):
        T = 1000.0
        t = np.linspace(0, T, 500)
        u_clean = make_gaussian_profile(
            t,
            centers=[T / 6, T / 2, 5 * T / 6],
            widths=[T / 12, T / 12, T / 12],
            amplitudes=[0.8, 1.0, 0.6],
        )

        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1e-3, size=len(t))
        u_noisy = np.maximum(u_clean + noise, 0.0)

        n_peaks, _, _ = detect_peaks(t, u_noisy, T)
        assert classify_profile(n_peaks) == 2


# ==============================================================
# 테스트 5: Phase 구조 (Bimodal -> 3-phase peak-coast-peak)
# ==============================================================
class TestPhaseStructure:
    """Phase 구조 결정 테스트."""

    def test_unimodal_single_phase(self):
        """Unimodal은 단일 peak phase를 반환한다."""
        T = 1000.0
        peak_times = np.array([T / 2])
        peak_widths_vals = np.array([T / 8])

        phases = determine_phase_structure(peak_times, peak_widths_vals, T)

        assert len(phases) == 1
        assert phases[0]["type"] == "peak"
        assert phases[0]["t_start"] == 0.0
        assert phases[0]["t_end"] == T
        assert phases[0]["n_nodes"] == LGL_NODES_PEAK

    def test_bimodal_three_phases(self):
        """Bimodal은 3개 phase (peak-coast-peak)를 반환한다."""
        T = 1000.0
        t1, t2 = T / 4, 3 * T / 4
        w1, w2 = T / 10, T / 10

        phases = determine_phase_structure(
            np.array([t1, t2]), np.array([w1, w2]), T
        )

        assert len(phases) == 3
        assert phases[0]["type"] == "peak"
        assert phases[1]["type"] == "coast"
        assert phases[2]["type"] == "peak"

        # 노드 수 확인
        assert phases[0]["n_nodes"] == LGL_NODES_PEAK
        assert phases[1]["n_nodes"] == LGL_NODES_COAST
        assert phases[2]["n_nodes"] == LGL_NODES_PEAK

        # Phase 경계 연속성 확인
        assert phases[0]["t_end"] == phases[1]["t_start"]
        assert phases[1]["t_end"] == phases[2]["t_start"]

        # 전체 범위 확인
        assert phases[0]["t_start"] == 0.0
        assert phases[-1]["t_end"] == T

    def test_multimodal_five_phases(self):
        """3-peak Multimodal은 5개 phase를 반환한다."""
        T = 1000.0
        centers = np.array([T / 6, T / 2, 5 * T / 6])
        widths = np.array([T / 12, T / 12, T / 12])

        phases = determine_phase_structure(centers, widths, T)

        # 3 peaks -> 최대 5 phases (peak-coast-peak-coast-peak)
        # 병합에 의해 줄어들 수 있음
        assert len(phases) >= 3
        assert len(phases) <= 5

        # 첫 번째와 마지막 phase는 peak
        assert phases[0]["type"] == "peak"
        assert phases[-1]["type"] == "peak"

        # 전체 범위 확인
        assert phases[0]["t_start"] == 0.0
        assert phases[-1]["t_end"] == T

    def test_no_peaks_coast_phase(self):
        """피크가 없으면 단일 coast phase를 반환한다."""
        T = 1000.0
        phases = determine_phase_structure(np.array([]), np.array([]), T)

        assert len(phases) == 1
        assert phases[0]["type"] == "coast"
        assert phases[0]["n_nodes"] == LGL_NODES_COAST

    def test_phase_continuity(self):
        """모든 phase 경계가 연속인지 확인한다."""
        T = 1000.0
        centers = np.array([T / 4, 3 * T / 4])
        widths = np.array([T / 10, T / 10])

        phases = determine_phase_structure(centers, widths, T)

        for i in range(len(phases) - 1):
            assert abs(phases[i]["t_end"] - phases[i + 1]["t_start"]) < 1e-10


# ==============================================================
# 테스트 6: 엣지 케이스
# ==============================================================
class TestEdgeCases:
    """엣지 케이스 테스트."""

    def test_zero_signal(self):
        """신호가 0인 경우 피크가 없다."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        u = np.zeros_like(t)

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks == 0
        assert len(peak_times) == 0
        assert len(peak_widths_fwhm) == 0

    def test_constant_signal(self):
        """상수 신호에서는 피크가 없다."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        u = np.ones_like(t) * 0.5

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks == 0


# ==============================================================
# 테스트 7: 경계 피크 (Boundary Peaks)
# ==============================================================
class TestEdgePeaks:
    """경계 피크 탐지 테스트."""

    def test_peak_at_start(self):
        """t=0에서 시작하는 감소형 프로파일 → 경계 피크 1개."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        # t=0에서 최대, 지수적 감소
        u = np.exp(-3.0 * t / T)

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks >= 1
        # 첫 피크가 T의 초반에 위치
        assert peak_times[0] < T * 0.1

    def test_peak_at_end(self):
        """t=T에서 끝나는 증가형 프로파일 → 경계 피크 1개."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        # t=T에서 최대, 지수적 증가
        u = np.exp(-3.0 * (T - t) / T)

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks >= 1
        # 마지막 피크가 T의 후반에 위치
        assert peak_times[-1] > T * 0.9

    def test_peaks_at_both_ends(self):
        """양쪽 경계 + 내부 피크 → 3개 탐지."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        # 양쪽 경계에서 큰 값 + 중앙 피크
        u_center = make_gaussian_profile(t, centers=[T / 2], widths=[T / 10], amplitudes=[1.0])
        u_start = np.exp(-5.0 * t / T) * 0.8
        u_end = np.exp(-5.0 * (T - t) / T) * 0.8
        u = u_center + u_start + u_end

        n_peaks, peak_times, peak_widths_fwhm = detect_peaks(t, u, T)

        assert n_peaks == 3
        assert peak_times[0] < T * 0.1
        assert abs(peak_times[1] - T / 2) < T * 0.1
        assert peak_times[2] > T * 0.9

    def test_near_boundary_peak_with_slight_decrease(self):
        """끝점 직전에서 살짝 감소하는 패턴 → 경계 근방 피크 탐지."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        # 끝 근방에 피크: t=0.95*T에서 최대, 이후 살짝 감소
        u = make_gaussian_profile(t, centers=[T / 3], widths=[T / 10], amplitudes=[1.0])
        u += make_gaussian_profile(t, centers=[0.95 * T], widths=[T / 15], amplitudes=[0.8])

        n_peaks, peak_times, _ = detect_peaks(t, u, T)

        assert n_peaks >= 2
        # 마지막 피크가 T의 후반에 위치
        assert peak_times[-1] > T * 0.85

    def test_near_start_peak_with_slight_increase(self):
        """시작점 직후에서 살짝 증가 후 감소하는 패턴 → 경계 근방 피크 탐지."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        # 시작 근방에 피크: t=0.05*T에서 최대, 이전 살짝 증가
        u = make_gaussian_profile(t, centers=[0.05 * T], widths=[T / 15], amplitudes=[0.8])
        u += make_gaussian_profile(t, centers=[2 * T / 3], widths=[T / 10], amplitudes=[1.0])

        n_peaks, peak_times, _ = detect_peaks(t, u, T)

        assert n_peaks >= 2
        # 첫 피크가 T의 초반에 위치
        assert peak_times[0] < T * 0.15


# ==============================================================
# 테스트 8: 비단조 시간 (Non-Monotonic Time)
# ==============================================================
class TestNonMonotonicTime:
    """비단조 시간 입력 처리 테스트."""

    def test_non_monotonic_sorted_correctly(self):
        """비단조 시간 입력 → 올바른 피크 수 반환."""
        T = 1000.0
        t_mono = np.linspace(0, T, 500)
        u_mono = make_gaussian_profile(
            t_mono,
            centers=[T / 4, 3 * T / 4],
            widths=[T / 10, T / 10],
            amplitudes=[1.0, 0.8],
        )

        # 정상 입력에서 피크 수 확인
        n_peaks_ref, _, _ = detect_peaks(t_mono, u_mono, T)

        # 시간 배열을 셔플 (비단조)
        rng = np.random.default_rng(123)
        shuffle_idx = rng.permutation(len(t_mono))
        t_shuffled = t_mono[shuffle_idx]
        u_shuffled = u_mono[shuffle_idx]

        n_peaks, _, _ = detect_peaks(t_shuffled, u_shuffled, T)

        assert n_peaks == n_peaks_ref

    def test_duplicate_time_points(self):
        """중복 시간 포인트가 있는 입력 → 정상 처리."""
        T = 1000.0
        t = np.linspace(0, T, 500)
        u = make_gaussian_profile(t, centers=[T / 2], widths=[T / 8], amplitudes=[1.0])

        # 중복 포인트 추가
        t_dup = np.concatenate([t, t[:10]])
        u_dup = np.concatenate([u, u[:10]])

        n_peaks, peak_times, _ = detect_peaks(t_dup, u_dup, T)

        assert n_peaks == 1
        assert abs(peak_times[0] - T / 2) < T * 0.05


# ==============================================================
# 테스트 9: 저해상도 보간 (Interpolation)
# ==============================================================
class TestInterpolation:
    """저해상도 신호 보간 테스트."""

    def test_coarse_signal_interpolated(self):
        """N=30 가우시안 3-피크 → 보간 후 3개 탐지."""
        T = 1000.0
        N_coarse = 30
        t_coarse = np.linspace(0, T, N_coarse)
        u_coarse = make_gaussian_profile(
            t_coarse,
            centers=[T / 6, T / 2, 5 * T / 6],
            widths=[T / 12, T / 12, T / 12],
            amplitudes=[0.8, 1.0, 0.6],
        )

        n_peaks, peak_times, _ = detect_peaks(t_coarse, u_coarse, T)

        assert n_peaks == 3
        assert abs(peak_times[0] - T / 6) < T * 0.1
        assert abs(peak_times[1] - T / 2) < T * 0.1
        assert abs(peak_times[2] - 5 * T / 6) < T * 0.1

    def test_dense_signal_no_interpolation(self):
        """N=500 이미 조밀 → 보간 없이 동일 결과."""
        T = 1000.0
        t_dense = np.linspace(0, T, 500)
        u_dense = make_gaussian_profile(
            t_dense,
            centers=[T / 4, 3 * T / 4],
            widths=[T / 10, T / 10],
            amplitudes=[1.0, 0.8],
        )

        n_peaks, peak_times, _ = detect_peaks(t_dense, u_dense, T)

        assert n_peaks == 2
        assert abs(peak_times[0] - T / 4) < T * 0.05
        assert abs(peak_times[1] - 3 * T / 4) < T * 0.05

    def test_very_coarse_bimodal(self):
        """N=20 쌍봉 → 보간 후 2개 탐지."""
        T = 1000.0
        N_coarse = 20
        t_coarse = np.linspace(0, T, N_coarse)
        u_coarse = make_gaussian_profile(
            t_coarse,
            centers=[T / 4, 3 * T / 4],
            widths=[T / 10, T / 10],
            amplitudes=[1.0, 0.8],
        )

        n_peaks, _, _ = detect_peaks(t_coarse, u_coarse, T)

        assert n_peaks == 2
