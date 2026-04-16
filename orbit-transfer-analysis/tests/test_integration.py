"""Multi-Phase LGL Collocation + Warm Start + TwoPassOptimizer 통합 테스트."""

import numpy as np
import pytest

from orbit_transfer.types import TransferConfig
from orbit_transfer.collocation.interpolation import interpolate_pass1_to_pass2
from orbit_transfer.optimizer.two_pass import TwoPassOptimizer


class TestInterpolation:
    def test_interpolation_shapes(self):
        """보간 결과 shape 확인."""
        # 합성 Pass 1 데이터
        N = 61
        t = np.linspace(0, 5400, N)
        x = np.random.randn(6, N)
        u = np.random.randn(3, N) * 0.001
        phases = [
            {'t_start': 0, 't_end': 2000, 'n_nodes': 15, 'type': 'peak'},
            {'t_start': 2000, 't_end': 3400, 'n_nodes': 8, 'type': 'coast'},
            {'t_start': 3400, 't_end': 5400, 'n_nodes': 15, 'type': 'peak'},
        ]
        t_ph, x_ph, u_ph = interpolate_pass1_to_pass2(t, x, u, phases)
        assert len(t_ph) == 3
        assert t_ph[0].shape == (15,)
        assert x_ph[1].shape == (6, 8)


class TestPhaseStructureContinuity:
    """Phase 구조 연속성 및 시간 단조성 테스트."""

    def test_overlapping_peaks_no_zero_phase(self):
        """피크 폭이 겹치는 경우 영-길이 phase가 없어야 한다."""
        from orbit_transfer.classification.classifier import determine_phase_structure
        T = 5000.0
        # 두 피크가 가까이 있고 폭이 넓어서 겹침
        peak_times = np.array([1500.0, 2500.0])
        peak_widths = np.array([800.0, 800.0])  # 겹침: 1500+800 > 2500-800
        phases = determine_phase_structure(peak_times, peak_widths, T)
        for p in phases:
            assert p['t_end'] > p['t_start'], f"영-길이 phase: {p}"

    def test_phases_cover_full_interval(self):
        """Phase가 [0, T] 전체를 커버해야 한다."""
        from orbit_transfer.classification.classifier import determine_phase_structure
        T = 6000.0
        peak_times = np.array([1000.0, 3000.0, 5000.0])
        peak_widths = np.array([400.0, 400.0, 400.0])
        phases = determine_phase_structure(peak_times, peak_widths, T)
        assert phases[0]['t_start'] == 0.0
        assert phases[-1]['t_end'] == T

    def test_phases_contiguous(self):
        """인접 phase 경계가 일치해야 한다."""
        from orbit_transfer.classification.classifier import determine_phase_structure
        T = 5000.0
        peak_times = np.array([1200.0, 3800.0])
        peak_widths = np.array([500.0, 500.0])
        phases = determine_phase_structure(peak_times, peak_widths, T)
        for i in range(len(phases) - 1):
            np.testing.assert_allclose(
                phases[i]['t_end'], phases[i + 1]['t_start'],
                atol=1e-10,
                err_msg=f"Phase {i} 끝 != Phase {i+1} 시작",
            )

    def test_enforce_monotonicity_basic(self):
        """_enforce_monotonicity가 중복/역전 시간을 수정한다."""
        from orbit_transfer.collocation.multiphase_lgl import _enforce_monotonicity
        t = np.array([0.0, 1.0, 1.0, 2.0, 1.5, 3.0])
        x = np.random.randn(6, 6)
        u = np.random.randn(3, 6)
        t_fix, x_fix, u_fix = _enforce_monotonicity(t, x, u)
        dt = np.diff(t_fix)
        assert np.all(dt > 0), f"비단조 시간: {t_fix}"

    def test_enforce_monotonicity_preserves_count(self):
        """_enforce_monotonicity가 중복 제거 후 개수가 줄어야 한다."""
        from orbit_transfer.collocation.multiphase_lgl import _enforce_monotonicity
        t = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        x = np.random.randn(6, 5)
        u = np.random.randn(3, 5)
        t_fix, _, _ = _enforce_monotonicity(t, x, u)
        assert len(t_fix) == 4  # 중복 1개 제거


class TestTwoPassOptimizer:
    @pytest.mark.slow
    def test_R1_coplanar(self):
        """R1: da=200, di=0, T=2T0."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        opt = TwoPassOptimizer(config)
        result = opt.solve()
        assert result.converged
        assert result.cost > 0
        if result.pass1_cost is not None:
            assert result.cost <= result.pass1_cost * 1.05  # Pass2 <= Pass1 (5% margin)

    @pytest.mark.slow
    def test_R1_time_monotonic(self):
        """수렴된 결과의 시간 배열이 엄격히 단조증가해야 한다."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=0.8)
        opt = TwoPassOptimizer(config)
        result = opt.solve()
        if result.converged:
            dt = np.diff(result.t)
            assert np.all(dt > 0), (
                f"비단조 시간 발견: min(dt)={dt.min():.2e}"
            )

    @pytest.mark.slow
    def test_R3_bimodal_time_monotonic(self):
        """Bimodal 케이스의 Multi-Phase 결과에서 시간 단조성 확인."""
        config = TransferConfig(h0=400, delta_a=500, delta_i=3, T_max_normed=1.2)
        opt = TwoPassOptimizer(config)
        result = opt.solve()
        if result.converged:
            dt = np.diff(result.t)
            assert np.all(dt > 0), (
                f"비단조 시간 발견: min(dt)={dt.min():.2e}"
            )

    @pytest.mark.slow
    def test_R4_lowering(self):
        """R4: da=-200, di=0, T=1.5T0."""
        config = TransferConfig(h0=400, delta_a=-200, delta_i=0, T_max_normed=1.5)
        opt = TwoPassOptimizer(config)
        result = opt.solve()
        assert result.converged
