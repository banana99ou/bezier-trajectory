"""타원궤도 (e != 0) 지원 테스트."""

import numpy as np
import pytest

from orbit_transfer.types import TransferConfig
from orbit_transfer.collocation.hermite_simpson import HermiteSimpsonCollocation
from orbit_transfer.optimizer.initial_guess import linear_interpolation_guess


class TestEccentricOrbits:
    @pytest.mark.slow
    def test_e0_nonzero(self):
        """e0=0.05 출발궤도 수렴."""
        config = TransferConfig(
            h0=400, delta_a=200, delta_i=0, T_max_normed=0.8, e0=0.05,
        )
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(config, hs.N_points)
        result = hs.solve(x_guess=x_g, u_guess=u_g, nu0_guess=nu0_g, nuf_guess=nuf_g)
        assert result.converged, "e0=0.05 수렴 실패"

    @pytest.mark.slow
    def test_ef_nonzero(self):
        """ef=0.05 도착궤도 수렴."""
        config = TransferConfig(
            h0=400, delta_a=200, delta_i=0, T_max_normed=0.8, ef=0.05,
        )
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(config, hs.N_points)
        result = hs.solve(x_guess=x_g, u_guess=u_g, nu0_guess=nu0_g, nuf_guess=nuf_g)
        assert result.converged, "ef=0.05 수렴 실패"

    @pytest.mark.slow
    def test_both_eccentric(self):
        """e0=0.05, ef=0.05 양쪽 타원 수렴 (TwoPass retry 포함)."""
        from orbit_transfer.optimizer.two_pass import TwoPassOptimizer
        config = TransferConfig(
            h0=400, delta_a=200, delta_i=3, T_max_normed=1.0, e0=0.05, ef=0.05,
        )
        opt = TwoPassOptimizer(config)
        result = opt.solve()
        assert result.converged, "양쪽 타원 수렴 실패"

    def test_eccentric_config_properties(self):
        """타원궤도 config 속성 확인."""
        config = TransferConfig(
            h0=400, delta_a=200, delta_i=3, T_max_normed=1.0, e0=0.05, ef=0.08,
        )
        assert config.e0 == 0.05
        assert config.ef == 0.08
        assert config.T_max > 0
        assert config.T_min > 0
        assert config.T_min < config.T_max

    def test_initial_guess_eccentric(self):
        """타원궤도 초기값 생성이 정상 동작."""
        config = TransferConfig(
            h0=400, delta_a=200, delta_i=0, T_max_normed=0.8, e0=0.05, ef=0.05,
        )
        t, x_g, u_g, nu0, nuf = linear_interpolation_guess(config, 61)
        assert t.shape == (61,)
        assert x_g.shape == (6, 61)
        # 첫 점은 타원궤도 위에 있어야 함
        r0_mag = np.linalg.norm(x_g[:3, 0])
        # 타원궤도이므로 a*(1-e) <= r <= a*(1+e) 범위
        assert r0_mag >= config.a0 * (1 - config.e0) - 1.0
        assert r0_mag <= config.a0 * (1 + config.e0) + 1.0
