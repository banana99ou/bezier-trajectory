"""T_f 자유변수 동작 테스트."""

import numpy as np
import pytest

from orbit_transfer.types import TransferConfig
from orbit_transfer.collocation.hermite_simpson import HermiteSimpsonCollocation
from orbit_transfer.optimizer.initial_guess import linear_interpolation_guess


class TestTfFree:
    @pytest.mark.slow
    def test_tf_converges_below_tmax(self):
        """T_f <= T_max로 수렴하는지 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=1.0)
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(config, hs.N_points)
        result = hs.solve(x_guess=x_g, u_guess=u_g, nu0_guess=nu0_g, nuf_guess=nuf_g)
        assert result.converged, "수렴 실패"
        assert result.T_f <= config.T_max + 1e-3
        assert result.T_f >= config.T_min - 1e-3

    @pytest.mark.slow
    def test_tf_stored_in_result(self):
        """T_f가 TrajectoryResult에 올바르게 저장되는지 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=0.8)
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(config, hs.N_points)
        result = hs.solve(x_guess=x_g, u_guess=u_g, nu0_guess=nu0_g, nuf_guess=nuf_g)
        assert result.T_f > 0

    def test_tf_default_zero(self):
        """TrajectoryResult 기본 T_f=0."""
        from orbit_transfer.types import TrajectoryResult
        result = TrajectoryResult(
            converged=False, cost=float('inf'),
            t=np.zeros(10), x=np.zeros((6, 10)), u=np.zeros((3, 10)),
            nu0=0.0, nuf=0.0, n_peaks=0, profile_class=0,
        )
        assert result.T_f == 0.0

    def test_t_min_property(self):
        """TransferConfig.T_min property 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=1.0)
        from orbit_transfer.config import T_MIN_FACTOR
        expected = T_MIN_FACTOR * config.T0
        np.testing.assert_allclose(config.T_min, expected)

    def test_t_max_property(self):
        """TransferConfig.T_max property 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=1.0)
        expected = 1.0 * config.T0
        np.testing.assert_allclose(config.T_max, expected)
