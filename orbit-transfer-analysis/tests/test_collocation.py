"""Hermite-Simpson collocation 및 초기값 생성 테스트."""

import numpy as np
import pytest

from orbit_transfer.types import TransferConfig
from orbit_transfer.collocation.hermite_simpson import HermiteSimpsonCollocation
from orbit_transfer.optimizer.initial_guess import (
    linear_interpolation_guess,
    keplerian_guess,
)
from orbit_transfer.optimizer.solver import get_pass1_options, get_pass2_options


# ============================================================
# 초기값 생성 테스트
# ============================================================
class TestLinearInterpolationGuess:
    def test_shape(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = linear_interpolation_guess(config, 61)
        assert t.shape == (61,)
        assert x.shape == (6, 61)
        assert u.shape == (3, 61)

    def test_time_range(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = linear_interpolation_guess(config, 61)
        assert t[0] == 0.0
        np.testing.assert_allclose(t[-1], config.T_max)

    def test_control_is_zero(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = linear_interpolation_guess(config, 61)
        np.testing.assert_array_equal(u, np.zeros((3, 61)))

    def test_nu_values(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = linear_interpolation_guess(config, 61)
        assert nu0 == 0.0
        assert nuf == np.pi

    def test_boundary_positions(self):
        """출발/도착점이 올바른 궤도 반지름을 가지는지 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = linear_interpolation_guess(config, 61)
        r0_mag = np.linalg.norm(x[:3, 0])
        rf_mag = np.linalg.norm(x[:3, -1])
        np.testing.assert_allclose(r0_mag, config.a0, rtol=1e-10)
        np.testing.assert_allclose(rf_mag, config.af, rtol=1e-10)


class TestKeplerianGuess:
    def test_shape(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = keplerian_guess(config, 61)
        assert t.shape == (61,)
        assert x.shape == (6, 61)
        assert u.shape == (3, 61)

    def test_coplanar_nu0(self):
        """Coplanar 전이에서 nu0=0."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = keplerian_guess(config, 61)
        assert nu0 == 0.0

    def test_plane_change_nu0(self):
        """Plane change에서 nu0=pi/2."""
        config = TransferConfig(h0=400, delta_a=0, delta_i=5.0, T_max_normed=3.0)
        t, x, u, nu0, nuf = keplerian_guess(config, 61)
        assert nu0 == np.pi / 2

    def test_keplerian_radius_conservation(self):
        """케플러 전파 시 원궤도 반지름이 보존되는지 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        t, x, u, nu0, nuf = keplerian_guess(config, 61)
        # 원궤도이므로 모든 점에서 반지름이 일정해야 함
        r_mag = np.linalg.norm(x[:3, :], axis=0)
        np.testing.assert_allclose(r_mag, config.a0, rtol=1e-8)


# ============================================================
# 솔버 옵션 테스트
# ============================================================
class TestSolverOptions:
    def test_pass1_defaults(self):
        opts = get_pass1_options()
        assert opts["ipopt.tol"] == 1e-4
        assert opts["ipopt.max_iter"] == 500

    def test_pass1_overrides(self):
        opts = get_pass1_options(**{"ipopt.max_iter": 1000})
        assert opts["ipopt.max_iter"] == 1000
        assert opts["ipopt.tol"] == 1e-4  # 나머지는 기본값 유지

    def test_pass2_defaults(self):
        opts = get_pass2_options()
        assert opts["ipopt.tol"] == 1e-6
        assert opts["ipopt.max_iter"] == 1000

    def test_pass2_overrides(self):
        opts = get_pass2_options(**{"ipopt.tol": 1e-8})
        assert opts["ipopt.tol"] == 1e-8


# ============================================================
# Hermite-Simpson Collocation 테스트
# ============================================================
class TestHermiteSimpsonCollocation:
    def test_init(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        hs = HermiteSimpsonCollocation(config)
        assert hs.M == 30
        assert hs.N_points == 61

    def test_custom_segments(self):
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        hs = HermiteSimpsonCollocation(config, M=10)
        assert hs.M == 10
        assert hs.N_points == 21

    @pytest.mark.slow
    def test_coplanar_transfer_R1(self):
        """R1: da=200, di=0, T=2T0 수렴 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
            config, hs.N_points
        )
        result = hs.solve(
            x_guess=x_g, u_guess=u_g,
            nu0_guess=nu0_g, nuf_guess=nuf_g,
        )
        assert result.converged, "R1 수렴 실패"
        assert result.cost > 0
        # 추력 상한 확인
        u_mag = np.linalg.norm(result.u, axis=0)
        assert np.all(u_mag <= config.u_max + 1e-6)

    @pytest.mark.slow
    def test_orbit_lowering_R4(self):
        """R4: da=-200, di=0, T=1.5T0 수렴 확인."""
        config = TransferConfig(h0=400, delta_a=-200, delta_i=0, T_max_normed=1.5)
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
            config, hs.N_points
        )
        result = hs.solve(
            x_guess=x_g, u_guess=u_g,
            nu0_guess=nu0_g, nuf_guess=nuf_g,
        )
        assert result.converged, "R4 수렴 실패"

    @pytest.mark.slow
    def test_minimum_altitude_constraint(self):
        """최소 고도 제약조건 만족 확인."""
        from orbit_transfer.constants import R_E
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
            config, hs.N_points
        )
        result = hs.solve(
            x_guess=x_g, u_guess=u_g,
            nu0_guess=nu0_g, nuf_guess=nuf_g,
        )
        if result.converged:
            r_mag = np.linalg.norm(result.x[:3, :], axis=0)
            altitudes = r_mag - R_E
            assert np.all(altitudes >= config.h_min - 1.0), (
                f"최소 고도 위반: {altitudes.min():.1f} km < {config.h_min} km"
            )

    @pytest.mark.slow
    def test_result_fields(self):
        """TrajectoryResult 필드 확인."""
        config = TransferConfig(h0=400, delta_a=200, delta_i=0, T_max_normed=2.0)
        hs = HermiteSimpsonCollocation(config)
        t, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(
            config, hs.N_points
        )
        result = hs.solve(
            x_guess=x_g, u_guess=u_g,
            nu0_guess=nu0_g, nuf_guess=nuf_g,
        )
        assert hasattr(result, 'converged')
        assert hasattr(result, 'cost')
        assert hasattr(result, 't')
        assert hasattr(result, 'x')
        assert hasattr(result, 'u')
        assert hasattr(result, 'nu0')
        assert hasattr(result, 'nuf')
        assert hasattr(result, 'n_peaks')
        assert hasattr(result, 'profile_class')
        assert result.t.shape == (hs.N_points,)
        assert result.x.shape == (6, hs.N_points)
        assert result.u.shape == (3, hs.N_points)
