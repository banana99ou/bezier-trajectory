"""normalize.py 단위 테스트."""

import math
import numpy as np
import pytest

from bezier_orbit.normalize import (
    CanonicalUnits,
    MU_EARTH,
    R_EARTH,
    from_orbit,
    standard_earth,
)


class TestCanonicalUnits:
    """CanonicalUnits 기본 동작 테스트."""

    def test_standard_earth_mu_star(self):
        cu = standard_earth()
        assert cu.mu_star == pytest.approx(1.0)

    def test_standard_earth_values(self):
        cu = standard_earth()
        # TU ≈ 806.81 s
        assert cu.TU == pytest.approx(806.81, rel=1e-3)
        # VU ≈ 7.905 km/s
        assert cu.VU == pytest.approx(7.905, rel=1e-3)

    def test_adaptive_normalization(self):
        # LEO a0 = 6778 km
        cu = from_orbit(6778.0)
        assert cu.DU == 6778.0
        assert cu.mu_star == pytest.approx(1.0)
        # R_earth_star < 1 (a0 > R_earth)
        assert cu.R_earth_star < 1.0

    def test_invalid_a0(self):
        with pytest.raises(ValueError):
            from_orbit(-100.0)


class TestRoundTrip:
    """정규화 ↔ 물리 단위 왕복 변환 테스트."""

    @pytest.fixture
    def cu(self):
        return from_orbit(6778.0)

    def test_pos_roundtrip(self, cu):
        r = np.array([7000.0, 100.0, -200.0])
        r_star = cu.nondim_pos(r)
        r_back = cu.dim_pos(r_star)
        np.testing.assert_allclose(r_back, r, rtol=1e-14)

    def test_vel_roundtrip(self, cu):
        v = np.array([0.5, 7.5, -0.1])
        v_star = cu.nondim_vel(v)
        v_back = cu.dim_vel(v_star)
        np.testing.assert_allclose(v_back, v, rtol=1e-14)

    def test_time_roundtrip(self, cu):
        t = 3600.0
        t_star = cu.nondim_time(t)
        t_back = cu.dim_time(t_star)
        assert t_back == pytest.approx(t, rel=1e-14)

    def test_accel_roundtrip(self, cu):
        a = np.array([1e-4, -2e-5, 3e-6])
        a_star = cu.nondim_accel(a)
        a_back = cu.dim_accel(a_star)
        np.testing.assert_allclose(a_back, a, rtol=1e-14)

    def test_state_roundtrip(self, cu):
        x = np.array([7000.0, 100.0, -200.0, 0.5, 7.5, -0.1])
        x_star = cu.nondim_state(x)
        x_back = cu.dim_state(x_star)
        np.testing.assert_allclose(x_back, x, rtol=1e-14)


class TestScaleOrder:
    """정규화된 변수가 O(1) 스케일인지 확인."""

    def test_leo_o1(self):
        cu = from_orbit(6778.0)
        # LEO 위치 ≈ 6778 km → r* ≈ 1
        r_star = cu.nondim_pos(np.array([6778.0, 0.0, 0.0]))
        assert 0.1 < np.linalg.norm(r_star) < 10.0

        # LEO 속도 ≈ 7.67 km/s → v* ≈ 1
        v_star = cu.nondim_vel(np.array([0.0, 7.67, 0.0]))
        assert 0.1 < np.linalg.norm(v_star) < 10.0

    def test_geo_adaptive_o1(self):
        # GEO a0 = 42164 km
        cu = from_orbit(42164.0)
        r_star = cu.nondim_pos(np.array([42164.0, 0.0, 0.0]))
        assert np.linalg.norm(r_star) == pytest.approx(1.0)
