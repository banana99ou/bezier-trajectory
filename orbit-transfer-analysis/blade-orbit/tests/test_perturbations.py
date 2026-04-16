"""orbit/perturbations.py 단위 테스트."""

import numpy as np
import pytest

from bezier_orbit.orbit.perturbations import (
    accel_jn,
    accel_jn_sum,
    exponential_density,
    accel_drag,
    accel_third_body,
    J_COEFFS,
)
from bezier_orbit.orbit.dynamics import _accel_j2
from bezier_orbit.normalize import J2_EARTH


class TestJnAcceleration:
    """J2–J6 가속도 테스트."""

    def test_j2_matches_dynamics(self):
        """perturbations.accel_jn(n=2) vs dynamics._accel_j2 일치."""
        r = np.array([1.2, -0.3, 0.8])
        r_mag = np.linalg.norm(r)
        Re_star = 0.94

        a1 = accel_jn(r, 2, Re_star=Re_star)
        a2 = _accel_j2(r, r_mag, Re_star, J2_EARTH)
        np.testing.assert_allclose(a1, a2, atol=1e-14)

    def test_j3_finite_diff(self):
        """J3 해석적 vs 수치 기울기."""
        r = np.array([1.5, 0.2, 0.6])
        Re_star = 0.9
        eps = 1e-7

        a_analytic = accel_jn(r, 3, Re_star=Re_star)

        a_fd = np.zeros(3)
        from bezier_orbit.orbit.perturbations import _potential_jn
        for j in range(3):
            rp = r.copy(); rp[j] += eps
            rm = r.copy(); rm[j] -= eps
            Up = _potential_jn(rp, 3, Re_star, J_COEFFS[3])
            Um = _potential_jn(rm, 3, Re_star, J_COEFFS[3])
            a_fd[j] = -(Up - Um) / (2.0 * eps)

        np.testing.assert_allclose(a_analytic, a_fd, atol=1e-5)

    def test_j4_finite_diff(self):
        """J4 해석적 vs 수치 기울기."""
        r = np.array([1.5, 0.2, 0.6])
        Re_star = 0.9
        eps = 1e-7

        a_analytic = accel_jn(r, 4, Re_star=Re_star)

        a_fd = np.zeros(3)
        from bezier_orbit.orbit.perturbations import _potential_jn
        for j in range(3):
            rp = r.copy(); rp[j] += eps
            rm = r.copy(); rm[j] -= eps
            Up = _potential_jn(rp, 4, Re_star, J_COEFFS[4])
            Um = _potential_jn(rm, 4, Re_star, J_COEFFS[4])
            a_fd[j] = -(Up - Um) / (2.0 * eps)

        np.testing.assert_allclose(a_analytic, a_fd, atol=1e-5)

    def test_higher_order_small(self):
        """J3–J6 합이 J2보다 훨씬 작음."""
        r = np.array([1.0, 0.0, 0.5])
        Re_star = 1.0
        a_j2 = np.linalg.norm(accel_jn(r, 2, Re_star=Re_star))
        a_higher = np.linalg.norm(accel_jn_sum(r, Re_star=Re_star))
        assert a_higher < a_j2 * 0.01  # 1% 미만


class TestAtmosphere:
    """대기 밀도 및 항력."""

    def test_density_surface(self):
        """해수면 밀도 ≈ 1.225 kg/m^3."""
        rho = exponential_density(0.0)
        assert rho == pytest.approx(1.225, rel=0.01)

    def test_density_decreasing(self):
        """고도 증가 → 밀도 감소."""
        rho_100 = exponential_density(100.0)
        rho_200 = exponential_density(200.0)
        rho_400 = exponential_density(400.0)
        assert rho_200 < rho_100
        assert rho_400 < rho_200

    def test_density_high_altitude_tiny(self):
        """고고도 밀도 극히 작음."""
        rho = exponential_density(800.0)
        assert rho < 1e-13

    def test_drag_vanishes_high(self):
        """1000 km 이상에서 항력 = 0."""
        r = np.array([2.5, 0.0, 0.0])  # 정규화, r_mag=2.5
        v = np.array([0.0, 0.6, 0.0])
        a = accel_drag(r, v, Re_star=1.0, DU=6378.137, TU=806.81)
        np.testing.assert_allclose(a, 0.0, atol=1e-20)


class TestThirdBody:
    """3체 섭동."""

    def test_tidal(self):
        """3체 섭동의 조석(tidal) 성질 확인.

        제3체 방향 위성: 직접항 > 간접항 → 양의 가속도 (제3체 방향)
        제3체 반대편 위성: 간접항 > 직접항 → 음의 가속도 (제3체 반대편)
        → 조석력 차이가 존재.
        """
        r_body = np.array([100.0, 0.0, 0.0])
        r1 = np.array([1.0, 0.0, 0.0])
        r2 = np.array([-1.0, 0.0, 0.0])
        mu_star = 0.001

        a1 = accel_third_body(r1, r_body, mu_body_star=mu_star)
        a2 = accel_third_body(r2, r_body, mu_body_star=mu_star)

        # r1이 제3체에 더 가까우므로 x 가속도가 더 큼 (조석 효과)
        assert a1[0] > a2[0]
