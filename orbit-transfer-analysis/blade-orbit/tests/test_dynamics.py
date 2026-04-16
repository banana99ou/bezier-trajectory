"""orbit/dynamics.py 단위 테스트."""

import numpy as np
import pytest

from bezier_orbit.orbit.dynamics import (
    eom_twobody_j2,
    jacobian_twobody_j2,
    propagate_rk4,
)
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.normalize import from_orbit, J2_EARTH


class TestTwoBody:
    """2체 동역학 기본 테스트."""

    def test_circular_orbit_acceleration(self):
        """원형 궤도에서 구심가속도 = 1/r² (mu*=1)."""
        r = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        x = np.concatenate([r, v])

        xdot = eom_twobody_j2(x, include_j2=False)

        # v̇ = -r/r³ = [-1, 0, 0]
        np.testing.assert_allclose(xdot[3:], [-1.0, 0.0, 0.0], atol=1e-15)
        # ṙ = v
        np.testing.assert_allclose(xdot[:3], v, atol=1e-15)

    def test_j2_perturbation_small(self):
        """J2 섭동이 중심 중력 대비 O(10^{-3})임을 확인."""
        r = np.array([1.0, 0.0, 0.5])
        v = np.array([0.0, 0.8, 0.0])
        x = np.concatenate([r, v])

        xdot_no_j2 = eom_twobody_j2(x, include_j2=False)
        xdot_j2 = eom_twobody_j2(x, include_j2=True, Re_star=1.0)

        diff = np.linalg.norm(xdot_j2[3:] - xdot_no_j2[3:])
        grav = np.linalg.norm(xdot_no_j2[3:])

        # J2 / gravity ~ O(10^{-3})
        ratio = diff / grav
        assert ratio < 0.01  # 1% 미만


class TestJacobian:
    """상태 Jacobian 유한차분 검증."""

    def test_jacobian_finite_diff(self):
        """해석적 Jacobian vs 유한차분 비교."""
        r = np.array([1.2, -0.3, 0.8])
        v = np.array([-0.1, 0.9, 0.2])
        x = np.concatenate([r, v])
        Re_star = 0.94

        A_analytic = jacobian_twobody_j2(x, Re_star=Re_star, include_j2=True)

        # 유한차분
        eps = 1e-7
        A_fd = np.zeros((6, 6))
        for j in range(6):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = eom_twobody_j2(xp, include_j2=True, Re_star=Re_star)
            fm = eom_twobody_j2(xm, include_j2=True, Re_star=Re_star)
            A_fd[:, j] = (fp - fm) / (2.0 * eps)

        np.testing.assert_allclose(A_analytic, A_fd, atol=1e-5)

    def test_jacobian_twobody_only(self):
        """J2 없이 2체 Jacobian 유한차분."""
        r = np.array([2.0, 0.5, -0.3])
        v = np.array([0.0, 0.7, 0.1])
        x = np.concatenate([r, v])

        A_analytic = jacobian_twobody_j2(x, include_j2=False)

        eps = 1e-7
        A_fd = np.zeros((6, 6))
        for j in range(6):
            xp = x.copy()
            xm = x.copy()
            xp[j] += eps
            xm[j] -= eps
            fp = eom_twobody_j2(xp, include_j2=False)
            fm = eom_twobody_j2(xm, include_j2=False)
            A_fd[:, j] = (fp - fm) / (2.0 * eps)

        np.testing.assert_allclose(A_analytic, A_fd, atol=1e-6)


class TestPropagation:
    """궤도 전파 테스트."""

    def test_circular_orbit_period(self):
        """원형 궤도 1주기 전파 후 초기 상태 복원 (2체)."""
        # 정규화 단위에서 a* = 1, mu* = 1 → T* = 2π
        r0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.0, 0.0])
        x0 = np.concatenate([r0, v0])

        t_f = 2.0 * np.pi  # 1주기
        tau_arr, x_arr = propagate_rk4(
            x0, (0.0, 1.0), n_steps=2000,
            include_j2=False, t_f=t_f,
        )

        # 1주기 후 초기 상태 복원
        np.testing.assert_allclose(x_arr[-1, :3], r0, atol=1e-6)
        np.testing.assert_allclose(x_arr[-1, 3:], v0, atol=1e-6)

    def test_energy_conservation_twobody(self):
        """2체 문제 에너지 보존."""
        r0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.1, 0.0])  # 약간 타원
        x0 = np.concatenate([r0, v0])

        t_f = 10.0
        tau_arr, x_arr = propagate_rk4(
            x0, (0.0, 1.0), n_steps=5000,
            include_j2=False, t_f=t_f,
        )

        def energy(x):
            r = np.linalg.norm(x[:3])
            v = np.linalg.norm(x[3:])
            return 0.5 * v**2 - 1.0 / r

        e0 = energy(x0)
        energies = np.array([energy(x_arr[k]) for k in range(len(x_arr))])
        # 에너지 변동 < 1e-8
        assert np.max(np.abs(energies - e0)) < 1e-8
