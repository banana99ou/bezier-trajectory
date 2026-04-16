"""드리프트 적분 3가지 방법 단위 테스트.

- position_control_points: 형상, 초기조건
- 자기일관 중력 보정 검증
- 각 방법의 SCP 수렴 및 최종 비용 비교
"""

import numpy as np
import pytest

from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.drift import (
    DriftConfig,
    compute_drift,
    position_control_points,
    _corrected_position,
    _gravity_accel_vec,
)
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop, SCPResult
from bezier_orbit.bezier.basis import bernstein_eval


def _make_problem(**overrides):
    """LEO → 1.2 a₀ 공면 원형 전이 문제 생성."""
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
    defaults = dict(r0=r0, v0=v0, rf=rf, vf=vf, t_f=4.0, N=8)
    defaults.update(overrides)
    return SCPProblem(**defaults)


# ── 1. position_control_points ────────────────────────────────


class TestPositionControlPoints:

    def test_shape(self):
        N = 12
        P_u = np.zeros((N + 1, 3))
        r0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.0, 0.0])
        q_r = position_control_points(P_u, r0, v0, t_f=3.0, N=N)
        assert q_r.shape == (N + 3, 3)

    def test_initial_position(self):
        N = 12
        P_u = np.random.randn(N + 1, 3) * 0.1
        r0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.0, 0.0])
        q_r = position_control_points(P_u, r0, v0, t_f=3.0, N=N)
        np.testing.assert_allclose(q_r[0], r0, atol=1e-14)

    def test_zero_control(self):
        N = 8
        P_u = np.zeros((N + 1, 3))
        r0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.0, 0.0])
        t_f = 3.0
        q_r = position_control_points(P_u, r0, v0, t_f, N)
        r_end = bernstein_eval(N + 2, q_r, 1.0)
        expected = r0 + t_f * v0
        np.testing.assert_allclose(r_end, expected, atol=1e-12)


# ── 2. 자기일관 중력 보정 검증 ────────────────────────────────


class TestCorrectedPosition:
    """_corrected_position이 RK4 궤적에 가까운 위치를 생성."""

    def test_correction_improves_position(self):
        """보정 후 위치가 RK4 궤적에 훨씬 가까워짐."""
        from bezier_orbit.scp.inner_loop import _propagate_reference

        prob = _make_problem(drift_config=DriftConfig(method="rk4"))
        res = solve_inner_loop(prob)
        P_u = res.P_u_opt

        # RK4 참조 궤적
        ref_traj = _propagate_reference(prob, P_u, 1.0, n_steps=200)
        tau_mid = np.array([0.5])
        r_rk4_mid = ref_traj[100, :3]

        N = prob.N
        # 보정 없는 위치
        q_r_uncorr = position_control_points(P_u, prob.r0, prob.v0, prob.t_f, N)
        r_uncorr_mid = bernstein_eval(N + 2, q_r_uncorr, tau_mid).flatten()

        # 보정된 위치 (대수적)
        q_r_corr, _ = _corrected_position(
            P_u, prob.r0, prob.v0, prob.t_f, N, 1.0, True, n_iter=8,
            correction_mode="algebraic",
        )
        r_corr_mid = bernstein_eval(N + 2, q_r_corr, tau_mid).flatten()

        err_uncorr = np.linalg.norm(r_uncorr_mid - r_rk4_mid)
        err_corr = np.linalg.norm(r_corr_mid - r_rk4_mid)

        # 보정 후 위치 오차가 10배 이상 줄어야 함
        assert err_corr < err_uncorr / 10, (
            f"Corrected err {err_corr:.4f} not much better than uncorrected {err_uncorr:.4f}"
        )

    def test_algebraic_vs_numerical(self):
        """대수적 보정과 수치적 보정이 유사한 결과를 산출."""
        from bezier_orbit.scp.inner_loop import _propagate_reference

        prob = _make_problem(drift_config=DriftConfig(method="rk4"))
        res = solve_inner_loop(prob)
        P_u = res.P_u_opt
        N = prob.N

        q_r_alg, _ = _corrected_position(
            P_u, prob.r0, prob.v0, prob.t_f, N, 1.0, True,
            n_iter=8, correction_mode="algebraic",
        )
        q_r_num, _ = _corrected_position(
            P_u, prob.r0, prob.v0, prob.t_f, N, 1.0, True,
            n_iter=8, correction_mode="numerical",
        )

        # τ=0.5에서 두 보정 결과 비교
        tau_mid = np.array([0.5])
        r_alg = bernstein_eval(N + 2, q_r_alg, tau_mid).flatten()
        r_num = bernstein_eval(N + 2, q_r_num, tau_mid).flatten()

        diff = np.linalg.norm(r_alg - r_num)
        scale = np.linalg.norm(r_alg)
        assert diff / scale < 0.05, (
            f"Algebraic vs numerical diff {diff/scale:.4f} > 5%"
        )


# ── 3. 드리프트 방법 자체 정합성 ──────────────────────────────


class TestDriftAffine:

    def test_convergence_with_K(self):
        """K 증가 시 c_v 값이 수렴."""
        prob = _make_problem()
        P_u = np.random.randn(prob.N + 1, 3) * 0.01

        cv_values = []
        for K in [4, 8, 16, 32]:
            p = _make_problem(
                drift_config=DriftConfig(method="affine", affine_K=K),
            )
            cv, _ = compute_drift(p, P_u, 1.0)[:2]
            cv_values.append(cv)

        diff_low = np.linalg.norm(cv_values[0] - cv_values[1])
        diff_high = np.linalg.norm(cv_values[2] - cv_values[3])
        assert diff_high < diff_low

    def test_order1_runs(self):
        prob = _make_problem(
            drift_config=DriftConfig(method="affine", affine_K=8, affine_order=1),
        )
        P_u = np.random.randn(prob.N + 1, 3) * 0.01
        cv, cr = compute_drift(prob, P_u, 1.0)[:2]
        assert cv.shape == (3,)
        assert cr.shape == (3,)
        assert np.all(np.isfinite(cv))


class TestDriftBernstein:

    def test_K_convergence(self):
        """K 증가 시 c_v가 수렴."""
        prob = _make_problem()
        P_u = np.random.randn(prob.N + 1, 3) * 0.01

        cv_values = []
        for K in [4, 6, 8, 10]:
            p = _make_problem(
                drift_config=DriftConfig(method="bernstein", bernstein_K=K),
            )
            cv, _ = compute_drift(p, P_u, 1.0)[:2]
            cv_values.append(cv)

        diff_low = np.linalg.norm(cv_values[0] - cv_values[1])
        diff_high = np.linalg.norm(cv_values[2] - cv_values[3])
        assert diff_high < diff_low


# ── 4. SCP 수렴 테스트 ────────────────────────────────────────


class TestInnerLoopWithBernstein:

    def test_converges(self):
        prob = _make_problem(
            max_iter=50, tol_ctrl=1e-3, tol_bc=0.02,
            drift_config=DriftConfig(method="bernstein"),
        )
        result = solve_inner_loop(prob)
        assert isinstance(result, SCPResult)
        assert result.cost > 0
        assert result.cost < float("inf")
        assert result.converged

    def test_cost_close_to_rk4(self):
        prob_rk4 = _make_problem(
            max_iter=50, tol_ctrl=1e-3, tol_bc=0.02,
            drift_config=DriftConfig(method="rk4"),
        )
        prob_bern = _make_problem(
            max_iter=50, tol_ctrl=1e-3, tol_bc=0.02,
            drift_config=DriftConfig(method="bernstein"),
        )

        res_rk4 = solve_inner_loop(prob_rk4)
        res_bern = solve_inner_loop(prob_bern)

        rel_diff = abs(res_bern.cost - res_rk4.cost) / max(abs(res_rk4.cost), 1e-12)
        assert rel_diff < 0.05, f"Cost relative diff {rel_diff:.4f} > 5%"


class TestInnerLoopWithAffine:

    def test_converges(self):
        prob = _make_problem(
            max_iter=50, tol_ctrl=1e-3, tol_bc=0.05,
            drift_config=DriftConfig(method="affine", affine_K=16),
        )
        result = solve_inner_loop(prob)
        assert isinstance(result, SCPResult)
        assert result.cost > 0
        assert result.cost < float("inf")
        assert result.converged


class TestAllMethodsConverge:

    def test_three_methods_similar_cost(self):
        configs = {
            "rk4": DriftConfig(method="rk4"),
            "affine": DriftConfig(method="affine", affine_K=16),
            "bernstein": DriftConfig(method="bernstein"),
        }
        costs = {}
        for name, cfg in configs.items():
            prob = _make_problem(
                max_iter=50, tol_ctrl=1e-3, tol_bc=0.05,
                drift_config=cfg,
            )
            res = solve_inner_loop(prob)
            costs[name] = res.cost

        cost_vals = list(costs.values())
        mean_cost = np.mean(cost_vals)
        for name, c in costs.items():
            rel = abs(c - mean_cost) / mean_cost
            assert rel < 0.10, f"{name} cost {c:.6f} deviates {rel:.2%} from mean"
