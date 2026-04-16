"""SCP (problem, inner_loop, outer_loop) 단위 테스트."""

import math
import numpy as np
import pytest

from bezier_orbit.normalize import from_orbit
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import (
    SCPProblem,
    build_boundary_constraints,
    build_lifted_boundary_constraints,
)
from bezier_orbit.scp.drift import DriftConfig
from bezier_orbit.scp.inner_loop import solve_inner_loop, SCPResult
from bezier_orbit.scp.outer_loop import grid_search, golden_section_search
from bezier_orbit.bezier.basis import int_matrix, double_int_matrix

# Level 1/2 섭동 테스트는 드리프트 적분에 고차 섭동이 반영되어야 하므로 RK4 사용
_RK4_CONFIG = DriftConfig(method="rk4")


class TestBoundaryConstraints:
    """경계조건 선형 제약 구성."""

    def test_shapes(self):
        N = 8
        prob = SCPProblem(
            r0=np.array([1.0, 0.0, 0.0]),
            v0=np.array([0.0, 1.0, 0.0]),
            rf=np.array([0.0, 1.5, 0.0]),
            vf=np.array([-0.8, 0.0, 0.0]),
            t_f=5.0, N=N,
        )
        A, b = build_boundary_constraints(prob)
        assert A.shape == (6, 3 * (N + 1))
        assert b.shape == (6,)

    def test_lifted_shapes(self):
        N = 8
        prob = SCPProblem(
            r0=np.array([1.0, 0.0, 0.0]),
            v0=np.array([0.0, 1.0, 0.0]),
            rf=np.array([0.0, 1.5, 0.0]),
            vf=np.array([-0.8, 0.0, 0.0]),
            t_f=5.0, N=N,
        )
        A, b = build_lifted_boundary_constraints(prob)
        assert A.shape == (6, 3 * (N + 1))
        assert b.shape == (6,)

    def test_lifting_equivalence(self):
        """Z = t_f · P_u 치환 후 등가성: A_lifted @ vec(Z) = A_orig @ vec(P_u)."""
        N = 6
        t_f = 3.0
        prob = SCPProblem(
            r0=np.array([1.0, 0.0, 0.0]),
            v0=np.array([0.0, 1.0, 0.0]),
            rf=np.array([0.0, 1.5, 0.0]),
            vf=np.array([-0.8, 0.0, 0.0]),
            t_f=t_f, N=N,
        )

        A_orig, b_orig = build_boundary_constraints(prob)
        A_lift, b_lift = build_lifted_boundary_constraints(prob)

        # 임의 P_u로 검증: A_orig @ vec(P_u) = b_orig → A_lift @ vec(Z) = b_lift
        rng = np.random.default_rng(42)
        P_u = rng.standard_normal((N + 1, 3))
        Z = t_f * P_u

        vec_Pu = np.concatenate([P_u[:, k] for k in range(3)])
        vec_Z = np.concatenate([Z[:, k] for k in range(3)])

        lhs_orig = A_orig @ vec_Pu
        lhs_lift = A_lift @ vec_Z

        # 둘 다 같은 b (c_v=c_r=0이면 b_orig == b_lift)
        np.testing.assert_allclose(b_orig, b_lift, atol=1e-12)
        np.testing.assert_allclose(lhs_orig, lhs_lift, atol=1e-10)


class TestInnerLoopQP:
    """Inner loop QP (추력 무제약) 테스트."""

    @pytest.fixture
    def simple_problem(self):
        """단순 2체 원형궤도 → 약간 타원궤도 전이."""
        # 정규화: a0 = 1.0 (DU), mu* = 1
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
        return SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8,
            u_max=None,  # QP
            perturbation_level=0,
            max_iter=10,
            tol_ctrl=1e-4,
            tol_bc=1e-3,
        )

    def test_qp_runs(self, simple_problem):
        """QP가 에러 없이 실행되고 유한 비용 반환."""
        result = solve_inner_loop(simple_problem)
        assert isinstance(result, SCPResult)
        assert result.cost < float("inf")
        assert result.n_iter >= 1
        assert result.P_u_opt.shape == (9, 3)

    def test_qp_cost_positive(self, simple_problem):
        """비용이 양수."""
        result = solve_inner_loop(simple_problem)
        assert result.cost > 0.0


class TestInnerLoopSOCP:
    """Inner loop SOCP (추력 제약) 테스트."""

    @pytest.fixture
    def constrained_problem(self):
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
        return SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8,
            u_max=5.0,  # SOCP (큰 값으로 feasibility 확보)
            perturbation_level=0,
            max_iter=10,
            tol_ctrl=1e-4,
            tol_bc=1e-3,
        )

    def test_socp_runs(self, constrained_problem):
        """SOCP가 에러 없이 실행."""
        result = solve_inner_loop(constrained_problem)
        assert isinstance(result, SCPResult)
        assert result.cost < float("inf")

    def test_socp_cost_ge_qp(self):
        """추력 제약 있는 비용 ≥ 무제약 비용."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

        prob_qp = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8, u_max=None, max_iter=5,
        )
        prob_socp = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=8, u_max=5.0, max_iter=5,
        )

        r_qp = solve_inner_loop(prob_qp)
        r_socp = solve_inner_loop(prob_socp)

        # SOCP 비용 ≥ QP 비용 (제약이 추가되므로)
        # 미수렴 상태에서는 수치 오차 허용
        if r_qp.cost < float("inf") and r_socp.cost < float("inf"):
            assert r_socp.cost >= r_qp.cost - 1e-3


class TestInnerLoopLevel1:
    """Inner loop Level 1 섭동 (J3–J6, 대기항력) 테스트."""

    @pytest.fixture
    def level1_problem(self):
        """Level 1 섭동 포함 LEO 전이 (CanonicalUnits 필요)."""
        from bezier_orbit.normalize import CanonicalUnits

        mu_km = 398600.4418
        a0 = 6678.137
        cu = CanonicalUnits(a0=a0, mu=mu_km)

        r0, v0 = keplerian_to_cartesian(a0, 0.001, 0.01, 0.0, 0.0, 0.0, mu_km)
        rf, vf = keplerian_to_cartesian(a0 * 1.2, 0.001, 0.01, 0.0, 0.0, np.pi, mu_km)

        return SCPProblem(
            r0=r0 / cu.DU, v0=v0 / cu.VU,
            rf=rf / cu.DU, vf=vf / cu.VU,
            t_f=3.62, N=8,
            perturbation_level=1,
            canonical_units=cu,
            max_iter=30,
            tol_ctrl=1e-4, tol_bc=1e-2,
            Cd_A_over_m=0.01, max_jn_degree=6,
            drift_config=_RK4_CONFIG,
        )

    def test_level1_runs(self, level1_problem):
        """Level 1이 에러 없이 실행되고 유한 비용 반환."""
        result = solve_inner_loop(level1_problem)
        assert isinstance(result, SCPResult)
        assert result.cost < float("inf")
        assert result.n_iter >= 1

    def test_level1_runs_and_finite(self, level1_problem):
        """Level 1이 유한 비용으로 수렴 (J3-J6+drag 효과는 LEO에서 미미할 수 있음)."""
        import dataclasses
        prob0 = dataclasses.replace(level1_problem, perturbation_level=0)
        res0 = solve_inner_loop(prob0)
        res1 = solve_inner_loop(level1_problem)
        # Level 1이 Level 0과 유사한 비용으로 수렴해야 함
        assert res1.cost < float("inf")
        assert abs(res1.cost - res0.cost) / max(res0.cost, 1e-12) < 0.1


class TestInnerLoopLevel2:
    """Inner loop Level 2 섭동 (SRP, 3체 섭동) 테스트."""

    def test_level2_runs(self):
        """Level 2가 에러 없이 실행."""
        from bezier_orbit.normalize import CanonicalUnits
        from bezier_orbit.orbit.ephemeris import make_body_func, compute_mu_star

        mu_km = 398600.4418
        a0 = 6678.137
        cu = CanonicalUnits(a0=a0, mu=mu_km)

        r0, v0 = keplerian_to_cartesian(a0, 0.001, 0.01, 0.0, 0.0, 0.0, mu_km)
        rf, vf = keplerian_to_cartesian(a0 * 1.2, 0.001, 0.01, 0.0, 0.0, np.pi, mu_km)

        jd_epoch = 2461281.5
        t_f_norm = 3.62
        t_f_sec = t_f_norm * cu.TU

        prob = SCPProblem(
            r0=r0 / cu.DU, v0=v0 / cu.VU,
            rf=rf / cu.DU, vf=vf / cu.VU,
            t_f=t_f_norm, N=8,
            perturbation_level=2,
            canonical_units=cu,
            max_iter=30, tol_ctrl=1e-4, tol_bc=1e-2,
            Cd_A_over_m=0.01, max_jn_degree=6,
            Cr_A_over_m=0.01,
            r_sun_func=make_body_func(jd_epoch, t_f_sec, "sun", cu),
            r_moon_func=make_body_func(jd_epoch, t_f_sec, "moon", cu),
            mu_sun_star=compute_mu_star("sun", cu),
            mu_moon_star=compute_mu_star("moon", cu),
            drift_config=_RK4_CONFIG,
        )
        result = solve_inner_loop(prob)
        assert result.cost < float("inf")
        assert result.n_iter >= 1


class TestNonCoplanar:
    """비공면 3차원 궤도전이 테스트."""

    def test_inclination_change(self):
        """LEO 경사각 변경 (i=0° → i=28.5°)."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.0, 0.0, np.radians(28.5), 0.0, 0.0, np.pi, mu=1.0)
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=12, u_max=None,
            perturbation_level=0,
            max_iter=50, tol_ctrl=1e-6, tol_bc=1e-3,
            drift_config=_RK4_CONFIG,
        )
        res = solve_inner_loop(prob)
        assert res.converged
        assert res.bc_violation < 1e-3

    def test_eccentric_inclined(self):
        """원형→타원(e=0.3) + 경사각 15° 전이."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, np.radians(15), 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.3, 0.3, np.radians(15), 0.0, np.radians(90), np.pi, mu=1.0)
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=5.0, N=12, u_max=None,
            perturbation_level=0,
            max_iter=50, tol_ctrl=1e-6, tol_bc=1e-3,
            drift_config=_RK4_CONFIG,
        )
        res = solve_inner_loop(prob)
        assert res.converged
        assert res.bc_violation < 1e-3

    def test_raan_change_with_path_constraint(self):
        """경사각+RAAN 변경 + r_min 경로 제약."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.01, np.radians(10), 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.2, 0.01, np.radians(30), np.radians(20), 0.0, np.pi, mu=1.0)
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=5.0, N=12, u_max=None,
            perturbation_level=0,
            max_iter=50, tol_ctrl=1e-6, tol_bc=1e-3,
            r_min=0.9, r_max=None,
            drift_config=_RK4_CONFIG,
        )
        res = solve_inner_loop(prob)
        assert res.converged
        assert res.bc_violation < 1e-3

    def test_socp_noncoplanar_with_path(self):
        """SOCP + 비공면 (i=0→20°) + r_min 경로 제약."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.2, 0.0, np.radians(20), 0.0, 0.0, np.pi, mu=1.0)
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=5.0, N=12, u_max=3.0,
            perturbation_level=0,
            max_iter=50, tol_ctrl=1e-6, tol_bc=1e-3,
            r_min=0.9, r_max=None,
            drift_config=_RK4_CONFIG,
        )
        res = solve_inner_loop(prob)
        assert res.converged
        assert res.bc_violation < 1e-3


class TestOuterLoop:
    """Outer loop t_f 탐색 테스트."""

    @pytest.fixture
    def transfer_problem(self):
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.1, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
        return SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=3.0,  # 초기값 (outer loop에서 덮어씀)
            N=8, u_max=None,
            max_iter=5, tol_ctrl=1e-3, tol_bc=1e-2,
            drift_config=_RK4_CONFIG,
        )

    def test_grid_search(self, transfer_problem):
        """격자 탐색이 여러 t_f에서 실행."""
        t_f_grid = np.array([2.0, 3.0, 4.0])
        result = grid_search(transfer_problem, t_f_grid)

        assert len(result.cost_history) == 3
        assert result.t_f_opt in t_f_grid
        assert result.inner_result is not None

    def test_golden_section(self, transfer_problem):
        """황금분할 탐색 실행."""
        result = golden_section_search(
            transfer_problem, (2.0, 5.0),
            tol=0.5, max_eval=4,
        )
        assert len(result.t_f_history) >= 2
        assert result.t_f_opt >= 2.0
        assert result.t_f_opt <= 5.0
