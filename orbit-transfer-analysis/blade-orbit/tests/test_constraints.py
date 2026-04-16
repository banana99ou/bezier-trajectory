"""bezier/constraints.py 단위 테스트."""

import numpy as np
import pytest

from bezier_orbit.bezier.constraints import (
    convex_hull_bound,
    find_extrema,
    extremum_constraint_matrix,
    thrust_socp_matrices,
    thrust_norm_at_grid,
    check_thrust_feasibility,
    path_constraint_matrices,
)
from bezier_orbit.bezier.basis import bernstein_eval, diff_matrix
from bezier_orbit.orbit.elements import keplerian_to_cartesian


class TestConvexHull:
    """볼록껍질 제약."""

    def test_scalar_upper(self):
        P = np.array([1.0, 3.0, 2.0])
        A, b = convex_hull_bound(P, upper=5.0)
        assert A.shape == (3, 3)
        assert np.all(b == 5.0)
        # A = I → p_i ≤ 5
        np.testing.assert_allclose(A, np.eye(3))

    def test_scalar_lower(self):
        P = np.array([1.0, 3.0, 2.0])
        A, b = convex_hull_bound(P, lower=-1.0)
        assert A.shape == (3, 3)
        # -I @ p ≤ -(-1) = 1
        np.testing.assert_allclose(A, -np.eye(3))
        np.testing.assert_allclose(b, 1.0)


class TestExtrema:
    """극값점 계산."""

    def test_monotone_no_extrema(self):
        """단조 증가 곡선 → 내부 극값 없음."""
        P = np.array([0.0, 1.0, 2.0, 3.0])  # N=3
        tau_ext = find_extrema(P, 3)
        assert len(tau_ext) == 0

    def test_symmetric_extremum(self):
        """대칭 곡선 → τ=0.5에서 극값."""
        P = np.array([0.0, 1.0, 1.0, 0.0])  # N=3
        tau_ext = find_extrema(P, 3)
        assert len(tau_ext) >= 1
        # 대칭이므로 0.5 근처에 극값 있어야 함
        assert any(abs(t - 0.5) < 0.1 for t in tau_ext)

    def test_constant_no_extrema(self):
        """상수 곡선 → 극값 없음 (미분이 0 함수)."""
        P = np.array([2.0, 2.0, 2.0, 2.0])
        tau_ext = find_extrema(P, 3)
        # 모든 점이 극값이지만 상수이므로 의미 있는 극값은 없음
        # 우리 구현에서는 빈 배열 반환
        assert len(tau_ext) == 0

    def test_constraint_matrix_shape(self):
        """제약 행렬 크기 확인."""
        N = 5
        P_ref = np.array([0.0, 2.0, -1.0, 3.0, 1.0, 0.0])
        C, tau = extremum_constraint_matrix(P_ref, N)
        # 최소 끝점 2개
        assert C.shape[0] >= 2
        assert C.shape[1] == N + 1
        assert len(tau) == C.shape[0]
        assert tau[0] == 0.0
        assert tau[-1] == 1.0


class TestThrustSOCP:
    """추력 SOCP 제약."""

    def test_grid_shape(self):
        N = 8
        tau, B = thrust_socp_matrices(N, M=16)
        assert tau.shape == (17,)
        assert B.shape == (17, 9)

    def test_default_grid(self):
        N = 10
        tau, B = thrust_socp_matrices(N)
        assert tau.shape == (21,)  # M=2N=20 → 21 points

    def test_norm_calculation(self):
        """추력 노름 계산 검증."""
        N = 4
        rng = np.random.default_rng(42)
        P_u = rng.standard_normal((N + 1, 3))
        tau, B = thrust_socp_matrices(N, M=8)

        norms = thrust_norm_at_grid(P_u, B)
        assert norms.shape == (9,)

        # 수동 계산과 비교
        for j in range(len(tau)):
            u_j = bernstein_eval(N, P_u, tau[j])
            assert norms[j] == pytest.approx(np.linalg.norm(u_j), rel=1e-10)

    def test_feasibility_check(self):
        """작은 제어점 → 큰 u_max에서 feasible."""
        N = 5
        P_u = 0.01 * np.ones((N + 1, 3))
        feasible, viol = check_thrust_feasibility(P_u, u_max=1.0, N=N)
        assert feasible
        assert viol < 0.0


class TestPathConstraint:
    """경로 제약 (궤도 반경 부등식)."""

    def _make_circular_setup(self, N=8):
        """원형궤도 근처 참조 제어점 생성."""
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        P_u_ref = np.zeros((N + 1, 3))  # 무추력 → 원형궤도 유지
        t_f = 3.0
        return P_u_ref, r0, v0, t_f, N

    def test_no_constraints_empty(self):
        """r_min, r_max 모두 None이면 빈 행렬."""
        P_u_ref, r0, v0, t_f, N = self._make_circular_setup()
        A, b = path_constraint_matrices(P_u_ref, r0, v0, t_f, N)
        assert A.shape[0] == 0
        assert b.shape[0] == 0

    def test_output_shapes(self):
        """출력 행렬 크기 확인 — K 구간 × (N+3) 부분 제어점."""
        N = 8
        P_u_ref, r0, v0, t_f, _ = self._make_circular_setup(N)
        K = 4
        A, b = path_constraint_matrices(
            P_u_ref, r0, v0, t_f, N,
            r_max=1.5, K_subdiv=K,
        )
        n_sub_cp = K * (N + 3)  # 각 구간에 (N+3)개 부분 제어점
        assert A.shape == (n_sub_cp, 3 * (N + 1))
        assert b.shape == (n_sub_cp,)

    def test_both_bounds_double_rows(self):
        """r_min + r_max → 행 수 2배."""
        N = 8
        P_u_ref, r0, v0, t_f, _ = self._make_circular_setup(N)
        K = 4
        A1, b1 = path_constraint_matrices(
            P_u_ref, r0, v0, t_f, N,
            r_max=1.5, K_subdiv=K,
        )
        A2, b2 = path_constraint_matrices(
            P_u_ref, r0, v0, t_f, N,
            r_min=0.8, r_max=1.5, K_subdiv=K,
        )
        assert A2.shape[0] == 2 * A1.shape[0]

    def test_loose_constraint_feasible(self):
        """느슨한 제약 → 참조 궤적이 이미 feasible (b_ub > 0)."""
        P_u_ref, r0, v0, t_f, N = self._make_circular_setup()
        # 무추력 직선궤적은 반경 1~3.16이므로 매우 느슨한 범위 설정
        A, b = path_constraint_matrices(
            P_u_ref, r0, v0, t_f, N,
            r_min=0.1, r_max=5.0,
        )
        Z_ref = t_f * P_u_ref
        z_ref_vec = np.concatenate([Z_ref[:, k] for k in range(3)])
        residual = A @ z_ref_vec - b
        assert np.all(residual <= 1e-10), f"max violation = {np.max(residual)}"

    def test_tight_rmax_infeasible_at_ref(self):
        """r_max < max(‖r_ref‖) → 참조 궤적 자체가 제약 위반."""
        P_u_ref, r0, v0, t_f, N = self._make_circular_setup()
        # 무추력 직선궤적은 반경 최대 ~3.16, r_max=2.0이면 후반부 위반
        A, b = path_constraint_matrices(
            P_u_ref, r0, v0, t_f, N,
            r_max=2.0,
        )
        Z_ref = t_f * P_u_ref
        z_ref_vec = np.concatenate([Z_ref[:, k] for k in range(3)])
        residual = A @ z_ref_vec - b
        assert np.any(residual > 0), "제약이 위반되어야 함"
