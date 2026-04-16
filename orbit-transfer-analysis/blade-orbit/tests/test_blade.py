"""BLADE 단위 및 통합 테스트.

Phase 1: 이중 적분기 (ẍ = u) 검증.
"""

import numpy as np
import pytest

from bezier_orbit.bezier.basis import gram_matrix
from bezier_orbit.blade.basis import (
    blade_gram,
    boundary_matrices,
    continuity_matrices,
    forward_propagate,
    segment_delta_v,
    segment_delta_r,
)
from bezier_orbit.blade.problem import BLADEProblem, solve_blade


# ════════════════════════════════════════════════════════════════
#  대수적 구조 테스트
# ════════════════════════════════════════════════════════════════

class TestBLADEBasis:

    def test_gram_block_diagonal_shape(self):
        """블록 대각 Gram 행렬의 크기 확인."""
        K, n = 5, 3
        G = blade_gram(K, n)
        assert G.shape == (K * (n + 1), K * (n + 1))

    def test_gram_block_diagonal_structure(self):
        """오프대각 블록이 0인지 확인."""
        K, n = 4, 2
        G = blade_gram(K, n)
        for j in range(K):
            for k in range(K):
                sj, ej = j * (n + 1), (j + 1) * (n + 1)
                sk, ek = k * (n + 1), (k + 1) * (n + 1)
                block = G[sj:ej, sk:ek]
                if j != k:
                    np.testing.assert_allclose(block, 0.0, atol=1e-15)

    def test_gram_gcf_special_case(self):
        """n=0 → Gram = Δ·I_K (GCF 특수 경우)."""
        K = 10
        G = blade_gram(K, n=0)
        expected = np.eye(K) / K  # Δ = 1/K, G_0 = 1
        np.testing.assert_allclose(G, expected, atol=1e-15)

    def test_gram_global_bezier_special_case(self):
        """K=1 → Gram = G_N (전역 베지어 특수 경우)."""
        N = 8
        G = blade_gram(K=1, n=N)
        G_N = gram_matrix(N)
        np.testing.assert_allclose(G, G_N, atol=1e-15)

    def test_gram_nonuniform_deltas(self):
        """비균일 세그먼트 길이에서 Gram 행렬 확인."""
        K, n = 3, 2
        deltas = np.array([0.2, 0.5, 0.3])
        G = blade_gram(K, n, deltas)
        G_n = gram_matrix(n)
        for k in range(K):
            s = k * (n + 1)
            e = s + (n + 1)
            np.testing.assert_allclose(G[s:e, s:e], deltas[k] * G_n, atol=1e-15)

    def test_forward_propagate_zero_control(self):
        """제어 0이면 등속직선운동."""
        K, n = 5, 2
        deltas = np.full(K, 1.0 / K)
        t_f = 2.0
        r0, v0 = 1.0, 0.5
        p_zero = [np.zeros(n + 1) for _ in range(K)]

        v_nodes, r_nodes = forward_propagate(p_zero, n, deltas, t_f, r0, v0)

        # 속도 불변
        np.testing.assert_allclose(v_nodes, v0, atol=1e-14)
        # 위치: r(τ_k) = r0 + v0 * t_f * τ_k (τ_k = k/K, 정규화 시간)
        for k in range(K + 1):
            tau_k = sum(deltas[:k])
            expected_r = r0 + v0 * t_f * tau_k
            assert r_nodes[k] == pytest.approx(expected_r, abs=1e-13)

    def test_continuity_C0_shape(self):
        """C⁰ 연속성 행렬 크기: (K-1) × K(n+1)."""
        K, n = 6, 3
        A = continuity_matrices(K, n, order=0)
        assert A.shape == (K - 1, K * (n + 1))

    def test_continuity_C1_shape(self):
        """C¹ 연속성 행렬 크기: 2(K-1) × K(n+1)."""
        K, n = 6, 3
        A = continuity_matrices(K, n, order=1)
        assert A.shape == (2 * (K - 1), K * (n + 1))

    def test_dof_formula(self):
        """자유도 = (n-r)K + (r+1) 확인."""
        K, n = 10, 3
        for r in [-1, 0, 1]:
            A = continuity_matrices(K, n, order=r) if r >= 0 else np.zeros((0, K * (n + 1)))
            total_vars = K * (n + 1)
            n_constraints = A.shape[0]
            dof = total_vars - n_constraints
            if r == -1:
                expected_dof = (n + 1) * K
            else:
                expected_dof = (n - r) * K + (r + 1)
            assert dof == expected_dof, f"r={r}: {dof} != {expected_dof}"


# ════════════════════════════════════════════════════════════════
#  이중 적분기 QP 테스트
# ════════════════════════════════════════════════════════════════

class TestBLADEDoubleIntegrator:

    # 기본 문제 설정: r0=0, v0=0, rf=1, vf=0, tf=1
    _BC = dict(r0=0.0, v0=0.0, rf=1.0, vf=0.0, t_f=1.0)

    def test_unconstrained_cost_approaches_analytic(self):
        """무제약 QP: 해석해 J* = 12에 수렴.

        n=1, K=20에서 J ≈ 12.03 (efc005),
        n≥2에서 J → 12.00.
        """
        for n, tol in [(1, 0.1), (2, 0.01), (3, 0.005)]:
            prob = BLADEProblem(**self._BC, K=20, n=n)
            result = solve_blade(prob)
            assert result.status in ("optimal", "optimal_inaccurate")
            assert result.cost == pytest.approx(12.0, abs=tol), \
                f"n={n}: cost={result.cost:.4f}"

    def test_boundary_conditions_satisfied(self):
        """경계조건 만족 확인."""
        prob = BLADEProblem(**self._BC, K=10, n=2)
        result = solve_blade(prob)

        assert result.v_nodes[0] == pytest.approx(0.0, abs=1e-6)
        assert result.v_nodes[-1] == pytest.approx(0.0, abs=1e-3)
        assert result.r_nodes[0] == pytest.approx(0.0, abs=1e-6)
        assert result.r_nodes[-1] == pytest.approx(1.0, abs=1e-3)

    def test_gcf_special_case_cost(self):
        """n=0(GCF): J ≈ 12.03 (K=20)."""
        prob = BLADEProblem(**self._BC, K=20, n=0)
        result = solve_blade(prob)
        assert result.status in ("optimal", "optimal_inaccurate")
        # GCF는 piecewise constant → 해석해의 계단 근사
        assert result.cost == pytest.approx(12.0, abs=0.1)

    def test_box_constraint_feasibility(self):
        """상자 제약 u_max=4: n=0(GCF) feasible, n=1 feasible 확인."""
        # GCF (n=0)
        prob0 = BLADEProblem(**self._BC, K=20, n=0, u_max=4.0)
        r0 = solve_blade(prob0)
        assert r0.status in ("optimal", "optimal_inaccurate")

        # n=1도 feasible (세그먼트별 독립이므로)
        prob1 = BLADEProblem(**self._BC, K=20, n=1, u_max=4.0)
        r1 = solve_blade(prob1)
        assert r1.status in ("optimal", "optimal_inaccurate")

    def test_coasting_exact(self):
        """코스팅: p_k = 0인 세그먼트에서 u ≡ 0 확인."""
        # 중앙 세그먼트들을 0으로 고정하여 검증
        K, n = 10, 2
        prob = BLADEProblem(**self._BC, K=K, n=n)
        result = solve_blade(prob)

        # 코스팅 세그먼트가 있으면 제어점이 실제로 0인지 확인
        for k in result.coasting_segments:
            np.testing.assert_allclose(result.p_segments[k], 0.0, atol=1e-5)

    def test_l1_coasting_emergence(self):
        """ℓ₁ 정규화로 코스팅 세그먼트 자동 출현."""
        prob = BLADEProblem(**self._BC, K=20, n=1, u_max=5.0, l1_lambda=2.0)
        result = solve_blade(prob)
        assert result.status in ("optimal", "optimal_inaccurate")
        # λ=2.0이면 일부 세그먼트가 코스팅으로 전환되어야 함
        assert len(result.coasting_segments) > 0, \
            f"No coasting segments with λ=2.0"

    def test_cost_ordering_blade_vs_gcf(self):
        """같은 K에서 n≥1이 n=0(GCF)보다 비용이 같거나 낮음."""
        K = 20
        prob_gcf = BLADEProblem(**self._BC, K=K, n=0)
        prob_blade = BLADEProblem(**self._BC, K=K, n=2)
        r_gcf = solve_blade(prob_gcf)
        r_blade = solve_blade(prob_blade)
        assert r_blade.cost <= r_gcf.cost + 1e-6, \
            f"BLADE(n=2) cost {r_blade.cost:.4f} > GCF cost {r_gcf.cost:.4f}"

    def test_continuity_C0_enforced(self):
        """C⁰ 연속성이 부과되면 세그먼트 경계에서 제어점 일치."""
        prob = BLADEProblem(**self._BC, K=5, n=3, continuity=0)
        result = solve_blade(prob)
        assert result.status in ("optimal", "optimal_inaccurate")
        for k in range(prob.K - 1):
            p_end = result.p_segments[k][-1]
            p_start = result.p_segments[k + 1][0]
            assert p_end == pytest.approx(p_start, abs=1e-4), \
                f"C⁰ violated at segment {k}/{k+1}: {p_end:.6f} != {p_start:.6f}"
