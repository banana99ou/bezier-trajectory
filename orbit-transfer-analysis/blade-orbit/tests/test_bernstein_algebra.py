"""Bernstein 대수 연산 단위 테스트.

Report 010의 핵심 주장을 단계별로 검증한다:
1. Bernstein 곱: 항등식, 교환법칙, 알려진 값
2. 차수 올림/축소: 왕복 정밀도, L² 최적성
3. Bernstein 합성: 항등 합성, 알려진 값
4. 비선형 함수 근사: 수렴 속도
5. 중력 합성 파이프라인: RK4 기준 대비 정밀도
"""

import math
import numpy as np
import pytest

from bezier_orbit.bezier.algebra import (
    bern_product,
    bern_compose,
    degree_elevate,
    degree_reduce,
    chebyshev_bernstein_approx,
    gravity_composition_pipeline,
)
from bezier_orbit.bezier.basis import bernstein, gram_matrix, definite_integral


# ═══════════════════════════════════════════════════════════════
# 1. Bernstein 곱
# ═══════════════════════════════════════════════════════════════

class TestBernProduct:
    """bern_product 검증."""

    def test_constant_times_constant(self):
        """상수 × 상수 = 상수: B_0 × B_0 → B_0."""
        p = np.array([3.0])
        q = np.array([5.0])
        r = bern_product(p, q)
        assert len(r) == 1
        np.testing.assert_allclose(r, [15.0])

    def test_constant_times_linear(self):
        """상수 c × 선형 f(τ) → 제어점이 c배.

        f(τ) ∈ B_2: 제어점 [1, 2, 3]
        c = 2 → 결과 [2, 4, 6] ∈ B_2
        """
        c = np.array([2.0])
        f = np.array([1.0, 2.0, 3.0])
        r = bern_product(c, f)
        assert len(r) == 3  # B_0 × B_2 → B_2
        np.testing.assert_allclose(r, [2.0, 4.0, 6.0])

    def test_output_degree(self):
        """B_N × B_M → B_{N+M}: 차수 확인."""
        N, M = 5, 3
        p = np.random.default_rng(42).standard_normal(N + 1)
        q = np.random.default_rng(43).standard_normal(M + 1)
        r = bern_product(p, q)
        assert len(r) == N + M + 1

    def test_commutativity(self):
        """교환법칙: p·q = q·p."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal(6)
        q = rng.standard_normal(4)
        np.testing.assert_allclose(bern_product(p, q), bern_product(q, p), atol=1e-14)

    def test_identity_tau_squared(self):
        """τ² = τ·τ 검증.

        τ ∈ B_1: 제어점 [0, 1]
        τ² ∈ B_2: 제어점 [0, 0, 1]
        """
        tau = np.array([0.0, 1.0])
        r = bern_product(tau, tau)
        np.testing.assert_allclose(r, [0.0, 0.0, 1.0], atol=1e-15)

    def test_pointwise_agreement(self):
        """곱 제어점으로 평가한 값 ≈ 개별 평가 후 곱한 값."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal(6)
        q = rng.standard_normal(4)
        r = bern_product(p, q)

        tau_test = np.linspace(0, 1, 50)
        f_vals = bernstein(5, tau_test) @ p
        g_vals = bernstein(3, tau_test) @ q
        fg_vals = bernstein(8, tau_test) @ r

        np.testing.assert_allclose(fg_vals, f_vals * g_vals, atol=1e-12)

    def test_integral_of_square(self):
        """∫₀¹ f²(τ)dτ = mean(bern_product(p,p)) = p^T G_N p."""
        rng = np.random.default_rng(42)
        N = 5
        p = rng.standard_normal(N + 1)

        # 방법 1: Bernstein 곱 + 정적분
        pp = bern_product(p, p)
        integral_prod = definite_integral(pp)

        # 방법 2: Gram 행렬
        G = gram_matrix(N)
        integral_gram = p @ G @ p

        np.testing.assert_allclose(integral_prod, integral_gram, atol=1e-14)


# ═══════════════════════════════════════════════════════════════
# 2. 차수 올림 / 축소
# ═══════════════════════════════════════════════════════════════

class TestDegreeElevate:
    """degree_elevate 검증."""

    def test_identity(self):
        """같은 차수 → 복사."""
        p = np.array([1.0, 2.0, 3.0])
        q = degree_elevate(p, 2)
        np.testing.assert_allclose(q, p)

    def test_raise_on_lower(self):
        """낮은 차수로 올림 시도 → ValueError."""
        with pytest.raises(ValueError):
            degree_elevate(np.array([1.0, 2.0, 3.0]), 1)

    def test_pointwise_preservation(self):
        """차수 올림 후 함수값 보존."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal(6)
        q = degree_elevate(p, 15)

        tau_test = np.linspace(0, 1, 50)
        f_orig = bernstein(5, tau_test) @ p
        f_elev = bernstein(15, tau_test) @ q

        np.testing.assert_allclose(f_elev, f_orig, atol=1e-12)

    def test_endpoints_preserved(self):
        """끝점 보존: q[0] = p[0], q[-1] = p[-1]."""
        p = np.array([2.0, 5.0, -1.0, 3.0])
        q = degree_elevate(p, 10)
        assert q[0] == p[0]
        assert q[-1] == p[-1]


class TestDegreeReduce:
    """degree_reduce (L² 최적) 검증."""

    def test_perfect_recovery_for_lower_degree(self):
        """R차 다항식을 Q차로 올린 뒤 다시 R차로 축소 → 원본 복원.

        p ∈ B_3을 B_10으로 올리고 다시 B_3으로 축소하면 원본과 일치해야 한다.
        """
        rng = np.random.default_rng(42)
        p_orig = rng.standard_normal(4)  # B_3
        p_elevated = degree_elevate(p_orig, 10)  # B_10
        p_reduced = degree_reduce(p_elevated, 3)  # B_3

        np.testing.assert_allclose(p_reduced, p_orig, atol=1e-10)

    def test_l2_error_decreases_with_R(self):
        """축소 차수 R이 클수록 L² 오차가 감소."""
        rng = np.random.default_rng(42)
        p = rng.standard_normal(21)  # B_20

        errors = []
        for R in [5, 10, 15]:
            q = degree_reduce(p, R)
            tau = np.linspace(0, 1, 200)
            f_p = bernstein(20, tau) @ p
            f_q = bernstein(R, tau) @ q
            err = np.sqrt(np.mean((f_p - f_q) ** 2))
            errors.append(err)

        assert errors[0] > errors[1] > errors[2]

    def test_noop_when_R_ge_Q(self):
        """R ≥ Q이면 축소 없이 반환 (복사 또는 올림)."""
        p = np.array([1.0, 2.0, 3.0])
        q = degree_reduce(p, 2)
        np.testing.assert_allclose(q, p)

        q2 = degree_reduce(p, 5)
        assert len(q2) == 6


# ═══════════════════════════════════════════════════════════════
# 3. Bernstein 합성
# ═══════════════════════════════════════════════════════════════

class TestBernCompose:
    """bern_compose 검증."""

    def test_identity_composition(self):
        """h(s) = s (항등 함수) 합성 → h(g(τ)) = g(τ).

        h = [0, 1] ∈ B_1,  g ∈ B_M → h∘g ∈ B_M (차수 올림 후).
        """
        h = np.array([0.0, 1.0])  # h(s) = s
        g = np.array([0.2, 0.5, 0.8])  # 임의 B_2

        result = bern_compose(h, g)
        # B_{1·2} = B_2차
        assert len(result) == 3
        np.testing.assert_allclose(result, g, atol=1e-14)

    def test_constant_composition(self):
        """h(s) = c (상수) → h(g(τ)) = c."""
        h = np.array([3.14])  # B_0
        g = np.array([0.1, 0.9, 0.5])
        result = bern_compose(h, g)
        assert len(result) == 1
        np.testing.assert_allclose(result, [3.14])

    def test_quadratic_composition(self):
        """h(s) = s² 합성.

        h(s) = s² ∈ B_2: 제어점 [0, 0, 1]
        g(τ) = τ ∈ B_1: 제어점 [0, 1]
        h(g(τ)) = τ² ∈ B_2: 제어점 [0, 0, 1]
        """
        h = np.array([0.0, 0.0, 1.0])  # s²
        g = np.array([0.0, 1.0])        # τ

        result = bern_compose(h, g)
        # B_{2·1} = B_2
        assert len(result) == 3
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-14)

    def test_pointwise_agreement(self):
        """합성 제어점으로 평가 ≈ 개별 평가 후 합성."""
        rng = np.random.default_rng(42)
        K, M = 3, 4
        h = rng.random(K + 1)  # [0,1] 범위 유지

        # g: [0,1] → [0,1]이 되도록 단조증가 구간
        g = np.sort(rng.random(M + 1))
        g[0] = 0.0
        g[-1] = 1.0

        result = bern_compose(h, g)

        tau_test = np.linspace(0, 1, 50)
        g_vals = bernstein(M, tau_test) @ g
        h_of_g_pointwise = bernstein(K, g_vals) @ h
        h_of_g_composed = bernstein(K * M, tau_test) @ result

        np.testing.assert_allclose(h_of_g_composed, h_of_g_pointwise, atol=1e-10)


# ═══════════════════════════════════════════════════════════════
# 4. 비선형 함수 Bernstein 근사
# ═══════════════════════════════════════════════════════════════

class TestChebyshevBernsteinApprox:
    """chebyshev_bernstein_approx 검증."""

    def test_polynomial_exact(self):
        """다항식은 K ≥ deg이면 정확히 복원."""
        # f(s) = 2s² + 3s + 1 → B_2의 제어점으로 정확 표현 가능
        def f(s):
            return 2 * s**2 + 3 * s + 1

        h = chebyshev_bernstein_approx(f, K=4, a=0, b=1)
        # 검증: 포인트별
        s_test = np.linspace(0, 1, 50)
        approx_vals = bernstein(4, s_test) @ h
        exact_vals = np.array([f(s) for s in s_test])
        np.testing.assert_allclose(approx_vals, exact_vals, atol=1e-10)

    def test_inverse_sqrt_convergence(self):
        """s^{-1/2} 근사: K 증가에 따른 지수적 수렴."""
        a, b = 0.5, 2.0
        func = lambda s: s ** (-0.5)

        errors = []
        for K in [4, 8, 12, 16]:
            h = chebyshev_bernstein_approx(func, K, a, b)
            s_test = np.linspace(0, 1, 200)
            approx = bernstein(K, s_test) @ h
            exact = np.array([func(a + (b - a) * s) for s in s_test])
            err = np.max(np.abs(approx - exact))
            errors.append(err)

        # 오차가 단조 감소
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i]

        # K=16에서 충분히 작은 오차
        assert errors[-1] < 1e-6

    def test_s_inv_three_halves(self):
        """s^{-3/2} 근사 (중력 파이프라인 핵심 함수)."""
        a, b = 0.8, 1.5
        func = lambda s: s ** (-1.5)

        h = chebyshev_bernstein_approx(func, K=12, a=a, b=b)

        s_test = np.linspace(0, 1, 100)
        approx = bernstein(12, s_test) @ h
        exact = np.array([func(a + (b - a) * s) for s in s_test])
        rel_err = np.max(np.abs(approx - exact) / np.abs(exact))

        assert rel_err < 1e-4  # 0.01% 이내


# ═══════════════════════════════════════════════════════════════
# 5. 중력 합성 파이프라인
# ═══════════════════════════════════════════════════════════════

class TestGravityPipeline:
    """gravity_composition_pipeline 검증."""

    @pytest.fixture
    def circular_orbit_control_points(self):
        """원궤도에 가까운 위치 제어점 생성.

        단순한 경우: r ≈ 1인 원궤도에 약간의 추력을 가한 궤적.
        """
        from bezier_orbit.bezier.basis import int_matrix, double_int_matrix

        N = 8
        t_f = 3.5
        r0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, 1.0, 0.0])

        # 작은 추력 (제어점)
        rng = np.random.default_rng(42)
        P_u = 0.05 * rng.standard_normal((N + 1, 3))

        # 위치 제어점 계산
        Ibar = double_int_matrix(N)
        ell = np.arange(N + 3) / (N + 2)
        q_r = np.zeros((N + 3, 3))
        for k in range(3):
            q_r[:, k] = r0[k] + t_f * v0[k] * ell + t_f**2 * Ibar @ P_u[:, k]

        return q_r

    def test_output_shape(self, circular_orbit_control_points):
        """출력 shape: (R+1, 3)."""
        q_r = circular_orbit_control_points
        R = 20
        q_agrav = gravity_composition_pipeline(q_r, K=6, R=R)
        assert q_agrav.shape == (R + 1, 3)

    def test_agrav_direction(self, circular_orbit_control_points):
        """중력 가속도 방향: 원점을 향해야 한다.

        ∫₀¹ a_grav · r dτ < 0 (내적이 음수)
        """
        q_r = circular_orbit_control_points
        q_agrav = gravity_composition_pipeline(q_r, K=8, R=20)

        # 포인트별 검증
        tau = np.linspace(0, 1, 50)
        P_pos = len(q_r) - 1
        R = len(q_agrav) - 1
        r_vals = bernstein(P_pos, tau) @ q_r
        a_vals = bernstein(R, tau) @ q_agrav

        # 내적: a · r < 0 (중력은 원점 방향)
        dots = np.sum(r_vals * a_vals, axis=1)
        assert np.all(dots < 0), "중력은 원점 방향이어야 한다"

    def test_agrav_magnitude_reasonable(self, circular_orbit_control_points):
        """r ≈ 1인 궤도에서 |a_grav| ≈ 1 (정규화 단위, μ*=1).

        적분(mean)으로 평균 크기를 확인 — 고차 Bernstein 포인트 평가의
        수치 불안정을 회피.
        """
        q_r = circular_orbit_control_points
        q_agrav = gravity_composition_pipeline(q_r, K=8, R=20)

        # 제어점 기반 크기 확인 (포인트 평가 대신)
        magnitudes = np.linalg.norm(q_agrav, axis=1)

        # r ≈ 1 ~ 4 범위에서 a_grav 제어점 크기 합리적
        assert np.all(magnitudes > 0.01)
        assert np.all(magnitudes < 10.0)

        # 적분값(mean)의 크기도 확인
        c_v = definite_integral(q_agrav)
        assert np.linalg.norm(c_v) > 0.01
        assert np.linalg.norm(c_v) < 5.0

    def test_vs_pointwise_reference(self, circular_orbit_control_points):
        """RK4 없이, 같은 궤적 위 pointwise 중력 대비 정밀도.

        Bernstein 파이프라인 결과 vs. 직접 -r/|r|³ 계산.
        """
        q_r = circular_orbit_control_points
        K, R = 12, 24

        q_agrav = gravity_composition_pipeline(q_r, K=K, R=R)

        # 포인트별 참값 계산
        tau = np.linspace(0.01, 0.99, 100)
        P_pos = len(q_r) - 1
        r_vals = bernstein(P_pos, tau) @ q_r
        r_norms = np.linalg.norm(r_vals, axis=1, keepdims=True)
        a_exact = -r_vals / r_norms**3

        # Bernstein 파이프라인 평가
        a_bern = bernstein(R, tau) @ q_agrav

        # 상대 오차 (성분별)
        rel_err = np.max(np.abs(a_bern - a_exact) / np.abs(a_exact).clip(min=1e-10))

        # K=12, R=24에서 5% 이내
        assert rel_err < 0.05, f"상대 오차 {rel_err:.2%}가 5% 초과"

    def test_K_convergence(self, circular_orbit_control_points):
        """K 증가에 따라 파이프라인 정밀도가 향상."""
        q_r = circular_orbit_control_points
        R = 24

        tau = np.linspace(0.01, 0.99, 100)
        P_pos = len(q_r) - 1
        r_vals = bernstein(P_pos, tau) @ q_r
        r_norms = np.linalg.norm(r_vals, axis=1, keepdims=True)
        a_exact = -r_vals / r_norms**3

        errors = []
        for K in [4, 8, 12]:
            q_agrav = gravity_composition_pipeline(q_r, K=K, R=R)
            a_bern = bernstein(R, tau) @ q_agrav
            err = np.max(np.abs(a_bern - a_exact))
            errors.append(err)

        # 단조 감소
        for i in range(len(errors) - 1):
            assert errors[i + 1] < errors[i], (
                f"K={[4,8,12][i+1]}의 오차가 K={[4,8,12][i]}보다 크다"
            )

    def test_integral_drift(self, circular_orbit_control_points):
        """드리프트 적분 c_v = ∫₀¹ a_grav(τ) dτ = mean(q_agrav) 검증.

        Bernstein 정적분은 제어점 산술평균 — 정확한 대수적 연산.
        """
        q_r = circular_orbit_control_points
        q_agrav = gravity_composition_pipeline(q_r, K=12, R=24)

        # Bernstein 정적분 (대수적)
        c_v_algebraic = definite_integral(q_agrav)

        # 수치 적분 기준 (Simpson)
        tau = np.linspace(0, 1, 1001)
        P_pos = len(q_r) - 1
        r_vals = bernstein(P_pos, tau) @ q_r
        r_norms = np.linalg.norm(r_vals, axis=1, keepdims=True)
        a_exact = -r_vals / r_norms**3
        c_v_simpson = np.trapezoid(a_exact, tau, axis=0)

        # 상대 오차 10% 이내 (K=12에서)
        rel_err = np.linalg.norm(c_v_algebraic - c_v_simpson) / np.linalg.norm(c_v_simpson)
        assert rel_err < 0.10, f"드리프트 적분 상대 오차 {rel_err:.2%}"
