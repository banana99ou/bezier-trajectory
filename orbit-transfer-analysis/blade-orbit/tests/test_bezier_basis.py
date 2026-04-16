"""bezier/basis.py 단위 테스트."""

import math
import numpy as np
import pytest

from bezier_orbit.bezier.basis import (
    bernstein,
    bernstein_eval,
    diff_matrix,
    int_matrix,
    double_int_matrix,
    gram_matrix,
    block_hessian,
    definite_integral,
)


class TestBernstein:
    """Bernstein 기저함수 성질."""

    @pytest.mark.parametrize("N", [3, 8, 12, 20])
    def test_partition_of_unity(self, N):
        """∑ B_i^N(τ) = 1 for all τ."""
        for tau in np.linspace(0, 1, 50):
            B = bernstein(N, tau)
            assert B.sum() == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("N", [3, 8, 12])
    def test_nonnegative(self, N):
        """B_i^N(τ) ≥ 0."""
        for tau in np.linspace(0, 1, 100):
            B = bernstein(N, tau)
            assert np.all(B >= -1e-15)

    @pytest.mark.parametrize("N", [3, 8, 12])
    def test_endpoint_interpolation(self, N):
        """B_N(0) = e_0, B_N(1) = e_N."""
        B0 = bernstein(N, 0.0)
        assert B0[0] == pytest.approx(1.0)
        assert np.sum(np.abs(B0[1:])) < 1e-15

        B1 = bernstein(N, 1.0)
        assert B1[N] == pytest.approx(1.0)
        assert np.sum(np.abs(B1[:N])) < 1e-15

    def test_vectorized(self):
        """배열 입력 테스트."""
        N = 5
        tau = np.array([0.0, 0.5, 1.0])
        B = bernstein(N, tau)
        assert B.shape == (3, N + 1)


class TestDiffMatrix:
    """미분 행렬 D_N 테스트."""

    def test_shape(self):
        N = 5
        D = diff_matrix(N)
        assert D.shape == (N, N + 1)

    def test_derivative_constant(self):
        """상수 베지어 곡선(모든 제어점 동일)의 미분 = 0."""
        N = 5
        P = np.ones(N + 1)
        D = diff_matrix(N)
        deriv_ctrl = D @ P
        np.testing.assert_allclose(deriv_ctrl, 0.0, atol=1e-15)


class TestIntMatrix:
    """적분 행렬 I_N 테스트."""

    def test_shape(self):
        N = 5
        I = int_matrix(N)
        assert I.shape == (N + 2, N + 1)

    def test_definite_integral_identity(self):
        """∫₀¹ q(τ) dτ = mean(P) (정적분 = 산술평균).

        보고서 004, 식 (15).
        """
        N = 8
        rng = np.random.default_rng(42)
        P = rng.standard_normal(N + 1)

        # 적분 행렬 경유
        I = int_matrix(N)
        e_last = np.zeros(N + 2)
        e_last[N + 1] = 1.0
        integral_via_matrix = e_last @ I @ P

        # 직접 산술평균
        integral_direct = np.mean(P)

        assert integral_via_matrix == pytest.approx(integral_direct, abs=1e-14)

    def test_first_row_zero(self):
        """I_N 의 첫 행은 0 (∫₀⁰ = 0)."""
        N = 5
        I = int_matrix(N)
        np.testing.assert_allclose(I[0, :], 0.0)


class TestDoubleIntMatrix:
    """이중 적분 행렬 Ī_N = I_{N+1} · I_N."""

    def test_shape(self):
        N = 5
        Ibar = double_int_matrix(N)
        assert Ibar.shape == (N + 3, N + 1)

    def test_equals_product(self):
        """Ī_N == I_{N+1} · I_N 직접 곱과 동일."""
        N = 8
        Ibar = double_int_matrix(N)
        product = int_matrix(N + 1) @ int_matrix(N)
        np.testing.assert_allclose(Ibar, product, atol=1e-15)


class TestGramMatrix:
    """Gram 행렬 G_N 테스트."""

    @pytest.mark.parametrize("N", [3, 8, 12])
    def test_symmetric(self, N):
        G = gram_matrix(N)
        np.testing.assert_allclose(G, G.T, atol=1e-15)

    @pytest.mark.parametrize("N", [3, 8, 12])
    def test_positive_definite(self, N):
        G = gram_matrix(N)
        eigvals = np.linalg.eigvalsh(G)
        assert np.all(eigvals > 0)

    def test_trace_identity(self):
        """tr(G_N) = ∑ ∫₀¹ B_i^N(τ)² dτ = ∑ C(N,i)²/[C(2N,2i)(2N+1)]."""
        N = 5
        G = gram_matrix(N)
        expected_trace = sum(
            math.comb(N, i) ** 2 / (math.comb(2 * N, 2 * i) * (2 * N + 1))
            for i in range(N + 1)
        )
        assert np.trace(G) == pytest.approx(expected_trace, rel=1e-12)

    def test_integral_via_gram(self):
        """∫₀¹ ‖q(τ)‖² dτ = p^T G_N p 검증 (수치 적분 대조)."""
        N = 6
        rng = np.random.default_rng(123)
        P = rng.standard_normal(N + 1)
        G = gram_matrix(N)

        # 행렬 경유
        integral_matrix = P @ G @ P

        # 수치 적분: 끝점 포함, bernstein_eval 사용
        tau = np.linspace(0, 1, 10000)
        q = bernstein_eval(N, P, tau)  # (10000,)
        integral_numeric = np.trapezoid(q**2, tau)

        assert integral_matrix == pytest.approx(integral_numeric, rel=1e-5)


class TestBlockHessian:

    def test_shape(self):
        N = 5
        H = block_hessian(N, d=3)
        assert H.shape == (3 * (N + 1), 3 * (N + 1))

    def test_block_diagonal(self):
        N = 4
        G = gram_matrix(N)
        H = block_hessian(N, d=3)
        for k in range(3):
            s = k * (N + 1)
            e = s + (N + 1)
            np.testing.assert_allclose(H[s:e, s:e], G)


class TestDefiniteIntegral:

    def test_mean(self):
        P = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = definite_integral(P)
        np.testing.assert_allclose(result, [3.0, 4.0])
