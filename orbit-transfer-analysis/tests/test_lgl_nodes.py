"""LGL 노드, 가중치, 미분행렬 테스트."""

import numpy as np
import pytest

from orbit_transfer.collocation.lgl_nodes import (
    compute_differentiation_matrix,
    compute_lgl_nodes,
    compute_lgl_weights,
    legendre_poly,
    legendre_poly_derivative,
)


# 테스트 대상 다항식 차수
LGL_ORDERS = [4, 8, 16, 32]


class TestLGLNodes:
    """LGL 노드 기본 성질 테스트."""

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_endpoint_inclusion(self, N):
        """끝점 포함 확인: tau[0] = -1, tau[-1] = 1."""
        tau = compute_lgl_nodes(N)
        assert tau[0] == pytest.approx(-1.0, abs=1e-15)
        assert tau[-1] == pytest.approx(1.0, abs=1e-15)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_node_count(self, N):
        """노드 개수 확인: N+1개."""
        tau = compute_lgl_nodes(N)
        assert len(tau) == N + 1

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_node_symmetry(self, N):
        """노드 대칭성: tau[i] = -tau[N-i] for all i."""
        tau = compute_lgl_nodes(N)
        for i in range(N + 1):
            assert tau[i] == pytest.approx(-tau[N - i], abs=1e-14), (
                f"N={N}, i={i}: tau[{i}]={tau[i]}, -tau[{N-i}]={-tau[N-i]}"
            )

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_nodes_sorted(self, N):
        """노드가 오름차순으로 정렬되어 있는지 확인."""
        tau = compute_lgl_nodes(N)
        assert np.all(np.diff(tau) > 0)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_nodes_are_lgl_points(self, N):
        """내부 노드가 P'_N의 영점인지 확인."""
        tau = compute_lgl_nodes(N)
        inner = tau[1:-1]
        dP = legendre_poly_derivative(N, inner)
        np.testing.assert_allclose(dP, 0.0, atol=1e-11)


class TestLGLWeights:
    """LGL 가중치 테스트."""

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_weights_sum(self, N):
        """가중치 합 = 2.0 (적분 구간 [-1, 1]의 길이)."""
        tau = compute_lgl_nodes(N)
        w = compute_lgl_weights(N, tau)
        assert np.sum(w) == pytest.approx(2.0, abs=1e-14)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_weights_positive(self, N):
        """모든 가중치가 양수인지 확인."""
        tau = compute_lgl_nodes(N)
        w = compute_lgl_weights(N, tau)
        assert np.all(w > 0)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_polynomial_integration_exactness(self, N):
        """다항식 적분 정확도: LGL 구적법은 2N-1차까지 정확.

        f(x) = x^k, k=0,1,...,2N-1에 대해
        integral_{-1}^{1} x^k dx 와 sum(w*f(tau))의 오차 < 1e-12.
        """
        tau = compute_lgl_nodes(N)
        w = compute_lgl_weights(N, tau)

        for k in range(2 * N):
            # 해석적 적분값
            if k % 2 == 0:
                exact = 2.0 / (k + 1)
            else:
                exact = 0.0

            # 수치 적분
            numerical = np.sum(w * tau**k)

            assert numerical == pytest.approx(exact, abs=1e-12), (
                f"N={N}, k={k}: numerical={numerical}, exact={exact}"
            )


class TestDifferentiationMatrix:
    """미분 행렬 테스트."""

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_row_sum_zero(self, N):
        """미분 행렬의 행 합이 0 (상수 함수의 미분 = 0)."""
        tau = compute_lgl_nodes(N)
        D = compute_differentiation_matrix(N, tau)
        row_sums = np.sum(D, axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_matrix_shape(self, N):
        """미분 행렬 크기: (N+1) x (N+1)."""
        tau = compute_lgl_nodes(N)
        D = compute_differentiation_matrix(N, tau)
        assert D.shape == (N + 1, N + 1)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_differentiate_linear(self, N):
        """선형 함수 f(x) = 3x + 2의 미분: f'(x) = 3."""
        tau = compute_lgl_nodes(N)
        D = compute_differentiation_matrix(N, tau)

        f = 3.0 * tau + 2.0
        df_numerical = D @ f
        df_exact = 3.0 * np.ones(N + 1)

        np.testing.assert_allclose(df_numerical, df_exact, atol=1e-12)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_differentiate_polynomial(self, N):
        """다항식 f(x) = x^3의 미분: f'(x) = 3x^2."""
        tau = compute_lgl_nodes(N)
        D = compute_differentiation_matrix(N, tau)

        f = tau**3
        df_numerical = D @ f
        df_exact = 3.0 * tau**2

        np.testing.assert_allclose(df_numerical, df_exact, atol=1e-10)

    def test_differentiate_sinusoidal(self):
        """삼각함수 미분: D @ sin(pi*tau) vs pi*cos(pi*tau).

        N=8에서 sin(pi*x)는 비다항식이므로 spectral accuracy에 한계가 있다.
        N=16 이상에서 1e-8 정밀도를 확인한다.
        """
        N = 16
        tau = compute_lgl_nodes(N)
        D = compute_differentiation_matrix(N, tau)

        f = np.sin(np.pi * tau)
        df_numerical = D @ f
        df_exact = np.pi * np.cos(np.pi * tau)

        np.testing.assert_allclose(df_numerical, df_exact, atol=1e-8)

    @pytest.mark.parametrize("N", LGL_ORDERS)
    def test_corner_values(self, N):
        """특수 대각 원소: D[0,0] = -N(N+1)/4, D[N,N] = N(N+1)/4."""
        tau = compute_lgl_nodes(N)
        D = compute_differentiation_matrix(N, tau)

        assert D[0, 0] == pytest.approx(-N * (N + 1) / 4.0, abs=1e-14)
        assert D[N, N] == pytest.approx(N * (N + 1) / 4.0, abs=1e-14)


class TestLegendrePoly:
    """Legendre 다항식 보조 함수 테스트."""

    def test_P0(self):
        """P_0(x) = 1."""
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(legendre_poly(0, x), np.ones(4))

    def test_P1(self):
        """P_1(x) = x."""
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        np.testing.assert_allclose(legendre_poly(1, x), x)

    def test_P2(self):
        """P_2(x) = (3x^2 - 1)/2."""
        x = np.array([-1.0, 0.0, 0.5, 1.0])
        expected = (3.0 * x**2 - 1.0) / 2.0
        np.testing.assert_allclose(legendre_poly(2, x), expected)

    def test_P3(self):
        """P_3(x) = (5x^3 - 3x)/2."""
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        expected = (5.0 * x**3 - 3.0 * x) / 2.0
        np.testing.assert_allclose(legendre_poly(3, x), expected)

    def test_endpoint_values(self):
        """P_N(1) = 1, P_N(-1) = (-1)^N."""
        for N in range(10):
            assert legendre_poly(N, np.array([1.0]))[0] == pytest.approx(1.0)
            assert legendre_poly(N, np.array([-1.0]))[0] == pytest.approx((-1.0) ** N)

    def test_derivative_at_endpoints(self):
        """P'_N(1) = N(N+1)/2, P'_N(-1) = (-1)^{N+1} N(N+1)/2."""
        for N in range(1, 10):
            dP_plus = legendre_poly_derivative(N, np.array([1.0]))[0]
            dP_minus = legendre_poly_derivative(N, np.array([-1.0]))[0]
            assert dP_plus == pytest.approx(N * (N + 1) / 2.0, abs=1e-12)
            assert dP_minus == pytest.approx(
                (-1.0) ** (N + 1) * N * (N + 1) / 2.0, abs=1e-12
            )
