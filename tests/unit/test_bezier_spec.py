"""
Unit tests asserting Project Spec math for Bézier: D/E/G matrices,
BezierCurve point(t) vs Bernstein sum, velocity/acceleration vs finite difference and shape.
"""

import numpy as np
import pytest
from scipy.special import comb

from orbital_docking.bezier import (
    BezierCurve,
    get_D_matrix,
    get_E_matrix,
    get_G_matrix,
)


# ---- D matrix (Project Spec: D[i,i]=-N, D[i,i+1]=N, else 0) ----

@pytest.mark.parametrize("N", [2, 3, 4])
def test_D_matrix_shape(N):
    """D has shape (N, N+1)."""
    D = get_D_matrix(N)
    assert D.shape == (N, N + 1)


@pytest.mark.parametrize("N", [2, 3, 4])
def test_D_matrix_diagonal_and_superdiagonal(N):
    """D[i,i]=-N, D[i,i+1]=N, all other entries 0."""
    D = get_D_matrix(N)
    for i in range(N):
        assert D[i, i] == -N
        assert D[i, i + 1] == N
    for i in range(N):
        for j in range(N + 1):
            if j not in (i, i + 1):
                assert D[i, j] == 0


# ---- E matrix (Project Spec: first/last row ones, middle (i-1)/(N+1) and (N+2-i)/(N+1)) ----

@pytest.mark.parametrize("N", [1, 2, 3])
def test_E_matrix_shape(N):
    """E_{N→N+1} has shape (N+2, N+1)."""
    E = get_E_matrix(N)
    assert E.shape == (N + 2, N + 1)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_E_matrix_first_row_ones(N):
    """First row is (1, 0, ..., 0)."""
    E = get_E_matrix(N)
    expected = np.zeros(N + 1)
    expected[0] = 1.0
    np.testing.assert_array_almost_equal(E[0], expected)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_E_matrix_last_row_ones(N):
    """Last row is (0, ..., 0, 1)."""
    E = get_E_matrix(N)
    expected = np.zeros(N + 1)
    expected[N] = 1.0
    np.testing.assert_array_almost_equal(E[N + 1], expected)


@pytest.mark.parametrize("N", [1, 2, 3])
def test_E_matrix_middle_rows_structure(N):
    """Middle rows: E[row, row-1] = (i-1)/(N+1), E[row, row] = (N+2-i)/(N+1) in 1-indexed i = row+1."""
    E = get_E_matrix(N)
    for row in range(1, N + 1):  # 0-indexed middle rows 1..N
        i_1idx = row + 1  # 1-indexed row index
        assert E[row, row - 1] == pytest.approx((i_1idx - 1) / (N + 1))
        assert E[row, row] == pytest.approx((N + 2 - i_1idx) / (N + 1))
    # All other entries of middle rows should be zero
    for row in range(1, N + 1):
        for col in range(N + 1):
            if col not in (row - 1, row):
                assert E[row, col] == pytest.approx(0.0)


# ---- G matrix: G[i,j] = C(N,i)C(N,j) / (C(2N,i+j)(2N+1)) ----

def _G_spec_entry(N, i, j):
    return (comb(N, i) * comb(N, j)) / (comb(2 * N, i + j) * (2 * N + 1))


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("i,j", [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
def test_G_matrix_formula(N, i, j):
    """G[i,j] = C(N,i)C(N,j)/(C(2N,i+j)(2N+1)) for N=2,3 and selected (i,j)."""
    if i > N or j > N:
        pytest.skip("index out of range for this N")
    G = get_G_matrix(N)
    expected = _G_spec_entry(N, i, j)
    assert G[i, j] == pytest.approx(expected)


# ---- BezierCurve point(t) vs explicit Bernstein sum ----

def _bernstein_point(N, P, tau):
    """Explicit Bernstein sum: sum_i B_{N,i}(tau) P_i, B_{N,i}(t) = C(N,i) t^i (1-t)^{N-i}."""
    P = np.asarray(P)
    d = P.shape[1]
    out = np.zeros(d)
    for i in range(N + 1):
        b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
        out += b * P[i]
    return out


@pytest.fixture
def P_3d(rng):
    """Random (4, 3) control points for degree-3 curve in 3D."""
    return rng.uniform(-1.0, 1.0, (4, 3))


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_bezier_point_equals_bernstein_sum(P_3d, tau):
    """BezierCurve.point(t) equals explicit Bernstein sum for given P and t."""
    curve = BezierCurve(P_3d)
    N = curve.degree
    got = curve.point(tau)
    expected = _bernstein_point(N, P_3d, tau)
    np.testing.assert_allclose(got, expected)


# ---- velocity(t) and acceleration(t) vs finite-difference from point(t) ----

def _finite_diff_velocity(curve, tau, h=1e-6):
    return (curve.point(tau + h) - curve.point(tau - h)) / (2 * h)


def _finite_diff_acceleration(curve, tau, h=1e-6):
    return (curve.point(tau + h) - 2 * curve.point(tau) + curve.point(tau - h)) / (h ** 2)


@pytest.fixture
def curve_degree_3_3d(P_3d):
    """BezierCurve of degree 3 in 3D."""
    return BezierCurve(P_3d)


@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_velocity_matches_finite_difference(curve_degree_3_3d, tau):
    """velocity(t) matches (point(t+h)-point(t-h))/(2h) with h=1e-6, tolerance ~1e-8."""
    curve = curve_degree_3_3d
    v = curve.velocity(tau)
    v_fd = _finite_diff_velocity(curve, tau, h=1e-6)
    np.testing.assert_allclose(v, v_fd, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_acceleration_matches_finite_difference(curve_degree_3_3d, tau):
    """acceleration(t) matches second-order finite difference from point(t)."""
    curve = curve_degree_3_3d
    a = curve.acceleration(tau)
    a_fd = _finite_diff_acceleration(curve, tau, h=1e-6)
    # Second-derivative FD amplifies roundoff by 1/h^2; use looser tolerance than velocity
    np.testing.assert_allclose(a, a_fd, atol=5e-4, rtol=1e-3)


# ---- velocity/acceleration output shape (3,) for degree 0,1,2,3,4 ----

@pytest.fixture
def control_points_3d_by_degree(rng):
    """Control points (N+1, 3) for degree N in 3D (uses conftest rng)."""
    def _get(degree):
        return rng.uniform(-1.0, 1.0, (degree + 1, 3))
    return _get


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_velocity_output_shape_is_3(control_points_3d_by_degree, degree):
    """velocity(t) returns array of shape (3,) for degree 0,1,2,3,4."""
    P = control_points_3d_by_degree(degree)
    curve = BezierCurve(P)
    v = curve.velocity(0.5)
    assert v.shape == (3,)


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_acceleration_output_shape_is_3(control_points_3d_by_degree, degree):
    """acceleration(t) returns array of shape (3,) for degree 0,1,2,3,4."""
    P = control_points_3d_by_degree(degree)
    curve = BezierCurve(P)
    a = curve.acceleration(0.5)
    assert a.shape == (3,)
