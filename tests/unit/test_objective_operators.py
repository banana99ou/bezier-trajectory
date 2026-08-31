"""Item B8: the objective measures what its name says, and the control-point
difference operators are the exact derivative maps they claim to be.

Two independent claims are checked here.

1. **The cost is blind to timing.** ``build_smoothness_regularizer`` never writes
   into the time column, so two control polygons that share every spatial
   coordinate and differ arbitrarily in time must score identically.
   FAILS IF the objective ever acquires a time-coordinate term -- which is what
   would happen if someone "re-derived the acceleration energy matrix" (a thing
   that cannot exist; see idea/spacetime.md sec. "The derivation behind the decisions").

2. **The difference operators are exact.** The relation between control-point
   differences and parameter-domain derivatives is polynomial and exact, so it is
   testable against numerical differentiation to near machine precision.
   FAILS IF a stencil, a degree factor, or the row count is wrong -- e.g.
   dropping the ``N`` in ``N (P[i+1] - P[i])`` shows up as a factor-N mismatch at
   every sample point.

Nothing here is checked against physical time. Physical velocity is a ratio of
Beziers and is deliberately out of scope for these operators.
"""

import numpy as np
import pytest

from orbital_docking.bezier import BezierCurve
from spacetime_bezier.objective import (
    build_energy_objective,
    build_smoothness_regularizer,
    derivative_control_point_matrix,
    first_difference_matrix,
    lift_operator,
    second_derivative_control_point_matrix,
    second_difference_matrix,
)


def _bezier_point(ctrl: np.ndarray, tau: float) -> np.ndarray:
    """Evaluate the Bezier with the given control points at ``tau``."""
    return BezierCurve(np.asarray(ctrl, dtype=float)).point(tau)


# ---------------------------------------------------------------------------
# 1. The regularizer is blind to timing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [3, 4])
def test_regularizer_ignores_the_time_coordinate(dim):
    """Identical spatial control points, wildly different timing -> same cost."""
    rng = np.random.default_rng(20260819)
    N = 6
    H = build_smoothness_regularizer(N, dim)

    spatial = rng.normal(size=(N + 1, dim - 1))

    # Uniform timing.
    P_cruise = np.hstack([spatial, np.linspace(0.0, 10.0, N + 1)[:, None]])
    # Wait, then dash: same spatial polygon, violently non-uniform timing.
    t_dash = np.array([0.0, 8.6, 8.8, 9.0, 9.2, 9.4, 10.0])
    P_dash = np.hstack([spatial, t_dash[:, None]])

    cost_cruise = 0.5 * P_cruise.ravel() @ H @ P_cruise.ravel()
    cost_dash = 0.5 * P_dash.ravel() @ H @ P_dash.ravel()

    assert cost_cruise == pytest.approx(cost_dash, rel=0.0, abs=1e-12)
    # And the time column really is structurally absent, not merely cancelling.
    time_cols = np.arange(N + 1) * dim + (dim - 1)
    assert np.all(H[time_cols, :] == 0.0)
    assert np.all(H[:, time_cols] == 0.0)


def test_deprecated_alias_still_works_and_warns():
    """The old name is kept so imports do not break, but it announces itself."""
    with pytest.warns(DeprecationWarning):
        old = build_energy_objective(5, 3)
    np.testing.assert_allclose(old, build_smoothness_regularizer(5, 3))


# ---------------------------------------------------------------------------
# 2. The difference operators are the exact parameter-domain derivative maps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [2, 3, 5, 8])
def test_first_difference_matrix_is_the_raw_gap(N):
    rng = np.random.default_rng(N)
    P = rng.normal(size=(N + 1, 3))
    got = first_difference_matrix(N) @ P
    np.testing.assert_allclose(got, P[1:] - P[:-1], atol=1e-12)
    assert got.shape == (N, 3)


@pytest.mark.parametrize("N", [2, 3, 5, 8])
def test_second_difference_matrix_is_the_raw_second_gap(N):
    rng = np.random.default_rng(100 + N)
    P = rng.normal(size=(N + 1, 3))
    got = second_difference_matrix(N) @ P
    expected = P[2:] - 2.0 * P[1:-1] + P[:-2]
    np.testing.assert_allclose(got, expected, atol=1e-12)
    assert got.shape == (N - 1, 3)


@pytest.mark.parametrize("N", [3, 5, 8])
@pytest.mark.parametrize("tau", [0.0, 0.17, 0.5, 0.83, 1.0])
def test_derivative_operator_matches_numerical_differentiation(N, tau):
    """d/d(parameter) of the curve equals the Bezier drawn by D @ P.

    Numerical differentiation of the ORIGINAL curve is the independent second
    computation: it never touches the operator under test.
    """
    rng = np.random.default_rng(1000 + N)
    P = rng.normal(size=(N + 1, 4)) * 3.0

    d_ctrl = derivative_control_point_matrix(N) @ P
    analytic = _bezier_point(d_ctrl, tau)

    # Central difference, deliberately NOT clamped to [0, 1]: the Bernstein form
    # is a polynomial and evaluates fine just outside the unit interval, and a
    # one-sided stencil at the endpoints would be first-order accurate, which
    # would force a tolerance loose enough to hide a real stencil error.
    h = 1e-5
    numeric = (_bezier_point(P, tau + h) - _bezier_point(P, tau - h)) / (2.0 * h)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("N", [3, 5, 8])
@pytest.mark.parametrize("tau", [0.1, 0.4, 0.73, 0.95])
def test_second_derivative_operator_matches_numerical_differentiation(N, tau):
    rng = np.random.default_rng(2000 + N)
    P = rng.normal(size=(N + 1, 4)) * 3.0

    dd_ctrl = second_derivative_control_point_matrix(N) @ P
    analytic = _bezier_point(dd_ctrl, tau)

    h = 1e-4
    numeric = (
        _bezier_point(P, tau + h) - 2.0 * _bezier_point(P, tau) + _bezier_point(P, tau - h)
    ) / (h * h)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("N,dim", [(4, 3), (6, 4)])
def test_lift_operator_acts_on_the_flattened_polygon(N, dim):
    """The lifted operator reproduces the unlifted one on flattened variables."""
    rng = np.random.default_rng(3000 + N)
    P = rng.normal(size=(N + 1, dim))
    M = first_difference_matrix(N)

    lifted = lift_operator(M, dim) @ P.ravel()
    np.testing.assert_allclose(lifted.reshape(N, dim), M @ P, atol=1e-12)


def test_operators_reject_degrees_they_cannot_serve():
    """A check that can fail: degree 1 has no second difference."""
    with pytest.raises(ValueError):
        second_difference_matrix(1)
    with pytest.raises(ValueError):
        first_difference_matrix(0)
