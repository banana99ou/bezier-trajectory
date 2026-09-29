"""
The wait start (added 2026-09-29): the straight space-time chord, travelled slowly
first and fast at the end. Space is warped, time is not. Used by one scene, `wall`.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from spacetime_bezier.objective import build_initial_guess


def _eval(P, u):
    N = len(P) - 1
    return sum(comb(N, i) * u**i * (1 - u) ** (N - i) * P[i] for i in range(N + 1))


@pytest.mark.parametrize("dim", [3, 4])
@pytest.mark.parametrize("N", [3, 8, 10])
@pytest.mark.parametrize("power", [2, 3])
def test_the_wait_start_is_exactly_the_warped_chord(power, N, dim):
    """FAILS IF a Bernstein coefficient is wrong: the curve must equal
    start + (end - start) * s**power in space and move uniformly in time, to
    rounding, at every parameter value."""
    rng = np.random.default_rng(N * 10 + dim)
    a = np.r_[rng.uniform(0, 10, dim - 1), 0.0]
    b = np.r_[rng.uniform(0, 10, dim - 1), 10.0]
    P = build_initial_guess(a, b, N + 1, {"mode": "wait", "power": power})
    for u in np.linspace(0, 1, 41):
        want = np.r_[a[:-1] + (b[:-1] - a[:-1]) * u**power, a[-1] + (b[-1] - a[-1]) * u]
        np.testing.assert_allclose(_eval(P, u), want, atol=1e-12)
    np.testing.assert_array_equal(P[0], a)
    np.testing.assert_allclose(P[-1], b, atol=1e-12)


def test_the_wait_start_is_behind_the_straight_one():
    """FAILS IF 'wait' does not wait: at mid-horizon it must have covered less of
    the chord than the straight start."""
    a, b = np.array([1.0, 5.0, 0.0]), np.array([8.0, 5.0, 10.0])
    assert _eval(build_initial_guess(a, b, 9, {"mode": "wait", "power": 3}), 0.5)[0] < _eval(build_initial_guess(a, b, 9), 0.5)[0]


def test_a_degree_below_the_power_is_refused():
    with pytest.raises(ValueError, match="needs degree"):
        build_initial_guess([0, 0, 0], [1, 1, 1], 3, {"mode": "wait", "power": 3})
