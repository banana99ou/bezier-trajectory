"""One objective, written twice. Pin the two together or delete one.

`spacetime_bezier.objective.build_smoothness_regularizer` is the readable form of
the derivation. `build_smoothness_regularizer_h` in `spacetime_optimizer.rs` is
the matrix the solver actually uses. Nothing in the solve path calls the Python
one, so it can drift into documenting a matrix that no longer exists and no test
would notice.

The oracle is the Rust cost at ``max_iter=0``: the loop runs zero iterations,
returns the initial guess, and reports ``info["cost"]`` -- the objective
evaluated by the solver's own code at a point we choose.
"""

import numpy as np
import pytest

from spacetime_bezier.objective import build_smoothness_regularizer

bezier_opt = pytest.importorskip("bezier_opt")


def _rust_cost(P, time_weight=0.0):
    """The solver's own objective at ``P``, with no iterations in between."""
    P = np.asarray(P, dtype=float)
    spatial_dim = P.shape[1] - 1
    _, info = bezier_opt.optimize_spacetime_bezier(
        p_init=P,
        obstacle_pos0=np.zeros((0, spatial_dim)),
        obstacle_vel=np.zeros((0, spatial_dim)),
        obstacle_r=np.zeros((0,)),
        n_seg=4,
        max_iter=0,
        tol=1e-6,
        scp_prox_weight=0.0,
        scp_trust_radius=0.5,
        elastic_weight=100.0,
        min_dt=0.1,
        coord_lb=-1e3,
        coord_ub=1e3,
        time_lb=-1e3,
        time_ub=1e3,
        v_max=None,
        time_weight=time_weight,
        free_arrival_time=False,
    )
    return float(info["cost"])


def _random_polygon(rng, N, dim):
    P = rng.normal(size=(N + 1, dim)) * 3.0
    P[:, -1] = np.sort(rng.uniform(0.0, 10.0, N + 1))
    return P


@pytest.mark.parametrize("N", [2, 4, 6, 8, 10, 12])
@pytest.mark.parametrize("dim", [3, 4])
def test_python_regularizer_equals_the_rust_matrix(N, dim):
    """FAILS IF the two implementations of H ever diverge.

    Measured agreement today is 4.9e-16 relative, which is arithmetic noise on a
    sum of this many terms. The tolerance below is 1e-12 -- four orders looser
    than the observed agreement and ten orders tighter than any real difference
    in the matrix could be.
    """
    rng = np.random.default_rng(7 + N * 10 + dim)
    P = _random_polygon(rng, N, dim)

    H = build_smoothness_regularizer(N, dim)
    x = P.reshape(-1)
    python_cost = 0.5 * x @ H @ x
    rust = _rust_cost(P)

    assert rust == pytest.approx(python_cost, rel=1e-12, abs=1e-12)


def test_the_oracle_can_tell_two_matrices_apart():
    """A check that cannot fail is not evidence.

    If the Rust cost ignored H entirely -- returning zero, say -- the comparison
    above would still pass for any polygon whose Python cost happened to cancel
    to zero. This pins the oracle to a value that is far from zero and is
    different for two different polygons, so the comparison has something to
    disagree about.
    """
    rng = np.random.default_rng(11)
    P = _random_polygon(rng, 8, 3)
    Q = _random_polygon(rng, 8, 3)
    cost_p, cost_q = _rust_cost(P), _rust_cost(Q)
    assert abs(cost_p) > 1.0
    assert abs(cost_p - cost_q) > 1e-6


def test_the_rust_objective_is_blind_to_timing():
    """The property that makes it a PARAMETER-domain regularizer, measured.

    Same spatial control points, wildly different time coordinates: a cruise and
    a wait-then-dash must score identically, because the time column is never
    written into. FAILS IF the time coordinate is ever penalized by the
    smoothness term -- which would bias the arrival time through the cost instead
    of through the declared time penalty (item B10).
    """
    rng = np.random.default_rng(3)
    P = _random_polygon(rng, 8, 3)
    Q = P.copy()
    Q[:, -1] = np.sort(rng.uniform(0.0, 10.0, 9)) * 7.0 + 100.0
    assert _rust_cost(P) == _rust_cost(Q)


def test_the_linear_time_penalty_sits_on_top_of_the_quadratic():
    """`cost` must rise by exactly ``time_weight * t_last``.

    FAILS IF `f_linear` is placed on the wrong variable, or scaled -- both of
    which would leave the arrival-time sweep monotone and therefore invisible to
    the tests that only check monotonicity.
    """
    rng = np.random.default_rng(5)
    P = _random_polygon(rng, 8, 3)
    base = _rust_cost(P)
    for weight in (1.0, 7.5):
        assert _rust_cost(P, time_weight=weight) == pytest.approx(
            base + weight * float(P[-1, -1]), rel=1e-12, abs=1e-12
        )
