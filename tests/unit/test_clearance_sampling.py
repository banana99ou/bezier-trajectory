"""Two ways a uniform sample grid missed a real violation, and the injection.

`compute_min_clearance` and `compute_los_margin` are the independent cross-check
on the solver's certificate: they sample the returned curve against the TRUE
obstacle trajectories. A sampled minimum is a sufficient condition for
penetration and never a proof of clearance, and two mechanisms turned that from
a theoretical caveat into a measured miss.

Both are reproduced here at the reviewer's own constructions, and both are
asserted against an 8-million-sample reference computed in the test rather than
against a remembered number.
"""

import numpy as np
import pytest
from math import comb

from spacetime_bezier.geometry import (
    _MAX_INJECTED_PER_OBSTACLE,
    _sample_taus,
    _tau_at_time,
    compute_los_margin,
    compute_min_clearance,
)

# A straight flight, x = t, from (0,0,0) to (10,0,10). Chosen because every
# quantity below is then writable in closed form.
_STRAIGHT = np.array([[i * 10 / 8, 0.0, i * 10 / 8] for i in range(9)])


def _uniform_reference(P, obstacles, n_eval):
    """What a purely uniform grid of ``n_eval`` samples would have reported.

    This is the OLD behaviour, kept in the test rather than in the library, so
    every assertion below has something to be compared against.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0] - 1
    tau = np.linspace(0.0, 1.0, n_eval)
    basis = np.array([comb(n, i) * tau**i * (1 - tau) ** (n - i) for i in range(n + 1)])
    pts = basis.T @ P
    spatial = P.shape[1] - 1
    worst = np.inf
    for obs in obstacles:
        t = pts[:, -1]
        active = (t >= obs.get("t_start", -np.inf)) & (t <= obs.get("t_end", np.inf))
        if not active.any():
            continue
        centres = (
            np.asarray(obs["pos0"], float)[None, :]
            + np.asarray(obs["vel"], float)[None, :] * t[active, None]
        )
        d = np.linalg.norm(pts[active, :spatial] - centres, axis=1) - obs["r"]
        worst = min(worst, float(d.min()))
    return worst


# ---------------------------------------------------------------------------
# Mechanism 1 -- an active window shorter than the sample spacing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("half_window", [1e-2, 2e-3, 1e-3, 5e-4, 1e-4])
def test_a_short_active_window_is_no_longer_skipped(half_window):
    """A full-size obstacle, sitting exactly on the curve, for a brief moment.

    Only the WINDOW is short -- the body is r=1.0 and centred on the flight path,
    so the true clearance is -1.0 whatever the window. At 3000 uniform samples
    over 10 s the spacing is 3.3 ms, and a window of +-0.001 s falls entirely
    between two of them: the old answer was +0.5, i.e. half a metre of clearance
    reported for a curve passing through the centre of the obstacle.

    FAILS IF the window endpoints stop being injected.
    """
    obstacles = [
        {
            "pos0": [5.0, 0.0],
            "vel": [0.0, 0.0],
            "r": 1.0,
            "t_start": 5.0 - half_window,
            "t_end": 5.0 + half_window,
        }
    ]
    truth = _uniform_reference(_STRAIGHT, obstacles, 8_000_001)
    assert truth == pytest.approx(-1.0, abs=1e-6), "the reference itself is wrong"

    assert compute_min_clearance(_STRAIGHT, obstacles, 3, 3000) == pytest.approx(
        -1.0, abs=1e-6
    )


def test_the_uniform_grid_really_did_miss_it():
    """The other half of the pair: without injection the miss reproduces.

    A test that only checked the new answer could pass on a grid that never had
    the defect. FAILS IF 3000 uniform samples would have caught the +-0.001 s
    window, which would make the case above prove nothing.
    """
    obstacles = [
        {
            "pos0": [5.0, 0.0],
            "vel": [0.0, 0.0],
            "r": 1.0,
            "t_start": 4.999,
            "t_end": 5.001,
        }
    ]
    assert _uniform_reference(_STRAIGHT, obstacles, 3000) > 0.0
    assert compute_min_clearance(_STRAIGHT, obstacles, 3, 3000) < 0.0


# ---------------------------------------------------------------------------
# Mechanism 2 -- a small fast obstacle crossing between samples
# ---------------------------------------------------------------------------


def test_a_small_fast_obstacle_is_no_longer_missed():
    """r=0.05 crossing the path at |v|=50, active for the whole plan.

    No window trickery: the obstacle is always there, it is simply small and fast
    enough to pass through the curve between two uniform samples. Measured before
    the injection: +0.035 reported against a true -0.05 -- the sign of the answer
    was wrong, which is the only thing the gate reads.

    FAILS IF the injected spacing stops scaling with the obstacle's speed.
    """
    obstacles = [{"pos0": [5.0 + 50.0 * 5.0, 0.0], "vel": [-50.0, 0.0], "r": 0.05}]
    assert _uniform_reference(_STRAIGHT, obstacles, 3000) > 0.0, (
        "the uniform grid caught it, so this construction proves nothing"
    )
    assert compute_min_clearance(_STRAIGHT, obstacles, 3, 3000) == pytest.approx(
        -0.05, abs=1e-6
    )


@pytest.mark.parametrize(
    "radius,speed", [(0.02, 50.0), (0.05, 200.0)]
)
def test_beyond_the_injection_cap_the_sign_is_still_right(radius, speed):
    """The cap bites, and what survives it is the property the gate reads.

    These need more than ``_MAX_INJECTED_PER_OBSTACLE`` samples to resolve the
    closest approach, so the reported magnitude is short of the truth. The SIGN
    is not: penetration is still detected, which is what
    `figure_grade_failures` tests.

    FAILS IF the cap is lowered far enough to lose the sign, which would make the
    honest-limitation note in the docstrings an understatement rather than a
    caveat.
    """
    obstacles = [
        {"pos0": [5.0 + speed * 5.0, 0.0], "vel": [-speed, 0.0], "r": radius}
    ]
    truth = _uniform_reference(_STRAIGHT, obstacles, 8_000_001)
    got = compute_min_clearance(_STRAIGHT, obstacles, 3, 3000)
    assert truth < 0.0
    assert got < 0.0
    assert got > truth  # under-reported magnitude, correct sign


# ---------------------------------------------------------------------------
# The machinery
# ---------------------------------------------------------------------------


def test_time_inversion_lands_on_the_requested_times():
    """Bisection on a monotone time coordinate, to machine precision."""
    targets = np.array([0.0, 0.25, 3.0, 5.0, 9.999, 10.0])
    taus = _tau_at_time(_STRAIGHT, targets)
    n = _STRAIGHT.shape[0] - 1
    basis = np.array(
        [comb(n, i) * taus**i * (1 - taus) ** (n - i) for i in range(n + 1)]
    )
    assert np.allclose(basis.T @ _STRAIGHT[:, -1], targets, atol=1e-9)


def test_non_monotone_time_falls_back_to_the_uniform_grid():
    """The inversion has no unique answer, so it is not attempted.

    The solver's monotonicity rows forbid this, but these helpers are also called
    on hand-built polygons. FAILS IF the fallback is removed -- bisection would
    then return an arbitrary branch and the injected samples would be nonsense.
    """
    P = _STRAIGHT.copy()
    P[4, -1], P[5, -1] = P[5, -1], P[4, -1]  # time runs backwards across one leg
    obstacles = [{"pos0": [5.0, 0.0], "vel": [0.0, 0.0], "r": 1.0}]
    taus = _sample_taus(P, obstacles, 101)
    assert len(taus) == 101
    assert np.allclose(taus, np.linspace(0.0, 1.0, 101))


def test_a_run_with_no_obstacles_gets_no_injection():
    taus = _sample_taus(_STRAIGHT, [], 257)
    assert len(taus) == 257


def test_injection_is_bounded():
    """FAILS IF an extreme obstacle can make the cross-check unboundedly slow."""
    obstacles = [{"pos0": [0.0, 0.0], "vel": [1e6, 0.0], "r": 1e-6}]
    taus = _sample_taus(_STRAIGHT, obstacles, 3000)
    assert len(taus) <= 3000 + _MAX_INJECTED_PER_OBSTACLE


# ---------------------------------------------------------------------------
# The same treatment on the line-of-sight leg
# ---------------------------------------------------------------------------


def test_los_margin_also_covers_a_short_window():
    """`compute_los_margin` shares the grid and shared the defect.

    The occluder blocks the sight line for 2 ms around t=5. FAILS IF the
    injection is applied to clearance only: the uniform grid reports the sight
    line clear for the whole flight.
    """
    station = np.array([0.0, 0.0])
    occluder = {
        "pos0": [4.0, 0.0],
        "vel": [0.0, 0.0],
        "r": 1.0,
        "t_start": 4.999,
        "t_end": 5.001,
    }
    # Vehicle parked out along +x, so the sight segment runs through the body.
    P = np.array([[8.0, 0.0, i * 10 / 8] for i in range(9)])

    _, margins = compute_los_margin(P, station, [occluder], dim=3, n_eval=3000)
    finite = margins[np.isfinite(margins)]
    assert finite.size > 0
    assert float(finite.min()) == pytest.approx(-1.0, abs=1e-6)


def test_los_margin_returns_at_least_n_eval_samples_in_order():
    """The contract the figure scripts rely on: sorted, and never fewer.

    Ordered by curve PARAMETER, so the time axis is non-decreasing only up to
    floating-point noise -- an injected parameter and a uniform one a few ulps
    apart can order oppositely in the two. The tolerance below is what that
    costs; anything larger would be a real inversion.
    """
    station = np.array([0.0, 0.0])
    obstacles = [{"pos0": [4.0, -5.0], "vel": [0.0, 1.0], "r": 1.0}]
    P = np.array([[8.0, 0.0, i * 10 / 8] for i in range(9)])
    t_values, margins = compute_los_margin(P, station, obstacles, dim=3, n_eval=501)
    assert len(t_values) == len(margins) >= 501
    assert np.all(np.diff(t_values) >= -1e-12)
