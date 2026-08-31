"""The independent line-of-sight margin (item B12).

``compute_los_margin`` is the second leg of a pair. The first leg is the solver's
occlusion certificate, computed in Rust against a convex outer approximation the
solver built for itself; this one samples the returned curve and measures the
TRUE geometry, knowing nothing about shadows, half-spaces or approximations. Two
independent computations of the same physical fact must agree, and where they
disagree the disagreement is the finding.

So this file does not compare the margin to the solver. It compares it to closed
form, on trajectories built by hand for which the blocked interval can be written
down exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacetime_bezier.geometry import compute_los_margin


def _straight_polygon(p_start, p_end, n_cp):
    """Equally spaced collinear control points.

    A Bezier over equally spaced collinear control points reproduces the straight
    line exactly -- Bernstein polynomials reproduce affine functions of the index
    -- so the sampled curve is the analytic line and the closed form below
    describes the actual trajectory, not an approximation of it.
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    alphas = np.linspace(0.0, 1.0, n_cp)[:, None]
    return (1.0 - alphas) * p_start[None, :] + alphas * p_end[None, :]


# Every trajectory in this file runs 0 to 10 s, and a legacy `{pos0, vel, r}`
# obstacle no longer carries an implicit window: since the 2026-08-21
# formulation change the active window IS the first and last control point's
# time coordinate, so an obstacle written the legacy way has to say when it is
# there. Stamping the horizon here keeps each test's geometry paragraph about
# geometry. `test_inactive_time_window_blocks_nothing` overrides it on purpose.
_HORIZON = 10.0


def _body(pos0, vel, r):
    """A straight legacy obstacle, present for the whole horizon."""
    return {"pos0": pos0, "vel": vel, "r": r, "t_start": 0.0, "t_end": _HORIZON}


def test_margin_goes_negative_exactly_on_the_analytic_interval():
    """FAILS IF the blocked interval is off by more than the sampling step.

    Geometry, chosen so the answer is writable in closed form. The station is at
    the origin. A static ball of radius 1 sits at (4, 0). The vehicle flies the
    line x = 8 from y = -6 to y = +6 over ten seconds, so y(t) = -6 + 1.2 t.

    The foot of the perpendicular from the ball centre onto the sight segment is
    interior for every y on this path, so the segment distance equals the line
    distance:

        d(y) = |4 y| / sqrt(64 + y^2)

    and the margin d(y) - 1 is negative exactly when 15 y^2 < 64, i.e.
    |y| < 8/sqrt(15). Converting to time gives the interval asserted below.
    """
    station = np.array([0.0, 0.0])
    obstacles = [_body([4.0, 0.0], [0.0, 0.0], 1.0)]
    P = _straight_polygon([8.0, -6.0, 0.0], [8.0, 6.0, 10.0], n_cp=7)

    n_eval = 20001
    t_values, margins = compute_los_margin(P, station, obstacles, dim=3, n_eval=n_eval)
    step = 10.0 / (n_eval - 1)

    y_star = 8.0 / np.sqrt(15.0)
    t_lo = (6.0 - y_star) / 1.2
    t_hi = (6.0 + y_star) / 1.2

    negative = margins < 0.0
    assert negative.any(), "the hand-built path must pass behind the occluder"
    assert t_values[negative].min() == pytest.approx(t_lo, abs=2 * step)
    assert t_values[negative].max() == pytest.approx(t_hi, abs=2 * step)
    # Contiguous: one crossing in, one crossing out, nothing in between.
    idx = np.flatnonzero(negative)
    assert np.all(np.diff(idx) == 1)

    # And the values themselves, not merely the sign.
    y = -6.0 + 1.2 * t_values
    closed_form = 4.0 * np.abs(y) / np.sqrt(64.0 + y * y) - 1.0
    assert np.allclose(margins, closed_form, atol=1e-9)


def test_margin_uses_the_sight_segment_not_the_infinite_line():
    """FAILS IF the occluder blocks from BEHIND the vehicle.

    A body on the extension of the sight line past the vehicle occludes nothing:
    the line of sight ends at the vehicle. Here the infinite line through the
    station and the vehicle passes 0.2 from the body centre -- deep inside a
    radius-1 body -- while the segment stops short of it, so the correct margin
    is large and positive. An unclamped implementation reports about -0.8.
    """
    station = np.array([0.0, 0.0])
    obstacles = [_body([6.0, 0.2], [0.0, 0.0], 1.0)]
    P = _straight_polygon([2.0, 0.0, 0.0], [2.0, 0.0, 10.0], n_cp=5)

    _, margins = compute_los_margin(P, station, obstacles, dim=3, n_eval=101)
    # Closest point on the segment is the vehicle itself, at (2, 0).
    expected = np.hypot(6.0 - 2.0, 0.2) - 1.0
    assert np.allclose(margins, expected, atol=1e-9)
    assert margins.min() > 0.0

    # The line-based answer, which this must NOT be.
    assert 0.2 - 1.0 < 0.0


def test_a_moving_occluder_blocks_only_while_it_crosses():
    """The margin must follow the obstacle's OWN position at each sample time.

    The vehicle is parked; the occluder sweeps across the sight line. The blocked
    interval is therefore a statement about the obstacle's motion alone, and an
    implementation that evaluated every obstacle at a single time would return a
    constant.
    """
    station = np.array([0.0, 0.0])
    # Centre passes through (4, 0) at t = 5, moving in +y at 1.0 per second.
    obstacles = [_body([4.0, -5.0], [0.0, 1.0], 1.0)]
    P = _straight_polygon([8.0, 0.0, 0.0], [8.0, 0.0, 10.0], n_cp=5)

    n_eval = 20001
    t_values, margins = compute_los_margin(P, station, obstacles, dim=3, n_eval=n_eval)
    step = 10.0 / (n_eval - 1)

    # Sight segment is the x axis from (0,0) to (8,0); distance from the centre
    # is |y_obs| = |t - 5|, so the margin is |t - 5| - 1.
    assert np.allclose(margins, np.abs(t_values - 5.0) - 1.0, atol=1e-9)
    negative = margins < 0.0
    assert t_values[negative].min() == pytest.approx(4.0, abs=2 * step)
    assert t_values[negative].max() == pytest.approx(6.0, abs=2 * step)


def test_inactive_time_window_blocks_nothing():
    """An obstacle outside its window is not there, so it cannot occlude."""
    station = np.array([0.0, 0.0])
    blocking = _body([4.0, 0.0], [0.0, 0.0], 1.0)
    P = _straight_polygon([8.0, 0.0, 0.0], [8.0, 0.0, 10.0], n_cp=5)

    _, blocked = compute_los_margin(P, station, [blocking], dim=3, n_eval=201)
    assert blocked.max() < 0.0, "the control case must actually be blocked"

    windowed = dict(blocking, t_start=20.0, t_end=30.0)
    _, clear = compute_los_margin(P, station, [windowed], dim=3, n_eval=201)
    assert np.all(np.isinf(clear)), "nothing active means nothing occluding"


def test_margin_is_the_minimum_over_obstacles():
    """Several bodies: the sight line is blocked by the worst of them."""
    station = np.array([0.0, 0.0])
    near = _body([4.0, 0.5], [0.0, 0.0], 1.0)
    far = _body([4.0, 3.0], [0.0, 0.0], 1.0)
    P = _straight_polygon([8.0, 0.0, 0.0], [8.0, 0.0, 10.0], n_cp=5)

    _, both = compute_los_margin(P, station, [near, far], dim=3, n_eval=51)
    _, only_near = compute_los_margin(P, station, [near], dim=3, n_eval=51)
    _, only_far = compute_los_margin(P, station, [far], dim=3, n_eval=51)
    assert np.allclose(both, np.minimum(only_near, only_far))
    assert both.max() < only_far.min(), "the near body must be the binding one"


def test_station_dimension_is_checked():
    """A station of the wrong width is refused, not silently broadcast.

    A three-coordinate station handed to a two-dimensional problem would place
    the observer somewhere nobody asked for and produce a margin about a
    different scenario. Numbers from such a run are indistinguishable from real
    ones once they reach a table.
    """
    P = _straight_polygon([8.0, 0.0, 0.0], [8.0, 0.0, 10.0], n_cp=5)
    with pytest.raises(ValueError):
        compute_los_margin(P, [0.0, 0.0, 0.0], [], dim=3, n_eval=11)
