"""
The obstacle wire-format converters in ``spacetime_bezier.geometry``.

**These two functions stopped being a conversion.** They were written when the
solver knew only a straight capsule, so `{pos0, vel, r}` was one representation
and lifted control points were another, and `moving_obstacle_from_bezier` had to
refuse anything it could not push back into `pos0`/`vel`. Since the 2026-08-21
formulation change there is ONE representation the solver knows -- lifted control
points in (x, ..., t), with the active window intrinsic to the first and last
point's time coordinate -- and `{pos0, vel, r}` survives only as a way to WRITE a
straight obstacle. `bezier_obstacle_from_moving` is that writing shorthand's
reader; `moving_obstacle_from_bezier` is the identity on the canonical form.

So the assertions here are about what a legacy obstacle LIFTS TO, and about the
canonical form surviving a round trip untouched -- not about a value coming back
out as `pos0` and `vel`, which no longer happens.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacetime_bezier.geometry import (
    bezier_obstacle_from_moving,
    moving_obstacle_from_bezier,
)


def test_a_straight_legacy_obstacle_lifts_to_its_own_endpoints():
    """The two control points are the body's position at each end of its window.

    FAILS IF the lift stops evaluating `pos0 + vel*t` at the window ends -- the
    only thing that makes the degree-1 case a faithful rewriting of the legacy
    shorthand rather than a different obstacle.
    """
    T = 10.0
    legacy = {"pos0": [2.0, 8.0], "vel": [0.5, -0.7], "r": 0.8, "name": "A", "color": "#e74c3c"}
    bo = bezier_obstacle_from_moving(legacy, T)

    cps = np.asarray(bo["control_points"], dtype=float)
    assert cps.shape == (2, 3), "a straight obstacle is the degree-1 case"
    np.testing.assert_allclose(cps[0], [2.0, 8.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(cps[1], [2.0 + 0.5 * T, 8.0 - 0.7 * T, T], atol=1e-12)
    assert bo["radius"] == 0.8
    assert bo["name"] == "A"
    assert bo["color"] == "#e74c3c"


def test_the_canonical_form_survives_the_round_trip_untouched():
    """`moving_obstacle_from_bezier` is the identity now, and must stay one.

    FAILS IF it resamples, re-times, or drops a coordinate -- any of which would
    silently hand the solver a different obstacle than the one written down.
    """
    T = 10.0
    legacy = {"pos0": [2.0, 8.0], "vel": [0.5, -0.7], "r": 0.8, "name": "A"}
    bo = bezier_obstacle_from_moving(legacy, T)
    back = moving_obstacle_from_bezier(bo)

    np.testing.assert_allclose(
        np.asarray(back["control_points"], dtype=float),
        np.asarray(bo["control_points"], dtype=float),
        atol=0.0,
    )
    assert back["radius"] == bo["radius"]
    assert back["name"] == "A"


def test_finite_time_window_is_carried_by_the_control_points():
    """The window IS the first and last time coordinate. One source of truth.

    A wall-style obstacle that is only there for the first half of the horizon
    must lift to control points spanning [0, 5], not [0, T] -- otherwise the
    solver would keep out of a body that has gone.
    """
    T = 10.0
    legacy = {"pos0": [2.5, 5.0], "vel": [0.0, 0.0], "r": 0.5, "t_start": 0.0, "t_end": 5.0}
    bo = bezier_obstacle_from_moving(legacy, T)
    assert bo["control_points"][0] == [2.5, 5.0, 0.0]
    assert bo["control_points"][1] == [2.5, 5.0, 5.0]

    back = moving_obstacle_from_bezier(bo)
    cps = np.asarray(back["control_points"], dtype=float)
    assert (float(cps[0, -1]), float(cps[-1, -1])) == (0.0, 5.0)


def test_infinite_window_clamps_to_scenario_T():
    T = 10.0
    legacy = {"pos0": [1.0, 2.0], "vel": [0.1, 0.2], "r": 0.3}
    bo = bezier_obstacle_from_moving(legacy, T)
    cps = bo["control_points"]
    assert cps[0][-1] == 0.0
    assert cps[1][-1] == T


def test_a_legacy_obstacle_with_no_window_and_no_T_is_refused():
    """The window cannot be invented. FAILS IF the reader guesses a horizon.

    A silently-assumed horizon is how an obstacle ends up present at times its
    author never wrote, which is a keep-out zone nobody asked for.
    """
    with pytest.raises(ValueError, match="scenario duration T"):
        bezier_obstacle_from_moving({"pos0": [1.0, 2.0], "vel": [0.1, 0.2], "r": 0.3}, None)


def test_degree_two_is_accepted_and_preserved_exactly():
    """**The reverse of the assertion this replaces, and the reversal is B12.**

    `moving_obstacle_from_bezier` used to raise `NotImplementedError` above
    degree 1, because the Rust builder understood only a straight capsule. The
    center-surface builder understands the general curve, so there is nothing to
    convert down to and nothing to refuse -- and a converter that still refused
    would make `curve` and `loiter`, the two scenarios the paper's contribution
    rests on, unrepresentable at this boundary.

    FAILS IF the refusal comes back, or if a curved obstacle is quietly
    flattened to its endpoints -- which would look like acceptance and be a
    different obstacle.
    """
    bo = {
        "control_points": [[0.0, 0.0, 0.0], [1.0, 1.0, 5.0], [2.0, 2.0, 10.0]],
        "radius": 0.5,
    }
    back = moving_obstacle_from_bezier(bo)
    cps = np.asarray(back["control_points"], dtype=float)
    assert cps.shape == (3, 3), "the middle control point must survive"
    np.testing.assert_allclose(cps, np.asarray(bo["control_points"], dtype=float), atol=0.0)


def test_backwards_time_is_refused():
    """A window that runs backwards is not an obstacle, and must not pass."""
    with pytest.raises(ValueError, match="backwards"):
        moving_obstacle_from_bezier(
            {"control_points": [[0.0, 0.0, 5.0], [1.0, 1.0, 0.0]], "radius": 0.5}
        )
