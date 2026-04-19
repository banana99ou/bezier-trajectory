"""
Unit tests for the BezierObstacle wire-format converters in
``spacetime_bezier.geometry``. The sandbox boundary depends on these
conversions being lossless for degree-1 obstacles.
"""

from __future__ import annotations

import numpy as np

from spacetime_bezier.geometry import (
    bezier_obstacle_from_moving,
    moving_obstacle_from_bezier,
)


def test_moving_obstacle_roundtrips_through_bezier():
    T = 10.0
    legacy = {"pos0": [2.0, 8.0], "vel": [0.5, -0.7], "r": 0.8, "name": "A", "color": "#e74c3c"}
    bo = bezier_obstacle_from_moving(legacy, T)
    back = moving_obstacle_from_bezier(bo)
    np.testing.assert_allclose(back["pos0"], legacy["pos0"], atol=1e-12)
    np.testing.assert_allclose(back["vel"], legacy["vel"], atol=1e-12)
    assert back["r"] == legacy["r"]
    assert back["t_start"] == 0.0
    assert back["t_end"] == T
    assert back["name"] == "A"
    assert back["color"] == "#e74c3c"


def test_finite_time_window_is_preserved():
    """Wall-style obstacles with explicit t_start/t_end must round-trip exactly."""
    T = 10.0
    legacy = {"pos0": [2.5, 5.0], "vel": [0.0, 0.0], "r": 0.5, "t_start": 0.0, "t_end": 5.0}
    bo = bezier_obstacle_from_moving(legacy, T)
    # Control points sit at the active-window endpoints in (x, y, t).
    assert bo["control_points"][0] == [2.5, 5.0, 0.0]
    assert bo["control_points"][1] == [2.5, 5.0, 5.0]
    back = moving_obstacle_from_bezier(bo)
    np.testing.assert_allclose(back["pos0"], [2.5, 5.0], atol=1e-12)
    np.testing.assert_allclose(back["vel"], [0.0, 0.0], atol=1e-12)
    assert back["t_start"] == 0.0
    assert back["t_end"] == 5.0


def test_infinite_window_clamps_to_scenario_T():
    T = 10.0
    legacy = {"pos0": [1.0, 2.0], "vel": [0.1, 0.2], "r": 0.3}
    bo = bezier_obstacle_from_moving(legacy, T)
    cps = bo["control_points"]
    assert cps[0][-1] == 0.0
    assert cps[1][-1] == T


def test_degree_two_bezier_rejected():
    import pytest

    bo = {
        "control_points": [[0.0, 0.0, 0.0], [1.0, 1.0, 5.0], [2.0, 2.0, 10.0]],
        "radius": 0.5,
    }
    with pytest.raises(NotImplementedError, match="degree 2"):
        moving_obstacle_from_bezier(bo)
