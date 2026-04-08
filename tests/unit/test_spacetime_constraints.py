"""
Unit tests for the space-time Bezier constraint builders.
"""

import numpy as np

from orbital_docking.de_casteljau import segment_matrices_equal_params
from spacetime_bezier.constraints import (
    build_box_bounds,
    build_spacetime_koz_constraints,
    build_time_monotonicity,
)


def test_time_monotonicity_rows_reference_time_coordinate():
    constraint = build_time_monotonicity(n_cp=4, dim=3, min_dt=0.2)

    assert constraint.A.shape == (3, 12)
    np.testing.assert_allclose(constraint.lb, np.array([0.2, 0.2, 0.2]))
    np.testing.assert_allclose(constraint.ub, np.full(3, np.inf))

    expected = np.zeros((3, 12))
    expected[0, 2] = -1.0
    expected[0, 5] = 1.0
    expected[1, 5] = -1.0
    expected[1, 8] = 1.0
    expected[2, 8] = -1.0
    expected[2, 11] = 1.0
    np.testing.assert_allclose(constraint.A, expected)


def test_spacetime_koz_constraints_skip_inactive_obstacles():
    A_list = segment_matrices_equal_params(2, 2)
    P = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    obstacles = [
        {"pos0": [1.0, 1.0], "vel": [0.0, 0.0], "r": 0.5, "t_start": 5.0, "t_end": 6.0}
    ]

    assert build_spacetime_koz_constraints(A_list, P, obstacles, dim=3) is None


def test_spacetime_koz_constraints_build_rows_for_active_obstacle():
    A_list = segment_matrices_equal_params(2, 1)
    P = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    obstacles = [{"pos0": [0.0, 1.0], "vel": [0.0, 0.0], "r": 0.5}]

    constraint = build_spacetime_koz_constraints(A_list, P, obstacles, dim=3)

    assert constraint is not None
    np.testing.assert_allclose(constraint.A[:, ::3], np.eye(3))
    np.testing.assert_allclose(constraint.A[:, 1::3], 0.0)
    np.testing.assert_allclose(constraint.A[:, 2::3], 0.0)
    np.testing.assert_allclose(constraint.lb, np.full(3, 0.5))
    np.testing.assert_allclose(constraint.ub, np.full(3, np.inf))


def test_box_bounds_fix_endpoints_and_limit_time():
    P_init = np.array(
        [
            [0.5, 1.0, 0.0],
            [4.0, 4.0, 5.0],
            [8.5, 8.5, 10.0],
        ]
    )

    bounds = build_box_bounds(P_init, coord_lb=-2.0, coord_ub=12.0, time_lb=0.0, time_ub_scale=1.5)

    np.testing.assert_allclose(bounds.lb[:2], P_init[0, :2])
    np.testing.assert_allclose(bounds.ub[:2], P_init[0, :2])
    np.testing.assert_allclose(bounds.lb[-3:-1], P_init[-1, :2])
    np.testing.assert_allclose(bounds.ub[-3:-1], P_init[-1, :2])
    np.testing.assert_allclose(bounds.lb[2::3], np.zeros(3))
    np.testing.assert_allclose(bounds.ub[2::3], np.full(3, 15.0))
