"""Per-coordinate workspace bounds (the altitude band), item added 2026-08-26.

The box rows are physics-class: assembled before the keep-out block, outside
the elastic slack range, so no elastic weight can buy through them. What these
tests pin:

  * the band actually confines the returned control points (a solve that would
    otherwise leave the band comes back inside it);
  * a run WITHOUT the key solves the identical problem it always did;
  * malformed bands and bands that exclude a pinned endpoint are refused
    loudly, Python-side, before any solver runs.

The scenario used is deliberately tiny -- one static obstacle the curve must
climb over in z, where an unbounded run measurably uses altitude the band
forbids.
"""

import numpy as np
import pytest

from spacetime_bezier.geometry import bezier_curve
from spacetime_bezier.optimize import optimize_spacetime


def _solve(coord_bounds=None, **kw):
    # A wall of three balls across the corridor at x=5: over the top in z is
    # the cheap exit, so a ceiling on z is guaranteed to matter.
    obstacles = [
        {"pos0": [5.0, 3.0, 1.0], "vel": [0.0, 0.0, 0.0], "r": 1.2, "t_start": 0.0, "t_end": 10.0, "name": "W0"},
        {"pos0": [5.0, 5.0, 1.0], "vel": [0.0, 0.0, 0.0], "r": 1.2, "t_start": 0.0, "t_end": 10.0, "name": "W1"},
        {"pos0": [5.0, 7.0, 1.0], "vel": [0.0, 0.0, 0.0], "r": 1.2, "t_start": 0.0, "t_end": 10.0, "name": "W2"},
    ]
    return optimize_spacetime(
        N=8,
        dim=4,
        p_start=[0.5, 5.0, 1.0, 0.0],
        p_end=[9.5, 5.0, 1.0, 10.0],
        obstacles=obstacles,
        n_seg=4,
        max_iter=120,
        scp_trust_radius=0.5,
        coord_bounds=coord_bounds,
        verbose=False,
        **kw,
    )


def test_band_confines_the_returned_control_points():
    P_free, _ = _solve()
    P_band, info = _solve(coord_bounds=[(-20.0, 20.0), (-20.0, 20.0), (0.2, 20.0)])
    # The premise must hold or the test proves nothing: unbounded, this solver
    # DIVES (measured z down to -0.79 -- ducking under the wall, the same
    # behaviour the loiter scenario showed against its ground station). The
    # floor is therefore the side of the band this fixture exercises.
    assert P_free[:, 2].min() < 0.2 - 1e-6, (
        f"premise broken: the unbounded run bottomed at z={P_free[:, 2].min():.3f}, "
        "which the floor would not even touch -- raise the floor"
    )
    # Control points inside the band => the whole curve is, by the convex hull
    # property. 1e-7 covers Clarabel's interior-point tolerance.
    assert P_band[:, 2].min() >= 0.2 - 1e-7
    assert P_band[:, 2].max() <= 20.0 + 1e-7
    curve = bezier_curve(P_band, 400)
    assert curve[:, 2].min() >= 0.2 - 1e-6


def test_absent_key_reproduces_the_unbounded_problem():
    P_a, _ = _solve()
    P_b, _ = _solve(coord_bounds=None)
    np.testing.assert_array_equal(P_a, P_b)


def test_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="one per"):
        _solve(coord_bounds=[(0.0, 1.9)])  # 1 pair for 3 spatial coordinates


def test_reversed_interval_is_refused():
    with pytest.raises(ValueError, match="empty or reversed"):
        _solve(coord_bounds=[(-20.0, 20.0), (-20.0, 20.0), (1.9, 0.0)])


def test_band_excluding_a_pinned_endpoint_is_refused():
    # Endpoints sit at z=1.0; a band [2, 3] excludes them. The rows exempt
    # pinned endpoints, so without this refusal the curve would leap from the
    # endpoint into the band -- a scenario bug, not a solvable problem.
    with pytest.raises(ValueError, match="excludes the start point"):
        _solve(coord_bounds=[(-20.0, 20.0), (-20.0, 20.0), (2.0, 3.0)])
