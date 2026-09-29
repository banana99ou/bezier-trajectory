"""
What the families put in front of the solver, counted at the initial guess with
the exact row builder. This is the linear-size claim as a test: walls are one
per (segment, obstacle, approach), a straight tube has one approach, so for a
straight-traffic member the wall count is at most segments x obstacles and it
must GROW with the obstacle count. A family whose rows did not grow would be a
scaling curve over nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

bezier_opt = pytest.importorskip("bezier_opt")

from orbital_docking.de_casteljau import segment_matrices_equal_params  # noqa: E402
from spacetime_bezier.families import DEFAULT_CONFIG, family_members  # noqa: E402
from spacetime_bezier.geometry import obstacle_array_bundle  # noqa: E402
from spacetime_bezier.objective import build_initial_guess  # noqa: E402


def _walls_at_initial_guess(sc, trust=0.5):
    N, n_seg = DEFAULT_CONFIG
    P0 = build_initial_guess(sc["start"], sc["end"], N + 1, init_curve=sc.get("init_curve"))
    dim = P0.shape[1]
    ctrl, radii = obstacle_array_bundle(sc["obstacles"], dim - 1)
    res = bezier_opt.spacetime_koz_rows_exact(
        p=P0, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
        trust_radius=float(trust), sound_clip=True, stations=None)
    seg, obs, comp = (np.asarray(res[2]), np.asarray(res[4]), np.asarray(res[5]))
    walls = {(int(s), int(o), int(c)) for s, o, c in zip(seg, obs, comp)}
    return walls, n_seg, len(sc["obstacles"])


def test_traffic_walls_grow_with_aircraft_and_stay_linear():
    """FAILS IF adding aircraft adds no walls (vacuous axis) or if walls ever
    exceed segments x obstacles (an approach split on a straight tube, which
    the construction says cannot happen)."""
    counts = []
    for n, sc in family_members("traffic"):
        walls, n_seg, n_obs = _walls_at_initial_guess(sc)
        assert len(walls) <= n_seg * n_obs, f"n={n}: {len(walls)} walls > {n_seg}x{n_obs}"
        assert len(walls) >= 1, f"n={n}: no wall at the initial guess"
        counts.append((n, len(walls)))
    ns = [c[0] for c in counts]
    ws = [c[1] for c in counts]
    assert ws == sorted(ws), f"walls not monotone in aircraft count: {counts}"
    assert ws[-1] > 2 * ws[0], f"twenty aircraft should cost far more than one: {counts}"


def test_horizon_walls_are_bounded_by_segments_times_obstacles_at_every_T():
    """The horizon axis must not grow the problem. FAILS IF a longer horizon
    breaks the segments x obstacles bound."""
    for T, sc in family_members("horizon"):
        walls, n_seg, n_obs = _walls_at_initial_guess(sc)
        assert 1 <= len(walls) <= n_seg * n_obs, f"T={T}: {len(walls)} walls"


def test_dimension_walls_are_bounded_in_both_dimensions():
    for d, sc in family_members("dimension"):
        walls, n_seg, n_obs = _walls_at_initial_guess(sc)
        assert 1 <= len(walls) <= n_seg * n_obs, f"dim={d}: {len(walls)} walls"


def test_curved_traffic_puts_walls_in_front_of_the_solver_and_grows():
    """The curved arm is the reliability arm (SPEC.md sec. 4.7). A curved tube
    may be approached more than once, so the straight arm's segments x
    obstacles bound does not apply here. What must hold is that the arm is
    not vacuous.

    FAILS IF a curved member produces no wall at the initial guess, or if
    twenty turning aircraft cost no more than twice what one does.
    """
    counts = []
    for n, sc in family_members("traffic_curved"):
        walls, _, _ = _walls_at_initial_guess(sc)
        assert len(walls) >= 1, f"curved n={n}: no wall at the initial guess"
        counts.append((n, len(walls)))
    assert counts[-1][1] > 2 * counts[0][1], f"curved arm does not grow: {counts}"
