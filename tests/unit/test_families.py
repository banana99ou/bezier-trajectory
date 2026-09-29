"""
The scenario families in ``spacetime_bezier.families``.

A family exists so a curve can be drawn along ONE knob. Each test here pins a
property that, if it silently failed, would make the curve measure something
other than the knob: an aircraft the plan never meets (count axis vacuous), a
horizon member whose spatial picture drifted (horizon axis confounded), a 3+t
member with a different schedule from its 2+t twin (dimension axis confounded).
"""

from __future__ import annotations

import numpy as np
import pytest

from spacetime_bezier.families import (
    FAMILY_MAP,
    family_dimension,
    family_horizon,
    family_members,
    family_motion,
    family_traffic,
    traffic_crossings,
)

REQUIRED = ("name", "title", "init_curve", "obstacles", "start", "end", "T", "family")


def _all_members():
    for name in FAMILY_MAP:
        for v, sc in family_members(name):
            yield name, v, sc


def _obstacle_position(cps: np.ndarray, t: float) -> np.ndarray | None:
    """Spatial position of a canonical (lifted Bezier) obstacle at time ``t``,
    or None outside its window. Bisection on the monotone time coordinate."""
    if t < cps[0, -1] - 1e-12 or t > cps[-1, -1] + 1e-12:
        return None
    N = len(cps) - 1
    from math import comb

    def ev(u):
        return sum(comb(N, i) * (1 - u) ** (N - i) * u ** i * cps[i] for i in range(N + 1))

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ev(mid)[-1] < t:
            lo = mid
        else:
            hi = mid
    return ev(0.5 * (lo + hi))[:-1]


def _nominal(sc, t):
    a = np.asarray(sc["start"][:-1], float)
    b = np.asarray(sc["end"][:-1], float)
    return a + (b - a) * (t / sc["T"])


def test_every_member_has_the_scenario_contract():
    """FAILS IF a member could not be handed to ``optimize_scenario`` as-is."""
    for name, v, sc in _all_members():
        for k in REQUIRED:
            assert k in sc, f"{name}={v}: missing {k}"
        dim = len(sc["start"])
        assert len(sc["end"]) == dim
        assert sc["start"][-1] == 0.0 and sc["end"][-1] == sc["T"] > 0
        assert len(sc["obstacles"]) >= 1
        for o in sc["obstacles"]:
            cps = np.asarray(o["control_points"], float)
            assert cps.shape[1] == dim, f"{name}={v}: obstacle {o.get('name')} is {cps.shape[1]}-d in a {dim}-d scene"
            assert o["radius"] > 0
            assert cps[0, -1] <= cps[-1, -1], "obstacle window runs backwards"


@pytest.mark.parametrize("family", ["traffic", "traffic_curved"])
def test_the_initial_guess_conflicts_with_every_aircraft(family):
    """The count axis is real only if every aircraft is a conflict.

    FAILS IF any traffic member, straight or curved arm, contains an aircraft
    whose body never overlaps the straight, uniform-speed initial plan at the
    same instant -- such an aircraft adds no keep-out row and the curve would
    be counting decorations.
    """
    ts = np.linspace(0.0, 1.0, 801)
    for v, sc in family_members(family):
        for o in sc["obstacles"]:
            cps = np.asarray(o["control_points"], float)
            best = np.inf
            for f in ts:
                t = f * sc["T"]
                pos = _obstacle_position(cps, t)
                if pos is None:
                    continue
                best = min(best, float(np.linalg.norm(pos - _nominal(sc, t))))
            assert best < o["radius"], (
                f"{family} n={v}: {o['name']} closest approach to the plan is {best:.3f} "
                f"> radius {o['radius']:.3f}; it is not a conflict")


def test_conflicts_are_spread_not_clumped():
    """Twenty aircraft must be twenty separate encounters. FAILS IF two
    crossings land within 2 percent of the transit of each other."""
    s = sorted(o["cross_s"] for o in traffic_crossings(20))
    gaps = np.diff(s)
    assert gaps.min() > 0.02, f"min crossing gap {gaps.min():.4f}"
    assert 0.15 <= s[0] and s[-1] <= 0.85


def test_the_seed_is_the_only_randomness():
    a = family_traffic(8, seed=3)
    b = family_traffic(8, seed=3)
    c = family_traffic(8, seed=4)
    assert a == b
    assert a["obstacles"] != c["obstacles"]


def test_the_horizon_family_keeps_the_spatial_picture():
    """The horizon knob must move the clock and nothing else.

    FAILS IF a crossing point or angle changes with T, or if a velocity does
    not scale exactly as 1/T -- either would let horizon length smuggle in a
    geometry change.
    """
    ref = traffic_crossings(8, T=10.0, seed=0)
    for T in (2.0, 5.0, 20.0):
        got = traffic_crossings(8, T=T, seed=0)
        for o_ref, o in zip(ref, got):
            np.testing.assert_allclose(o["pos0"], o_ref["pos0"], atol=1e-12)
            np.testing.assert_allclose(np.asarray(o["vel"]) * T,
                                       np.asarray(o_ref["vel"]) * 10.0, atol=1e-12)
            assert o["r"] == o_ref["r"] and o["cross_s"] == o_ref["cross_s"]
    for T, sc in family_members("horizon"):
        assert sc["T"] == T and sc["family"]["axis"] == "horizon"


def test_the_dimension_family_shares_one_schedule():
    """2+t and 3+t of one seed are the same conflicts, lifted. FAILS IF the
    extra dimension changed a crossing fraction, radius or speed."""
    two = traffic_crossings(8, dim_spatial=2, seed=0)
    three = traffic_crossings(8, dim_spatial=3, seed=0)
    for a, b in zip(two, three):
        assert a["cross_s"] == b["cross_s"]
        assert a["r"] == b["r"]
        assert np.isclose(np.linalg.norm(a["vel"]), np.linalg.norm(b["vel"]))
        assert len(b["pos0"]) == 3 and len(a["pos0"]) == 2
    sc3 = family_dimension(3)
    assert len(sc3["start"]) == 4 and all(len(o["control_points"][0]) == 4 for o in sc3["obstacles"])


def test_motion_kinds_are_three_different_tubes():
    """Straight is degree 1, arc is a single cubic, circle is a chain of four
    cubics on contiguous windows that closes. FAILS IF any of those collapses
    into another -- the axis would then compare a tube with itself."""
    s = family_motion("straight")["obstacles"]
    a = family_motion("arc")["obstacles"]
    c = family_motion("circle")["obstacles"]
    assert [len(o["control_points"]) for o in s] == [2]
    assert [len(o["control_points"]) for o in a] == [4]
    assert [len(o["control_points"]) for o in c] == [4, 4, 4, 4]
    ends = [np.asarray(o["control_points"], float) for o in c]
    for k in range(3):
        np.testing.assert_allclose(ends[k][-1], ends[k + 1][0], atol=1e-12)
    np.testing.assert_allclose(ends[3][-1][:-1], ends[0][0][:-1], atol=1e-12)
    with pytest.raises(ValueError, match="unknown motion kind"):
        family_motion("helix")


def test_every_motion_kind_crosses_the_plan():
    """The motion axis compares tubes; each must actually be in the way.
    FAILS IF a kind's body never overlaps the plan at the same instant."""
    ts = np.linspace(0.0, 1.0, 801)
    for kind, sc in family_members("motion"):
        best = np.inf
        for o in sc["obstacles"]:
            cps = np.asarray(o["control_points"], float)
            for f in ts:
                t = f * sc["T"]
                pos = _obstacle_position(cps, t)
                if pos is not None:
                    best = min(best, float(np.linalg.norm(pos - _nominal(sc, t))) - o["radius"])
        assert best < 0.0, f"motion={kind}: nearest approach to the plan is {best:+.3f}, not a conflict"


def _quad(cps: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray]:
    """Point and derivative of a quadratic Bezier at ``u``."""
    p0, p1, p2 = cps
    x = (1 - u) ** 2 * p0 + 2 * u * (1 - u) * p1 + u ** 2 * p2
    dx = 2 * (1 - u) * (p1 - p0) + 2 * u * (p2 - p1)
    return x, dx


@pytest.mark.parametrize("dim_spatial", [2, 3])
def test_the_curved_arm_shares_the_straight_schedule(dim_spatial):
    """The two arms must differ in the turn and nothing else, or the arm
    comparison measures a different traffic picture instead of curvature.

    FAILS IF a curved aircraft has a different crossing fraction or radius
    than its straight twin, is not AT the straight twin's crossing point at the
    crossing time, or does not have the straight twin's velocity there.
    """
    T = 10.0
    for seed in range(5):
        straight = traffic_crossings(8, dim_spatial, T, seed)
        curved = traffic_crossings(8, dim_spatial, T, seed, curved=True)
        for a, b in zip(straight, curved):
            assert a["cross_s"] == b["cross_s"] and a["r"] == b["radius"]
            cps = np.asarray(b["control_points"], float)
            t_c = a["cross_s"] * T
            x, dx = _quad(cps, a["cross_s"])
            x_cross = np.asarray(a["pos0"]) + np.asarray(a["vel"]) * t_c
            np.testing.assert_allclose(x[-1], t_c, atol=1e-12)
            np.testing.assert_allclose(x[:-1], x_cross, atol=1e-9)
            # d(space)/dt = d(space)/du / d(t)/du
            np.testing.assert_allclose(dx[:-1] / dx[-1], a["vel"], atol=1e-9)


@pytest.mark.parametrize("dim_spatial", [2, 3])
def test_the_curved_arm_actually_turns(dim_spatial):
    """A curved arm whose turn collapsed to a straight line would compare the
    straight arm with itself, and the tube would be convex again.

    FAILS IF any curved aircraft turns its heading by less than 25 degrees
    over its window.
    """
    for seed in range(5):
        for o in traffic_crossings(20, dim_spatial, 10.0, seed, curved=True):
            cps = np.asarray(o["control_points"], float)
            _, d0 = _quad(cps, 0.0)
            _, d1 = _quad(cps, 1.0)
            h0, h1 = d0[:-1], d1[:-1]
            cosang = h0 @ h1 / (np.linalg.norm(h0) * np.linalg.norm(h1))
            turn = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            assert turn > 25.0, f"seed {seed} {o['name']}: turns only {turn:.1f} deg"


def test_the_curved_arm_keeps_its_picture_across_horizons():
    """FAILS IF the horizon knob changes a curved aircraft's spatial path --
    the curved arm must obey the same one-knob rule as the straight arm."""
    ref = traffic_crossings(8, T=10.0, seed=0, curved=True)
    for T in (2.0, 5.0, 20.0):
        got = traffic_crossings(8, T=T, seed=0, curved=True)
        for o_ref, o in zip(ref, got):
            a = np.asarray(o_ref["control_points"], float)
            b = np.asarray(o["control_points"], float)
            np.testing.assert_allclose(b[:, :-1], a[:, :-1], atol=1e-12)
            np.testing.assert_allclose(b[:, -1] * 10.0, a[:, -1] * T, atol=1e-12)
