"""The center surface: the shadow is the same keep-out zone, one stretch out.

`spacetime_generator` reads an obstacle's centreline through a point light source
and gets a SURFACE where the tube had a curve:

    Sigma(s,u) = ( p + u*(x(s) - p),  t(s) ),   u >= 1
    K          = union over (s,u) of Ball( Sigma(s,u), u*r_m )

`u = 1` is the obstacle itself, so body and shadow are one connected set and one
row set. These tests are the evidence for that claim at the level the solver
actually runs: through the Rust builder, at real control polygons.

The tube's own evidence stays in `test_spacetime_koz_geometry.py`, which must not
move at all — that file is the regression test for "a generalization does not
change the special case".
"""

import numpy as np
import pytest

from spacetime_bezier.geometry import compute_los_margin, obstacle_array_bundle

bezier_opt = pytest.importorskip("bezier_opt")

DIM = 3
SPATIAL = 2
TRUST = 0.5

# One segment, so the De Casteljau matrix is the identity: the centroid is the
# mean of the control points and each row's lower bound IS its offset `b`.
N_SEG = 1

STATION = np.array([[0.0, 0.0]])

# A degree-2 occluder arcing across the sight lines, times 0 -> 4. Curved on
# purpose: a straight one sweeps a convex tube and would pass with machinery far
# weaker than this.
OCCLUDER = {
    "control_points": [[2.0, -1.0, 0.0], [3.0, 1.5, 2.0], [2.0, 3.0, 4.0]],
    "r": 0.35,
}

# A segment beyond the occluder, where its shadow falls, and close enough that
# the obstacle's own zone is in reach too -- so one call produces BOTH readings
# and the body/shadow comparison below is not quantifying over an empty set.
SEGMENT = np.array(
    [
        [3.6, 0.9, 1.4],
        [4.2, 1.6, 2.0],
        [4.6, 2.4, 2.6],
    ]
)


def _rows(P, obstacles, stations, n_seg=N_SEG, trust=TRUST):
    ctrl, radii = obstacle_array_bundle(obstacles, SPATIAL)
    out = bezier_opt.spacetime_koz_rows_exact(
        p=np.ascontiguousarray(np.asarray(P, dtype=float)),
        obstacle_ctrl=ctrl,
        obstacle_r=radii,
        n_seg=n_seg,
        trust_radius=trust,
        stations=None if stations is None else np.asarray(stations, dtype=float),
    )
    (normals, lbs, seg, cp, obs, comp, sta, rho, sound,
     dropped, dropped_shadow, unsound) = out
    return {
        "normals": np.asarray(normals, dtype=float).reshape(-1, DIM),
        "lbs": np.asarray(lbs, dtype=float),
        "seg": np.asarray(seg, dtype=int),
        "cp": np.asarray(cp, dtype=int),
        "obs": np.asarray(obs, dtype=int),
        "comp": np.asarray(comp, dtype=int),
        "sta": np.asarray(sta, dtype=int),
        "rho": np.asarray(rho, dtype=float),
        "sound": np.asarray(sound, dtype=bool),
        "dropped": int(dropped),
        "dropped_shadow": int(dropped_shadow),
        "unsound": int(unsound),
    }


def _bezier(ctrl, s):
    ctrl = np.asarray(ctrl, dtype=float)
    n = len(ctrl) - 1
    from math import comb

    return sum(
        comb(n, i) * (1 - s) ** (n - i) * s**i * ctrl[i] for i in range(n + 1)
    )


def test_the_shadow_generator_leaves_the_obstacles_own_wall_untouched():
    """`u = 1` is the obstacle, so adding a station cannot move its wall.

    The body rows built WITH a station present must equal, to the bit, the rows
    built without one. They are the same generator: stretch pinned at one.

    FAILS IF the station leaks into the body reading — through a shared clip
    radius, a shared band, or a reach rule that changed. That would mean the
    generalization altered the special case, which is the one thing it may not
    do.
    """
    plain = _rows(SEGMENT, [OCCLUDER], None)
    both = _rows(SEGMENT, [OCCLUDER], STATION)

    assert np.all(plain["sta"] == -1)
    body = both["sta"] == -1
    assert body.sum() == len(plain["lbs"]) > 0, "the body rows changed in number"

    assert np.array_equal(both["normals"][body], plain["normals"])
    assert np.array_equal(both["lbs"][body], plain["lbs"])
    assert np.array_equal(both["rho"][body], plain["rho"])
    assert np.array_equal(both["comp"][body], plain["comp"])


def test_a_station_actually_adds_walls():
    """The premise of every test below.

    FAILS IF passing a station produces no shadow rows at all, which would make
    the containment and agreement tests vacuous — they would be quantifying over
    an empty set and passing for the wrong reason.
    """
    both = _rows(SEGMENT, [OCCLUDER], STATION)
    assert (both["sta"] >= 0).sum() > 0
    assert both["dropped"] == 0 and both["dropped_shadow"] == 0


def test_every_point_of_the_true_shadow_inside_the_clip_ball_is_behind_its_wall():
    """The soundness invariant, and the only reason a shadow row certifies.

    Sample the TRUE keep-out zone — `Ball(Sigma(s,u), u*r_m)` over the stretch,
    which is body and umbra together — keep what lies inside the clip ball, and
    demand the wall contains every bit of it.

    FAILS IF the offset drops below the material's true support along `n`: part
    of the shadow would then sit on the ALLOWED side, and a curve satisfying the
    row could be in it. Detection floor: the offset is a rigorous CEILING, so an
    error smaller than the De Casteljau slack is conservative, not unsound, and
    is not caught here.
    """
    rows = _rows(SEGMENT, [OCCLUDER], STATION)
    centroid = SEGMENT.mean(axis=0)
    ctrl = np.asarray(OCCLUDER["control_points"], dtype=float)
    r_m = OCCLUDER["r"]
    p = STATION[0]

    shadow = np.flatnonzero(rows["sta"] >= 0)
    assert shadow.size > 0
    checked = 0
    for j in sorted(set(rows["comp"][shadow].tolist())):
        idx = shadow[rows["comp"][shadow] == j]
        n = rows["normals"][idx][0]
        b = float(rows["lbs"][idx][0])
        rho = float(rows["rho"][idx][0])
        # Directions on the unit sphere of the lifted space, plus the wall's own
        # normal, which is where the support is attained.
        dirs = [np.zeros(DIM), n, -n]
        for k in range(DIM):
            e = np.zeros(DIM)
            e[k] = 1.0
            dirs += [e, -e]
        for s in np.linspace(0.0, 1.0, 61):
            g = _bezier(ctrl, s)
            v = g[:SPATIAL] - p
            for u in np.linspace(1.0, 6.0, 41):
                centre = np.concatenate([p + u * v, [g[SPATIAL]]])
                for d in dirs:
                    z = centre + u * r_m * d
                    if np.linalg.norm(z - centroid) > rho + 1e-12:
                        continue  # outside the clip ball: not claimed
                    checked += 1
                    assert float(n @ z) <= b + 1e-9, (
                        f"shadow material escaped its wall at s={s:.3f}, "
                        f"u={u:.3f}: n.z = {float(n @ z):.6f} > b = {b:.6f}"
                    )
    assert checked > 0, "no shadow material fell inside any clip ball to check"


def test_a_shadow_wall_is_slanted_in_time():
    """Shadow walls are space-time walls, like every other wall here.

    The center surface carries the obstacle's own time coordinate, so its wall's
    normal has a time component for the same reason the tube's does. The prism
    with time-parallel walls that this replaced had an exactly zero one by
    construction — which made the formulation's claim about normals false of half
    its own rows.

    FAILS IF a shadow wall comes back time-blind beside a moving occluder, which
    is the G1 signature wherever it appears.
    """
    rows = _rows(SEGMENT, [OCCLUDER], STATION)
    shadow = rows["sta"] >= 0
    times = np.abs(rows["normals"][shadow][:, -1])
    assert times.max() > 1e-9, "every shadow wall here is time-blind"


def test_the_sweep_is_the_analytic_umbra_for_a_static_body():
    """The union of balls IS the tangent cone, not an approximation of it.

    For a static body the umbra is the cone from the station tangent to it, half
    angle `asin(r/d)`. A point at angle `theta` from the axis, beyond the body,
    is occluded exactly when `theta <= asin(r/d)`; the sweep must agree on both
    sides of that boundary.

    FAILS IF the radius stops scaling with the stretch — the failure this whole
    construction turns on. A constant radius makes the modelled shadow a cylinder
    instead of a cone, and it declares genuinely occluded positions clear.
    """
    p = np.zeros(2)
    x = np.array([3.0, 0.0])
    r = 0.5
    d = float(np.linalg.norm(x - p))
    alpha = np.arcsin(r / d)

    def swept(q, scale):
        us = np.linspace(1.0, 60.0, 60000)
        centres = p + us[:, None] * (x - p)
        radii = us * r if scale else np.full_like(us, r)
        return float(np.min(np.linalg.norm(q - centres, axis=1) - radii)) <= 0.0

    inside = 0
    outside = 0
    for theta in np.linspace(0.0, 3.0 * alpha, 40):
        for reach in (2.0, 4.0, 8.0):
            q = reach * d * np.array([np.cos(theta), np.sin(theta)])
            truth = theta <= alpha
            if abs(theta - alpha) < 1e-3:
                continue  # boundary: sampling cannot decide it
            assert swept(q, True) == bool(truth), (
                f"the sweep disagrees with the analytic cone at theta={theta:.4f}"
            )
            inside += int(truth)
            outside += int(not truth)
    assert inside > 0 and outside > 0, "the fixture must cover both sides"

    # And the constant-radius version must get one of them wrong, or this test
    # is not discriminating.
    far = 8.0 * d * np.array([np.cos(0.9 * alpha), np.sin(0.9 * alpha)])
    assert swept(far, True) is True
    assert swept(far, False) is False, (
        "a constant radius must MISS this occluded point — if it does not, this "
        "test cannot tell the two models apart"
    )


def test_the_builder_and_the_independent_oracle_agree_on_a_lost_link():
    """Two computations that share no code must agree about the same trajectory.

    `compute_los_margin` is pure Python: it samples the curve, reconstructs the
    occluder at each sample's own time and measures the sight segment directly.
    It has never heard of a half-space. When it says the link is lost, the
    solver's own shadow rows must be violated too.

    FAILS IF the builder reports every row satisfied for a trajectory the oracle
    says is blocked — the silent-success this construction exists to make
    impossible.
    """
    # Straight through the shadow: control points on the far side of the
    # occluder from the station, at the times the occluder is there.
    blocked = np.array(
        [
            [4.0, -2.0, 0.5],
            [5.6, 2.8, 2.0],
            [4.0, 6.0, 3.5],
        ]
    )
    _, margins = compute_los_margin(
        blocked, STATION[0], [OCCLUDER], dim=DIM, n_eval=2001
    )
    finite = margins[np.isfinite(margins)]
    assert finite.size and float(finite.min()) < 0.0, (
        "the fixture must actually lose the link"
    )

    rows = _rows(blocked, [OCCLUDER], STATION, n_seg=4)
    shadow = rows["sta"] >= 0
    assert shadow.sum() > 0
    hulls = np.asarray(blocked, dtype=float)
    # n_seg=4 on a degree-2 polygon: reconstruct each row's own control point via
    # the De Casteljau matrices the builder used.
    from orbital_docking.de_casteljau import segment_matrices_equal_params

    a_list = [np.asarray(a, float) for a in segment_matrices_equal_params(2, 4)]
    worst = np.inf
    for i in np.flatnonzero(shadow):
        q = (a_list[rows["seg"][i]] @ hulls)[rows["cp"][i]]
        worst = min(worst, float(rows["normals"][i] @ q) - rows["lbs"][i])
    assert worst < 0.0, (
        f"the oracle says the link is lost by {float(finite.min()):.4f} but every "
        f"shadow row is satisfied (worst margin {worst:.4e})"
    )
