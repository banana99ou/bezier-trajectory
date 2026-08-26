"""
Scenario registry for the space-time Bezier demos.

**Scenarios are WRITTEN one way and LEAVE another way.** A straight obstacle is
readable as ``{pos0, vel, r}`` and there is no reason to stop writing it that
way. What leaves this module is always the canonical form -- lifted Bezier
control points in (x, ..., t), with the active window intrinsic to the first and
last control point's time coordinate -- because that is the only obstacle the
solver knows about since the 2026-08-21 formulation change. ``@_canonical`` does
the conversion, and constant velocity is simply the degree-1 case.

A curved obstacle is written directly as ``{control_points, radius}``.
"""

from __future__ import annotations

import functools

import numpy as np

from .geometry import normalize_obstacle


def _canonical(fn):
    """Convert a scenario's authored obstacles to lifted control points."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        sc = fn(*args, **kwargs)
        duration = float(sc["T"])
        lifted = []
        for obs in sc.get("obstacles", []):
            norm = normalize_obstacle(obs, duration)
            out = {
                "control_points": np.asarray(norm["control_points"], dtype=float).tolist(),
                "radius": float(norm["radius"]),
            }
            for key in ("name", "color"):
                if norm.get(key) is not None:
                    out[key] = norm[key]
            lifted.append(out)
        sc["obstacles"] = lifted
        return sc

    return wrapper


def make_wall(
    p1,
    p2,
    thickness: float = 0.5,
    spacing: float = 0.7,
    color: str = "#e67e22",
    name_prefix: str = "W",
    vel=None,
    t_start=None,
    t_end=None,
) -> list[dict]:
    """Create a wall obstacle as a row of overlapping circles between p1 and p2."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    length = np.linalg.norm(p2 - p1)
    n_circles = max(2, int(length / spacing) + 1)
    obstacles = []
    for idx in range(n_circles):
        alpha = idx / (n_circles - 1)
        pos = ((1.0 - alpha) * p1 + alpha * p2).tolist()
        obstacle = {
            "pos0": pos,
            "vel": vel or [0.0, 0.0],
            "r": thickness,
            "color": color,
            "name": f"{name_prefix}{idx}",
        }
        if t_start is not None:
            obstacle["t_start"] = float(t_start)
        if t_end is not None:
            obstacle["t_end"] = float(t_end)
        obstacles.append(obstacle)
    return obstacles


@_canonical
def scenario_original() -> dict:
    return {
        "name": "original",
        "title": "3 Moving Obstacles",
        "init_curve": {
            "mode": "quadratic_bow",
            "bow": 2.3,
            "side": 1.0,
            "workspace_center": [5.0, 5.0],
        },
        "obstacles": [
            {"pos0": [2.0, 8.0], "vel": [0.5, -0.7], "r": 0.8, "color": "#e74c3c", "name": "A"},
            {"pos0": [6.0, 2.0], "vel": [-0.3, 0.5], "r": 0.7, "color": "#2980b9", "name": "B"},
            {"pos0": [4.5, 5.5], "vel": [0.1, -0.3], "r": 0.6, "color": "#27ae60", "name": "C"},
        ],
        "start": [0.5, 1.0, 0.0],
        "end": [8.5, 8.5, 10.0],
        "T": 10.0,
    }


@_canonical
def scenario_diverse() -> dict:
    return {
        "name": "diverse",
        "title": "Diverse Moving Obstacles",
        "init_curve": {"mode": "straight"},
        "obstacles": [
            {"pos0": [1.0, 6.0], "vel": [0.8, -0.1], "r": 0.6, "color": "#e74c3c", "name": "A"},
            {"pos0": [5.0, 5.0], "vel": [0.05, -0.15], "r": 1.0, "color": "#2980b9", "name": "B"},
            {"pos0": [8.0, 1.0], "vel": [-0.6, 0.4], "r": 0.5, "color": "#27ae60", "name": "C"},
            {"pos0": [3.0, 1.5], "vel": [0.3, 0.6], "r": 0.4, "color": "#9b59b6", "name": "D"},
            {"pos0": [4.0, 8.0], "vel": [0.1, -0.5], "r": 0.9, "color": "#e67e22", "name": "E"},
            {"pos0": [7.0, 7.0], "vel": [-0.4, -0.3], "r": 0.35, "color": "#1abc9c", "name": "F"},
            {"pos0": [2.5, 4.0], "vel": [0.0, 0.0], "r": 0.7, "color": "#f39c12", "name": "G"},
        ],
        "start": [0.5, 0.5, 0.0],
        "end": [9.0, 9.0, 10.0],
        "T": 10.0,
    }


@_canonical
def scenario_wall() -> dict:
    """Wall that disappears early enough for the curve to wait and pass through.

    Spacing densified 0.8 -> 0.5 on 2026-08-24 (user request: the row of circles
    must READ as a wall). A denser wall is a different problem -- more rows, a
    slightly larger forbidden union between the old centres -- so the measured
    numbers were re-taken the same day; see README sec. Measurements.
    """
    wall_obs = make_wall(
        p1=[2.5, 0.0],
        p2=[2.5, 10.0],
        thickness=0.5,
        spacing=0.5,
        color="#e67e22",
        name_prefix="W",
        t_start=0.0,
        t_end=5.0,
    )
    other_obs = [
        {"pos0": [7.0, 3.0], "vel": [0.0, 0.3], "r": 0.6, "color": "#2980b9", "name": "M1"},
        {"pos0": [6.5, 8.0], "vel": [0.2, -0.2], "r": 0.5, "color": "#27ae60", "name": "M2"},
    ]
    return {
        "name": "wall",
        "title": "Disappearing Wall (t<5)",
        "init_curve": {"mode": "straight"},
        "obstacles": wall_obs + other_obs,
        "start": [1.0, 5.0, 0.0],
        "end": [8.0, 5.0, 10.0],
        "T": 10.0,
    }


@_canonical
def scenario_fence3d() -> dict:
    """Item B11 -- three spatial coordinates plus time.

    Renamed from `wall3d` 2026-08-24: in this repository "wall" means the
    disappearing door, and this scenario is not that -- it is a MOVING fence,
    and the motion is load-bearing: it is what makes waiting futile, so the
    demonstrated behaviour is the climb. (`door3d` is the waiting one.)
    Measurements recorded under the old name carry over unchanged -- the
    problem definition is byte-identical, only the key moved.

    `make_wall` is already dimension-agnostic, so this is the fence idea lifted:
    a wide, low fence advancing in +x, and the vehicle climbs over it. Every
    start/end point and every obstacle `pos0`/`vel` carries three spatial
    components; nothing in the solver changes.

    The third dimension is load-bearing, not decorative. The fence spans
    y in [0.5, 9.5] with radius 0.9, so going around costs a lateral excursion of
    about 5.4 while going over costs a climb of about 0.4 above the flight
    altitude. Measured: the identical scenario with the third spatial coordinate
    removed penetrates by 0.47 to 0.78 and converges at no config, while this one
    clears +0.1623 and certifies at N8_seg2. That is a local-solver statement,
    not an infeasibility proof -- the lateral detour is a different passing class
    that a straight seed does not reach.

    The fence sits at a single height on purpose. A second layer beneath it was
    measured and changed nothing to 1e-10: the solver goes over either way, so
    the lower row is 11 obstacles of runtime that never bind.
    """
    fence = make_wall(
        p1=[4.0, 0.5, 0.0],
        p2=[4.0, 9.5, 0.0],
        thickness=0.9,
        spacing=0.9,
        color="#e67e22",
        name_prefix="F",
        vel=[0.25, 0.0, 0.0],
    )
    movers = [
        {"pos0": [7.5, 2.0, 0.8], "vel": [-0.15, 0.35, 0.02], "r": 0.7,
         "color": "#2980b9", "name": "M1"},
        {"pos0": [8.5, 8.5, 0.2], "vel": [-0.2, -0.4, 0.05], "r": 0.6,
         "color": "#27ae60", "name": "M2"},
    ]
    return {
        "name": "fence3d",
        "title": "Moving Fence, 3 Spatial Dimensions",
        "init_curve": {"mode": "straight"},
        "obstacles": fence + movers,
        "start": [0.5, 5.0, 0.5, 0.0],
        "end": [9.5, 5.0, 0.5, 10.0],
        "T": 10.0,
    }


@_canonical
def scenario_door3d() -> dict:
    """A doorway in time: a static wall in 3D that stands until t=6.5, then opens.

    The demonstrated behaviour is WAITING -- the complement of `fence3d`, whose
    motion makes waiting futile so the solver climbs. Here the wall is static,
    tall, and gone after t=6.5; the cheap answer is to hold short and pass
    through the opening, visible in the lift as a steep stretch of curve along
    the time axis at the wall plane. The straight seed crosses x=4 around
    t=4.7, inside the window, so the seed genuinely conflicts.

    The wall is ONE row of large overlapping spheres (r=1.6, centres at z=0.9,
    y-spacing 0.8, spanning y in [-2, 12]), sealed up to z of about 2.45 with no
    interior gap. Every parameter here was forced by a measured escape route
    (all 2026-08-24), which is why the recorded CROSSING TIME is the assertion
    that matters, never the figure-grade flag alone:

    * two stacked rows of r=0.9 spheres left a diagonal pore (centre distance
      1.84 against a radii sum of 1.8) -- threaded at +0.007 clearance, t=2.6;
    * a sealed row spanning y in [0.5, 9.5] was simply flown around at its
      y-end -- +0.084 clearance, t=4.8, never higher than the flight altitude;
    * the window closes at t=5.0, not 6.5, because the capsule's rounded time
      cap keeps forbidding the wall plane for about r after the window -- with
      r=1.6 the door EFFECTIVELY opens near t=6.5. That conservatism is the
      declared SPACETIME_AXIS_SCALE=1 modelling choice, demonstrated.

    What this scenario may NOT be used to claim: nothing here makes the third
    spatial dimension load-bearing -- a 2D cut would wait just the same (`wall`
    is that scenario). The load-bearing-z evidence is `fence3d`'s, where motion
    closes the waiting strategy.

    Measured behaviour lives beside its assertion, not here -- see README
    sec. Measurements for the recorded pass.
    """
    row = make_wall(p1=[4.0, -2.0, 0.9], p2=[4.0, 12.0, 0.9],
                    thickness=1.6, spacing=0.8, color="#8e44ad",
                    name_prefix="D", vel=[0.0, 0.0, 0.0],
                    t_start=0.0, t_end=5.0)
    return {
        "name": "door3d",
        "title": "Door in 3D — Wait for the Opening (t>5)",
        "init_curve": {"mode": "straight"},
        "obstacles": row,
        "start": [0.5, 5.0, 0.5, 0.0],
        "end": [8.0, 5.0, 0.5, 10.0],
        "T": 10.0,
    }


@_canonical
def scenario_station_fence() -> dict:
    """Item B12 -- keep line of sight to a fixed station past a moving fence.

    Three spatial coordinates plus time, and the third spatial coordinate is
    LOAD-BEARING rather than decorative: the vehicle climbs, and the only reason
    it climbs is the occlusion constraint.

    Geometry, and why each piece of it is where it is.

    * The **station** sits off to the -y side at low altitude, roughly abeam the
      mid-path. The vehicle flies +x at y = 5, so the fence sits between them.
    * The **fence** is wide in x and low in z, and it never comes within a body
      radius of the vehicle's corridor -- the gap in y is at least 1.5 against a
      radius of 0.8. It is therefore not an obstacle the vehicle has to dodge:
      the keep-out rows are present and are not expected to bind. That is the
      point, not an oversight -- staying visible to the station already implies
      staying out of the body, so occlusion subsumes collision for this body and
      the keep-out machinery is demonstrated by the other scenarios.
    * The fence's **x extent is finite** (3.0 to 7.0). At the start and end of
      the horizon the vehicle is far enough along x that the sight line crosses
      the fence's plane beyond its ends, so the pinned endpoints are visible and
      the problem is feasible. In the middle the sight line crosses the fence
      head on, and the only way through is over the top.
    * The fence's path is **non-straight**: it descends in y and drifts in +x,
      then reverses on both. That is expressed as a CHAIN of two straight pieces
      on adjacent time windows whose caps overlap at the joint, which is the
      construction PAPER_1 sec. "Occluder geometry" requires -- a capsule around
      a curved centreline is not convex and would break the certificate, while
      each straight piece is convex on its own.

    The baseline that must be able to fail: with the occlusion rows removed the
    fence obstructs nothing, so the solver returns an essentially straight
    trajectory at the flight altitude and loses line of sight across most of the
    horizon. Two runs, identical but for one constraint block. Both assertions
    live in `tests/integration/test_station_fence_scenario.py`, not in this
    docstring.
    """
    body_z = 0.1
    radius = 0.8
    # Piece 1, t in [0, 5.2]: centre path from (x, 3.6) at t=0 to (x+0.5, 2.0)
    # at t=5. `pos0` is the position at t=0, which is what the solver extrapolates
    # from, so the piece endpoints are written at t=0 and the window does the
    # clipping.
    piece_a = make_wall(
        p1=[3.0, 3.6, body_z],
        p2=[7.0, 3.6, body_z],
        thickness=radius,
        spacing=0.8,
        color="#e67e22",
        name_prefix="A",
        vel=[0.1, -0.32, 0.0],
        t_start=0.0,
        t_end=5.2,
    )
    # Piece 2, t in [4.8, 10]: the reversal. Its position at t=5 matches piece
    # one's, so `pos0` is that point walked back to t=0 along the new velocity.
    # The windows OVERLAP on [4.8, 5.2] so the union covers the joint with no
    # gap for the curve to slip through.
    piece_b = make_wall(
        p1=[4.0, 0.4, body_z],
        p2=[8.0, 0.4, body_z],
        thickness=radius,
        spacing=0.8,
        color="#d35400",
        name_prefix="B",
        vel=[-0.1, 0.32, 0.0],
        t_start=4.8,
        t_end=10.0,
    )
    return {
        "name": "station_fence",
        "title": "Line of Sight Past a Moving Fence",
        "init_curve": {"mode": "straight"},
        "obstacles": piece_a + piece_b,
        "start": [0.5, 5.0, 0.5, 0.0],
        "end": [9.5, 5.0, 0.5, 10.0],
        # One of the two scenarios carrying this key (`station_gate` is the
        # other). Its absence everywhere else is what keeps every other
        # scenario's problem bit-identical to pre-B12.
        "stations": [[5.0, -2.0, 0.3]],
        "T": 10.0,
    }


def scenario_station_gate() -> dict:
    """A holding stack sweeps past the corridor, and the link is lost unless the vehicle climbs.

    The paper's demo scenario, and the replacement for `station_fence`, whose
    occluder is a chain of straight pieces on adjacent time windows -- the
    pre-2026-08-21 construction, which needed every piece convex. Here the
    occluder is ONE degree-4 Bezier per body, so its tube is genuinely
    non-convex and has no global supporting half-space. That is the case the
    clipped-volume construction exists for.

    The vehicle transits an air-mobility corridor past a vertiport. Traffic
    holding at that vertiport flies a pattern whose leg swings toward the
    corridor at t=3, away from it at t=5, and back at t=7, passing between the
    vehicle and the ground station on the way.

    Geometry, and why each piece of it is where it is.

    * The **body** is six overlapping spheres of radius 0.8, spaced 0.8 apart
      across x in [3, 7]: sequenced traffic on one holding track, not a single
      aircraft. It is written wide because the sight line has to be blocked
      head-on -- a single sphere is walked around in x for nothing. Reading a
      row of spheres as a queue of aircraft is a MODELLING CHOICE and is stated
      as one; so is the fact that a real corridor's lateral bounds, which this
      scenario format cannot express, are part of why going around is not the
      obvious answer.
    * The centreline's **y control points are [3.5, 3.5, -4.5, 3.5, 3.5]**,
      putting the curve at y=3.50 at t=3 and t=7 and at y=0.50 at t=5. The
      control polygon dips to -4.5 while the curve only reaches 0.50; that is
      ordinary Bezier behaviour, and the obstacle IS the curve, so nothing is
      approximated.
    * The body therefore **never comes within a body radius of the corridor**:
      the gap in y is at least 1.5 against a radius of 0.8. The keep-out rows
      are present and are not expected to bind, exactly as in `station_fence`
      and for the same reason -- staying visible already implies staying out of
      the body, and the keep-out machinery is demonstrated elsewhere.
    * The **time control points are evenly spaced and z is constant**. Neither
      is stylistic: `geometry.obstacle_positions_at` inverts time by division
      and assumes the time column is affine in the curve parameter.
    * The body is **active only on [3, 7]** of a horizon of 10, and that window
      is intrinsic to the first and last control point's time coordinate, so the
      pinned endpoints sit outside it and the problem is feasible.
    * The **station** sits at (5.0, -6.0, 0.5), abeam the mid-path on the far
      side in y, so the sight line crosses the body head-on while the body is
      swung out and the only way through is over the top.

    What this scenario DOES exercise, and what it does not.

    The clipped keep-out volume of this occluder **splits into two connected
    components** -- the leg approaching and the leg departing, separated in TIME
    rather than in space -- and the two walls that come out of it carry time
    components of opposite sign: one says *after it passes*, the other says
    *before it returns*. On the straight seed at N=8, 8 segments, trust 0.5,
    four of the forty-eight (segment, obstacle) pairs have two components. It is
    the only scenario in this repository that does.

    It is **not** the case that the returned trajectory threads that gap. The
    occlusion rows drive the vehicle up and away from the body, and a segment
    that far away has a clip ball that reaches only one leg, so on the converged
    iterate the count is back to one everywhere. The two demonstrations pull in
    opposite directions and this scenario resolves it in favour of occlusion: an
    altitude ceiling would be needed to make the time slot the cheaper answer,
    and the scenario format has no way to write one. Do not claim the temporal
    gate as the demonstrated behaviour on the strength of this scenario.

    The baseline that must be able to fail: with the occlusion rows removed the
    vehicle never leaves cruise altitude and the sight line is blocked for the
    whole of the body's active window. All of it is re-measured by
    `tests/integration/test_station_gate_scenario.py` -- a number in a docstring
    cannot fail.
    """
    radius = 0.8
    body_z = 0.45
    # Nearest the corridor at the window's ends, swung out at its middle. y=-4.5
    # is a control point, not a place the obstacle is ever at: the curve runs
    # from y=3.50 at t=3 to y=0.50 at t=5 and back.
    y_ctrl = [3.5, 3.5, -4.5, 3.5, 3.5]
    t_ctrl = [3.0, 4.0, 5.0, 6.0, 7.0]
    xs = [3.0, 3.8, 4.6, 5.4, 6.2, 7.0]
    obstacles = [
        {
            "control_points": [[x, y, body_z, t] for y, t in zip(y_ctrl, t_ctrl)],
            "radius": radius,
            "color": "#e67e22",
            "name": f"H{idx}",
        }
        for idx, x in enumerate(xs)
    ]
    return {
        "name": "station_gate",
        "title": "Holding Stack Sweeps the Corridor",
        "init_curve": {"mode": "straight"},
        "obstacles": obstacles,
        "start": [0.5, 5.0, 0.5, 0.0],
        "end": [9.5, 5.0, 0.5, 10.0],
        "stations": [[5.0, -6.0, 0.5]],
        "T": 10.0,
    }


# (degree, segment count) pairs tried per scenario.
#
# Low segment counts were added 2026-08-18. The pre-2026-08-17 geometry used one
# half-space per control point, so more segments meant more planes and generally
# better numbers, and these lists were built around that. With one plane per
# segment the trade reverses: each plane now binds every control point of its
# segment, so few segments is a genuinely different -- and here better -- regime.
# `original` and `diverse` had no config below 8 segments at all, which is why
# their best results were being missed. Measured at degree 8, 4 segments:
# original +0.6204 (converged, certified) against +0.3231 at 8 segments;
# diverse +0.0045 (feasible) against -0.40 at every count that was listed.
# Elastic (exact-penalty) weight the live sandbox uses per scenario.
#
# The batch path walks ELASTIC_WEIGHT_LADDER in optimize.py and reports which
# weight certified; the sandbox re-solves on every slider move and cannot afford
# a ladder, so it starts from the weight that is known to certify that scenario.
# These are measured, not guessed -- see the sweep recorded in CLAUDE.md.
#
# `wall` and `diverse` were recorded as infeasible for months. They are not:
# both clear and certify once the penalty exceeds the scenario's threshold. At
# the old fixed 100 the QP was being asked a different question, one whose
# optimum genuinely is a penetrating curve.
SCENARIO_ELASTIC_WEIGHT = {
    "original": 100.0,
    "diverse": 800.0,
    # `wall` is the hardest: it certifies only at 1e5, and only at 16 segments
    # (measured +0.0751, certificate 0.000, 19 iterations at N10_seg16). Lower
    # weights leave it penetrating by ~0.09 regardless of segment count.
    "wall": 100000.0,
    # `station_fence` needs the top rung too, and for the same reason: below it
    # the exact penalty is cheaper to pay than to satisfy, and the run returns a
    # trajectory standing on occlusion slack. Measured at N8_seg8 -- occlusion
    # certificate 0.61 at 100, 0.29 at 800, 0.50 at 3000, and 0.0 at 1e5.
    "station_fence": 100000.0,
    # PROVISIONAL -- copied from `station_fence` because it is the other
    # occlusion scenario, NOT measured on this one. Run the ladder before any
    # number from this scenario is quoted.
    "station_gate": 100000.0,
}


# NOT a scenario factory despite the name — it maps a scenario key to a weight,
# so it must not be decorated.
def scenario_elastic_weight(name: str) -> float:
    """Elastic weight for a named scenario, or the shared default."""
    from .optimize import DEFAULT_ELASTIC_WEIGHT

    return float(SCENARIO_ELASTIC_WEIGHT.get(name, DEFAULT_ELASTIC_WEIGHT))


@_canonical
def scenario_curve() -> dict:
    """One obstacle on a CURVED path — the case the whole construction is for.

    A straight obstacle sweeps a convex capsule in the lifted space, so a plane
    touching it anywhere already supports the whole tube; the pre-2026-08-21
    builder was sound on every scenario above for that reason alone. This
    obstacle's lifted centreline is a genuine Bezier arc, so its tube is NOT
    convex, it has no supporting half-space, and the clip-and-hull construction
    is doing real work rather than reproducing an easier answer.

    The arc sweeps across the corridor and back, so the segment centroid spends
    part of the run INSIDE the turn — the configuration where the tangent plane
    leaves 12.35% of the clipped piece on the safe side.
    """
    return {
        "name": "curve",
        "title": "Curved Obstacle Path",
        "init_curve": {"mode": "straight"},
        "obstacles": [
            {
                # Degree 3 in (x, y, t): out across the corridor and back.
                "control_points": [
                    [8.5, 1.0, 0.0],
                    [1.0, 3.5, 3.3],
                    [1.0, 6.5, 6.7],
                    [8.5, 9.0, 10.0],
                ],
                "radius": 0.9,
                "color": "#e74c3c",
                "name": "arc",
            },
            {"pos0": [3.0, 8.5], "vel": [0.35, -0.45], "r": 0.5, "color": "#2980b9", "name": "B"},
        ],
        "start": [0.5, 1.0, 0.0],
        "end": [9.5, 9.0, 10.0],
        "T": 10.0,
    }


SCENARIO_MAP = {
    # FIRST ON PURPOSE. The frontend has no explicit default scenario: the page
    # fills its <select> from the catalog's key order and the browser shows the
    # first option, so whatever is registered first here is what
    # `python3 -m spacetime_bezier` opens on. Moving this entry silently changes
    # that, which is why `tests/integration/test_frontend.py` asserts the order.
    #
    # Two configs, matching `station_fence` for the reason recorded there: the
    # occlusion rows re-aim every iteration, so these runs are an order of
    # magnitude longer than a keep-out-only one.
    "station_gate": (scenario_station_gate, [(8, 8), (8, 16)]),
    "original": (scenario_original, [(4, 4), (4, 8), (6, 8), (8, 4), (8, 8)]),
    # The only scenario whose obstacle tube is non-convex, which makes it the
    # only one that distinguishes the hull-projection plane from the tangent
    # plane. Everything else would pass with either.
    "curve":    (scenario_curve,    [(8, 4), (8, 8), (8, 16), (10, 8)]),
    "diverse":  (scenario_diverse,  [(8, 4), (8, 8), (8, 16), (10, 4), (10, 16)]),
    "wall":     (scenario_wall,     [(8, 2), (8, 3), (8, 4), (8, 16), (10, 16), (10, 24)]),
    # Four coordinates, not three. Everything downstream reads the dimension off
    # the array shape -- except the viewer, which plots columns 0, 1, 2 as
    # (x, y, t) and would render this one's z as if it were time. `viewer.py`
    # therefore filters it out of the catalog rather than drawing a false
    # picture; see the comment there. Renamed from `wall3d` 2026-08-24.
    "fence3d":  (scenario_fence3d,  [(8, 2), (8, 4), (8, 8), (10, 8)]),
    # The waiting demo: static tall wall, opens at t=6.5. Configs measured
    # 2026-08-24; see README sec. Measurements.
    "door3d":   (scenario_door3d,   [(8, 4), (8, 8)]),
    # Also four coordinates, so the viewer filters it out for the same reason.
    # Short list on purpose: the occlusion rows re-aim every iteration, so the
    # run is an order of magnitude longer than a keep-out-only one, and only
    # these two configurations were measured to converge with the occlusion
    # certificate at zero.
    "station_fence": (scenario_station_fence, [(8, 8), (8, 16)]),
}
