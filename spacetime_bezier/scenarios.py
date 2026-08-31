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
    # MEASURED 2026-08-30 at N8_seg8, priced run (free arrival, time_weight 10,
    # v_max 5), center-surface builder, `sound_clip=True` -- the configuration
    # the paper figure is drawn from. Every rung certifies and returns the SAME
    # trajectory (arrival 55.747, min line-of-sight +10.642, clearance 30.524,
    # agreeing to 1e-6); they differ only in how long they take to get there:
    #
    #   1e4  121 iterations, 2.6 s     1e5  12 iterations, 0.3 s
    #   1e6   11 iterations, 0.2 s
    #
    # 1e5 is registered: a decade above the rung that needs ten times the
    # iterations for the identical answer, and a decade below one that buys
    # nothing further. The two earlier readings were of other problems -- the
    # 2026-08-28 rescale measured 1e5 standing on slack (4.2e-05) under the
    # RETIRED occlusion builder, and the center-surface builder was measured at
    # a tenth of these lengths (1e4 certified, 1e5 hit the iteration cap).
    "loiter": 100000.0,
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


@_canonical
def scenario_loiter() -> dict:
    """A body ORBITING the ground station, and a corridor its shadow sweeps ALONG.

    **The numbers are metres and seconds.** A ground control station on the soil,
    another aircraft loitering 50 m above it on a 30 m circle at 2.4 m/s, and a
    small unmanned aircraft crossing 200 m of corridor at 62.5 m altitude and
    5 m/s while the loiterer's shadow sweeps that corridor.

    **Timing is the only escape, and that is measured, not asserted.** Take the
    constrained run's spatial path, graft the BASELINE's schedule onto it, and
    re-measure line of sight: it goes to -2.109 and the link is lost (the
    reverse graft, the baseline's path on the constrained schedule, keeps it at
    +10.620). The same path flown on the earlier schedule fails, so the retiming
    is what saves the run. The spatial path is not doing the work and cannot:
    the constrained run deviates 0.08 m laterally over 200 m and its path is
    200.00 m long, +0.00 percent against the straight line. (Re-measured
    2026-08-30 under the center-surface builder with the reach floor; the
    2026-08-28 readings under the retired builder were -1.699 and 0.00 m.)

    Two design choices buy that, and the geometry earns both:

    * **The corridor is TANGENT to the shadow ring, not a chord of it.** At
      altitude z the shadow of the body falls on a ring of radius
      ``orbit_r * z / body_z`` -- 37.5 m at z = 62.5 -- and its spot there has
      radius ``body_r * z / body_z`` = 7.5 m. Put the corridor ON that ring
      (y = 37.5) and the spot arrives moving PARALLEL to the corridor instead of
      across it, so it sits on the path instead of crossing it. An earlier
      version ran the corridor through the ring's centre line; the spot crossed
      transversally, a sidestep cleared it, and the run's own retiming was
      decoration -- grafting the baseline schedule onto that path still kept the
      link at +2.393.
    * **The corridor has a WIDTH, 5 m either side.** The spot's radius is 7.5 m
      and at tangency it is centred on the corridor axis, so no lateral offset
      the corridor allows can clear it. Without the width the solver sidesteps
      15.8 m and never has to wait. Narrower does not work: at +-3, +-2 and +-1
      the run stops certifying (occlusion certificate 5 to 13, standing on
      slack, at both 1e6 and 1e7).

    **The phase is scanned, and what it sets is WHEN the shadow reaches the
    corridor relative to the baseline's transit.** Tangency alone does not pick
    it: phase 0 puts the body at the tangency point at a quarter lap, which is
    also when the baseline crosses that point, and it gives the deepest baseline
    failure (-6.000) -- but the constrained run does not certify there
    (occlusion certificate 3.13, standing on slack). Scanned 2026-08-28 at the
    registered weight, 9 of the phases tried satisfy all three conditions at
    once: the run certifies, the baseline loses the link, and the graft above
    fails. 51 deg is the one where the spatial path is untouched. (That scan
    ran under the retired occlusion builder; only 51 deg has been re-measured
    since.) Measured there, 2026-08-30, center-surface builder with the reach
    floor: the baseline loses the link over [14.96, 16.52] s at min
    line-of-sight -2.169 -- bit-identical to the 2026-08-28 reading, as it must
    be, since the baseline has no shadow rows and the builder change cannot
    reach it -- and the constrained run keeps it at +10.642 by arriving 15.7 s
    later, clearing the body by 30.5 m, in 12 iterations with both certificates
    0.0 and zero unsound clips. The deeper-baseline alternatives cost path
    purity -- 12 deg gave -5.781 but 3.19 m of lateral deviation.

    The remaining pieces close the ways the solver was measured to squirm out
    (2026-08-26, at a tenth of these lengths):

    * **Station at the origin, ON the soil.** The floor of the altitude band is
      what makes the soil real -- without it the solver dives, because z < 0 is
      shadow-free (a sight line to a ground station never passes below the
      ground; measured at the authoring scale: it went to z = -1.86).
    * **The band sits above the body's reach** (floor 60 > body top 56), so the
      keep-out rows are present and never binding -- the constrained run clears
      by 30.5 m. What forces the timing is the line-of-sight rows and nothing
      else.
    * **One full lap over the horizon** (T = 80), so the shadow visits the
      corridor once and "wait for it to pass" is meaningful. A cubic cannot
      close a circle, so the lap is a CHAIN of four quarter-arc pieces on
      contiguous time windows, each piece a cubic arc;
      the quarter-circle constant k = 0.5522847498 overshoots the radius by
      0.03 percent, far below the body radius 6.

    Two solver knobs are lengths, so they do NOT come along for free when the
    scene is scaled:

    * ``trust_radius`` 5.0, and it is a scenario key for a measured reason. The
      clip radius is built from it (``reach = seg_radius + trust * sqrt(dim)``
      in ``clip_band``), so the stock 0.5 against a 200 m scene is degenerate:
      the constrained run then pushes arrival to the horizon end and still loses
      the link. Callers must forward it the way they forward ``coord_bounds``.
    * The elastic weight, 1e5 in ``SCENARIO_ELASTIC_WEIGHT``, with the ladder
      it was read off recorded beside it. The exact-penalty threshold scales
      with the problem, which is why it is re-measured whenever the scene or
      the builder changes.

    ``min_dt`` is the exception that needs no scaling: at an 80 s horizon the
    floor never binds.

    The arrival time is meant to be FREE. That is a solve flag, not a scenario
    key: tick ``free_arrival_time`` with a ``time_weight > 0`` and a ``v_max``,
    or the solver refuses by design (a freed arrival that nothing prices is an
    artifact generator -- see ``optimize.py``). The paper figure runs it with
    ``time_weight=10.0``, ``v_max=5.0`` and ``sound_clip=True`` -- the reach
    floor on the clip radius (PAPER_1 statement 8), so the certificate speaks
    for the whole keep-out zone. Measured to matter here: without it the
    returned iterate has 24 (segment, obstacle) pairs whose wall covers only a
    clipped piece, and the figure tool refuses to draw it.
    """
    orbit_r = 30.0   # m, radius of the loiter circle
    body_z = 50.0    # m, altitude of the loiter circle
    body_r = 6.0     # m, keep-out radius of the loitering body
    corridor_z = 62.5    # m, altitude of the transit
    corridor_half = 5.0  # m, lateral half-width of the corridor
    # The shadow ring at the corridor's altitude. Putting the corridor here is
    # the whole design: the spot arrives moving along the corridor rather than
    # across it. Derived, not tuned -- change either altitude and it follows.
    corridor_y = orbit_r * corridor_z / body_z   # 37.5 m
    k = 0.5522847498  # cubic quarter-circle constant
    # Scanned, not derived -- see the docstring. This is the phase where the run
    # certifies AND the baseline fails AND the constrained path flown on the
    # baseline's schedule still loses the link, with the spatial path untouched.
    phase = np.deg2rad(51.0)
    lap = [
        # (x, y) control points of each quarter, counterclockwise from (orbit_r, 0).
        [(orbit_r, 0.0), (orbit_r, orbit_r * k), (orbit_r * k, orbit_r), (0.0, orbit_r)],
        [(0.0, orbit_r), (-orbit_r * k, orbit_r), (-orbit_r, orbit_r * k), (-orbit_r, 0.0)],
        [(-orbit_r, 0.0), (-orbit_r, -orbit_r * k), (-orbit_r * k, -orbit_r), (0.0, -orbit_r)],
        [(0.0, -orbit_r), (orbit_r * k, -orbit_r), (orbit_r, -orbit_r * k), (orbit_r, 0.0)],
    ]
    cs, sn = float(np.cos(phase)), float(np.sin(phase))
    quarter = 20.0  # s, T / 4
    obstacles = [
        {
            "control_points": [
                [x * cs - y * sn, x * sn + y * cs, body_z, i * quarter + j * quarter / 3.0]
                for j, (x, y) in enumerate(arc)
            ],
            "radius": body_r,
            "color": "#e74c3c",
            "name": f"orbit_q{i + 1}",
        }
        for i, arc in enumerate(lap)
    ]
    return {
        "name": "loiter",
        "title": "Body Orbiting the Ground Station",
        "init_curve": {"mode": "straight"},
        "obstacles": obstacles,
        "start": [-100.0, corridor_y, corridor_z, 0.0],
        "end":   [ 100.0, corridor_y, corridor_z, 80.0],
        "stations": [[0.0, 0.0, 0.0]],
        # The corridor as a TUBE, enforced as HARD box rows outside the elastic
        # slack range -- the penalty cannot buy through them. The lateral pair
        # is what makes timing the only escape (docstring); the altitude pair
        # keeps the run above the body's reach and out of the shadow-free space
        # below the soil. Endpoints are pinned and exempt, and a band that
        # excludes an endpoint is refused loudly.
        "coord_bounds": [
            [-120.0, 120.0],
            [corridor_y - corridor_half, corridor_y + corridor_half],
            [60.0, 65.0],
        ],
        # A LENGTH, and this scene is 200 m across. See the docstring: the
        # stock 0.5 leaves the constrained run losing the link with its arrival
        # pinned against the horizon. Callers that solve a scenario must
        # forward this the way they forward `coord_bounds`.
        "trust_radius": 5.0,
        "T": 80.0,
    }

SCENARIO_MAP = {
    "original": (scenario_original, [(4, 4), (4, 8), (6, 8), (8, 4), (8, 8)]),
    # The only scenario whose obstacle tube is non-convex, which makes it the
    # only one that distinguishes the hull-projection plane from the tangent
    # plane. Everything else would pass with either.
    "curve":    (scenario_curve,    [(8, 4), (8, 8), (8, 16), (10, 8)]),
    # THE paper's demo scenario, and the only one with a station since
    # `station_fence` was removed. Its obstacle RETURNS, so its tube can be cut
    # twice by one clipping ball. Four coordinates: the viewer filters it out
    # rather than draw z as if it were time. Short list on purpose -- the
    # occlusion rows re-aim every iteration, so a run costs an order of
    # magnitude more than a keep-out-only one. Measured 2026-08-30; see
    # README sec. Measurements.
    "loiter":   (scenario_loiter,   [(8, 8), (8, 16)]),
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
}
