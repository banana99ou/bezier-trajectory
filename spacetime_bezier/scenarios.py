"""
Scenario registry for the space-time Bezier demos.
"""

from __future__ import annotations

import numpy as np


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


def scenario_wall() -> dict:
    """Wall that disappears early enough for the curve to wait and pass through."""
    wall_obs = make_wall(
        p1=[2.5, 0.0],
        p2=[2.5, 10.0],
        thickness=0.5,
        spacing=0.8,
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


def scenario_wall3d() -> dict:
    """Item B11 -- three spatial coordinates plus time.

    `make_wall` is already dimension-agnostic, so this is the `wall` idea lifted:
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
        "name": "wall3d",
        "title": "Moving Fence, 3 Spatial Dimensions",
        "init_curve": {"mode": "straight"},
        "obstacles": fence + movers,
        "start": [0.5, 5.0, 0.5, 0.0],
        "end": [9.5, 5.0, 0.5, 10.0],
        "T": 10.0,
    }


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
        # The only scenario carrying this key. Its absence everywhere else is
        # what keeps every other scenario's problem bit-identical to pre-B12.
        "stations": [[5.0, -2.0, 0.3]],
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
}


def scenario_elastic_weight(name: str) -> float:
    """Elastic weight for a named scenario, or the shared default."""
    from .optimize import DEFAULT_ELASTIC_WEIGHT

    return float(SCENARIO_ELASTIC_WEIGHT.get(name, DEFAULT_ELASTIC_WEIGHT))


SCENARIO_MAP = {
    "original": (scenario_original, [(4, 4), (4, 8), (6, 8), (8, 4), (8, 8)]),
    "diverse":  (scenario_diverse,  [(8, 4), (8, 8), (8, 16), (10, 4), (10, 16)]),
    "wall":     (scenario_wall,     [(8, 2), (8, 3), (8, 4), (8, 16), (10, 16), (10, 24)]),
    # Four coordinates, not three. Everything downstream reads the dimension off
    # the array shape -- except the viewer, which plots columns 0, 1, 2 as
    # (x, y, t) and would render this one's z as if it were time. `viewer.py`
    # therefore filters it out of the catalog rather than drawing a false
    # picture; see the comment there.
    "wall3d":   (scenario_wall3d,   [(8, 2), (8, 4), (8, 8), (10, 8)]),
    # Also four coordinates, so the viewer filters it out for the same reason.
    # Short list on purpose: the occlusion rows re-aim every iteration, so the
    # run is an order of magnitude longer than a keep-out-only one, and only
    # these two configurations were measured to converge with the occlusion
    # certificate at zero.
    "station_fence": (scenario_station_fence, [(8, 8), (8, 16)]),
}
