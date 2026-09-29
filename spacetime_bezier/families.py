"""
Parametric scenario families for the scaling experiment.

``scenarios.py`` holds hand-authored scenes, each a fixed picture. A *family* is
one picture with one knob, so a curve can be drawn along that knob and nothing
else moves. Four knobs, one per axis the decomposition-free claim is about:

* ``traffic``   -- obstacle COUNT, 1 to 20 crossing aircraft on filed (known)
                   straight trajectories. Osburn's Table I axis, with the half
                   they cannot do: every obstacle moves.
* ``traffic_curved`` -- the SAME seeds and schedule, every aircraft turning
                   through its crossing instead of flying straight. The
                   reliability arm (decided 2026-09-29, SPEC.md sec. 4.7): a
                   straight track lifts to a convex capsule, so the straight
                   arm never exercises the clip construction; the curved arm
                   does, and the difference between the two arms is what the
                   non-convex case costs.
* ``dimension`` -- 2+t against 3+t, same conflict schedule.
* ``horizon``   -- the same spatial conflict picture on a 2 s clock and on a
                   20 s clock. The row count must not move; a time-gridded
                   method's cost must. (The SPOT argument.)
* ``motion``    -- one obstacle, straight / arc / circle, so the certificate
                   is measured against a convex tube, a bent one, and one that
                   returns.

**Every member returns the same dict a ``SCENARIO_MAP`` entry does**, canonical
obstacles included, so ``optimize_scenario``, the frontend and the figure tools
take it unchanged. The extra ``family`` key is provenance for the bench and is
ignored downstream.

**The scene is the nondimensional 10-unit box of ``original`` / ``diverse``**,
horizon 10 by default. The stock solver knobs (trust radius, ``min_dt``, the
coordinate box) were tuned there; ``loiter`` is the documented cost of leaving
it. Mission units -- a 2 km corridor, 40 m/s aircraft, 150 m separation -- are a
scale factor applied when a figure is drawn, never here.

**The count axis is real by construction, not by luck.** Every traffic aircraft
is placed so that it is AT the nominal plan's position AT the nominal plan's
time, i.e. the straight-line initial guess conflicts with every one of them.
An obstacle the initial guess never meets adds no row and would make the curve
measure nothing; ``tests/unit/test_families.py`` asserts the conflict for
every member.
"""

from __future__ import annotations

import numpy as np

from .scenarios import _canonical

# The 10-unit scene. Start and end sit on the corridor axis y = 5 (z = 5 in 3+t).
SCENE = 10.0
CORRIDOR_Y = 5.0
X_START, X_END = 0.5, 9.5
DEFAULT_T = 10.0
DEFAULT_CONFIG = (8, 8)  # (degree, segments) held fixed along every axis

_PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#9b59b6", "#e67e22",
            "#1abc9c", "#f39c12", "#34495e", "#c0392b", "#16a085"]


def _endpoints(dim_spatial: int, T: float) -> tuple[list[float], list[float]]:
    if dim_spatial == 2:
        return [X_START, CORRIDOR_Y, 0.0], [X_END, CORRIDOR_Y, T]
    if dim_spatial == 3:
        return [X_START, CORRIDOR_Y, CORRIDOR_Y, 0.0], [X_END, CORRIDOR_Y, CORRIDOR_Y, T]
    raise ValueError(f"dim_spatial must be 2 or 3, got {dim_spatial}")


def _nominal(start, end, s: float) -> np.ndarray:
    """Spatial position of the straight, uniform-speed plan at fraction ``s``."""
    a = np.asarray(start[:-1], dtype=float)
    b = np.asarray(end[:-1], dtype=float)
    return (1.0 - s) * a + s * b


def traffic_crossings(n_obs: int, dim_spatial: int = 2, T: float = DEFAULT_T,
                      seed: int = 0, curved: bool = False) -> list[dict]:
    """``n_obs`` aircraft on known trajectories, each crossing the nominal
    plan at a stratified fraction of the transit.

    Stratified, not uniform, so 20 aircraft are 20 separate conflicts spread
    along the corridor rather than a clump. The crossing angle, speed ratio
    and radius are drawn from the seed; the crossing POINT and TIME are not
    -- they are pinned to the nominal plan, which is what makes the conflict
    certain.

    ``curved=True`` flies every aircraft on a turn instead of a straight line,
    keeping everything above: the same crossing point, crossing time, velocity
    AT the crossing, radius and speed. The turn is a quadratic in lifted space,

        x(u) = x_cross + V (u - u_c) + W (u - u_c)^2,   t(u) = T u,

    with ``V = vel * T``, ``u_c = t_cross / T`` and ``W`` perpendicular to
    ``V`` with ``|W| = kappa |V|``, kappa drawn from [0.3, 0.6]. The heading
    therefore turns by ``atan(2 kappa u_c) + atan(2 kappa (1 - u_c))`` over the
    window, about 33 to 62 degrees for kappa at its bounds and u_c = 0.5.
    The curvature comes from its OWN generator seeded from ``seed``, so the
    straight arm's draws -- and every straight member ever generated -- are
    unchanged by the curved option existing. ``V`` and ``W`` do not depend on
    ``T`` (velocity scales as ``1/T``), so the horizon property holds for the
    curved arm too.
    """
    if n_obs < 1:
        raise ValueError("a traffic family member has at least one aircraft")
    rng = np.random.default_rng(seed)
    # A separate stream, so drawing the turn never shifts the straight arm's
    # draws. Drawn for every aircraft whether used or not, in aircraft order.
    rng_turn = np.random.default_rng([seed, 1])
    start, end = _endpoints(dim_spatial, T)
    axis = np.asarray(end[:-1], dtype=float) - np.asarray(start[:-1], dtype=float)
    v_nom = float(np.linalg.norm(axis)) / T
    axis_hat = axis / np.linalg.norm(axis)

    out = []
    for i in range(n_obs):
        # Fraction of the transit at which this aircraft crosses: stratified
        # over (0.15, 0.85) with within-bin jitter.
        lo, hi = 0.15, 0.85
        s = lo + (hi - lo) * (i + rng.uniform(0.2, 0.8)) / n_obs
        t_cross = s * T
        x_cross = _nominal(start, end, s)

        # Crossing direction: 30..150 deg off the corridor axis, alternating
        # side, with a modest elevation component in 3+t.
        # Every draw happens in the same order whatever the dimension, so the
        # 2+t and 3+t members of one seed share their schedule: same crossing
        # fractions, angles, speeds and radii. The elevation draw is simply
        # unused in 2+t.
        theta = np.deg2rad(rng.uniform(30.0, 150.0)) * (1 if i % 2 == 0 else -1)
        phi = np.deg2rad(rng.uniform(-30.0, 30.0))
        speed = v_nom * rng.uniform(0.6, 1.4)
        radius = float(rng.uniform(0.35, 0.6))
        if dim_spatial == 2:
            perp = np.array([-axis_hat[1], axis_hat[0]])
            direction = np.cos(theta) * axis_hat + np.sin(theta) * perp
        else:
            perp = np.array([-axis_hat[1], axis_hat[0], 0.0])
            up = np.array([0.0, 0.0, 1.0])
            direction = (np.cos(theta) * axis_hat + np.sin(theta) * perp) * np.cos(phi) + up * np.sin(phi)
        direction /= np.linalg.norm(direction)

        kappa = float(rng_turn.uniform(0.3, 0.6))
        turn_sign = 1.0 if rng_turn.uniform() < 0.5 else -1.0

        vel = speed * direction
        common = {
            "color": _PALETTE[i % len(_PALETTE)],
            "name": f"AC{i + 1:02d}",
            "cross_s": float(s),
        }
        if not curved:
            pos0 = x_cross - vel * t_cross
            out.append({"pos0": pos0.tolist(), "vel": vel.tolist(), "r": radius, **common})
            continue

        # Lateral direction: in 2+t the in-plane normal; in 3+t the horizontal
        # normal to the heading (the elevation is at most 30 degrees, so the
        # cross product with vertical is never degenerate).
        if dim_spatial == 2:
            lateral = np.array([-direction[1], direction[0]])
        else:
            lateral = np.cross(direction, np.array([0.0, 0.0, 1.0]))
            lateral /= np.linalg.norm(lateral)
        V = vel * T
        W = turn_sign * kappa * float(np.linalg.norm(V)) * lateral
        u_c = t_cross / T
        # Power basis a + b u + c u^2, then the quadratic Bernstein points.
        a = x_cross - V * u_c + W * u_c ** 2
        b = V - 2.0 * W * u_c
        c = W
        spatial = [a, a + 0.5 * b, a + b + c]
        cps = [np.concatenate([q, [tq]]).tolist() for q, tq in zip(spatial, (0.0, 0.5 * T, T))]
        out.append({"control_points": cps, "radius": radius, "kappa": kappa, **common})
    return out


def _scenario(name: str, title: str, obstacles: list[dict], dim_spatial: int, T: float,
              family: dict) -> dict:
    start, end = _endpoints(dim_spatial, T)
    return {
        "name": name,
        "title": title,
        "init_curve": {"mode": "straight"},
        "obstacles": obstacles,
        "start": start,
        "end": end,
        "T": float(T),
        "family": family,
    }


@_canonical
def family_traffic(n_obs: int, dim_spatial: int = 2, T: float = DEFAULT_T, seed: int = 0,
                   curved: bool = False) -> dict:
    """Scheduled-traffic deconfliction: thread between ``n_obs`` aircraft whose
    4D trajectories are filed and therefore known."""
    obs = traffic_crossings(n_obs, dim_spatial, T, seed, curved=curved)
    return _scenario(
        f"traffic_n{n_obs:02d}" + ("" if dim_spatial == 2 else f"_{dim_spatial}d")
        + ("" if T == DEFAULT_T else f"_T{T:g}") + ("_curved" if curved else ""),
        f"Scheduled Traffic, {n_obs} {'Turning ' if curved else ''}Aircraft",
        obs, dim_spatial, T,
        {"axis": "obstacle_count", "value": n_obs, "dim_spatial": dim_spatial, "T": T, "seed": seed,
         "motion": "curved" if curved else "straight"},
    )


def family_traffic_curved(n_obs: int, dim_spatial: int = 2, T: float = DEFAULT_T,
                          seed: int = 0) -> dict:
    """The curved arm: ``family_traffic`` with every aircraft turning."""
    return family_traffic(n_obs, dim_spatial=dim_spatial, T=T, seed=seed, curved=True)


def family_dimension(dim_spatial: int, n_obs: int = 8, seed: int = 0) -> dict:
    """The same eight-aircraft conflict schedule in 2+t and in 3+t."""
    sc = family_traffic(n_obs, dim_spatial=dim_spatial, T=DEFAULT_T, seed=seed)
    sc["name"] = f"dim_{dim_spatial}d"
    sc["title"] = f"{dim_spatial} Spatial Dimensions + Time"
    sc["family"] = {"axis": "dimension", "value": dim_spatial, "n_obs": n_obs, "seed": seed}
    return sc


def family_horizon(T: float, n_obs: int = 8, seed: int = 0) -> dict:
    """The same spatial conflict picture on a different clock.

    Crossing points, angles and radii are identical across ``T`` (same seed);
    every velocity scales as ``1/T`` so each aircraft still meets the nominal
    plan where and when the plan is there. Rows should not move with ``T``.
    """
    sc = family_traffic(n_obs, dim_spatial=2, T=T, seed=seed)
    sc["name"] = f"horizon_T{T:g}"
    sc["title"] = f"Horizon {T:g} s"
    sc["family"] = {"axis": "horizon", "value": T, "n_obs": n_obs, "seed": seed}
    return sc


@_canonical
def family_motion(kind: str, T: float = DEFAULT_T) -> dict:
    """One obstacle crossing the corridor at mid-transit, on a straight,
    an arc, or a returning circle. Radius 0.9 throughout."""
    start, end = _endpoints(2, T)
    mid = _nominal(start, end, 0.5)
    r = 0.9
    if kind == "straight":
        vel = np.array([0.0, -1.2]) * (DEFAULT_T / T)
        obstacles = [{"pos0": (mid - vel * (T / 2)).tolist(), "vel": vel.tolist(),
                      "r": r, "color": "#e74c3c", "name": "straight"}]
    elif kind == "arc":
        # Out across the corridor and back, like `curve`, centred on mid.
        # The x offsets are +3.0 / -1.0 so the cubic's midpoint is EXACTLY the
        # plan's mid-transit position (2*(+3) + 6*(-1) = 0 over the Bernstein
        # weights 1,3,3,1), which is what makes the arc a conflict rather than
        # a decoration; the time column is uniform, so u = 0.5 IS t = T/2.
        obstacles = [{
            "control_points": [
                [mid[0] + 3.0, mid[1] + 4.0, 0.0],
                [mid[0] - 1.0, mid[1] + 1.5, T / 3],
                [mid[0] - 1.0, mid[1] - 1.5, 2 * T / 3],
                [mid[0] + 3.0, mid[1] - 4.0, T],
            ],
            "radius": r, "color": "#e74c3c", "name": "arc",
        }]
    elif kind == "circle":
        # One lap over the horizon as a chain of four cubic quarter arcs (the
        # `loiter` construction), centred just off the corridor so the body
        # sweeps across it once. The offset is NEGATIVE: on the +0.6*R side the
        # lap's two corridor crossings both trail the plan and the nearest
        # approach is +0.683 (a decoration, no row); on the -0.6*R side the
        # first crossing lands on the plan at t ~ 3.5.
        R = 2.5
        k = 0.5522847498
        cx, cy = mid[0], mid[1] - R * 0.6
        lap = [
            [(R, 0.0), (R, R * k), (R * k, R), (0.0, R)],
            [(0.0, R), (-R * k, R), (-R, R * k), (-R, 0.0)],
            [(-R, 0.0), (-R, -R * k), (-R * k, -R), (0.0, -R)],
            [(0.0, -R), (R * k, -R), (R, -R * k), (R, 0.0)],
        ]
        q = T / 4
        obstacles = [{
            "control_points": [[cx + x, cy + y, i * q + j * q / 3.0] for j, (x, y) in enumerate(arc)],
            "radius": r, "color": "#e74c3c", "name": f"circle_q{i + 1}",
        } for i, arc in enumerate(lap)]
    else:
        raise ValueError(f"unknown motion kind {kind!r}; want straight, arc or circle")
    return _scenario(f"motion_{kind}", f"Obstacle Motion: {kind}", obstacles, 2, T,
                     {"axis": "motion", "value": kind, "T": T})


# name -> (generator, knob values, (degree, segments))
FAMILY_MAP = {
    "traffic":   (family_traffic,   [1, 2, 4, 8, 12, 16, 20], DEFAULT_CONFIG),
    "traffic_curved": (family_traffic_curved, [1, 2, 4, 8, 12, 16, 20], DEFAULT_CONFIG),
    "dimension": (family_dimension, [2, 3],                   DEFAULT_CONFIG),
    "horizon":   (family_horizon,   [2.0, 5.0, 10.0, 20.0],   DEFAULT_CONFIG),
    "motion":    (family_motion,    ["straight", "arc", "circle"], DEFAULT_CONFIG),
}


def family_members(name: str):
    """Yield ``(value, scenario)`` along one family's knob."""
    gen, values, _ = FAMILY_MAP[name]
    for v in values:
        yield v, gen(v)
