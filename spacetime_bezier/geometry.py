"""
Shared geometry helpers for space-time Bezier demos and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_docking.bezier import BezierCurve


@dataclass(frozen=True)
class MovingObstacle:
    """An obstacle whose motion is a polynomial in time, as lifted control points.

    ``control_points`` is ``(n_ctrl, spatial_dim + 1)``; the last coordinate is
    time, and it is AFFINE in the curve parameter (PAPER_1 statement 3 says the
    motion is a polynomial *in time*, and lifting such a motion gives affine time
    by definition). The active window is the first and last control point's time
    coordinate, so it is derived rather than carried alongside.

    Constant velocity is the degree-1 case and nothing more. It is still the
    convenient way to *write* an obstacle -- see ``normalize_obstacle`` -- but it
    is no longer what the solver assumes.
    """

    control_points: np.ndarray
    radius: float
    color: str | None = None
    name: str | None = None

    @classmethod
    def from_dict(cls, obstacle: dict) -> "MovingObstacle":
        norm = normalize_obstacle(obstacle)
        return cls(
            control_points=np.asarray(norm["control_points"], dtype=float),
            radius=float(norm["radius"]),
            color=norm.get("color"),
            name=norm.get("name"),
        )

    @property
    def t_start(self) -> float:
        return float(self.control_points[0, -1])

    @property
    def t_end(self) -> float:
        return float(self.control_points[-1, -1])

    @property
    def degree(self) -> int:
        return int(self.control_points.shape[0] - 1)

    def position(self, t_value: float) -> np.ndarray:
        """Spatial position at an instant. Extrapolates outside the window;
        callers gate on ``is_active`` exactly as they always did."""
        return obstacle_positions_at(self.control_points, np.array([float(t_value)]))[0]

    def is_active(self, t_value: float) -> bool:
        t_value = float(t_value)
        return self.t_start <= t_value <= self.t_end

    def to_dict(self) -> dict:
        data = {
            "control_points": self.control_points.tolist(),
            "radius": float(self.radius),
        }
        if self.color is not None:
            data["color"] = self.color
        if self.name is not None:
            data["name"] = self.name
        return data

    def tube_mesh(self, t_range, n_circ: int = 24, n_t: int = 30) -> list[np.ndarray]:
        """Generate a tube mesh in (x, y, t) space."""
        ts = np.linspace(float(t_range[0]), float(t_range[1]), n_t)
        theta = np.linspace(0.0, 2.0 * np.pi, n_circ)
        verts = []
        for t_value in ts:
            cx, cy = self.position(t_value)[:2]
            ring = np.column_stack(
                [
                    cx + self.radius * np.cos(theta),
                    cy + self.radius * np.sin(theta),
                    np.full_like(theta, t_value),
                ]
            )
            verts.append(ring)
        return verts



# ===========================================================================
# The obstacle, canonically: lifted Bezier control points
# ===========================================================================
#
# One representation reaches the solver -- ``control_points`` in (x, ..., t) with
# time affine in the parameter -- and everything else converts into it at the
# edge. ``{pos0, vel, r}`` survives as a way to WRITE a straight obstacle, not as
# a thing the solver knows about.


def normalize_obstacle(obstacle: dict, T: float | None = None) -> dict:
    """Return the canonical ``{control_points, radius, ...}`` form.

    Accepts either the canonical form or the legacy ``{pos0, vel, r,
    [t_start], [t_end]}`` writing shorthand. ``T`` supplies the scenario duration
    when a legacy obstacle has no explicit window; it is unused otherwise.
    """
    if "control_points" in obstacle:
        cps = np.asarray(obstacle["control_points"], dtype=float)
        if cps.ndim != 2 or cps.shape[0] < 2:
            raise ValueError(f"control_points must be (n_ctrl>=2, dim), got {cps.shape}")
        if cps[-1, -1] < cps[0, -1]:
            raise ValueError("obstacle control point times run backwards")
        out = {"control_points": cps, "radius": float(obstacle.get("radius", obstacle.get("r")))}
        for key in ("name", "color"):
            if obstacle.get(key) is not None:
                out[key] = obstacle[key]
        return out

    pos0 = np.asarray(obstacle["pos0"], dtype=float)
    vel = np.asarray(obstacle["vel"], dtype=float)
    t_start = float(obstacle.get("t_start", 0.0))
    t_end = float(obstacle.get("t_end", T if T is not None else np.inf))
    if not np.isfinite(t_start):
        t_start = 0.0
    if not np.isfinite(t_end):
        if T is None:
            raise ValueError(
                "a legacy obstacle with an unbounded window needs the scenario duration T; "
                "the active window is now intrinsic to the control points"
            )
        t_end = float(T)
    if T is not None:
        t_start = max(0.0, min(t_start, float(T)))
        t_end = max(t_start, min(t_end, float(T)))
    cps = np.array(
        [
            np.concatenate([pos0 + vel * t_start, [t_start]]),
            np.concatenate([pos0 + vel * t_end, [t_end]]),
        ],
        dtype=float,
    )
    out = {"control_points": cps, "radius": float(obstacle["r"])}
    for key in ("name", "color"):
        if obstacle.get(key) is not None:
            out[key] = obstacle[key]
    return out


def normalize_obstacles(obstacles: list[dict], T: float | None = None) -> list[dict]:
    return [normalize_obstacle(o, T) for o in obstacles or []]


def elevate_to_degree(control_points: np.ndarray, degree: int) -> np.ndarray:
    """Degree-elevate a Bezier to ``degree``. Exact -- the curve is unchanged.

    This is what lets obstacles of mixed degree share one rectangular array
    without any of them being resampled or approximated.
    """
    from orbital_docking.bezier import get_E_matrix

    cps = np.asarray(control_points, dtype=float)
    while cps.shape[0] - 1 < degree:
        cps = get_E_matrix(cps.shape[0] - 1) @ cps
    return cps


def obstacle_positions_at(control_points: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """Spatial positions ``pi_m(tau)`` at the given instants, shaped (len, spatial_dim).

    Time is affine in the parameter, so the inversion is division, not a root
    solve. This is the only obstacle quantity the certificate touches.
    """
    cps = np.asarray(control_points, dtype=float)
    t_values = np.asarray(t_values, dtype=float)
    t0, t1 = float(cps[0, -1]), float(cps[-1, -1])
    span = t1 - t0
    s = (t_values - t0) / span if span > 1e-15 else np.zeros_like(t_values)
    return _eval_at(cps, np.clip(s, 0.0, 1.0))[:, :-1]


def obstacle_speed_bound(control_points: np.ndarray) -> float:
    """Upper bound on ``||pi_m'(tau)||`` from the control polygon.

    ``N * max_l ||dG_l(spatial)|| / (t1 - t0)`` -- the standard Bezier derivative
    bound, divided by the affine time scale.
    """
    cps = np.asarray(control_points, dtype=float)
    span = float(cps[-1, -1] - cps[0, -1])
    if span <= 1e-15:
        return 0.0
    legs = np.diff(cps[:, :-1], axis=0)
    if legs.shape[0] == 0:
        return 0.0
    return float(cps.shape[0] - 1) * float(np.max(np.linalg.norm(legs, axis=1))) / span


def bezier_curve(control_points: np.ndarray, num_pts: int = 200) -> np.ndarray:
    """Evaluate a Bezier curve from control points shaped (N+1, dim)."""
    curve = BezierCurve(np.asarray(control_points, dtype=float))
    taus = np.linspace(0.0, 1.0, int(num_pts))
    return np.array([curve.point(tau) for tau in taus], dtype=float)


# ---------------------------------------------------------------------------
# Adaptive sampling for the two cross-checks
#
# Both `compute_min_clearance` and `compute_los_margin` sample the curve and
# report a minimum over the samples. That makes them SUFFICIENT-condition
# detectors: a sample that lands inside an obstacle proves penetration, but no
# sample landing inside proves nothing about the samples that were not taken.
# Two measured ways the uniform grid missed a real violation:
#
#   1. An obstacle whose active window is shorter than the time between
#      consecutive samples can fall entirely BETWEEN them and be skipped. At 3000
#      samples over a 10 s plan a window of +-0.001 s around t=5, with a
#      full-size r=1.0 body sitting exactly on the curve, reported +0.5 -- while
#      an 8-million-sample reference reported -1.0.
#
#   2. Uniform in the curve PARAMETER is not uniform in TIME. On a clustered
#      curve the time spacing between neighbouring samples varied by 72.7x, so
#      the effective resolution where the curve waits is far worse than the
#      nominal one. A small fast obstacle (r=0.05, |v|=50) crossing the path
#      reported +0.035 against a true -0.05.
#
# The repair is to inject extra samples. Time is monotone along the curve -- the
# monotonicity constraint guarantees it -- so a target TIME can be inverted to a
# curve parameter by bisection, and the injected samples can be placed to cover
# every obstacle's active window at a spacing fine enough that the obstacle
# cannot pass through unseen.
#
# THE RESIDUAL LIMITATION IS NOT REMOVED. This is still sampling, and the
# spacing rule below is a heuristic bound on the closing rate, not a proof. What
# carries the guarantee is the hull certificate, which is evaluated against the
# exact half-spaces rebuilt at the returned control points and does not share
# this weakness at all. These two functions are the INDEPENDENT cross-check on
# that certificate, which is the only reason they exist.
# ---------------------------------------------------------------------------

# Ceiling on injected samples per obstacle. Reached only by geometry an order of
# magnitude more extreme than any registered scenario (the r=0.05 / |v|=50 case
# above needs about 2e4). A run that hits it is under-sampled and the docstrings
# say so.
_MAX_INJECTED_PER_OBSTACLE = 50_000


def _bernstein(n: int, taus: np.ndarray) -> np.ndarray:
    """Bernstein basis of degree ``n``, shaped ``(len(taus), n+1)``."""
    from math import comb

    taus = np.asarray(taus, dtype=float)[:, None]
    k = np.arange(n + 1)[None, :]
    coef = np.array([comb(n, int(i)) for i in range(n + 1)], dtype=float)[None, :]
    return coef * (taus**k) * ((1.0 - taus) ** (n - k))


def _eval_at(P: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Curve points at arbitrary parameters, shaped ``(len(taus), dim)``."""
    P = np.asarray(P, dtype=float)
    return _bernstein(P.shape[0] - 1, taus) @ P


def _tau_at_time(P: np.ndarray, targets: np.ndarray, n_bisect: int = 60) -> np.ndarray:
    """Invert the (monotone) time coordinate: parameters where ``t(tau) == target``.

    Bisection rather than a root solve because monotonicity is all that is
    assumed, and 60 halvings of [0, 1] is machine precision.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0] - 1
    t_cp = P[:, -1]
    targets = np.asarray(targets, dtype=float)
    lo = np.zeros_like(targets)
    hi = np.ones_like(targets)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        t_mid = _bernstein(n, mid) @ t_cp
        below = t_mid < targets
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _max_leg_speed(P: np.ndarray) -> float:
    """Upper bound on the curve's physical speed, from the control polygon.

    ``d(spatial)/d(time)`` is a ratio of two Beziers of equal degree, and a
    Bezier is a convex combination of its own control points, so the largest leg
    slant of the control polygon bounds the speed of the whole curve -- the same
    argument the speed cap stands on. Returns 0.0 if any time gap is
    non-positive, i.e. if time is not monotone and the argument does not apply.
    """
    P = np.asarray(P, dtype=float)
    gaps = np.diff(P, axis=0)
    dt = gaps[:, -1]
    if gaps.shape[0] == 0 or not np.all(dt > 0.0):
        return 0.0
    return float(np.max(np.linalg.norm(gaps[:, :-1], axis=1) / dt))


def _sample_taus(P: np.ndarray, obstacles: list[dict], n_eval: int) -> np.ndarray:
    """Uniform parameters, plus samples covering every obstacle's active window.

    Falls back to the uniform grid alone when time is not monotone along the
    curve, because the inversion has no unique answer then. That case cannot
    arise from the solver -- the monotonicity rows forbid it -- but this function
    is also called on hand-built polygons.
    """
    base = np.linspace(0.0, 1.0, int(n_eval))
    P = np.asarray(P, dtype=float)
    if not obstacles or P.shape[0] < 2:
        return base

    t_cp = P[:, -1]
    if not np.all(np.diff(t_cp) > 0.0):
        return base
    t_lo, t_hi = float(t_cp[0]), float(t_cp[-1])
    vehicle_speed = _max_leg_speed(P)

    extra = [base]
    for obs in normalize_obstacles(obstacles):
        cps = obs["control_points"]
        a = max(float(cps[0, -1]), t_lo)
        b = min(float(cps[-1, -1]), t_hi)
        if not (b >= a):
            continue
        radius = float(obs["radius"])
        # Bound on how fast the gap between vehicle and obstacle can close. For a
        # curved obstacle this is the control-polygon derivative bound, which is
        # the constant-velocity speed exactly when the degree is 1.
        closing = obstacle_speed_bound(cps) + vehicle_speed
        step = radius / (2.0 * max(closing, 1e-12))
        if step <= 0.0:
            continue
        count = int(min(np.ceil((b - a) / step) + 1, _MAX_INJECTED_PER_OBSTACLE))
        # At least three: both window endpoints and the midpoint. Bisection can
        # land an endpoint a few ulps outside its own window, where the active
        # mask drops it; the midpoint never is.
        count = max(count, 3)
        extra.append(_tau_at_time(P, np.linspace(a, b, count)))

    taus = np.unique(np.concatenate(extra))
    # Drop injected parameters that land on top of a uniform one. They cost a
    # sample each and, being separated by a few ulps, can order by tau in the
    # opposite sense to their times -- which makes the returned time axis
    # non-monotone at the 1e-16 level for no benefit.
    if taus.size > 1:
        taus = taus[np.concatenate(([True], np.diff(taus) > 1e-15))]
    return taus


def compute_min_clearance(P, obstacles: list[dict], dim: int, n_eval: int = 1500) -> float:
    """Minimum clearance of a Bezier curve to all obstacles, by sampling.

    ``n_eval`` sets the uniform sample count; additional samples are injected to
    cover each obstacle's active window at a spacing fine enough that a body of
    its radius cannot cross the curve between two of them. See the module
    comment above ``_MAX_INJECTED_PER_OBSTACLE``.

    **This is the cross-check, not the guarantee.** It is a sufficient condition
    for penetration and never a proof of clearance: a negative answer proves the
    curve enters an obstacle, a positive answer says only that no sample did. The
    guarantee is the control-point hull certificate, which is evaluated against
    the exact half-spaces and has no sampling weakness.
    """
    if not obstacles:
        return float("inf")

    P = np.asarray(P, dtype=float)
    pts = _eval_at(P, _sample_taus(P, obstacles, n_eval))
    spatial_dim = dim - 1
    worst = np.inf

    for obs in normalize_obstacles(obstacles):
        cps = obs["control_points"]
        radius = float(obs["radius"])
        t0, t1 = float(cps[0, -1]), float(cps[-1, -1])

        t_vals = pts[:, -1]
        active = (t_vals >= t0) & (t_vals <= t1)
        if not active.any():
            continue

        o_positions = obstacle_positions_at(cps, t_vals[active])
        dists = np.linalg.norm(pts[active, :spatial_dim] - o_positions, axis=1) - radius
        worst = min(worst, float(dists.min()))

    return float(worst)


def compute_los_margin(P, station, obstacles: list[dict], dim: int, n_eval: int = 1500):
    """Line-of-sight margin along the curve, sampled against the TRUE geometry.

    This is the independent leg of the pair. It does **not** call the Rust
    occlusion builder, does not know what a shadow plane is, and never sees the
    convex outer approximation the solver optimized against. It samples the
    returned curve, reconstructs each obstacle's own position at the sample's own
    time, and measures the sight segment directly. If it and the solver's
    certificate ever disagree, one of them is wrong and the disagreement is the
    finding -- which is the only reason to have two.

    At each sample the margin is

        distance( segment[station, vehicle position], obstacle centre at t )  -  r

    which is positive exactly when the sight line clears the body, because a
    sphere blocks the segment precisely when the segment passes within `r` of its
    centre. The reported margin is the minimum over the obstacles active at that
    time; an inactive obstacle blocks nothing, and a sample where nothing is
    active reports ``+inf``.

    Returns ``(t_values, margins)``, both shaped ``(n_samples,)`` and ordered by
    increasing curve parameter -- so the time axis is non-decreasing too, up to
    floating-point noise of order 1e-16. Times come back alongside because the
    margin-versus-time panel is the figure that proves the claim, and a margin
    array with no time axis cannot be plotted against one.

    ``n_samples >= n_eval``: the uniform grid is augmented with samples covering
    each obstacle's active window, for the two reasons in the module comment
    above ``_MAX_INJECTED_PER_OBSTACLE`` -- a short window can fall entirely
    between two uniform samples, and uniform in the curve parameter is not
    uniform in time. **The residual limitation stands**: this is sampling, so a
    negative margin proves the sight line was lost and a positive one does not
    prove it was kept. The occlusion certificate is the guarantee.
    """
    P = np.asarray(P, dtype=float)
    pts = _eval_at(P, _sample_taus(P, obstacles, int(n_eval)))
    spatial_dim = dim - 1
    t_values = pts[:, -1]
    positions = pts[:, :spatial_dim]
    s = np.asarray(station, dtype=float).reshape(-1)
    if s.shape[0] != spatial_dim:
        raise ValueError(f"Station must have {spatial_dim} spatial coordinates, got {s.shape[0]}")

    margins = np.full(t_values.shape, np.inf, dtype=float)
    if not obstacles:
        return t_values, margins

    seg = positions - s[None, :]              # station -> vehicle, per sample
    seg_sq = np.einsum("ij,ij->i", seg, seg)
    safe_sq = np.where(seg_sq > 1e-24, seg_sq, 1.0)

    for obs in normalize_obstacles(obstacles):
        cps = obs["control_points"]
        radius = float(obs["radius"])
        t0, t1 = float(cps[0, -1]), float(cps[-1, -1])
        active = (t_values >= t0) & (t_values <= t1)
        if not active.any():
            continue

        centers = obstacle_positions_at(cps, t_values)
        # Foot of the perpendicular from the obstacle centre onto the sight
        # SEGMENT -- clamped, because the body only blocks what lies between the
        # station and the vehicle, not what lies behind either of them.
        tau = np.einsum("ij,ij->i", centers - s[None, :], seg) / safe_sq
        tau = np.clip(np.where(seg_sq > 1e-24, tau, 0.0), 0.0, 1.0)
        foot = s[None, :] + tau[:, None] * seg
        dist = np.linalg.norm(foot - centers, axis=1) - radius
        margins = np.where(active, np.minimum(margins, dist), margins)

    return t_values, margins


def bezier_obstacle_from_moving(obstacle: dict, T: float) -> dict:
    """Legacy alias for :func:`normalize_obstacle`, kept for the wire format.

    Returns the canonical form with ``control_points`` as a plain list, which is
    what the JSON boundary wants.
    """
    out = dict(normalize_obstacle(obstacle, T))
    out["control_points"] = np.asarray(out["control_points"], dtype=float).tolist()
    return out


def moving_obstacle_from_bezier(bezier_obstacle: dict) -> dict:
    """Identity, up to normalising the shape.

    This used to raise ``NotImplementedError`` above degree 1, because the Rust
    KOZ builder only understood a straight capsule. It understands the general
    curve now, so there is nothing to convert down to and nothing to refuse.
    """
    out = dict(normalize_obstacle(bezier_obstacle))
    out["control_points"] = np.asarray(out["control_points"], dtype=float).tolist()
    return out


def obstacle_array_bundle(obstacles: list[dict], spatial_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense arrays for the Rust call: ``(ctrl, radii)``.

    ``ctrl`` is ``(n_obs, n_ctrl, spatial_dim + 1)``. Obstacles of differing
    degree are degree-elevated to the highest present, which is exact, so the
    array is rectangular without anything being resampled.

    There is no ``t_start`` / ``t_end`` pair any more: the active window is the
    first and last control point's time coordinate. One source of truth, so the
    two cannot disagree.
    """
    dim = spatial_dim + 1
    if not obstacles:
        return np.zeros((0, 2, dim), dtype=float), np.zeros(0, dtype=float)

    norm = normalize_obstacles(obstacles)
    degree = max(o["control_points"].shape[0] - 1 for o in norm)
    ctrl = np.empty((len(norm), degree + 1, dim), dtype=float)
    radii = np.empty(len(norm), dtype=float)
    for i, obs in enumerate(norm):
        cps = np.asarray(obs["control_points"], dtype=float)
        if cps.shape[1] != dim:
            raise ValueError(
                f"obstacle {i}: control points must have {dim} coordinates "
                f"(spatial_dim + 1), got {cps.shape[1]}"
            )
        ctrl[i] = elevate_to_degree(cps, degree)
        radii[i] = float(obs["radius"])
    return ctrl, radii

