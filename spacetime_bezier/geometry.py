"""
Shared geometry helpers for space-time Bezier demos and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_docking.bezier import BezierCurve


@dataclass(frozen=True)
class MovingObstacle:
    """A circular obstacle moving at constant velocity in 2D, optionally active only during [t_start, t_end]."""

    pos0: np.ndarray
    velocity: np.ndarray
    radius: float
    color: str | None = None
    name: str | None = None
    t_start: float = -np.inf
    t_end: float = np.inf

    @classmethod
    def from_dict(cls, obstacle: dict) -> "MovingObstacle":
        return cls(
            pos0=np.asarray(obstacle["pos0"], dtype=float),
            velocity=np.asarray(obstacle["vel"], dtype=float),
            radius=float(obstacle["r"]),
            color=obstacle.get("color"),
            name=obstacle.get("name"),
            t_start=float(obstacle.get("t_start", -np.inf)),
            t_end=float(obstacle.get("t_end", np.inf)),
        )

    def position(self, t_value: float) -> np.ndarray:
        return self.pos0 + self.velocity * float(t_value)

    def is_active(self, t_value: float) -> bool:
        t_value = float(t_value)
        return self.t_start <= t_value <= self.t_end

    def to_dict(self) -> dict:
        data = {
            "pos0": self.pos0.tolist(),
            "vel": self.velocity.tolist(),
            "r": float(self.radius),
        }
        if self.color is not None:
            data["color"] = self.color
        if self.name is not None:
            data["name"] = self.name
        if np.isfinite(self.t_start):
            data["t_start"] = float(self.t_start)
        if np.isfinite(self.t_end):
            data["t_end"] = float(self.t_end)
        return data

    def tube_mesh(self, t_range, n_circ: int = 24, n_t: int = 30) -> list[np.ndarray]:
        """Generate a tube mesh in (x, y, t) space."""
        ts = np.linspace(float(t_range[0]), float(t_range[1]), n_t)
        theta = np.linspace(0.0, 2.0 * np.pi, n_circ)
        verts = []
        for t_value in ts:
            cx, cy = self.position(t_value)
            ring = np.column_stack(
                [
                    cx + self.radius * np.cos(theta),
                    cy + self.radius * np.sin(theta),
                    np.full_like(theta, t_value),
                ]
            )
            verts.append(ring)
        return verts


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
    for obs in obstacles:
        a = max(float(obs.get("t_start", -np.inf)), t_lo)
        b = min(float(obs.get("t_end", np.inf)), t_hi)
        if not (b >= a):
            continue
        radius = float(obs["r"])
        # Bound on how fast the gap between vehicle and obstacle can close.
        closing = float(np.linalg.norm(np.asarray(obs["vel"], dtype=float))) + vehicle_speed
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

    for obs in obstacles:
        pos0 = np.asarray(obs["pos0"], dtype=float)
        vel = np.asarray(obs["vel"], dtype=float)
        radius = float(obs["r"])
        t0 = float(obs.get("t_start", -np.inf))
        t1 = float(obs.get("t_end", np.inf))

        t_vals = pts[:, -1]
        active = (t_vals >= t0) & (t_vals <= t1)
        if not active.any():
            continue

        o_positions = pos0[None, :] + vel[None, :] * t_vals[active, None]
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

    for obs in obstacles:
        pos0 = np.asarray(obs["pos0"], dtype=float)
        vel = np.asarray(obs["vel"], dtype=float)
        radius = float(obs["r"])
        t0 = float(obs.get("t_start", -np.inf))
        t1 = float(obs.get("t_end", np.inf))
        active = (t_values >= t0) & (t_values <= t1)
        if not active.any():
            continue

        centers = pos0[None, :] + vel[None, :] * t_values[:, None]
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
    """Convert a legacy ``{pos0, vel, r, [t_start], [t_end]}`` obstacle into the
    wire-format BezierObstacle shape used by the sandbox: two control points in
    (x, y, t) plus ``radius``. Active time window becomes the t-coordinates of
    the two control points (clamped to [0, T]).
    """
    pos0 = np.asarray(obstacle["pos0"], dtype=float)
    vel = np.asarray(obstacle["vel"], dtype=float)
    t_start = float(obstacle.get("t_start", 0.0))
    t_end = float(obstacle.get("t_end", T))
    # A missing/infinite window means "full scenario duration".
    if not np.isfinite(t_start):
        t_start = 0.0
    if not np.isfinite(t_end):
        t_end = float(T)
    t_start = max(0.0, min(t_start, float(T)))
    t_end = max(t_start, min(t_end, float(T)))

    p0 = (pos0 + vel * t_start).tolist() + [t_start]
    p1 = (pos0 + vel * t_end).tolist() + [t_end]
    out = {
        "control_points": [p0, p1],
        "radius": float(obstacle["r"]),
    }
    if obstacle.get("name") is not None:
        out["name"] = obstacle["name"]
    if obstacle.get("color") is not None:
        out["color"] = obstacle["color"]
    return out


def moving_obstacle_from_bezier(bezier_obstacle: dict) -> dict:
    """Convert a BezierObstacle (wire format) back to the legacy straight-capsule
    dict consumed by ``optimize_spacetime`` / Rust. Only degree-1 obstacles
    (two control points) are supported here — the N=2+ path extends the Rust
    KOZ builder and will be handled separately.
    """
    cps = np.asarray(bezier_obstacle["control_points"], dtype=float)
    if cps.ndim != 2 or cps.shape[0] < 2:
        raise ValueError(f"BezierObstacle must have >=2 control points, got shape {cps.shape}")
    if cps.shape[0] > 2:
        raise NotImplementedError(
            f"BezierObstacle degree {cps.shape[0] - 1} not yet supported in the Rust KOZ builder"
        )

    p0, p1 = cps[0], cps[1]
    t0, t1 = float(p0[-1]), float(p1[-1])
    xy0, xy1 = p0[:-1], p1[:-1]
    if t1 > t0:
        vel = (xy1 - xy0) / (t1 - t0)
    else:
        vel = np.zeros_like(xy0)
    # ``pos0`` is the extrapolated position at t=0; only ``position(t)`` inside
    # [t_start, t_end] is meaningful since the KOZ is gated on that window.
    pos0 = xy0 - vel * t0

    out = {
        "pos0": pos0.tolist(),
        "vel": vel.tolist(),
        "r": float(bezier_obstacle["radius"]),
        "t_start": t0,
        "t_end": t1,
    }
    if bezier_obstacle.get("name") is not None:
        out["name"] = bezier_obstacle["name"]
    if bezier_obstacle.get("color") is not None:
        out["color"] = bezier_obstacle["color"]
    return out


def obstacle_array_bundle(obstacles: list[dict], spatial_dim: int) -> tuple[np.ndarray, ...]:
    """Convert dict obstacles into dense arrays for vectorized math and Rust calls."""
    if not obstacles:
        empty_pos = np.zeros((0, spatial_dim), dtype=float)
        empty_scalars = np.zeros(0, dtype=float)
        return empty_pos, empty_pos.copy(), empty_scalars, empty_scalars.copy(), empty_scalars.copy()

    pos0 = np.array([obs["pos0"] for obs in obstacles], dtype=float)
    vel = np.array([obs["vel"] for obs in obstacles], dtype=float)
    radius = np.array([obs["r"] for obs in obstacles], dtype=float)
    t_start = np.array([obs.get("t_start", -np.inf) for obs in obstacles], dtype=float)
    t_end = np.array([obs.get("t_end", np.inf) for obs in obstacles], dtype=float)
    if pos0.shape[1] != spatial_dim or vel.shape[1] != spatial_dim:
        raise ValueError(
            f"Obstacle arrays must have spatial_dim={spatial_dim}; got pos0={pos0.shape}, vel={vel.shape}"
        )
    return pos0, vel, radius, t_start, t_end
