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


def compute_min_clearance(P, obstacles: list[dict], dim: int, n_eval: int = 1500) -> float:
    """Evaluate the minimum clearance of a Bezier curve to all obstacles."""
    if not obstacles:
        return float("inf")

    pts = bezier_curve(np.asarray(P, dtype=float), num_pts=n_eval)
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
