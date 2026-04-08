"""
Shared geometry helpers for space-time Bezier demos and optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital_docking.bezier import BezierCurve


@dataclass(frozen=True)
class MovingObstacle:
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
