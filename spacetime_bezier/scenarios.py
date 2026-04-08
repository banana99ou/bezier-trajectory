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
    """Create a row of overlapping circles between p1 and p2."""
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
        "init_curve": {
            "mode": "quadratic_bow",
            "bow": 3.2,
            "side": 1.0,
            "workspace_center": [5.0, 5.0],
        },
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
        t_end=8.0,
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


SCENARIO_MAP = {
    "original": (scenario_original, [(4, 8),  (6, 8),   (8, 8)]  ),
    "diverse":  (scenario_diverse,  [(8, 8),  (8, 16),  (10, 16)]),
    "wall":     (scenario_wall,     [(8, 16), (10, 16), (10, 24)]),
}
