"""
Space-time Bezier optimization package.
"""

from .geometry import MovingObstacle, bezier_curve, obstacle_array_bundle
from .io import load_outputs, main, save_outputs
from .optimize import (
    compute_min_clearance,
    optimize_scenario,
    optimize_scenarios,
    optimize_spacetime,
    optimize_spacetime_from_control_points,
)
from .scenarios import SCENARIO_MAP, scenario_diverse, scenario_original, scenario_wall

__all__ = [
    "MovingObstacle",
    "SCENARIO_MAP",
    "bezier_curve",
    "compute_min_clearance",
    "load_outputs",
    "main",
    "obstacle_array_bundle",
    "optimize_scenario",
    "optimize_scenarios",
    "optimize_spacetime",
    "optimize_spacetime_from_control_points",
    "save_outputs",
    "scenario_diverse",
    "scenario_original",
    "scenario_wall",
]
