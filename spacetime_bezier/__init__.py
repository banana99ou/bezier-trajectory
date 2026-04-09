"""
Space-time Bezier optimization package.
"""

from .geometry import MovingObstacle, bezier_curve, obstacle_array_bundle
from .io import load_outputs, main, save_outputs
from .optimize import (
    compute_min_clearance,
    create_spacetime_debug_stepper,
    create_spacetime_debug_stepper_from_control_points,
    optimize_scenario,
    optimize_scenarios,
    optimize_spacetime,
    optimize_spacetime_from_control_points,
)
from .debug_session import OptimizerDebugSession, SessionConfig
from .scenarios import SCENARIO_MAP, scenario_diverse, scenario_original, scenario_wall

__all__ = [
    "MovingObstacle",
    "SCENARIO_MAP",
    "OptimizerDebugSession",
    "SessionConfig",
    "bezier_curve",
    "compute_min_clearance",
    "create_spacetime_debug_stepper",
    "create_spacetime_debug_stepper_from_control_points",
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
