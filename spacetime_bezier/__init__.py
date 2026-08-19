"""
Space-time Bezier optimization package.
"""

from .geometry import MovingObstacle, bezier_curve, compute_min_clearance, obstacle_array_bundle
# `main` is deliberately NOT re-exported. It blocks in serve_forever(), and a
# blocking function on a package's public surface is how a test run ended up
# owning port 8765 for ten hours. Reach it through `python3 -m spacetime_bezier`.
from .io import load_outputs, save_outputs
from .optimize import (
    optimize_scenario,
    optimize_scenarios,
    optimize_spacetime,
    optimize_spacetime_from_control_points,
)
from .rust_debug_stepper import (
    create_spacetime_debug_stepper,
    create_spacetime_debug_stepper_from_control_points,
)
from .debug_session import OptimizerDebugSession, SessionConfig
from .scenarios import (
    SCENARIO_MAP,
    scenario_diverse,
    scenario_original,
    scenario_wall,
    scenario_wall3d,
)

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
    "obstacle_array_bundle",
    "optimize_scenario",
    "optimize_scenarios",
    "optimize_spacetime",
    "optimize_spacetime_from_control_points",
    "save_outputs",
    "scenario_diverse",
    "scenario_original",
    "scenario_wall",
    "scenario_wall3d",
]
