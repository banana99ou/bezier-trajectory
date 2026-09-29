"""
`wall` starts by waiting (2026-09-29). The wall stands until t = 5; the straight
start reaches it at t ~ 2.2, inside it, and the wait start at t ~ 6.0, after it.
A scene-specific start, like `original`'s bow.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bezier_opt")

from spacetime_bezier.optimize import (  # noqa: E402
    optimize_scenario,
    optimize_spacetime_from_control_points,
)
from spacetime_bezier.scenarios import SCENARIO_MAP, scenario_wall  # noqa: E402

WALL_CONFIGS = SCENARIO_MAP["wall"][1]


@pytest.mark.parametrize("N,n_seg", WALL_CONFIGS)
def test_every_wall_setting_certifies_from_the_scene_start(N, n_seg):
    """FAILS IF any registered `wall` setting stops certifying from the scene's
    own start."""
    assert scenario_wall()["init_curve"]["mode"] == "wait"
    row = optimize_scenario(scenario_wall(), [(N, n_seg)], verbose=False)["results"][f"N{N}_seg{n_seg}"]
    assert row["figure_grade"], row["figure_grade_reasons"]


@pytest.mark.parametrize("N,n_seg", [(8, 2), (8, 3), (8, 4)])
def test_the_straight_start_is_why(N, n_seg):
    """The reason the scene has its own start. FAILS IF the straight start
    certifies these coarse settings -- then the scene-specific start is no
    longer needed and should be reconsidered."""
    sc = dict(scenario_wall(), init_curve={"mode": "straight"})
    row = optimize_scenario(sc, [(N, n_seg)], verbose=False)["results"][f"N{N}_seg{n_seg}"]
    assert not row["figure_grade"]


def test_a_certified_start_comes_back_certified():
    """Start at the answer. The solver handed a trajectory it already certified
    must return one it certifies. FAILS IF the solver can lose a certified
    start -- a defect no starting guess could excuse."""
    row = optimize_scenario(scenario_wall(), [(8, 16)], verbose=False)["results"]["N8_seg16"]
    assert row["figure_grade"]
    _, info = optimize_spacetime_from_control_points(
        np.asarray(row["control_points"]), scenario_wall()["obstacles"], n_seg=16, max_iter=200,
        tol=1e-6, scp_prox_weight=0.3, scp_trust_radius=0.5, elastic_weight=100.0, min_dt=0.1,
        verbose=False)
    assert bool(info["converged"]) and float(info["koz_violation_reference"]) <= 1e-6
    assert float(info["total_koz_slack_returned"]) <= 1e-6
