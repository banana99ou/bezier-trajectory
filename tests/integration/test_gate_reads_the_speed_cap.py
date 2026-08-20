"""The feasibility gate must be able to fail on the speed cap.

`speed_cap_violation` was written into every result row and read by nothing:
`figure_grade_failures` never looked at it. A cap that the returned curve broke
would have been recorded next to `figure_grade: True`. The cap is a HARD
constraint (formulation decision 4) -- it has no slack variable and the elastic
penalty may not buy its way out of it -- so a violation is a broken constraint,
not a priced relaxation, and it belongs in the gate.
"""

import numpy as np
import pytest

from spacetime_bezier.optimize import (
    figure_grade_failures,
    is_figure_grade,
    optimize_scenario,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import scenario_original

pytest.importorskip("bezier_opt")


def _passing_row() -> dict:
    return {
        "converged": True,
        "stop_label": "stationary",
        "certificate_violation": 0.0,
        "occlusion_violation": 0.0,
        "occlusion_planes_dropped": 0.0,
        "min_clearance": 0.62,
        "total_slack": 1e-14,
        "speed_cap_violation": 0.0,
    }


def test_a_speed_cap_violation_alone_sinks_the_gate():
    """FAILS IF `speed_cap_violation` is decorative.

    Before this condition existed, flipping this one field on an otherwise
    perfect row left the gate reporting figure_grade True. The synthetic value is
    1e9 so no tolerance question is involved: either the field is read or it is
    not.
    """
    row = _passing_row()
    assert figure_grade_failures(row) == []

    row["speed_cap_violation"] = 1e9
    reasons = figure_grade_failures(row)
    assert len(reasons) == 1, f"expected exactly one failure, got {reasons}"
    assert "speed cap" in reasons[0]
    assert is_figure_grade(row) is False


def test_a_capless_run_is_unaffected():
    """No cones means nothing to violate, so the key defaults to 0.0.

    FAILS IF the new condition defaults to NaN: every scenario in the table runs
    without a speed cap and would stop being figure-grade.
    """
    row = _passing_row()
    del row["speed_cap_violation"]
    assert figure_grade_failures(row) == []

    out = optimize_scenario(
        scenario_original(), [(8, 4)], elastic_weight=100.0, verbose=False
    )
    result = out["results"]["N8_seg4"]
    assert result["figure_grade"] is True, result["figure_grade_reasons"]


def test_a_capless_run_reports_positive_zero_not_negative_zero():
    """Rust's `Sum for f64` folds from -0.0, so an EMPTY cone list summed to -0.0.

    Harmless to the comparison, and wrong in every table and log line it reached:
    "-0.0" in a violation column reads as a measurement, not as an absence.

    FAILS IF the `+ 0.0` normalization at export is removed -- measured: before
    it, `original` N8_seg4 with v_max=None reported `-0.0`.
    """
    scenario = scenario_original()
    _, info = optimize_spacetime(
        N=8,
        dim=3,
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=scenario["obstacles"],
        n_seg=4,
        max_iter=200,
        tol=1e-6,
        elastic_weight=100.0,
        init_curve=scenario["init_curve"],
        v_max=None,
        verbose=False,
    )
    value = float(info["speed_cap_violation"])
    assert value == 0.0
    assert np.copysign(1.0, value) > 0.0, "speed_cap_violation came back as -0.0"


def test_a_real_capped_run_still_passes_the_new_condition():
    """The cap binds and is satisfied, so the condition must be silent.

    FAILS IF the tolerance is tighter than the cone solve can deliver: `original`
    at v_max=2 rides the cap on at least one leg, so its violation is the
    interior-point residue and not identically zero.
    """
    scenario = scenario_original()
    _, info = optimize_spacetime(
        N=8,
        dim=3,
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=scenario["obstacles"],
        n_seg=4,
        max_iter=200,
        tol=1e-6,
        elastic_weight=100.0,
        init_curve=scenario["init_curve"],
        v_max=2.0,
        verbose=False,
    )
    row = _passing_row()
    row["speed_cap_violation"] = float(info["speed_cap_violation"])
    assert figure_grade_failures(row) == []
