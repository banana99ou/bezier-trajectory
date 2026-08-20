"""Item B7: a run standing on elastic slack cannot produce a figure.

Every subproblem in this solver is solved elastically -- slack variables on the
keep-out rows, priced by the exact-penalty weight -- so "the loop converged" and
"the loop obeyed the obstacles" are different statements. The gate makes the
difference explicit and refuses to let one stand in for the other.

The four conditions are tested for INDEPENDENCE, not merely jointly: each one is
flipped on its own against an otherwise-passing row, and each alone must sink the
gate. A gate whose conditions cannot be shown to bind separately is a gate that
might be passing on one condition and reporting four.
"""

import numpy as np
import pytest

from spacetime_bezier.optimize import (
    FIGURE_GRADE_CERTIFICATE_TOL,
    FIGURE_GRADE_SLACK_TOL,
    figure_grade_failures,
    is_figure_grade,
    optimize_scenario,
)
from spacetime_bezier.scenarios import scenario_original, scenario_wall

pytest.importorskip("bezier_opt")


def _passing_row() -> dict:
    """A row that satisfies every gate condition, as the baseline to break."""
    return {
        "converged": True,
        "stop_label": "stationary",
        "certificate_violation": 0.0,
        "min_clearance": 0.62,
        "total_slack": 1e-14,
    }


# ---------------------------------------------------------------------------
# The gate predicate itself
# ---------------------------------------------------------------------------


def test_a_row_meeting_every_condition_is_figure_grade():
    assert figure_grade_failures(_passing_row()) == []
    assert is_figure_grade(_passing_row()) is True


@pytest.mark.parametrize(
    "field,bad_value,expected_fragment",
    [
        ("converged", False, "not converged"),
        ("certificate_violation", FIGURE_GRADE_CERTIFICATE_TOL * 10.0, "certificate"),
        ("min_clearance", -1e-9, "penetrates"),
        ("total_slack", FIGURE_GRADE_SLACK_TOL * 10.0, "slack"),
    ],
)
def test_each_condition_alone_sinks_the_gate(field, bad_value, expected_fragment):
    """FAILS IF any condition is decorative -- e.g. if `total_slack` were merely
    recorded rather than gated on, flipping it alone would leave the row passing.
    """
    row = _passing_row()
    row[field] = bad_value
    reasons = figure_grade_failures(row)
    assert len(reasons) == 1, f"expected exactly one failure, got {reasons}"
    assert expected_fragment in reasons[0]
    assert is_figure_grade(row) is False


@pytest.mark.parametrize("field", ["converged", "certificate_violation", "min_clearance", "total_slack"])
def test_a_missing_field_is_not_figure_grade(field):
    """No evidence is not the same as evidence of feasibility.

    A row that never reported a quantity (NaN, or the key absent) must fail the
    gate rather than default to passing. NaN fails every comparison, which is why
    the predicate is written as `not (x <= tol)` and never as `x > tol`.
    """
    row = _passing_row()
    del row[field]
    assert is_figure_grade(row) is False


# ---------------------------------------------------------------------------
# The gate on real runs
# ---------------------------------------------------------------------------


def test_every_result_row_carries_total_slack_and_a_verdict():
    out = optimize_scenario(
        scenario_original(), [(8, 4)], elastic_weight=100.0, verbose=False
    )
    row = out["results"]["N8_seg4"]
    assert "total_slack" in row
    assert "figure_grade" in row
    assert "figure_grade_reasons" in row
    assert np.isfinite(row["total_slack"])
    assert row["figure_grade"] is True, row["figure_grade_reasons"]
    assert row["total_slack"] <= FIGURE_GRADE_SLACK_TOL


def test_an_unsolvable_scenario_is_not_figure_grade():
    """A blob that swallows both endpoints.

    The elastic relaxation makes this SOLVABLE as a penalized problem -- the QP
    returns an answer at every iteration and the loop runs to completion -- which
    is precisely the failure the gate exists to catch. Measured: clearance
    -0.689, hull certificate 6.52, total slack 6.52.
    """
    scenario = scenario_original()
    scenario["obstacles"] = [
        {"pos0": [4.5, 4.5], "vel": [0.0, 0.0], "r": 6.0, "color": "#000", "name": "BLOB"}
    ]
    out = optimize_scenario(scenario, [(8, 4)], elastic_weight=100.0, verbose=False)
    row = out["results"]["N8_seg4"]

    assert row["min_clearance"] < 0.0
    assert row["total_slack"] > FIGURE_GRADE_SLACK_TOL
    assert row["figure_grade"] is False
    assert row["figure_grade_reasons"]


def test_a_penetrating_run_below_the_penalty_threshold_is_not_figure_grade():
    """`wall` at elastic_weight=100 is the historical case.

    Below the scenario's exact-penalty threshold a penetrating curve is genuinely
    the cheaper answer, so the solver returns one. It is not figure-grade, and it
    is the run that used to be quoted as "wall renders a trajectory 0.112 inside
    an obstacle".
    """
    out = optimize_scenario(scenario_wall(), [(8, 2)], elastic_weight=100.0, verbose=False)
    row = out["results"]["N8_seg2"]
    assert row["min_clearance"] < 0.0
    assert row["figure_grade"] is False
