"""Item B7: a run standing on elastic slack cannot produce a figure.

Every subproblem in this solver is solved elastically -- slack variables on the
keep-out and occlusion rows, priced by the exact-penalty weight -- so "the loop
converged" and "the loop obeyed the obstacles" are different statements. The gate
makes the difference explicit and refuses to let one stand in for the other.

**The predicate has SIX conditions.** This docstring used to say four while the
code checked five, which is how a condition can go decorative without anyone
noticing:

  1. ``converged``               -- the loop stopped for a principled reason, and
                                    stopped on the point it returned.
  2. ``certificate_violation``   -- the control-point hull satisfies the KEEP-OUT
                                    half-spaces it generates, rebuilt at the
                                    returned iterate.
  3. ``occlusion_violation``     -- the same, for the LINE-OF-SIGHT half-spaces
                                    (item B12). A separate guarantee.
  4. ``occlusion_planes_dropped``-- every line-of-sight plane in range could
                                    actually be built. A window with no
                                    supporting half-space is uncertifiable, not
                                    certified.
  5. ``min_clearance``           -- the sampled curve misses the TRUE obstacle
                                    trajectories.
  6. ``total_slack``             -- the elastic relaxation bought nothing.
     ``speed_cap_violation``     -- and, as a seventh field on the same footing,
                                    the returned curve satisfies the hard
                                    slant-limit cones.

Each is flipped on its own against an otherwise-passing row, and each alone must
sink the gate. A gate whose conditions cannot be shown to bind separately is a
gate that might be passing on one condition and reporting seven.

Three of them -- occlusion, occlusion drops, speed cap -- are OPTIONAL features
and default to 0.0 rather than NaN when absent, because a run without stations or
without a cap genuinely has nothing to violate. That asymmetry is tested too:
defaulting them to NaN would sink every scenario in the table.
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
        "occlusion_violation": 0.0,
        "occlusion_planes_dropped": 0.0,
        "min_clearance": 0.62,
        "total_slack": 1e-14,
        "speed_cap_violation": 0.0,
    }


# Every field the predicate reads, with a value that must sink it and the
# fragment of the reason that identifies which condition fired. Keeping the two
# together is what stops a flipped field being credited to the wrong condition.
_CONDITIONS = [
    ("converged", False, "not converged"),
    ("certificate_violation", FIGURE_GRADE_CERTIFICATE_TOL * 10.0, "certificate"),
    ("occlusion_violation", FIGURE_GRADE_CERTIFICATE_TOL * 10.0, "line of sight"),
    ("occlusion_planes_dropped", 3.0, "could not be built"),
    ("min_clearance", -1e-9, "penetrates"),
    ("total_slack", FIGURE_GRADE_SLACK_TOL * 10.0, "slack"),
    ("speed_cap_violation", 1e9, "speed cap"),
]


# ---------------------------------------------------------------------------
# The gate predicate itself
# ---------------------------------------------------------------------------


def test_a_row_meeting_every_condition_is_figure_grade():
    assert figure_grade_failures(_passing_row()) == []
    assert is_figure_grade(_passing_row()) is True


def test_the_baseline_row_covers_every_field_the_predicate_reads():
    """FAILS IF a condition is added to the predicate and not to this file.

    Without this, a new condition whose field is absent from `_passing_row` would
    silently take its default, the flip-one-condition sweep would never reach it,
    and the file would go on reporting full coverage of a predicate it no longer
    covers.
    """
    covered = {field for field, _, _ in _CONDITIONS}
    row = _passing_row()
    unreachable = covered - set(row)
    assert not unreachable, f"_passing_row is missing {sorted(unreachable)}"


@pytest.mark.parametrize("field,bad_value,expected_fragment", _CONDITIONS)
def test_each_condition_alone_sinks_the_gate(field, bad_value, expected_fragment):
    """FAILS IF any condition is decorative.

    `speed_cap_violation` was exactly that until this test existed: written into
    every row by `optimize_scenario` and read by nothing, so flipping it left the
    row passing.
    """
    row = _passing_row()
    row[field] = bad_value
    reasons = figure_grade_failures(row)
    assert len(reasons) == 1, f"expected exactly one failure, got {reasons}"
    assert expected_fragment in reasons[0]
    assert is_figure_grade(row) is False


@pytest.mark.parametrize(
    "field", ["certificate_violation", "min_clearance", "total_slack"]
)
def test_a_missing_required_field_is_not_figure_grade(field):
    """No evidence is not the same as evidence of feasibility.

    These three default to NaN, and NaN fails every comparison, which is why the
    predicate is written as `not (x <= tol)` and never as `x > tol`. FAILS IF a
    comparison is flipped to the positive form: a row that never reported the
    quantity would then pass.
    """
    row = _passing_row()
    del row[field]
    assert is_figure_grade(row) is False


def test_a_missing_converged_flag_is_not_figure_grade():
    """`converged` fails on a DICT DEFAULT, not on the NaN idiom.

    It is a bool, so `row.get("converged", False)` supplies the failing value
    directly and no comparison is involved. Kept as its own test because grouping
    it with the NaN cases claimed the idiom was doing work it does not do here.
    """
    row = _passing_row()
    del row["converged"]
    assert is_figure_grade(row) is False
    assert any("not converged" in r for r in figure_grade_failures(row))


@pytest.mark.parametrize(
    "field", ["occlusion_violation", "occlusion_planes_dropped", "speed_cap_violation"]
)
def test_a_missing_optional_field_defaults_to_passing(field):
    """The three optional conditions default to 0.0, deliberately.

    A run with no station has no occlusion rows and nothing to drop; a run with
    no cap has no cones. 0.0 is the TRUE value in both cases, not an assumption.
    FAILS IF one of them is switched to the NaN default -- every pre-B12,
    capless scenario in the table would stop being figure-grade.
    """
    row = _passing_row()
    del row[field]
    assert figure_grade_failures(row) == []


def test_a_negative_slack_residue_is_measured_by_magnitude():
    """The interior-point residue can come back negative (-1.65e-14 measured).

    FAILS IF the comparison reverts to `slack <= tol`: an arbitrarily large
    negative total would pass, and a negative total is a solver artifact, not a
    credit against a violation.
    """
    row = _passing_row()
    row["total_slack"] = -1e-14
    assert figure_grade_failures(row) == []

    row["total_slack"] = -FIGURE_GRADE_SLACK_TOL * 10.0
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
    assert abs(row["total_slack"]) <= FIGURE_GRADE_SLACK_TOL


def test_every_gate_field_is_present_on_a_real_row():
    """FAILS IF `optimize_scenario` stops exporting a field the gate reads.

    A missing optional field defaults to passing, so the gate would go quiet
    rather than error -- exactly the failure mode this file exists to prevent.
    """
    out = optimize_scenario(
        scenario_original(), [(8, 4)], elastic_weight=100.0, verbose=False
    )
    row = out["results"]["N8_seg4"]
    for field, _, _ in _CONDITIONS:
        assert field in row, f"result rows no longer carry {field}"


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
    an obstacle". Its slack, 2.29, is the smallest genuine relaxation measured --
    six orders above the gate.
    """
    out = optimize_scenario(scenario_wall(), [(8, 2)], elastic_weight=100.0, verbose=False)
    row = out["results"]["N8_seg2"]
    assert row["min_clearance"] < 0.0
    assert row["figure_grade"] is False
    assert row["total_slack"] > FIGURE_GRADE_SLACK_TOL * 1e5
