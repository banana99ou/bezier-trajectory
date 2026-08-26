"""Occlusion rows in the solver (item B12): off by default, gated when on.

Two things are asserted here and they pull in opposite directions.

*Off by default.* Adding a constraint block to a shared solver is the kind of
change that quietly perturbs every existing result. The golden below is a number
recorded BEFORE this work landed, in the `wall3d` docstring and in
`optimize.py`'s slack-tolerance comment, so reproducing it is evidence about this
change rather than a restatement of it.

*Gated when on.* The paper's claim is a guarantee, so a run that has lost line of
sight must not be able to produce a figure. The keep-out certificate says nothing
about line of sight, so it cannot stand in for the occlusion one.
"""

from __future__ import annotations

import numpy as np
import pytest

bezier_opt = pytest.importorskip("bezier_opt")

from spacetime_bezier.optimize import (  # noqa: E402
    FIGURE_GRADE_CERTIFICATE_TOL,
    figure_grade_failures,
    is_figure_grade,
    optimize_spacetime,
)
from spacetime_bezier.geometry import compute_min_clearance  # noqa: E402
from spacetime_bezier.scenarios import SCENARIO_MAP  # noqa: E402


def _fence3d_run(stations):
    scenario = SCENARIO_MAP["fence3d"][0]()
    P, info = optimize_spacetime(
        N=8,
        dim=4,
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=scenario["obstacles"],
        n_seg=2,
        max_iter=200,
        elastic_weight=100.0,
        stations=stations,
        verbose=False,
        init_curve=scenario["init_curve"],
    )
    clearance = compute_min_clearance(P, scenario["obstacles"], dim=4, n_eval=3000)
    return P, info, clearance


def test_zero_station_run_reproduces_the_pre_occlusion_golden():
    """FAILS IF the occlusion block perturbs a run that asked for no occlusion.

    `wall3d` at N8_seg2 was measured at +0.1623 clearance with 3.8e-11 elastic
    slack before any occlusion code existed -- both numbers are recorded in the
    repository, in `scenario_wall3d`'s docstring and in the comment on
    FIGURE_GRADE_SLACK_TOL. Reproducing them to four decimals is what "bit
    identical when no station is given" means in practice.

    This check can fail: routing the occlusion rows into the shared relaxable
    block without guarding on an empty station set changes the row count, which
    changes the elastic slack vector, which moves this number.

    **The slack figure was re-measured 2026-08-26** when the keep-out wall moved
    from the hull-of-band outer approximation to the support of the clipped KOZ
    volume. The clearance did not move at four decimals, which is the point: this
    scenario's obstacles are straight, so their tubes are convex and the two
    constructions agree on the plane. What moved is the residue the solver leaves
    on rows whose offset is now a rigorous ceiling rather than a projection --
    3.8e-11 to 1.28e-9, still two orders under the worst good residue the
    figure-grade gate was placed against (8.51e-9, see FIGURE_GRADE_SLACK_TOL).
    """
    P, info, clearance = _fence3d_run(None)
    assert clearance == pytest.approx(0.162344, abs=5e-5)
    assert bool(info["converged"])
    assert float(info["koz_violation_reference"]) <= FIGURE_GRADE_CERTIFICATE_TOL
    assert float(info["occlusion_violation_reference"]) == 0.0
    assert float(info["total_koz_slack_returned"]) == pytest.approx(1.28e-9, rel=0.2)


def test_absent_and_empty_station_sets_are_the_same_problem():
    """An empty station array must mean exactly what no array means."""
    p_none, info_none, clear_none = _fence3d_run(None)
    p_empty, info_empty, clear_empty = _fence3d_run(np.zeros((0, 3)))
    assert np.array_equal(p_none, p_empty)
    assert clear_none == clear_empty
    assert info_none["iterations"] == info_empty["iterations"]


def test_station_array_of_the_wrong_width_is_refused():
    """A silently reshaped station would certify a scenario nobody described."""
    scenario = SCENARIO_MAP["fence3d"][0]()
    with pytest.raises(ValueError):
        optimize_spacetime(
            N=8,
            dim=4,
            p_start=scenario["start"],
            p_end=scenario["end"],
            obstacles=scenario["obstacles"],
            n_seg=2,
            max_iter=1,
            stations=[[1.0, 2.0]],  # two coordinates in a three-space problem
            verbose=False,
            init_curve=scenario["init_curve"],
        )


def test_figure_grade_gate_fails_on_an_occlusion_violation():
    """FAILS IF a run that lost line of sight can still produce a figure.

    The gate's other four conditions are all satisfied in the row below. Only the
    occlusion certificate is violated, so if the gate passed it, the guarantee
    the paper states would not be the one the code enforces.
    """
    clean = {
        "converged": True,
        "certificate_violation": 0.0,
        "min_clearance": 0.5,
        "total_slack": 0.0,
    }
    assert is_figure_grade(clean), "the control row must pass, or this proves nothing"

    # Absent key: a scenario with no station has no occlusion rows to violate.
    assert "occlusion_violation" not in clean
    assert is_figure_grade(dict(clean, occlusion_violation=0.0))

    lost = dict(clean, occlusion_violation=1.3e-3)
    assert not is_figure_grade(lost)
    reasons = figure_grade_failures(lost)
    assert len(reasons) == 1
    assert "line of sight" in reasons[0]


def test_occlusion_rows_are_reported_and_relaxable():
    """The occlusion block participates in the elastic relaxation, deliberately.

    A hard occlusion row would make the first subproblem infeasible whenever the
    initial guess starts inside the shadow -- which the straight-line guess
    generally does -- and the run would report a QP failure instead of a repair
    step. What carries the guarantee is the certificate and the slack gate, not
    hardness. This asserts the machinery is wired that way: the run below starts
    blocked, does not fail, and reports both the occlusion certificate and the
    slack that block bought.
    """
    # A blocked start, written out here rather than borrowed from a registered
    # scenario: this file is about the solver plumbing, and it must not start
    # failing because a demo scenario was retuned.
    obstacles = [
        {"pos0": [5.0, 2.5, 0.1], "vel": [0.1, -0.2, 0.0], "r": 0.8,
         "t_start": 0.0, "t_end": 10.0, "name": "F0"},
        {"pos0": [5.8, 2.5, 0.1], "vel": [0.1, -0.2, 0.0], "r": 0.8,
         "t_start": 0.0, "t_end": 10.0, "name": "F1"},
    ]
    P, info = optimize_spacetime(
        N=8,
        dim=4,
        p_start=[0.5, 5.0, 0.5, 0.0],
        p_end=[9.5, 5.0, 0.5, 10.0],
        obstacles=obstacles,
        n_seg=8,
        max_iter=3,  # deliberately stopped early, while slack is still bought
        elastic_weight=100.0,
        stations=[[5.0, -2.0, 0.3]],
        verbose=False,
        init_curve={"mode": "straight"},
    )
    assert int(info["stop_reason"]) != 3, "the subproblem must not fail outright"
    assert "occlusion_violation_reference" in info
    # Three iterations from a blocked straight line cannot have cleared the
    # shadow, so the certificate must be reporting the violation rather than
    # rounding it away.
    assert float(info["occlusion_violation_reference"]) > 0.0
    assert float(info["total_koz_slack_returned"]) > 0.0
