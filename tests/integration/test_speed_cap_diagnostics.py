"""Three ways the speed cap produced an answer that said nothing.

1. A FEASIBLE problem reported `qp_failure` at iteration 1 -- indistinguishable
   from a genuinely unsolvable one -- because the cones carry no slack and the
   trust box was too small to reach cap-feasibility from the straight-line seed.
2. The "freed" arrival time is clamped at ``P_init[-1,-1] * time_ub_scale``, and
   a run that wanted more came back sitting exactly on the clamp, converged, with
   nothing in the result saying the clamp is what chose the number.
3. ``v_max`` was accepted unvalidated, including values at which the cone
   degenerates numerically.

None of these is repaired here. They are made VISIBLE, which is the difference
between a solver that gave up and a solver that gave up for a stated reason.
"""

import numpy as np
import pytest

from spacetime_bezier.optimize import (
    MAX_SPEED_CAP,
    SpeedCapOutOfRangeError,
    _STOP_REASONS,
    check_speed_cap_is_representable,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import scenario_original

pytest.importorskip("bezier_opt")

FIRST_QP_INFEASIBLE = 5
QP_FAILURE = 3


def _solve(**kwargs):
    scenario = scenario_original()
    defaults = dict(
        N=8,
        dim=3,
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=scenario["obstacles"],
        n_seg=4,
        max_iter=300,
        tol=1e-6,
        elastic_weight=1e5,
        init_curve=scenario["init_curve"],
        verbose=False,
    )
    defaults.update(kwargs)
    return optimize_spacetime(**defaults)


# ---------------------------------------------------------------------------
# (a) a first-iteration cone infeasibility is not a verdict on the problem
# ---------------------------------------------------------------------------


def test_a_feasible_problem_that_fails_at_iteration_one_says_so():
    """The reviewer's construction: same problem, two trust radii.

    `original` with v_max=1.0 and a priced free arrival is FEASIBLE -- trust 8.0
    converges on it and returns a curve that obeys the cone. At trust 0.5 the
    straight-line seed is more than one trust radius from cap-feasibility, the
    cones carry no slack, and iteration 1 is infeasible.

    FAILS IF the two are reported with the same label. The pair is what makes
    this checkable: if the larger trust radius did not solve the problem, the
    small-radius failure would be an honest `qp_failure` and this test would be
    asserting the wrong label onto a genuine one.
    """
    _, tight = _solve(
        v_max=1.0,
        free_arrival_time=True,
        time_weight=1.0,
        scp_trust_radius=0.5,
        time_ub_scale=6.0,
    )
    # Trust 4.0, not 8.0, and the reason is a measured interaction rather than a
    # tuning preference. The reach floor -- `sound_clip`, the default since
    # 2026-08-31 -- is the segment radius PLUS one trust step, so it grows with
    # the trust radius. On this 10-unit scene, trust 8.0 makes the floor about 14
    # and the clip stops localizing anything: measured, the run goes from
    # converging in 13 iterations to hitting the 300-iteration cap (its returned
    # iterate is still certified at 2.36e-07, so this is a convergence
    # declaration, not a correctness failure). Trust 2.0 and 4.0 are unaffected
    # -- 21 vs 22 and 13 vs 13 iterations with the floor off and on. 4.0 proves
    # feasibility just as well as 8.0, which is all this half of the pair is for.
    _, loose = _solve(
        v_max=1.0,
        free_arrival_time=True,
        time_weight=1.0,
        scp_trust_radius=4.0,
        time_ub_scale=6.0,
    )

    # The problem is feasible -- proven by solving it.
    assert bool(loose["converged"]) is True
    assert float(loose["speed_cap_violation"]) <= 1e-6
    assert int(loose["stop_reason"]) != FIRST_QP_INFEASIBLE

    assert int(tight["iterations"]) == 1
    assert int(tight["stop_reason"]) == FIRST_QP_INFEASIBLE
    assert bool(tight["converged"]) is False
    assert "trust radius" in _STOP_REASONS[FIRST_QP_INFEASIBLE]


def test_the_label_needs_a_cone_to_be_present():
    """Without a cap there is no cone, so a first-iteration failure is ordinary.

    FAILS IF the new label is applied to every iteration-1 failure: the KOZ rows
    are elastic, so a capless subproblem that fails did so for a reason the trust
    radius will not fix, and saying otherwise would send the reader somewhere
    useless.
    """
    from spacetime_bezier.optimize import _STOP_REASONS as labels

    assert labels[QP_FAILURE].startswith("qp_failure")
    assert labels[FIRST_QP_INFEASIBLE].startswith("first_qp_infeasible")

    # The capless default run solves, so the label cannot be reached from it.
    _, info = _solve()
    assert int(info["stop_reason"]) not in (QP_FAILURE, FIRST_QP_INFEASIBLE)


# ---------------------------------------------------------------------------
# (b) the freed arrival is clamped, and the clamp is now reported
# ---------------------------------------------------------------------------


def test_an_arrival_sitting_on_the_time_upper_bound_is_flagged():
    """`time_ub = P_init[-1,-1] * time_ub_scale` silently chose the answer.

    `original` seeded at t=10 with the default scale of 1.5 gives an upper bound
    of 15.0. At v_max=0.75 the chord of 10.966 needs about 14.6 s of flight, and
    the smoothness regularizer wants more, so the arrival lands exactly on 15.0
    -- converged, stationary, cone satisfied, and set by a scale factor rather
    than by the scenario.

    FAILS IF `arrival_on_time_ub` stops tracking the returned arrival: raising
    the scale to 3.0 moves the answer off the bound and the flag must go quiet,
    which a flag keyed on `free_arrival_time` alone could not do.

    **The released arrival is checked by an impossibility, not by a threshold.**
    It used to be asserted as `> 16.0`, chosen against a measured 19.78; the
    interior optimum is 15.5637 today and the assertion started failing without
    anything about the flag having changed. A number that moves when the keep-out
    construction changes is not what this test is about. What cannot be true if
    the bound is genuinely inactive is that moving the bound FURTHER AWAY changes
    the answer -- so the run is repeated at scale 6.0 and the two must agree.
    Measured 2026-08-31 across three time weights: scale 3.0 and 6.0 return
    15.6008 / 15.5637 / 15.4145 at weights 0.25 / 1.0 / 4.0, identical between
    the two scales to 1e-4, and monotone in the weight -- so the arrival is set
    by the time penalty, which is the thing that is supposed to set it.
    """
    _, clamped = _solve(
        v_max=0.75,
        free_arrival_time=True,
        time_weight=1.0,
        scp_trust_radius=8.0,
        time_ub_scale=1.5,
    )
    assert float(clamped["time_ub_used"]) == pytest.approx(15.0, abs=1e-12)
    assert float(clamped["arrival_time"]) == pytest.approx(15.0, abs=1e-6)
    assert float(clamped["arrival_on_time_ub"]) == 1.0
    # And it looked like a clean answer.
    assert bool(clamped["converged"]) is True

    _, released = _solve(
        v_max=0.75,
        free_arrival_time=True,
        time_weight=1.0,
        scp_trust_radius=8.0,
        time_ub_scale=3.0,
    )
    assert float(released["time_ub_used"]) == pytest.approx(30.0, abs=1e-12)
    assert float(released["arrival_on_time_ub"]) == 0.0
    arrival = float(released["arrival_time"])
    # It left the bound it used to sit on, and it is nowhere near the new one --
    # so "off the bound" is not a rounding artifact in either direction.
    assert arrival > 15.0
    assert arrival < 30.0 - 1.0

    # The impossibility: an inactive bound cannot choose the answer. Doubling it
    # again must leave the arrival where it was.
    _, further = _solve(
        v_max=0.75,
        free_arrival_time=True,
        time_weight=1.0,
        scp_trust_radius=8.0,
        time_ub_scale=6.0,
    )
    assert float(further["time_ub_used"]) == pytest.approx(60.0, abs=1e-12)
    assert float(further["arrival_on_time_ub"]) == 0.0
    assert float(further["arrival_time"]) == pytest.approx(arrival, abs=1e-4)


def test_a_pinned_arrival_never_flags_the_upper_bound():
    """`free_arrival_time=False` cannot sit on a bound it is not subject to."""
    _, info = _solve(v_max=2.0)
    assert float(info["arrival_on_time_ub"]) == 0.0
    assert np.isfinite(float(info["time_ub_used"]))


# ---------------------------------------------------------------------------
# (c) v_max is validated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", [0.0, -1.0, -0.0, float("inf"), float("nan"), MAX_SPEED_CAP * 10.0]
)
def test_an_unusable_speed_cap_is_refused(bad):
    """FAILS IF v_max is accepted unvalidated.

    Zero and negatives are the dangerous half: the Rust builder reads them as
    "no cap", so a sign error silently turns a constraint off rather than
    erroring.
    """
    with pytest.raises(SpeedCapOutOfRangeError):
        check_speed_cap_is_representable(bad)
    with pytest.raises(SpeedCapOutOfRangeError):
        _solve(v_max=bad)


@pytest.mark.parametrize("good", [None, 1e-3, 1.0, 2.0, MAX_SPEED_CAP])
def test_a_usable_speed_cap_is_accepted(good):
    """`None` is how "no cap" is spelled and must stay legal."""
    check_speed_cap_is_representable(good)


def test_the_bound_is_not_advertised_as_a_numerical_threshold():
    """The measured cone degradation begins BELOW the accepted bound.

    Documented in `MAX_SPEED_CAP`. This test pins the measurement so the comment
    cannot drift into claiming the bound is a safety threshold: on `original`
    N8_seg4 a cap of 3e4 reproduces the uncapped answer exactly while 1e6 -- an
    accepted value -- fails at iteration 1 with a violation of 0.00.

    FAILS IF the degradation moves above the bound, which would mean the bound
    HAS become a threshold and the comment should be rewritten to say so.
    """
    _, fine = _solve(v_max=3e4)
    assert bool(fine["converged"]) is True
    assert float(fine["min_clearance"]) == pytest.approx(0.6204434559492797, abs=1e-6)

    _, degenerate = _solve(v_max=MAX_SPEED_CAP)
    assert bool(degenerate["converged"]) is False
    assert int(degenerate["stop_reason"]) in (QP_FAILURE, FIRST_QP_INFEASIBLE)
    # The violation reads as a satisfied constraint. This is the trap.
    assert float(degenerate["speed_cap_violation"]) == 0.0
