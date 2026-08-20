"""Items B9 and B10: the slant-limit speed cap, and the freed arrival time.

The speed cap is a HARD constraint, not a cost (formulation decision 4). It is
written on the control polygon:

    || P[i+1, spatial] - P[i, spatial] ||  <=  v_max * ( P[i+1, t] - P[i, t] )

and it lands as a second-order cone, not as the per-axis linear fallback.
Sufficiency for the whole curve is the Bernstein-weights-plus-triangle-inequality
argument in PAPER_1; what is tested here is the *consequence*: the physical speed
of the sampled curve, ``d(spatial)/dt`` between consecutive samples, obeys the
bound.

Every assertion below is written so that it can fail:

* the unconstrained run is asserted to VIOLATE the cap. If the cap were slack the
  constrained run would prove nothing, and this assertion is what stops that
  from passing silently.
* the constrained run is asserted to obey it, with a margin far tighter than the
  gap between the two runs.
* the time-penalty trap is asserted against the number PAPER_1 predicts --
  ``min_dt`` times the number of control-point gaps -- not against a range.
"""

import numpy as np
import pytest

from orbital_docking.bezier import BezierCurve
from spacetime_bezier.optimize import (
    UncappedTimePenaltyError,
    optimize_scenario,
    optimize_spacetime,
)
from spacetime_bezier.objective import build_initial_guess
from spacetime_bezier.scenarios import scenario_original

bezier_opt = pytest.importorskip("bezier_opt")

MIN_DT = 0.1
N_DEGREE = 8
N_SEG = 4


def sampled_max_speed(P, n: int = 4000) -> float:
    """Largest physical speed on the curve: ||d(spatial)|| / d(time).

    Measured from samples of the CURVE, never from the control points, so it is
    an independent check on the control-polygon constraint rather than a restated
    form of it.
    """
    curve = BezierCurve(np.asarray(P, dtype=float))
    pts = np.array([curve.point(t) for t in np.linspace(0.0, 1.0, n)])
    step = np.diff(pts, axis=0)
    dt = step[:, -1]
    ds = np.linalg.norm(step[:, :-1], axis=1)
    assert np.all(dt > 0.0), "time is not monotone; the speed cap derivation does not apply"
    return float(np.max(ds / dt))


def _solve(**kwargs):
    scenario = scenario_original()
    defaults = dict(
        N=N_DEGREE,
        dim=len(scenario["start"]),
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=scenario["obstacles"],
        n_seg=N_SEG,
        max_iter=200,
        tol=1e-6,
        scp_prox_weight=0.3,
        elastic_weight=100.0,
        min_dt=MIN_DT,
        verbose=False,
        init_curve=scenario["init_curve"],
    )
    defaults.update(kwargs)
    return optimize_spacetime(**defaults)


# ---------------------------------------------------------------------------
# B9 -- the speed cap binds, and it binds on the curve
# ---------------------------------------------------------------------------

V_MAX = 2.0


def test_speed_cap_binds_where_the_uncapped_solution_violates_it():
    """FAILS IF the cap is not enforced, and equally if it was never needed.

    The uncapped assertion is the one that keeps this honest: `original` with a
    pinned 10 s arrival waits and then dashes, peaking around 7.9 -- almost four
    times the cap. If a future change made the uncapped run slow, this test would
    fail rather than quietly become a tautology.
    """
    P_free, _ = _solve(v_max=None)
    P_capped, info = _solve(v_max=V_MAX)

    speed_free = sampled_max_speed(P_free)
    speed_capped = sampled_max_speed(P_capped)

    assert speed_free > V_MAX * 1.5, (
        f"the uncapped solution only reaches {speed_free:.3f}; the cap at "
        f"{V_MAX} would not bind and the test would prove nothing"
    )
    assert speed_capped <= V_MAX * (1.0 + 1e-6), (
        f"capped run peaks at {speed_capped:.6f}, above v_max={V_MAX}"
    )
    # The solver's own view of the cone must agree with the sampled measurement.
    assert float(info["speed_cap_violation"]) <= 1e-8


def test_speed_cap_off_by_default_reproduces_the_pinned_run():
    """Default v_max is None, so nothing about an existing scenario changes."""
    P_default, info_default = _solve()
    P_explicit_none, _ = _solve(v_max=None)
    np.testing.assert_allclose(P_default, P_explicit_none, atol=0.0, rtol=0.0)
    assert float(info_default["speed_cap_violation"]) == 0.0


@pytest.mark.parametrize("v_max", [1.5, 2.0, 3.0])
def test_speed_cap_holds_at_several_limits(v_max):
    P, _ = _solve(v_max=v_max)
    assert sampled_max_speed(P) <= v_max * (1.0 + 1e-6)


def test_the_cap_is_a_cone_not_a_per_axis_box():
    """The landed variant is the second-order cone, and this distinguishes them.

    A per-axis linear cap bounds each coordinate separately, so a 45-degree leg
    could reach a norm of ``sqrt(2) * v_max``. Under the cone the norm itself is
    bounded, so the peak sits at ``v_max`` and not above it. `original` runs
    diagonally, so its legs are close enough to 45 degrees for the two to be
    distinguishable.

    FAILS IF someone swaps the cone for the documented linear fallback -- which
    is a legitimate change, but it must be a visible one.
    """
    P, _ = _solve(v_max=V_MAX)
    gaps = np.diff(np.asarray(P, dtype=float), axis=0)
    leg_speed = np.linalg.norm(gaps[:, :-1], axis=1) / gaps[:, -1]
    assert leg_speed.max() <= V_MAX * (1.0 + 1e-7)
    # And the cap is actually active on at least one leg, otherwise the above is
    # satisfied vacuously.
    assert leg_speed.max() >= V_MAX * (1.0 - 1e-3)


# ---------------------------------------------------------------------------
# B10 -- freed arrival time and the linear time penalty
# ---------------------------------------------------------------------------


def test_defaults_reproduce_the_recorded_original_run_exactly():
    """Golden: `original` N8_seg4 at the pinned elastic weight, to 1e-9.

    Recorded before B8/B9/B10 landed. FAILS IF freeing the arrival time, adding
    the linear cost term, or plumbing the cone changed the default problem --
    which none of them may, because they are all off by default.
    """
    out = optimize_scenario(
        scenario_original(), [(N_DEGREE, N_SEG)], elastic_weight=100.0, verbose=False
    )
    row = out["results"][f"N{N_DEGREE}_seg{N_SEG}"]
    assert row["min_clearance"] == pytest.approx(0.6204434559492797, abs=1e-9)
    assert row["converged"] is True
    assert row["certificate_violation"] == pytest.approx(0.0, abs=1e-12)


def test_arrival_time_is_pinned_unless_it_is_freed():
    P_pinned, info = _solve(v_max=V_MAX)
    assert float(P_pinned[-1, -1]) == pytest.approx(10.0, abs=1e-9)
    assert float(info["arrival_time"]) == pytest.approx(10.0, abs=1e-9)

    P_free, info_free = _solve(v_max=V_MAX, free_arrival_time=True, time_weight=1.0)
    assert float(P_free[-1, -1]) < 9.5
    assert float(info_free["arrival_time"]) == pytest.approx(float(P_free[-1, -1]), abs=1e-12)
    # Only the arrival TIME is freed. The spatial endpoint stays pinned.
    np.testing.assert_allclose(P_free[-1, :-1], scenario_original()["end"][:-1], atol=1e-7)
    np.testing.assert_allclose(P_free[0], scenario_original()["start"], atol=1e-7)


def test_time_penalty_without_a_speed_cap_is_refused():
    """The artifact-generating configuration must not be silently runnable.

    FAILS IF the guard is removed: the call would then return a plausible-looking
    trajectory whose arrival time is a property of `min_dt`, not of the scenario.
    """
    with pytest.raises(UncappedTimePenaltyError):
        _solve(v_max=None, time_weight=1.0, free_arrival_time=True)
    with pytest.raises(UncappedTimePenaltyError):
        _solve(v_max=0.0, time_weight=0.01, free_arrival_time=True)
    # The guard is about the COMBINATION. Either half alone is legitimate.
    _solve(v_max=V_MAX, time_weight=1.0, free_arrival_time=True)
    _solve(v_max=None, time_weight=0.0)


_N_CP = N_DEGREE + 1
_FLOOR = MIN_DT * (_N_CP - 1)
# Chord of the obstacle-free problem below, over the shortest time the
# monotonicity rows allow. A cap above this is SLACK: the collapsed trajectory
# already obeys it, so the cones constrain nothing.
_BINDING_THRESHOLD = float(
    np.hypot(8.5 - 0.5, 8.5 - 1.0) / (MIN_DT * N_DEGREE)
)  # ~= 13.7


def _obstacle_free_run(v_max, time_weight):
    """The artifact problem, straight through the Rust binding.

    No obstacles, so nothing but the constraint set and the objective can set
    the answer, and no Python guard is in the way.
    """
    P0 = build_initial_guess([0.5, 1.0, 0.0], [8.5, 8.5, 10.0], _N_CP)
    _, info = bezier_opt.optimize_spacetime_bezier(
        p_init=P0,
        obstacle_pos0=np.zeros((0, 2)),
        obstacle_vel=np.zeros((0, 2)),
        obstacle_r=np.zeros((0,)),
        n_seg=N_SEG,
        max_iter=400,
        tol=1e-6,
        scp_prox_weight=0.3,
        scp_trust_radius=0.5,
        elastic_weight=100.0,
        min_dt=MIN_DT,
        coord_lb=-20.0,
        coord_ub=20.0,
        time_lb=0.0,
        time_ub=15.0,
        v_max=v_max,
        time_weight=time_weight,
        free_arrival_time=True,
    )
    return info


def test_the_refused_configuration_really_does_collapse():
    """Evidence that the guard guards something, measured not assumed.

    Without a speed cap the arrival time lands on ``min_dt * gaps`` exactly and
    does not move when the weight changes by a factor of ten -- the signature of
    an answer that is a property of `min_dt` rather than of the problem. And the
    floor flag fires, which is what makes the collapse detectable rather than
    merely inferable from the number.
    """
    low = _obstacle_free_run(None, 0.5)
    high = _obstacle_free_run(None, 5.0)
    assert float(low["arrival_time"]) == pytest.approx(_FLOOR, abs=1e-6)
    assert float(high["arrival_time"]) == pytest.approx(_FLOOR, abs=1e-6)
    assert float(low["arrival_on_min_dt_floor"]) == 1.0
    assert float(high["arrival_on_min_dt_floor"]) == 1.0


@pytest.mark.parametrize(
    "v_max,expect_floor",
    [
        (2.0, False),
        (5.0, False),
        (13.0, False),
        (14.0, True),
        (50.0, True),
        (1e3, True),
    ],
)
def test_what_matters_is_whether_the_cap_binds_not_whether_it_exists(v_max, expect_floor):
    """The collapse is reproduced by a cap that is present and SLACK.

    ``v_max=2`` used to be the whole proof that "with the cap it lands where the
    geometry chose". It does -- but so would any value below
    ``chord / (min_dt * N) ~= 13.7``, and above that the arrival returns exactly
    ``0.800000 = min_dt * gaps`` again while `UncappedTimePenaltyError` stays
    silent, because a cap does exist. Parametrizing across the threshold is what
    makes that visible.

    FAILS IF ``arrival_on_min_dt_floor`` stops tracking the actual returned
    arrival -- e.g. if it were derived from `v_max` (it would then have to guess
    this threshold) or from `free_arrival_time` alone (it would fire on every
    row here).
    """
    info = _obstacle_free_run(v_max, 0.5)
    arrival = float(info["arrival_time"])
    flag = float(info["arrival_on_min_dt_floor"])

    assert (v_max > _BINDING_THRESHOLD) is expect_floor, (
        f"the parametrization disagrees with the measured threshold "
        f"{_BINDING_THRESHOLD:.3f}"
    )
    if expect_floor:
        assert arrival == pytest.approx(_FLOOR, abs=1e-6)
        assert flag == 1.0, "the collapse happened and the flag did not fire"
    else:
        assert arrival > _FLOOR * 1.05
        assert flag == 0.0, "the flag fired on an arrival the geometry chose"


def test_the_floor_flag_is_silent_when_the_arrival_is_pinned():
    """`free_arrival_time=False` cannot collapse, so the flag must never fire.

    FAILS IF the flag stops being conditioned on `free_arrival_time`. `original`
    is pinned at 10.0 against a floor of 0.8, so the two are far apart and only a
    flag that had lost its dependence on the returned arrival could fire.
    """
    _, pinned = _solve(v_max=V_MAX)
    assert float(pinned["arrival_time"]) == pytest.approx(10.0, abs=1e-9)
    assert float(pinned["arrival_on_min_dt_floor"]) == 0.0


TIME_WEIGHTS = (0.1, 1.0, 100.0)


def test_arrival_time_decreases_as_the_time_weight_rises():
    """The energy-versus-time trade curve, with the speed cap in place.

    The weight is a reported preference, not a threshold (formulation decision
    6), so what must hold is monotonicity, not any particular value.

    FAILS IF the linear cost term is dropped from the QP or from the merit -- the
    arrival time then stops responding to the weight and the sequence flattens,
    which the strict end-to-end margin below catches.
    """
    arrivals = []
    for weight in TIME_WEIGHTS:
        P, info = _solve(v_max=4.0, free_arrival_time=True, time_weight=weight)
        assert float(info["koz_violation_reference"]) <= 1e-6
        assert sampled_max_speed(P) <= 4.0 * (1.0 + 1e-6)
        arrivals.append(float(info["arrival_time"]))

    for lo, hi in zip(arrivals, arrivals[1:]):
        assert hi <= lo + 1e-6, f"arrival time rose with the weight: {arrivals}"
    assert arrivals[-1] < arrivals[0] - 0.1, (
        f"the time weight barely moved the arrival time: {arrivals}"
    )
