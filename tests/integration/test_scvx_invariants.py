"""Invariants on the solver's step-acceptance logic.

These are the only tests in the repository that exercise the Rust space-time
solver's outer loop. Every one is written so that it CAN fail, and each docstring
says what outcome would fail it -- a check that cannot fail is not evidence.

There is one solver. `test_one_solver_only` is what holds that: the batch loop
and the step debugger must produce the same trajectory, because they call the
same iteration function.

`test_ratio_is_exact_when_model_is_exact` caught two real bugs: obstacle-free
problems being routed into the QP-failure return, and a tolerance stated relative
to a merit that cancels to zero.
"""

import math

import numpy as np
import pytest

from spacetime_bezier.optimize import optimize_spacetime
from spacetime_bezier.scenarios import SCENARIO_MAP

# A non-positive radius means "unspecified" and the solver substitutes its
# default; this pins it so the tests do not silently drift with that default.
TRUST = 0.5
MAX_ITER = 200


def _run(scenario_name, N, n_seg, *, trust=TRUST, max_iter=MAX_ITER):
    fn, _ = SCENARIO_MAP[scenario_name]
    sc = fn()
    _, info = optimize_spacetime(
        N=N,
        dim=len(sc["start"]),
        p_start=sc["start"],
        p_end=sc["end"],
        obstacles=sc["obstacles"],
        n_seg=n_seg,
        max_iter=max_iter,
        tol=1e-6,
        scp_prox_weight=0.3,
        scp_trust_radius=trust,
        min_dt=0.1,
        verbose=False,
        init_curve=sc.get("init_curve"),
    )
    return dict(info)


def test_one_solver_only():
    """The batch loop and the step debugger must return the same trajectory.

    They are two entry points onto one `scp_iterate`. If they diverge, something
    has grown a second accept rule -- which is the failure this whole structure
    exists to prevent.

    FAILS IF: the two control-point sets differ by more than solver noise, e.g. if
    the stepper reintroduces its own convergence test or best-iterate tracking.
    """
    from spacetime_bezier.rust_debug_stepper import (
        create_spacetime_debug_stepper_from_control_points,
    )
    from spacetime_bezier.geometry import compute_min_clearance
    from spacetime_bezier.objective import build_initial_guess

    fn, _ = SCENARIO_MAP["wall"]
    sc = fn()
    P_init = build_initial_guess(sc["start"], sc["end"], 9, init_curve=sc.get("init_curve"))

    batch_P, batch_info = optimize_spacetime(
        N=8, dim=3, p_start=sc["start"], p_end=sc["end"], obstacles=sc["obstacles"],
        n_seg=2, max_iter=60, tol=1e-6, scp_trust_radius=TRUST, min_dt=0.1,
        verbose=False, init_curve=sc.get("init_curve"),
    )

    stepper = create_spacetime_debug_stepper_from_control_points(
        p_init=P_init, obstacles=sc["obstacles"], n_seg=2, max_iter=60, tol=1e-6,
        scp_trust_radius=TRUST, min_dt=0.1,
    )
    step_P, step_info = stepper.run_to_completion()

    assert np.allclose(np.asarray(batch_P), np.asarray(step_P), atol=1e-9), (
        "batch loop and step debugger produced different trajectories"
    )
    assert batch_info["iterations"] == step_info["iterations"]
    assert bool(batch_info["converged"]) == bool(step_info["converged"])



def _optimize_free_flight():
    """A straight run with no obstacles at all."""
    return optimize_spacetime(
        N=8,
        dim=3,
        p_start=[0.0, 0.0, 0.0],
        p_end=[10.0, 0.0, 10.0],
        obstacles=[],
        n_seg=8,
        max_iter=50,
        tol=1e-6,
        scp_trust_radius=TRUST,
        min_dt=0.1,
        verbose=False,
    )


def test_ratio_is_exact_when_model_is_exact():
    """With no obstacles there are no half-spaces, nothing re-aims, and the
    objective is exactly quadratic -- the convex subproblem IS the problem. No
    step can then be genuinely worse than its reference.

    FAILS IF: any step is rejected, or the run does not converge. Both happened
    during development: first because the unconditional-elastic solve had no
    elastic branch to fall into without obstacle rows, then because the merit
    tolerance was stated relative to a value that cancels to ~1e-12 for a
    straight line.
    """
    _, info = _optimize_free_flight()
    assert info["reject_count"] == 0.0, (
        "a step was rejected on a problem whose convex model is exact"
    )
    assert info["converged"] == 1.0
    # 4 == model stationarity. The straight line is already optimal, so the loop
    # should recognise that rather than run to the iteration cap.
    assert info["stop_reason"] == 4.0


def test_ratio_approaches_one_as_trust_shrinks():
    """The merit is continuous in the step, so forcing the step to zero must make
    the linearized and true violations agree. This is what makes the ratio a
    measurement of half-space movement rather than of arithmetic error.

    FAILS IF: the mean ratio at the smallest radius is no closer to 1 than at the
    largest -- which would mean the predicted and actual merits are not two
    evaluations of the same quantity, and every ratio reported anywhere else is
    meaningless.
    """
    radii = [0.5, 0.02, 0.0008]
    means = []
    for trust in radii:
        info = _run("original", 8, 8, trust=trust, max_iter=60)
        mean = info["rho_mean"]
        assert not math.isnan(mean), f"no graded steps at radius {trust}"
        means.append(mean)

    assert abs(means[-1] - 1.0) < abs(means[0] - 1.0)
    # At the smallest radius the model should be exact to solver precision.
    assert means[-1] == pytest.approx(1.0, abs=1e-6)


def test_convergence_requires_the_hull_certificate():
    """A run may only report convergence if its iterate satisfies the half-spaces
    its own control points generate. Clearance alone is not enough: the
    half-space is conservative, so a curve can clear every obstacle while its
    control-point hull does not.

    FAILS IF: any scenario reports convergence while its reference still violates
    its own rebuilt rows -- the exact failure mode that would let a penetrating
    trajectory become a figure.
    """
    # One config per scenario. `wall` at 2 segments is the only configuration that
    # currently converges, so it is what actually exercises the assertions.
    cases = [("original", 8, 8), ("diverse", 8, 8), ("wall", 8, 2)]
    converged_seen = 0
    for name, N, n_seg in cases:
        info = _run(name, N, n_seg)
        if info["converged"] == 1.0:
            converged_seen += 1
            assert info["koz_violation_reference"] <= 1e-6, (
                f"{name} N{N}_seg{n_seg} claimed convergence with certificate "
                f"violation {info['koz_violation_reference']:.3e}"
            )
            assert info["min_clearance"] > 0.0, (
                f"{name} N{N}_seg{n_seg} claimed convergence while penetrating"
            )
    # Guard against passing vacuously: if nothing converges the assertions above
    # never execute and this test proves nothing.
    assert converged_seen > 0, "no case converged, so the certificate check never ran"


def test_never_claims_convergence_while_penetrating():
    """Across every shipped configuration, `converged` and a penetrating curve
    must never appear together.

    FAILS IF: any config reports success while its returned trajectory is inside
    an obstacle -- which is the single outcome that would put a false figure in
    the paper.
    """
    bad = []
    for name, (fn, configs) in SCENARIO_MAP.items():
        for N, n_seg in configs:
            info = _run(name, N, n_seg)
            if info["converged"] == 1.0 and info["min_clearance"] <= 0.0:
                bad.append(f"{name} N{N}_seg{n_seg} clearance={info['min_clearance']:+.4f}")
    assert not bad, "converged while penetrating: " + "; ".join(bad)
