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
    # The tolerance is deliberately looser than what this actually measures: on
    # the no-obstacle problem the model is exact, so rho must be 1, and the
    # measured band is [0.9999999999999, 1.0000000000030]. Both sides of rho use
    # the same objective and the same penalty -- that is the invariant, and 1e-6
    # is the margin allowed before it counts as broken.
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


def _run_scene(scenario_name, N, n_seg, **overrides):
    """Like `_run`, but with the scene's own trust radius, stations and box --
    the same problem `optimize_scenario` poses, so a fact measured here is a
    fact about the battery's configs and not about a stripped-down cousin."""
    from spacetime_bezier.optimize import DEFAULT_TRUST_RADIUS

    fn, _ = SCENARIO_MAP[scenario_name]
    sc = fn()
    kwargs = dict(
        N=N,
        dim=len(sc["start"]),
        p_start=sc["start"],
        p_end=sc["end"],
        obstacles=sc["obstacles"],
        n_seg=n_seg,
        max_iter=MAX_ITER,
        tol=1e-6,
        scp_prox_weight=0.3,
        scp_trust_radius=float(sc.get("trust_radius", DEFAULT_TRUST_RADIUS)),
        min_dt=0.1,
        verbose=False,
        init_curve=sc.get("init_curve"),
        stations=sc.get("stations"),
        coord_bounds=sc.get("coord_bounds"),
    )
    kwargs.update(overrides)
    _, info = optimize_spacetime(**kwargs)
    return dict(info)


class TestElasticComplementarity:
    """The exactness margin is MEASURED: `max_koz_dual` against the weight.

    The elastic subproblem penalizes slack as `w * sum(s)`, so its KKT
    conditions carry, for every relaxable row, a multiplier `0 <= lambda <= w`
    and the complementarity `s * (w - lambda) = 0`. Read in both directions:

      (i)  some slack is active  =>  the largest multiplier sits AT `w`;
      (ii) every slack is zero   =>  the largest multiplier sits strictly below
           `w`, and `w - max lambda` is a margin that exists.

    Direction (ii) is the one the paper leans on -- "the penalty is exact at
    this weight" -- and it is only evidence because direction (i) is also
    checked: a dual that never reached the weight would be a number that is
    not the multiplier of the subproblem being described.

    The slack graded is `total_koz_slack`, the LAST subproblem's, never
    `total_koz_slack_returned`: when the best-iterate fallback fires the
    returned slack belongs to a different trajectory from the one the dual was
    computed for (measured on `curve` N8_seg4: 1.664 vs 2.443).

    Thresholds are set from a sweep of all 28 battery configs plus seven
    held-weight runs (2026-09-06): every active-slack run carried >= 1.66 of
    slack and every zero-slack run <= 1.9e-9 (interior-point residue), so the
    1e-6 split is six orders of magnitude from either population.

    FAILS IF: a run whose last subproblem bought slack reports a dual below
    the weight (the dual is not that subproblem's multiplier, or the export
    paired a post-raise weight with a pre-raise dual); or a run with zero
    slack reports a dual at or above the weight (no margin exists, and the
    exactness claim is unfounded); or either quantity is NaN -- present and
    NaN fails, so a stale extension or a refused pairing cannot pass.
    """

    # Runs that END with slack active. `wall` N8_seg2 is the deliberately red
    # config: it reaches the cap with 4.5 of slack. The held runs pin a weight
    # below the scene's threshold, which is what the hold exists for.
    ACTIVE = [
        ("wall", 8, 2, {}),
        ("wall", 10, 16, dict(elastic_weight=100.0, escalate_elastic_weight=False)),
        ("curve", 8, 4, dict(elastic_weight=100.0, escalate_elastic_weight=False)),
        ("diverse", 8, 4, dict(elastic_weight=100.0, escalate_elastic_weight=False)),
    ]
    # Runs that END with every slack zero, at three different final weights
    # (100 with no raise, 1000 after one, 10000 after two).
    ZERO = [
        ("original", 8, 8, {}),
        ("fence3d", 8, 2, {}),
        ("diverse", 8, 4, {}),
        ("wall", 10, 16, {}),
    ]

    @pytest.mark.parametrize("scenario,N,n_seg,overrides", ACTIVE)
    def test_active_slack_pins_the_dual_at_the_weight(self, scenario, N, n_seg, overrides):
        info = _run_scene(scenario, N, n_seg, **overrides)
        w = float(info["final_elastic_weight"])
        dual = float(info["max_koz_dual"])
        slack = float(info["total_koz_slack"])
        assert math.isfinite(w) and math.isfinite(dual) and math.isfinite(slack), (
            f"NaN in the pairing: w={w} dual={dual} slack={slack}"
        )
        assert slack > 1e-6, (
            f"{scenario} N{N}_seg{n_seg} was chosen as an active-slack case but "
            f"ended with slack {slack:.3e}; the case no longer tests direction (i)"
        )
        assert abs(dual - w) <= 1e-6 * w, (
            f"slack {slack:.3f} is active but the largest dual {dual:.6g} is not "
            f"at the weight {w:g} (ratio {dual / w:.7f})"
        )

    @pytest.mark.parametrize("scenario,N,n_seg,overrides", ZERO)
    def test_zero_slack_leaves_a_real_margin(self, scenario, N, n_seg, overrides):
        info = _run_scene(scenario, N, n_seg, **overrides)
        w = float(info["final_elastic_weight"])
        dual = float(info["max_koz_dual"])
        slack = float(info["total_koz_slack"])
        assert math.isfinite(w) and math.isfinite(dual) and math.isfinite(slack), (
            f"NaN in the pairing: w={w} dual={dual} slack={slack}"
        )
        assert slack <= 1e-6, (
            f"{scenario} N{N}_seg{n_seg} was chosen as a zero-slack case but "
            f"ended with slack {slack:.3e}; the case no longer tests direction (ii)"
        )
        assert dual < w, (
            f"slack is zero but the largest dual {dual:.6g} is not below the "
            f"weight {w:g}: no exactness margin exists"
        )

    def test_stepper_never_pairs_a_dual_with_a_weight_it_was_not_solved_at(self):
        """Per-iteration version, through the stepping API.

        On a raise iteration the state's weight is the NEXT subproblem's, and
        the dual just computed belongs to the previous one; the honest export
        is NaN. On every other iteration the pair describes one subproblem and
        the KKT bound `dual <= w` holds, with equality exactly when slack is
        active. `wall` N8_seg2 from 100 raises three times before the cap, so
        all three raise iterations are exercised.

        FAILS IF: a raise iteration exports a finite dual (the fictitious
        margin -- measured before the fix as dual/weight == 0.1 exactly); or a
        non-raise iteration exports a dual above its weight, a NaN dual, or an
        active-slack dual off the weight; or no raise happens at all (nothing
        exercised the pairing).
        """
        import bezier_opt
        from spacetime_bezier.geometry import obstacle_array_bundle
        from spacetime_bezier.objective import build_initial_guess

        fn, _ = SCENARIO_MAP["wall"]
        sc = fn()
        dim = len(sc["start"])
        P_init = build_initial_guess(sc["start"], sc["end"], 9, init_curve=sc.get("init_curve"))
        ctrl, radii = obstacle_array_bundle(sc["obstacles"], dim - 1)
        ctx = bezier_opt.SpacetimeScpContext(
            p_init=np.asarray(P_init, dtype=float),
            obstacle_ctrl=ctrl,
            obstacle_r=radii,
            n_seg=2,
            min_dt=0.1,
            coord_lb=-20.0,
            coord_ub=20.0,
            time_lb=0.0,
            time_ub=float(P_init[-1, -1]) * 1.5,
            scp_trust_radius=TRUST,
            elastic_weight=100.0,
            tol=1e-6,
            sound_clip=True,
        )
        prev_w = 100.0
        raises_seen = 0
        for _ in range(MAX_ITER):
            info = ctx.step()[1]
            w = float(info["elastic_weight"])
            dual = float(info["max_koz_dual"])
            slack = float(info["total_slack"])
            it = int(info["iteration"])
            if w != prev_w:
                raises_seen += 1
                assert math.isnan(dual), (
                    f"iteration {it}: weight raised {prev_w:g} -> {w:g} but a "
                    f"finite dual {dual:.6g} was exported beside the NEW weight"
                )
            else:
                assert math.isfinite(dual), f"iteration {it}: NaN dual with no raise"
                assert dual <= w * (1.0 + 1e-9), (
                    f"iteration {it}: dual {dual:.6g} above the weight {w:g}"
                )
                if slack > 1e-6:
                    assert abs(dual - w) <= 1e-6 * w, (
                        f"iteration {it}: slack {slack:.3f} active but dual "
                        f"{dual:.6g} is not at the weight {w:g}"
                    )
            prev_w = w
            if not info["running"]:
                break
        assert raises_seen >= 1, "no raise fired; the pairing was never exercised"


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
