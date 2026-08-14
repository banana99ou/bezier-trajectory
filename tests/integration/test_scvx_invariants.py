"""Invariants on the SCvx ratio test, and a regression lock on the legacy path.

These are the only tests in the repository that exercise the Rust space-time
solver's outer loop. Every one of them is written so that it CAN fail, and the
docstring for each says what outcome would fail it -- a check that cannot fail is
not evidence.

Two of these caught real bugs during the port:
  * `test_ratio_is_exact_when_model_is_exact` caught the SCvx path routing
    obstacle-free problems into the QP-failure return, and then caught a
    tolerance stated relative to a merit that cancels to zero.
  * `test_legacy_path_is_unchanged` is what licenses the claim that adding the
    SCvx path did not disturb the existing one.
"""

import math

import pytest

from spacetime_bezier.optimize import optimize_spacetime
from spacetime_bezier.scenarios import SCENARIO_MAP

# A positive trust radius is what enables the SCvx path at all; with 0.0 the
# solver takes the legacy branch regardless of `use_scvx`.
TRUST = 0.5
MAX_ITER = 200


def _run(scenario_name, N, n_seg, *, use_scvx, trust, max_iter=MAX_ITER):
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
        use_scvx=use_scvx,
        verbose=False,
        init_curve=sc.get("init_curve"),
    )
    return dict(info)


# Measured on the legacy path before the SCvx port was added. These are not
# targets -- two of the three are infeasible and stay that way. They exist so
# that a change to the shared code path shows up as a test failure instead of as
# a quietly different figure.
LEGACY_BASELINE = {
    ("original", 8, 8): +0.8348,
    ("diverse", 8, 8): -0.6968,
    ("wall", 10, 24): -0.0855,
}


@pytest.mark.parametrize(("key", "expected"), sorted(LEGACY_BASELINE.items()))
def test_legacy_path_is_unchanged(key, expected):
    """The legacy loop must reproduce its pre-SCvx clearance exactly.

    FAILS IF: any edit shifts the legacy result by more than 5e-4 -- which is
    what would happen if the SCvx branch leaked into the unconditional-accept
    path (a stray trust row, a skipped proximal term, a changed solve order).
    """
    scenario, N, n_seg = key
    info = _run(scenario, N, n_seg, use_scvx=False, trust=0.0)
    assert info["min_clearance"] == pytest.approx(expected, abs=5e-4)


def test_legacy_path_never_claims_convergence():
    """The legacy loop accepts every step without grading it, so it is never in a
    position to assert optimality.

    FAILS IF: `converged` is ever set on the legacy path -- e.g. if someone wires
    the legacy small-step exit to the convergence flag, restoring the original
    defect where "the step got small" was reported as success.
    """
    for scenario, N, n_seg in LEGACY_BASELINE:
        info = _run(scenario, N, n_seg, use_scvx=False, trust=0.0)
        assert info["converged"] == 0.0, f"{scenario} N{N}_seg{n_seg} claimed convergence"
        # 5 == legacy small step, 0 == iteration cap. Both mean "stopped", neither
        # means "optimal".
        assert info["stop_reason"] in (0.0, 5.0)


def test_ratio_is_exact_when_model_is_exact():
    """With no obstacles there are no half-spaces, nothing re-aims, and the
    objective is exactly quadratic -- the convex subproblem IS the problem. No
    step can then be genuinely worse than its reference.

    FAILS IF: any step is rejected, or the run does not converge. Both happened
    during the port: first because the unconditional-elastic solve had no elastic
    branch to fall into without obstacle rows, then because the merit tolerance
    was stated relative to a value that cancels to ~1e-12 for a straight line.
    """
    _, info = _optimize_free_flight()
    assert info["scvx_reject_count"] == 0.0, (
        "a step was rejected on a problem whose convex model is exact"
    )
    assert info["converged"] == 1.0
    # 4 == model stationarity. The straight line is already optimal, so the loop
    # should recognise that rather than run to the iteration cap.
    assert info["stop_reason"] == 4.0


def _optimize_free_flight():
    return optimize_spacetime(
        N=8,
        dim=3,
        p_start=[0.0, 0.0, 0.0],
        p_end=[10.0, 0.0, 10.0],
        obstacles=[],
        n_seg=8,
        max_iter=50,
        tol=1e-6,
        scp_prox_weight=0.3,
        scp_trust_radius=TRUST,
        min_dt=0.1,
        use_scvx=True,
        verbose=False,
    )


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
        info = _run("original", 8, 8, use_scvx=True, trust=trust, max_iter=60)
        mean = info["scvx_rho_mean"]
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

    FAILS IF: any scenario reports `converged` while its reference still violates
    its own rebuilt rows -- the exact failure mode that would let a penetrating
    trajectory become a figure.
    """
    # One config per scenario, chosen to cover both outcomes: `wall` N8_seg2 is
    # the only configuration that currently converges, so it is the one that
    # actually exercises the assertions rather than skipping past them. Sweeping
    # every config added two minutes of runtime and no additional coverage.
    cases = [("original", 8, 8), ("diverse", 8, 8), ("wall", 8, 2)]
    converged_seen = 0
    for name, N, n_seg in cases:
        info = _run(name, N, n_seg, use_scvx=True, trust=TRUST)
        if info["converged"] == 1.0:
            converged_seen += 1
            assert info["scvx_koz_violation_reference"] <= 1e-6, (
                f"{name} N{N}_seg{n_seg} claimed convergence with certificate "
                f"violation {info['scvx_koz_violation_reference']:.3e}"
            )
            assert info["min_clearance"] > 0.0, (
                f"{name} N{N}_seg{n_seg} claimed convergence while penetrating"
            )
    # Guard against the test passing vacuously: if nothing converges, the
    # assertions above never execute and this test proves nothing.
    assert converged_seen > 0, "no case converged, so the certificate check never ran"
