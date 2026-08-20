"""The invariant the ratio test cannot fail unless something is miswired.

rho = (actual merit reduction) / (predicted merit reduction). The prediction
comes from the convex subproblem; the actual comes from re-evaluating the merit
with the half-spaces REBUILT at the candidate. The two can differ for exactly one
reason -- the half-spaces moved -- because the objective is exactly quadratic
plus exactly linear, with no linearization anywhere in it.

With NO obstacles there are no half-spaces to move. The convex model IS the
problem, so rho must be exactly 1 and no step may ever be rejected. That is an
impossibility check rather than a plausibility one: no arrangement of a correct
implementation can produce anything else.

It is worth running specifically WITH the linear time penalty and the cap,
because those are the two terms that were added last and the linear one has to
appear in the merit as well as in the QP. If `f_linear` were in the QP alone the
solver would be minimizing a different function from the one being graded, and
rho would drift off 1 -- reading as model error when it is bookkeeping error.
"""

import pytest

from spacetime_bezier.optimize import optimize_spacetime

pytest.importorskip("bezier_opt")


def _no_obstacle_run(**kwargs):
    defaults = dict(
        N=8,
        dim=3,
        p_start=[0.0, 0.0, 0.0],
        p_end=[10.0, 0.0, 10.0],
        obstacles=[],
        n_seg=8,
        max_iter=80,
        tol=1e-6,
        scp_trust_radius=0.5,
        min_dt=0.1,
        verbose=False,
    )
    defaults.update(kwargs)
    return optimize_spacetime(**defaults)[1]


@pytest.mark.parametrize("time_weight", [0.1, 1.0, 100.0, 1e4])
def test_rho_is_exactly_one_with_a_time_penalty_and_a_cap(time_weight):
    """FAILS IF the linear time term is dropped from either merit.

    Measured: rho_mean = 1.0000000000 at weights 100 and above, and within
    1.5e-10 of 1 at weight 0.1 where the linear term is small enough for the
    quadratic's own cancellation to show. Zero rejections at every weight.
    """
    info = _no_obstacle_run(v_max=4.0, time_weight=time_weight, free_arrival_time=True)

    samples = int(info["rho_samples"])
    assert samples > 0, (
        "no ratios were recorded, so this test measured nothing -- the run "
        "took only bootstrap or null steps"
    )
    assert float(info["rho_min"]) == pytest.approx(1.0, abs=1e-8)
    assert float(info["rho_max"]) == pytest.approx(1.0, abs=1e-8)
    assert float(info["rho_mean"]) == pytest.approx(1.0, abs=1e-8)

    # A rejected step on an exact model is a contradiction, not a tolerance
    # question: the candidate minimizes the very function being graded.
    assert int(info["reject_count"]) == 0
    assert bool(info["converged"]) is True


def test_the_ratio_tightens_as_the_linear_term_dominates():
    """The residual at small weight is cancellation, not a missing term.

    A genuinely missing or misscaled linear term would push rho FURTHER from 1 as
    the weight grew, because the unmodelled part of the objective would grow with
    it. The opposite is measured, which is what distinguishes the two.

    FAILS IF that ordering reverses.
    """
    small = _no_obstacle_run(v_max=4.0, time_weight=0.1, free_arrival_time=True)
    large = _no_obstacle_run(v_max=4.0, time_weight=100.0, free_arrival_time=True)

    err_small = abs(float(small["rho_mean"]) - 1.0)
    err_large = abs(float(large["rho_mean"]) - 1.0)
    assert err_large <= err_small
