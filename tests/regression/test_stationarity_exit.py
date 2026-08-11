"""
Regression test: the stationarity exit must fire at numerically-zero `pred`.

Near a stationary point `pred` is a difference of two merits that agree to about
1e-14 absolute, so its SIGN is noise. The stationarity test in `optimizer.rs`
therefore compares |pred| against tol_f*|phi|. It formerly required
`pred >= 0.0`, which reset the streak on every noise-level negative pred, so the
exit could never fire: the loop rejected null steps and halved the trust region
to its floor instead, reporting stop_reason 2 (failure) at points that were
critical three orders inside the tolerance.

WHAT MAKES THIS TEST ABLE TO FAIL: both configurations below reported
stop_reason 2 before the 2026-08-11 fix, on this exact scenario. Reverting the
comparison to `pred >= 0.0` turns both assertions red. They are the only two
configurations in the suite that do so -- no verification pillar covers either,
which is why this file exists (design_freeze.md section 7, evidence 9).

The objective values are pinned alongside, because the fix must change WHERE the
loop stops, never WHAT it converges to. They were measured identical to all
digits before and after.
"""

import numpy as np
import pytest

from tools.verify import harness_common as H

# stop_reason: 1 = merit streak, 4 = model stationarity (both principled);
# 0 = iteration cap, 2 = trust collapse, 3 = QP failure (all "the loop gave up").
PRINCIPLED_STOPS = (1, 4)
TRUST_COLLAPSE = 2

# (degree, n_seg, objective measured 2026-08-11 both pre- and post-fix)
CASES = [
    (8, 16, 2.256657e-05),   # was stop 2 @ 27 iters; KKT residual 1.229e-02 (optimal)
    (7, 4, 7.039215e-05),    # was stop 2 @ 30 iters; both restart-invariant
]


@pytest.mark.parametrize("degree,n_seg,expected_cost", CASES)
def test_critical_point_exits_principled(degree, n_seg, expected_cost):
    """A critical iterate must not be reported as a trust-region collapse."""
    sc = H.make_scenario("phase120", N=degree)
    _, info = H.run_rust(sc, n_seg=n_seg)

    stop = int(info["scvx_stop_reason"])
    assert stop != TRUST_COLLAPSE, (
        f"N={degree}, n_seg={n_seg} stopped on trust-region collapse. This is the "
        f"pre-2026-08-11 regression: the iterate is critical (pred/|phi| ~ 1e-9 "
        f"against tol_f=1e-8) but the stationarity test rejected it for having a "
        f"noise-level NEGATIVE pred. Check the |pred| comparison in optimizer.rs."
    )
    assert stop in PRINCIPLED_STOPS, f"unexpected stop_reason {stop}"

    # The certificate must still hold -- a principled exit is only meaningful if
    # the point it stops at is one Proposition 1 certifies.
    assert float(info["final_hull_violation_km"]) <= 1e-6

    # And the answer itself must be untouched by the termination change.
    assert float(info["cost"]) == pytest.approx(expected_cost, rel=1e-6)


def test_uncertified_run_still_cannot_claim_stationarity():
    """The certificate guard must keep the exit from firing on an uncertified point.

    n_seg=2 is too coarse for the Proposition 1 condition to be satisfiable here,
    so it runs to the iteration cap. If this ever reports a principled stop, the
    `vlin_p <= 1e-6` guard on the stationarity test has been lost and the exit
    can no longer fail -- which would make the assertions above worthless.
    """
    sc = H.make_scenario("phase120", N=7)
    _, info = H.run_rust(sc, n_seg=2)

    assert float(info["final_hull_violation_km"]) > 1e-6, (
        "n_seg=2 now satisfies the certificate; this test no longer probes the guard"
    )
    assert int(info["scvx_stop_reason"]) not in PRINCIPLED_STOPS
