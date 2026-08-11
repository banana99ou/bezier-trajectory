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


def test_coarse_mesh_does_not_reach_a_principled_stop():
    """n_seg=2 must not reach a principled exit: it is nowhere near stationary.

    WHAT THIS DOES *NOT* TEST -- read before relying on it. An earlier version of
    this test claimed to probe the `vlin_p <= 1e-6` certificate guard on the
    stationarity exit. It does not, and an adversarial review (2026-08-11) proved
    it: rebuilding with `&& vlin_p <= 1e-6` DELETED from optimizer.rs leaves all
    three tests in this file green.

    The reason is measurable. Over n_seg=2's 1000 traced iterations on phase120,
    min |pred|/|phi| = 1.082e-05 against tol_f = 1e-8 -- the pred band never opens
    at all, on any iteration. The certificate is satisfied on 30 of those 1000
    iterations, but the two conditions co-occur on ZERO. So this case is gated
    entirely by the pred magnitude; the certificate guard is inert here and a
    regression in it would pass unnoticed.

    What this test DOES assert is still worth having: a mesh too coarse for the
    problem stays three orders outside the stationarity band, so the exit added
    above cannot fire on it. That is what keeps the two tests above from being
    vacuously true for every configuration.

    The certificate guard remains UNTESTED. Closing that gap needs a configuration
    that is model-stationary AND uncertified simultaneously; none is currently
    known.
    """
    sc = H.make_scenario("phase120", N=7)
    _, info = H.run_rust(sc, n_seg=2)

    assert float(info["final_hull_violation_km"]) > 1e-6, (
        "n_seg=2 now satisfies the certificate; re-derive what this test covers"
    )
    assert int(info["scvx_stop_reason"]) not in PRINCIPLED_STOPS
