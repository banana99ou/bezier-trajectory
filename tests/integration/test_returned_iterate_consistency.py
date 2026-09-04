"""What the solver returns, and what it says about what it returns.

Two claims, both about the best-feasible fallback:

  (a) `converged` must describe the RETURNED point. The loop's verdict is about
      the reference it stopped on, and when the fallback fires that is a
      different trajectory from the one handed back.
  (b) "Best" must mean best by every guarantee in play. It meant largest
      keep-out clearance alone, so with a station present the fallback could
      return the very trajectory the occlusion constraint exists to exclude.
"""

import numpy as np
import pytest

from spacetime_bezier.optimize import optimize_spacetime
from spacetime_bezier.geometry import obstacle_array_bundle
from spacetime_bezier.scenarios import SCENARIO_MAP

bezier_opt = pytest.importorskip("bezier_opt")


# ---------------------------------------------------------------------------
# (a) converged describes the returned point
# ---------------------------------------------------------------------------


def test_the_fallback_fires_and_does_not_report_convergence():
    """The one configuration measured to trigger the fallback.

    `diverse` N8_seg4 starting at w=100 with max_iter=15: the loop's final
    iterate penetrates, the best feasible iterate seen is returned instead, and
    the certificate at the returned point differs from the loop's final
    reference -- two different trajectories. (Was max_iter=40 at a FIXED weight
    100; the in-loop weight escalation now converges that run outright --
    measured 2026-08-31: figure-grade in 40 iterations, one raise -- so the cap
    is tightened to stop before the escalation can rescue it.)

    WHAT THIS DOES AND DOES NOT PROVE. Here the loop had not converged anyway
    (stop_reason is the iteration cap), so the pairing was already consistent.
    The change is defensive, and the gap it closes is narrow but real: `converged`
    is set on a certificate of <= 1e-6, while the fallback triggers on a SAMPLED
    clearance < 0, so an iterate certified at 1e-7 whose 1500-sample clearance
    reads -1e-9 would have satisfied both at once. No natural case was found in
    a 339-configuration sweep; the invariant below is what stops one appearing
    unnoticed.
    """
    sc = SCENARIO_MAP["diverse"][0]()
    _, info = optimize_spacetime(
        N=8,
        dim=3,
        p_start=sc["start"],
        p_end=sc["end"],
        obstacles=sc["obstacles"],
        n_seg=4,
        max_iter=15,
        tol=1e-6,
        scp_trust_radius=0.5,
        min_dt=0.1,
        elastic_weight=100.0,
        init_curve=sc.get("init_curve"),
        verbose=False,
    )
    assert float(info["returned_best_iterate"]) == 1.0, (
        "the fallback did not fire, so this test measured nothing"
    )
    assert bool(info["converged"]) is False
    # The loop's own verdict is still available and still separate.
    assert "stop_reason" in info
    # And the two certificates really are different points.
    assert float(info["koz_violation_reference"]) != float(
        info["koz_violation_last_reference"]
    )


@pytest.mark.parametrize(
    "name,N,n_seg,weight,max_iter",
    [
        ("diverse", 8, 4, 100.0, 40),
        ("diverse", 8, 4, 100.0, 12),
        ("diverse", 8, 8, 100.0, 12),
        ("original", 8, 4, 100.0, 5),
        ("wall", 8, 2, 100.0, 12),
        ("fence3d", 8, 2, 100.0, 8),
    ],
)
def test_converged_is_never_reported_alongside_the_fallback(
    name, N, n_seg, weight, max_iter
):
    """The invariant: those two flags may not both be 1.

    FAILS IF `converged` goes back to being copied from the loop state without
    regard to which point is returned.
    """
    sc = SCENARIO_MAP[name][0]()
    _, info = optimize_spacetime(
        N=N,
        dim=len(sc["start"]),
        p_start=sc["start"],
        p_end=sc["end"],
        obstacles=sc["obstacles"],
        n_seg=n_seg,
        max_iter=max_iter,
        tol=1e-6,
        scp_trust_radius=0.5,
        min_dt=0.1,
        elastic_weight=weight,
        init_curve=sc.get("init_curve"),
        verbose=False,
    )
    assert not (
        bool(info["converged"]) and float(info["returned_best_iterate"]) == 1.0
    )


# ---------------------------------------------------------------------------
# (b) "best" accounts for line of sight when a station is present
# ---------------------------------------------------------------------------


def _blocked_scene():
    """A station scenario whose STRAIGHT SEED is deeply inside the shadow.

    **This is a fixture, not a registry scenario, and that is deliberate.** The
    test below is about `update_best`'s ordering, and to distinguish two
    orderings it needs a run that actually passes through iterates with the line
    of sight lost. `station_fence` used to supply that -- its fence sat between
    the station and the corridor for the middle of the horizon, so the
    unconstrained optimum was inside the shadow -- and it was removed 2026-08-30.
    **`loiter` cannot replace it here, measured not assumed:** its corridor is
    TANGENT to the shadow ring by design, so the run escapes in three iterations
    and the largest candidate violation ever seen is 1.3e-07, below the 1e-6
    threshold. Scanned 2026-08-31 at weights 1, 10, 100, 1000 at defaults and
    again on the priced run (free arrival, `time_weight` 10, `v_max` 5): zero
    uncertified candidates in every one of the eight runs.

    So the scene is built here, where it can be chosen to make the condition
    hold: station at the origin, a static body parked squarely on the sight line,
    and a straight seed flying through it. Measured at w=100: 30 steps, every one
    of them carrying an uncertified candidate, peak violation 71.4.
    """
    T = 20.0
    return {
        "start": [10.0, -10.0, 0.0],
        "end": [10.0, 10.0, T],
        "stations": [[0.0, 0.0]],
        # Parked at (5, 0). The sight segment from the origin to the vehicle runs
        # straight through it when the vehicle is near y = 0, so the seed loses
        # the link outright rather than grazing it.
        "obstacles": [
            {"pos0": [5.0, 0.0], "vel": [0.0, 0.0], "r": 2.0, "t_start": 0.0, "t_end": T}
        ],
    }


def _step_blocked(weight, n_steps, n_seg=8):
    """Drive the canonical iteration one step at a time, recording what it does.

    The stepping API and the batch loop call the same `scp_iterate`, so what is
    observed here is what the batch loop does.
    """
    sc = _blocked_scene()
    N = 8
    P0 = np.array(
        [np.linspace(a, b, N + 1) for a, b in zip(sc["start"], sc["end"])]
    ).T
    ctrl, radii = obstacle_array_bundle(sc["obstacles"], spatial_dim=len(sc["start"]) - 1)
    ctx = bezier_opt.SpacetimeScpContext(
        p_init=P0,
        obstacle_ctrl=ctrl,
        obstacle_r=radii,
        n_seg=n_seg,
        min_dt=0.1,
        coord_lb=-30.0,
        coord_ub=30.0,
        time_lb=0.0,
        time_ub=30.0,
        scp_trust_radius=1.0,
        elastic_weight=weight,
        tol=1e-6,
        stations=np.array(sc["stations"], dtype=float),
    )

    history = []
    previous_best = -np.inf
    for _ in range(n_steps):
        out = ctx.step()
        info = out[1]
        best = float(info["best_clearance"])
        history.append(
            {
                "advanced": best > previous_best,
                "vtrue_c": float(info["koz_violation_candidate"]),
                "accepted": bool(info["accepted"]),
                "outcome": info["outcome"],
                "best_clearance": best,
                # The CANDIDATE's own clearance -- what the old ordering ranked
                # on. `best_clearance` is the state's running best, which is
                # already gated and therefore cannot show what the old rule
                # would have selected.
                "candidate_clearance": float(info["clearance"]),
            }
        )
        previous_best = max(previous_best, best)
        if not info["running"]:
            break
    return history


def test_best_never_advances_onto_an_uncertified_candidate_when_a_station_exists():
    """`update_best` used to rank on keep-out clearance alone.

    The scene in `_blocked_scene` spends its iterations with the line of sight
    genuinely lost -- measured at w=100: all 30 steps carry an uncertified
    candidate, peak violation 71.4 -- while the keep-out clearance is comfortably
    positive and rising. Under the old ordering those iterates were exactly the
    ones "best" selected, so the fallback's answer would be a trajectory that
    clears every obstacle and cannot see the station.

    `vtrue_c` covers BOTH relaxable blocks, which is why it is the right test.

    FAILS IF the certificate condition is dropped from `update_best`: the early
    high-clearance, high-violation iterates advance `best_clearance` again.
    """
    history = _step_blocked(weight=100.0, n_steps=30)

    uncertified = [h for h in history if h["vtrue_c"] > 1e-6]
    assert uncertified, (
        "no step had an uncertified candidate, so this run cannot distinguish "
        "the two orderings"
    )
    advanced_uncertified = [h for h in uncertified if h["advanced"]]
    assert not advanced_uncertified, (
        "best_clearance advanced onto candidates with relaxable violations "
        f"{[h['vtrue_c'] for h in advanced_uncertified]}"
    )


def test_the_old_ordering_would_have_advanced_here():
    """The other half of the pair, so the test above is not vacuous.

    Reconstructs what keep-out-clearance-alone would have selected from the same
    recorded steps. If that ordering would never have advanced onto an
    uncertified candidate either, the assertion above proves nothing.
    """
    history = _step_blocked(weight=100.0, n_steps=30)

    best = -np.inf
    would_have_advanced = []
    for h in history:
        if not h["accepted"]:
            continue
        clearance = h["candidate_clearance"]
        # Under the old rule any accepted candidate with positive clearance
        # greater than the running best became `best_p`, certificate or not.
        if clearance > 0.0 and clearance > best:
            if h["vtrue_c"] > 1e-6:
                would_have_advanced.append(h["vtrue_c"])
            best = clearance

    assert would_have_advanced, (
        "the old ordering would not have advanced onto an uncertified candidate "
        "in this run either, so the constraint is untested"
    )


def test_a_run_without_stations_keeps_the_old_ordering():
    """No station means no line of sight to lose, so nothing may change.

    FAILS IF the certificate condition is applied unconditionally: `diverse`
    N8_seg4 from w=100 would stop returning its best feasible iterate, because that
    iterate carries a keep-out violation of 2.264 and would no longer qualify.
    """
    sc = SCENARIO_MAP["diverse"][0]()
    P, info = optimize_spacetime(
        N=8,
        dim=3,
        p_start=sc["start"],
        p_end=sc["end"],
        obstacles=sc["obstacles"],
        n_seg=4,
        max_iter=15,
        tol=1e-6,
        scp_trust_radius=0.5,
        min_dt=0.1,
        elastic_weight=100.0,
        init_curve=sc.get("init_curve"),
        verbose=False,
    )
    assert float(info["returned_best_iterate"]) == 1.0
    assert float(info["koz_violation_reference"]) > 1.0
