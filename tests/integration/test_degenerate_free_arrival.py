"""A freed arrival time that no objective term prices is not a measurement.

`build_smoothness_regularizer` is blind to the time coordinate, and
``time_weight=0`` zeroes the only linear term. With ``free_arrival_time=True``
the arrival is therefore a free variable of a function that does not depend on
it, and whatever comes back is where the interior-point solver centred itself on
a flat optimal face. It reports `converged`, `stationary`, and a cost of ~1e-12,
which is exactly what a correct solve of a degenerate problem looks like.
"""

import numpy as np
import pytest

from spacetime_bezier.optimize import (
    DegenerateFreeArrivalError,
    check_free_arrival_is_costed,
    optimize_scenario,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import scenario_original

bezier_opt = pytest.importorskip("bezier_opt")


def test_the_degenerate_combination_is_refused():
    """FAILS IF the guard is removed.

    The call would then return a plausible trajectory whose arrival time is a
    property of the trust radius.
    """
    with pytest.raises(DegenerateFreeArrivalError):
        optimize_spacetime(
            N=8,
            dim=3,
            p_start=[0.0, 0.0, 0.0],
            p_end=[10.0, 0.0, 10.0],
            obstacles=[],
            n_seg=8,
            free_arrival_time=True,
            time_weight=0.0,
            verbose=False,
        )
    # A speed cap does not repair it: the cap bounds the arrival from below and
    # the objective is still flat above that bound.
    with pytest.raises(DegenerateFreeArrivalError):
        optimize_spacetime(
            N=8,
            dim=3,
            p_start=[0.0, 0.0, 0.0],
            p_end=[10.0, 0.0, 10.0],
            obstacles=[],
            n_seg=8,
            free_arrival_time=True,
            time_weight=0.0,
            v_max=4.0,
            verbose=False,
        )


def test_the_guard_is_about_the_combination():
    """Either half alone is legitimate and must stay runnable."""
    check_free_arrival_is_costed(False, 0.0)  # the default
    check_free_arrival_is_costed(True, 1.0)  # priced
    check_free_arrival_is_costed(False, 1.0)  # penalty on a pinned arrival
    with pytest.raises(DegenerateFreeArrivalError):
        check_free_arrival_is_costed(True, 0.0)


def test_the_objective_really_is_flat_in_the_arrival_time():
    """Evidence that the guard guards something, measured not assumed.

    Goes around the Python guard, straight to the Rust binding, on an
    obstacle-free problem. The trust radius is not part of the problem, so an
    answer that moves with it is not an answer.

    FAILS IF a future change makes the objective depend on the arrival time --
    which would be a legitimate change, and this test is what makes it a visible
    one rather than a silent repair of a guard nobody then needs.
    """
    P0 = np.array(
        [
            np.linspace(a, b, 9)
            for a, b in zip([0.0, 0.0, 0.0], [10.0, 0.0, 10.0])
        ]
    ).T

    arrivals = {}
    for trust in (0.1, 0.5, 2.0):
        _, info = bezier_opt.optimize_spacetime_bezier(
            p_init=P0,
            # No obstacles: the empty bundle shape, (n_obs, n_ctrl, dim).
            obstacle_ctrl=np.zeros((0, 2, 3)),
            obstacle_r=np.zeros((0,)),
            n_seg=8,
            max_iter=80,
            tol=1e-6,
            scp_prox_weight=0.0,
            scp_trust_radius=trust,
            elastic_weight=100.0,
            min_dt=0.1,
            coord_lb=-20.0,
            coord_ub=20.0,
            time_lb=0.0,
            time_ub=15.0,
            v_max=None,
            time_weight=0.0,
            free_arrival_time=True,
        )
        arrivals[trust] = float(info["arrival_time"])
        # The degeneracy is not visible in any success signal.
        assert bool(info["converged"]) is True
        assert abs(float(info["cost"])) < 1e-6

    spread = max(arrivals.values()) - min(arrivals.values())
    assert spread > 1.0, (
        f"the arrival barely moved with the trust radius ({arrivals}); the "
        "objective may no longer be flat in it"
    )


def test_the_default_configuration_is_bit_identical():
    """`free_arrival_time=False` + `time_weight=0` is the shipped default.

    The golden number is the one recorded before B8/B9/B10 landed. FAILS IF the
    new guard changed the default problem in any way -- it may not, because the
    default combination is precisely the one it does not refuse.
    """
    out = optimize_scenario(
        scenario_original(), [(8, 4)], elastic_weight=100.0, verbose=False
    )
    row = out["results"]["N8_seg4"]
    assert row["min_clearance"] == pytest.approx(0.6204432935456294, abs=1e-9)
    assert row["converged"] is True
    assert row["certificate_violation"] == pytest.approx(0.0, abs=1e-12)
