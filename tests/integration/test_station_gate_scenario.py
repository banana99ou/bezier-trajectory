"""The `station_gate` demo scenario.

Two things are being asserted, and they are different in kind.

**The pair.** Every line-of-sight claim comes in two runs, identical in
everything but the occlusion rows. A claim that also holds for the baseline is
not evidence about the constraint -- PAPER_1 sec. "The demo scenario and its
figure", "the baseline that must be able to fail".

**The component count.** This is the only scenario in the repository whose
clipped keep-out volume splits into two connected components, so it is the only
one that exercises the per-component wall on a real run rather than on a
synthetic unit test. That is asserted on the STRAIGHT SEED, not on the returned
trajectory, and the reason is written into the scenario's own docstring: the
occlusion rows drive the vehicle up and away from the body, and a segment that
far away has a clip ball reaching only one leg. Asserting it on the converged
iterate would fail, and asserting it on nothing would let the claim rot.

The line-of-sight numbers come from `compute_los_margin`, which samples the true
obstacle trajectories and never calls the Rust shadow builder.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("bezier_opt")

from orbital_docking.de_casteljau import segment_matrices_equal_params  # noqa: E402
from spacetime_bezier.geometry import (  # noqa: E402
    compute_los_margin,
    compute_min_clearance,
)
from spacetime_bezier.optimize import (  # noqa: E402
    DEFAULT_TRUST_RADIUS,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import (  # noqa: E402
    SCENARIO_ELASTIC_WEIGHT,
    SCENARIO_MAP,
    scenario_station_gate,
)

N_DEGREE = 8
N_SEG = 8
ELASTIC_WEIGHT = SCENARIO_ELASTIC_WEIGHT["station_gate"]

CRUISE_Z = 0.5
BODY_TOP_Z = 0.45 + 0.8          # centre altitude plus radius


def _solve(scenario, stations):
    P, info = optimize_spacetime(
        N=N_DEGREE,
        dim=4,
        p_start=scenario["start"],
        p_end=scenario["end"],
        obstacles=scenario["obstacles"],
        n_seg=N_SEG,
        max_iter=200,
        elastic_weight=ELASTIC_WEIGHT,
        stations=stations,
        verbose=False,
        init_curve=scenario["init_curve"],
    )
    return np.asarray(P, dtype=float), dict(info)


@pytest.fixture(scope="module")
def runs():
    """The pair: occlusion rows off, occlusion rows on. Nothing else differs."""
    scenario = scenario_station_gate()
    out = {"scenario": scenario}
    for label, stations in (("baseline", None), ("constrained", scenario["stations"])):
        P, info = _solve(scenario, stations)
        t_values, margins = compute_los_margin(
            P, scenario["stations"][0], scenario["obstacles"], dim=4, n_eval=3000
        )
        out[label] = {
            "P": P,
            "info": info,
            "t": t_values,
            "margin": margins,
            "clearance": compute_min_clearance(
                P, scenario["obstacles"], dim=4, n_eval=3000
            ),
        }
    return out


def _lost_interval(entry):
    """Length of the time interval over which the link is down."""
    finite = np.isfinite(entry["margin"])
    lost = finite & (entry["margin"] < 0.0)
    if not lost.any():
        return 0.0
    t = entry["t"][lost]
    return float(t.max() - t.min())


# ---------------------------------------------------------------------------
# The scenario definition
# ---------------------------------------------------------------------------


def test_scenario_is_three_spatial_dimensions_plus_time():
    scenario = scenario_station_gate()
    assert len(scenario["start"]) == 4
    assert len(scenario["end"]) == 4
    assert len(scenario["stations"][0]) == 3
    assert "station_gate" in SCENARIO_MAP


def test_the_occluder_is_one_curved_bezier_not_a_chain_of_straight_pieces():
    """FAILS IF the occluder degenerates to a straight or piecewise-straight path.

    A straight obstacle sweeps a CONVEX capsule in the lifted space, so a plane
    touching it anywhere already supports the whole tube and the clipped-volume
    construction is reproducing an easier answer. `station_fence` writes its
    occluder as a chain of straight pieces because the pre-2026-08-21 builder
    needed every piece convex; this one must not.
    """
    scenario = scenario_station_gate()
    for obstacle in scenario["obstacles"]:
        cps = np.asarray(obstacle["control_points"], dtype=float)
        assert cps.shape[0] >= 4, "a straight or quadratic path is not the case under test"
        # No t_start/t_end: the window is intrinsic to the control point times.
        assert "t_start" not in obstacle and "t_end" not in obstacle
        # The centreline must actually BEND -- collinear control points would be
        # a straight line written with extra points.
        legs = np.diff(cps, axis=0)
        units = legs / np.linalg.norm(legs, axis=1, keepdims=True)
        assert not np.allclose(units, units[0], atol=1e-6), "centreline is straight"


def test_the_time_column_is_affine_in_the_curve_parameter():
    """FAILS IF the control point times are unevenly spaced.

    `geometry.obstacle_positions_at` inverts time by division, which is only the
    right inversion when the time coordinate is affine in the parameter. Uneven
    times would put every obstacle position the certificate reads at the wrong
    instant, silently.
    """
    for obstacle in scenario_station_gate()["obstacles"]:
        times = np.asarray(obstacle["control_points"], dtype=float)[:, -1]
        assert np.allclose(np.diff(times), np.diff(times)[0])


def test_the_body_never_threatens_the_corridor():
    """The keep-out rows are present and are not expected to bind.

    Same design as `station_fence`, and stated for the same reason: staying
    visible already implies staying out of the body, so this scenario is
    evidence about occlusion and not about collision avoidance. If the gap ever
    closes, the two constraints start competing and the figure stops being clean.
    """
    scenario = scenario_station_gate()
    corridor_y = scenario["start"][1]
    for obstacle in scenario["obstacles"]:
        cps = np.asarray(obstacle["control_points"], dtype=float)
        us = np.linspace(0.0, 1.0, 2001)[:, None]
        basis = np.hstack([
            (1 - us) ** 4, 4 * us * (1 - us) ** 3, 6 * us**2 * (1 - us) ** 2,
            4 * us**3 * (1 - us), us**4,
        ])
        gap = np.abs((basis @ cps)[:, 1] - corridor_y).min()
        assert gap > obstacle["radius"] + 0.5, (
            f"{obstacle['name']} comes within {gap:.3f} of the corridor "
            f"against a radius of {obstacle['radius']}"
        )


# ---------------------------------------------------------------------------
# The two-component claim -- the only scenario that makes it
# ---------------------------------------------------------------------------


def test_the_clipped_volume_splits_in_two_on_the_straight_seed():
    """FAILS IF no (segment, obstacle) pair sees two legs of the sweep at once.

    This is what the scenario is for. The clip ball around a segment centroid in
    the middle of the horizon reaches the occluder's approaching leg and its
    departing leg, which are disjoint in the curve parameter and separated in
    TIME rather than in space. Fusing them into one interval -- the construction
    this replaces -- makes the count structurally impossible.

    Checked here on the geometry alone, without the solver: the distance from
    the centroid to the centreline must dip below the ball's reach at two
    separated parameter stretches. A single basin means one component and the
    scenario has stopped testing what it exists to test.
    """
    scenario = scenario_station_gate()
    start = np.asarray(scenario["start"], dtype=float)
    end = np.asarray(scenario["end"], dtype=float)
    seed = np.array([start + (end - start) * i / N_DEGREE for i in range(N_DEGREE + 1)])
    a_list = segment_matrices_equal_params(N_DEGREE, N_SEG)

    us = np.linspace(0.0, 1.0, 4001)[:, None]
    basis = np.hstack([
        (1 - us) ** 4, 4 * us * (1 - us) ** 3, 6 * us**2 * (1 - us) ** 2,
        4 * us**3 * (1 - us), us**4,
    ])

    split = 0
    for a in a_list:
        Q = np.asarray(a, dtype=float) @ seed
        centroid = Q.mean(axis=0)
        seg_radius = float(np.linalg.norm(Q - centroid, axis=1).max())
        for obstacle in scenario["obstacles"]:
            r_m = float(obstacle["radius"])
            centreline = basis @ np.asarray(obstacle["control_points"], dtype=float)
            dist = np.linalg.norm(centreline - centroid, axis=1)
            r_cap = r_m + seg_radius + DEFAULT_TRUST_RADIUS * np.sqrt(4.0)
            r_clip = min(float(dist.min()), r_cap)
            inside = dist <= r_clip + r_m
            runs_of_true = int(inside[0]) + int((np.diff(inside.astype(int)) == 1).sum())
            if runs_of_true >= 2:
                split += 1
    assert split >= 1, (
        "no (segment, obstacle) pair has a clipped volume in two pieces -- this "
        "scenario is the only one that is supposed to, and it no longer does"
    )


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


def test_baseline_loses_line_of_sight_for_a_measurable_interval(runs):
    """The constraint must be able to bind. Without it the link breaks."""
    lost = _lost_interval(runs["baseline"])
    assert lost > 1.0, (
        f"baseline keeps line of sight (lost for {lost:.2f}s) -- the constraint "
        "was slack and the pair proves nothing"
    )
    assert runs["baseline"]["margin"][np.isfinite(runs["baseline"]["margin"])].min() < -0.1


def test_constrained_run_keeps_line_of_sight_everywhere(runs):
    entry = runs["constrained"]
    finite = np.isfinite(entry["margin"])
    assert entry["margin"][finite].min() > 0.0
    assert _lost_interval(entry) == 0.0


def test_the_climb_exists_and_is_caused_by_the_occlusion_rows(runs):
    """FAILS IF the baseline climbs too -- then the climb is not the constraint's.

    The third spatial coordinate is load-bearing only if removing the occlusion
    rows removes the climb. Co-located is not causal.
    """
    baseline_z = float(runs["baseline"]["P"][:, 2].max())
    constrained_z = float(runs["constrained"]["P"][:, 2].max())
    assert baseline_z < CRUISE_Z + 0.2, (
        f"the baseline climbs to {baseline_z:.2f} on its own -- the climb is not "
        "evidence about occlusion"
    )
    assert constrained_z > BODY_TOP_Z + 0.5, (
        f"the constrained run only reaches {constrained_z:.2f}, which does not "
        f"clear the body top at {BODY_TOP_Z:.2f}"
    )


def test_constrained_run_certifies_and_the_keep_out_rows_do_not_bind(runs):
    """Figure-grade, and the clearance is against the TRUE obstacle trajectories."""
    info = runs["constrained"]["info"]
    assert info["converged"]
    # Both certificates are evaluated at the RETURNED iterate, not at the loop's
    # final reference point -- they are different trajectories whenever the
    # best-iterate fallback fires.
    assert float(info["occlusion_violation_reference"]) <= 1e-6
    assert float(info["koz_violation_reference"]) <= 1e-6
    assert float(info["total_koz_slack_returned"]) <= 1e-6
    assert runs["constrained"]["clearance"] > 0.0
    # Present but never binding -- the design, stated in the scenario docstring.
    assert runs["baseline"]["clearance"] > 0.0
