"""The `station_fence` demo scenario (item B12).

idea/spacetime.md sec. "The demo scenario and its figure" states what this scenario has to
do, and the sentence that matters is **"the baseline that must be able to fail"**:

    Run the same scenario with the occlusion rows removed: it must lose contact
    for a measurable interval. If the baseline also keeps line of sight, the
    constraint was slack and the figure proves nothing.

So every assertion below comes in a pair. Two runs, identical in everything but
one constraint block, and each claim is checked against both. A claim that holds
for the baseline too is not evidence about the constraint.

The line-of-sight numbers come from `compute_los_margin`, which samples the true
obstacle trajectories and never calls the Rust shadow builder. The solver's own
certificate is checked separately; the two are independent computations of the
same physical fact and are allowed to disagree, which is the point of having
both.
"""

from __future__ import annotations

import numpy as np
import pytest

bezier_opt = pytest.importorskip("bezier_opt")

from orbital_docking.de_casteljau import segment_matrices_equal_params  # noqa: E402
from spacetime_bezier.geometry import (  # noqa: E402
    bezier_curve,
    compute_los_margin,
    compute_min_clearance,
)
from spacetime_bezier.optimize import (  # noqa: E402
    figure_grade_failures,
    is_figure_grade,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import (  # noqa: E402
    SCENARIO_ELASTIC_WEIGHT,
    SCENARIO_MAP,
    scenario_station_fence,
)

# The measured configuration. `station_fence` is registered with this pair plus
# a 16-segment one; the 8-segment run is the cheaper of the two that reach a zero
# occlusion certificate, so it is what the tests grade.
N_DEGREE = 8
N_SEG = 8
ELASTIC_WEIGHT = SCENARIO_ELASTIC_WEIGHT["station_fence"]

# The fence body: centre altitude plus radius, from `scenario_station_fence`.
FENCE_TOP_Z = 0.1 + 0.8


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
    scenario = scenario_station_fence()
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
            "curve": bezier_curve(P, 3000),
            "clearance": compute_min_clearance(
                P, scenario["obstacles"], dim=4, n_eval=3000
            ),
        }
    return out


def _occlusion_violation_at(P, scenario):
    """Occlusion certificate recomputed in Python from the control points alone.

    Needed for the BASELINE, whose solver run was told about no station and so
    reports zero by construction. Asking what its trajectory would score against
    the rows it never saw is the only way to grade the two runs on one scale.
    """
    obstacles = scenario["obstacles"]
    out = bezier_opt.spacetime_occlusion_rows_exact(
        p=P,
        obstacle_pos0=np.array([o["pos0"] for o in obstacles], dtype=float),
        obstacle_vel=np.array([o["vel"] for o in obstacles], dtype=float),
        obstacle_r=np.array([o["r"] for o in obstacles], dtype=float),
        stations=np.array(scenario["stations"], dtype=float),
        obstacle_t_start=np.array([o.get("t_start", -1e18) for o in obstacles], dtype=float),
        obstacle_t_end=np.array([o.get("t_end", 1e18) for o in obstacles], dtype=float),
        n_seg=N_SEG,
    )
    margins = np.asarray(out[10], dtype=float)
    return float(np.maximum(0.0, -margins).sum()), margins


def _koz_margins(P, scenario):
    """Margin of every EXACT keep-out row at ``P``, recomputed from scratch."""
    obstacles = scenario["obstacles"]
    normals, lbs, seg, cp, _obs = bezier_opt.spacetime_koz_rows_exact(
        p=P,
        obstacle_pos0=np.array([o["pos0"] for o in obstacles], dtype=float),
        obstacle_vel=np.array([o["vel"] for o in obstacles], dtype=float),
        obstacle_r=np.array([o["r"] for o in obstacles], dtype=float),
        obstacle_t_start=np.array([o.get("t_start", -1e18) for o in obstacles], dtype=float),
        obstacle_t_end=np.array([o.get("t_end", 1e18) for o in obstacles], dtype=float),
        n_seg=N_SEG,
    )
    normals = np.asarray(normals, dtype=float).reshape(-1, P.shape[1])
    lbs = np.asarray(lbs, dtype=float)
    a_list = segment_matrices_equal_params(P.shape[0] - 1, N_SEG)
    return np.array([
        float(normals[i] @ (np.asarray(a_list[int(seg[i])], dtype=float) @ P)[int(cp[i])])
        - float(lbs[i])
        for i in range(len(lbs))
    ])


# ---------------------------------------------------------------------------
# The scenario definition
# ---------------------------------------------------------------------------


def test_scenario_is_three_spatial_dimensions_plus_time():
    scenario = scenario_station_fence()
    assert len(scenario["start"]) == 4
    assert len(scenario["end"]) == 4
    assert len(scenario["stations"][0]) == 3
    assert all(len(o["pos0"]) == 3 for o in scenario["obstacles"])
    assert "station_fence" in SCENARIO_MAP


def test_the_occluder_path_is_a_chain_of_straight_pieces_with_overlapping_caps():
    """FAILS IF the two pieces leave a gap in time, or bend the same capsule.

    idea/spacetime.md sec. "Occluder geometry": a capsule around a CURVED centreline is not
    convex and the certificate would have nothing to stand on. The remedy is a
    chain of straight pieces on adjacent windows with the caps overlapping at the
    joint, so the union covers the swept region with no seam.
    """
    scenario = scenario_station_fence()
    pieces = {}
    for obstacle in scenario["obstacles"]:
        pieces.setdefault(obstacle["name"][0], []).append(obstacle)
    assert set(pieces) == {"A", "B"}, "the fence must be a two-piece chain"

    a_window = (pieces["A"][0]["t_start"], pieces["A"][0]["t_end"])
    b_window = (pieces["B"][0]["t_start"], pieces["B"][0]["t_end"])
    assert b_window[0] < a_window[1], "the windows must OVERLAP, not merely abut"

    # Each piece is straight -- one constant velocity -- and the two velocities
    # differ, so the chain describes a genuinely non-straight path.
    vel_a = np.array(pieces["A"][0]["vel"], dtype=float)
    vel_b = np.array(pieces["B"][0]["vel"], dtype=float)
    assert not np.allclose(vel_a, vel_b)
    for piece, vel in (("A", vel_a), ("B", vel_b)):
        for obstacle in pieces[piece]:
            assert np.allclose(np.array(obstacle["vel"], dtype=float), vel)

    # Continuous at the joint: both pieces put the same fence in the same place
    # in the middle of the overlap.
    t_joint = 0.5 * (b_window[0] + a_window[1])
    for oa, ob in zip(pieces["A"], pieces["B"]):
        pa = np.array(oa["pos0"], dtype=float) + vel_a * t_joint
        pb = np.array(ob["pos0"], dtype=float) + vel_b * t_joint
        assert np.linalg.norm(pa - pb) < oa["r"], (
            "the caps must overlap at the joint, or the swept region has a hole"
        )


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


def test_baseline_loses_line_of_sight_for_a_measurable_interval(runs):
    """The constraint must be able to bind. Without it, the link breaks.

    If this passes only because the solver happened to fly somewhere unlucky, the
    scenario is decoration. It passes because the fence is placed between the
    station and the flight corridor for the middle of the horizon, so the
    unconstrained optimum -- which has no reason to leave the flight altitude --
    is inside the shadow.
    """
    baseline = runs["baseline"]
    assert bool(baseline["info"]["converged"]), "the baseline must be a real solution"
    assert baseline["clearance"] > 0.0

    negative = baseline["margin"] < 0.0
    assert negative.any(), "the baseline keeps line of sight; the constraint is slack"
    interval = float(baseline["t"][negative].max() - baseline["t"][negative].min())
    assert interval > 5.0, f"blocked for only {interval:.3f} s -- not a measurable interval"
    assert baseline["margin"].min() < -0.3

    # And it is genuinely a violation of the rows, not merely of the sampled
    # geometry: the trajectory the baseline returns fails the occlusion
    # certificate outright.
    violation, _ = _occlusion_violation_at(baseline["P"], runs["scenario"])
    assert violation > 1.0


def test_constrained_run_keeps_line_of_sight_everywhere(runs):
    """The claim, measured against the true geometry rather than the solver."""
    constrained = runs["constrained"]
    assert bool(constrained["info"]["converged"])
    assert constrained["margin"].min() > 0.0
    assert constrained["clearance"] > 0.0

    # The two independent legs must agree. `compute_los_margin` samples the true
    # obstacle trajectories; the certificate is rebuilt from the control points
    # against the convex outer approximation. Both say visible.
    violation, _ = _occlusion_violation_at(constrained["P"], runs["scenario"])
    assert violation == pytest.approx(0.0, abs=1e-6)
    assert float(constrained["info"]["occlusion_violation_reference"]) == pytest.approx(
        violation, abs=1e-6
    )


def test_constrained_run_is_figure_grade_and_the_baseline_is_not(runs):
    """The gate has to separate them, and on the occlusion condition alone.

    The baseline is converged, certified against the keep-out rows, clearing and
    standing on no slack -- it passes every condition the gate had before item
    B12. Only the occlusion certificate tells them apart, which is why that
    condition had to be added rather than folded into the keep-out one.
    """
    scenario = runs["scenario"]
    rows = {}
    for label in ("baseline", "constrained"):
        run = runs[label]
        violation, _ = _occlusion_violation_at(run["P"], scenario)
        rows[label] = {
            "converged": bool(run["info"]["converged"]),
            "stop_label": str(int(run["info"]["stop_reason"])),
            "certificate_violation": float(run["info"]["koz_violation_reference"]),
            "min_clearance": float(run["clearance"]),
            "total_slack": float(run["info"]["total_koz_slack_returned"]),
            "occlusion_violation": violation,
        }

    assert is_figure_grade(rows["constrained"]), figure_grade_failures(rows["constrained"])

    assert not is_figure_grade(rows["baseline"])
    reasons = figure_grade_failures(rows["baseline"])
    assert reasons == [r for r in reasons if "line of sight" in r], (
        f"the baseline must fail on occlusion and nothing else, got {reasons}"
    )


def test_the_climb_exists_and_is_caused_by_the_occlusion_rows(runs):
    """The third spatial dimension is load-bearing, not a rendering choice.

    The constrained run climbs above the fence; the baseline does not leave the
    flight altitude at all. The difference between the two runs is one constraint
    block, so the climb is attributable to it.
    """
    baseline_z = runs["baseline"]["curve"][:, 2]
    constrained_z = runs["constrained"]["curve"][:, 2]
    start_z = runs["scenario"]["start"][2]

    assert constrained_z.max() > FENCE_TOP_Z, (
        f"peak altitude {constrained_z.max():.4f} does not clear the fence top "
        f"{FENCE_TOP_Z:.4f}"
    )
    assert constrained_z.max() > baseline_z.max() + 1.0
    assert baseline_z.max() == pytest.approx(start_z, abs=1e-6), (
        "the baseline must have no reason to climb"
    )


def test_keep_out_rows_are_present_but_never_bind(runs):
    """Occlusion subsumes collision for this body, and the rows show it.

    A sight line that starts inside the occluder is blocked by definition, so a
    trajectory satisfying the occlusion rows is outside the body as well. The
    keep-out rows are still built and still checked -- they are what would catch
    the claim being wrong -- but in this scenario they sit far from active while
    the occlusion rows sit exactly on their bound.

    Stated as a comparison rather than a threshold: the tightest keep-out row has
    an order of magnitude more room than the tightest occlusion row, which is at
    zero.
    """
    scenario = runs["scenario"]
    constrained = runs["constrained"]

    koz = _koz_margins(constrained["P"], scenario)
    assert koz.size > 0, "the keep-out rows must actually be present"
    assert koz.min() > 0.5, f"a keep-out row is nearly active at {koz.min():.4f}"

    _, occlusion = _occlusion_violation_at(constrained["P"], scenario)
    assert occlusion.size > 0
    # Active to within interior-point residue: Clarabel approaches the bound
    # from the feasible side and never lands exactly on it.
    assert occlusion.min() == pytest.approx(0.0, abs=1e-4), (
        "the occlusion block must be the ACTIVE one, or the run was not shaped by it"
    )
    assert koz.min() > 100.0 * max(occlusion.min(), 1e-3)

    # The same holds for the true sampled geometry: the fence surface is never
    # approached, while the line of sight is the quantity running close to its
    # limit.
    assert constrained["clearance"] > 3.0 * constrained["margin"].min()
