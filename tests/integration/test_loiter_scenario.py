"""The `loiter` demo scenario, graded as a PAIR (item B12).

`idea/spacetime.md` states what this scenario has to do, and the sentence that
matters is **"the baseline that must be able to fail"**:

    Run the same scenario with the occlusion rows removed: it must lose contact
    for a measurable interval. If the baseline also keeps line of sight, the
    constraint was slack and the figure proves nothing.

So every assertion below comes in a pair. Two runs, identical in everything but
one constraint block, and each claim is checked against both. A claim that holds
for the baseline too is not evidence about the constraint.

**This file exists because removing `station_fence` lost coverage.** That scenario
carried the pair structure; `loiter` replaced it as the paper's demo and inherited
the frontend, plane-drop and obstacle-format coverage, but not this. Enumerated at
the time of the merge: of the seven tests `station_fence` carried, four had
equivalents, one asserted the retired chain-of-straight-pieces design and was
correctly dropped, and two were lost — the baseline-must-fail pair and the claim
that the TIMING is what saves the run. Those two are what this file restores.

**Why the assertions are on `compute_los_margin` and never on the baseline's own
certificate.** The baseline is solved with `stations=None`, so it builds no shadow
rows, so `occlusion_violation_reference` comes back 0.0 BY CONSTRUCTION for a
trajectory that loses the link outright. That false zero is asserted below rather
than merely avoided, so that nobody later "simplifies" this file by reading the
number the solver reports. `compute_los_margin` samples the true obstacle
trajectories and knows nothing about shadows, half-spaces or the outer
approximation; it is the independent leg, and the certificate is what is under
test.

Numbers measured 2026-08-31 on the post-merge build, at the paper figure's own
configuration (N=8, 8 segments, free arrival, time_weight 10, v_max 5, the reach
floor on, elastic weight 1e5 from the registry).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("bezier_opt")

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spacetime_bezier.geometry import (  # noqa: E402
    bezier_curve,
    compute_los_margin,
    compute_min_clearance,
    los_margin_at,
)
from spacetime_bezier.optimize import figure_grade_failures  # noqa: E402


def _make_paper_figure():
    """The FIGURE's own `solve_pair`, not a copy of it.

    `tools/` is not a package, so it is loaded by path. Importing it rather than
    reimplementing the pair is deliberate: the run this file grades is then the
    run the figure is drawn from, and a drift in the tool's parameters — the
    trust radius it forwards, the weight it takes from the registry, the fact
    that the baseline differs by the occlusion rows and nothing else — fails here
    instead of silently producing a figure nobody re-checked.
    """
    spec = importlib.util.spec_from_file_location(
        "_mpf_for_tests", REPO / "tools" / "make_paper_figure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The paper figure's configuration. `sound_clip` is the default now, and named
# anyway: a run whose certificate covers only the clipped pieces is not what the
# figure claims.
N_DEGREE, N_SEG = 8, 8
TIME_WEIGHT, V_MAX = 10.0, 5.0


@pytest.fixture(scope="module")
def runs():
    """The pair: occlusion rows off, occlusion rows on. Nothing else differs."""
    mpf = _make_paper_figure()
    sc, weight, (P_con, info_con), (P_base, info_base) = mpf.solve_pair(
        "loiter", N_DEGREE, N_SEG,
        free_arrival=True, time_weight=TIME_WEIGHT, v_max=V_MAX, sound_clip=True,
    )
    dim = len(sc["start"])
    station = sc["stations"][0]
    out = {"scenario": sc, "weight": weight, "dim": dim, "station": station}
    for label, P, info in (("constrained", P_con, info_con), ("baseline", P_base, info_base)):
        t, m = compute_los_margin(P, station, sc["obstacles"], dim=dim, n_eval=2001)
        out[label] = {
            "P": np.asarray(P, dtype=float),
            "info": dict(info),
            "t": t,
            "margin": m,
            # The same 4001-point sampling the figure grafts on, so the numbers
            # here and in the sidecar are the same numbers.
            "fine": bezier_curve(np.asarray(P, dtype=float), num_pts=4001),
            "clearance": compute_min_clearance(P, sc["obstacles"], dim, 20001),
        }
    return out


def _graft(runs, path_from: str, schedule_from: str) -> float:
    """Minimum line-of-sight margin of one run's PATH on the other's SCHEDULE.

    Same curve parameter, the other run's time coordinate. Nothing from the
    solver but the control points.
    """
    dim = runs["dim"]
    return float(np.min(los_margin_at(
        runs[path_from]["fine"][:, :dim - 1],
        runs[schedule_from]["fine"][:, -1],
        runs["station"],
        runs["scenario"]["obstacles"],
    )))


# ---------------------------------------------------------------------------
# The pair is a pair
# ---------------------------------------------------------------------------


def test_the_pair_differs_by_the_occlusion_rows_and_nothing_else(runs):
    """Neither half may be a failed solve, and the baseline's number lies.

    A baseline that did not converge would make "the baseline loses the link" a
    statement about a broken run rather than about the constraint.

    And its reported occlusion certificate is 0.0 for a trajectory that spends
    1.56 s blocked, because with `stations=None` there are no shadow rows to
    violate. That false zero is asserted rather than merely avoided, so that
    nobody later "simplifies" this file by reading the number the solver reports.

    FAILS IF: either half stops converging or starts penetrating, or the baseline
    ever begins building shadow rows -- in which case the pair no longer differs
    by one block and every other test here compares two different problems.
    """
    for label in ("constrained", "baseline"):
        assert bool(runs[label]["info"]["converged"]), f"{label} is not a solution"
        assert runs[label]["clearance"] > 0.0, f"{label} penetrates a keep-out zone"

    base = runs["baseline"]["info"]
    assert float(base["occlusion_violation_reference"]) == pytest.approx(0.0, abs=1e-12), (
        "the baseline built shadow rows; it is no longer the occlusion-free half"
    )
    assert runs["baseline"]["margin"].min() < 0.0, (
        "the baseline's certificate reads 0.0 AND its true margin is non-negative "
        "-- then the false zero this file warns about is not demonstrable here"
    )


# ---------------------------------------------------------------------------
# The baseline that must be able to fail
# ---------------------------------------------------------------------------


def test_the_baseline_loses_line_of_sight_for_a_measurable_interval(runs):
    """Without the occlusion rows the link breaks, measurably and for a reason.

    Measured: blocked over [14.960, 16.520] s, 1.560 s wide, minimum margin
    -2.1695 m. It is not luck — the corridor is TANGENT to the shadow ring at the
    transit altitude, so the spot arrives moving ALONG the corridor and sits on
    the path instead of crossing it, and the corridor's 5 m half-width is
    narrower than the spot's 7.5 m radius, so no lateral offset the band allows
    can clear it.

    Asserted on `compute_los_margin`, never on the certificate: see the module
    docstring.

    FAILS IF: the baseline keeps the link (the constraint would be decoration and
    the figure would prove nothing), or the blocked interval collapses to a
    graze.
    """
    base = runs["baseline"]
    negative = base["margin"] < 0.0
    assert negative.any(), (
        "the baseline keeps line of sight everywhere; the occlusion rows are "
        "decoration in this configuration and the figure proves nothing"
    )
    interval = float(base["t"][negative].max() - base["t"][negative].min())
    assert interval > 1.0, f"blocked for only {interval:.3f} s -- a graze, not an interval"
    assert base["margin"].min() < -1.0, (
        f"deepest block is {base['margin'].min():+.4f}; measured -2.1695"
    )


def test_the_constrained_run_holds_the_link_and_both_computations_agree(runs):
    """The certificate says clear and the independent sampling agrees.

    Two independent computations of one physical fact. They are allowed to
    disagree, and the disagreement would be the finding; here they must not.

    FAILS IF: the occlusion certificate leaves zero, a plane could not be built
    (uncertifiable is not certified), a clipped volume escapes the wall built
    against it, the independent margin goes negative while the certificate stays
    at zero, or the run stops being figure-grade.
    """
    con = runs["constrained"]
    info = con["info"]

    assert float(info["occlusion_violation_reference"]) == pytest.approx(0.0, abs=1e-6)
    assert float(info["occlusion_planes_dropped"]) == 0.0
    assert float(info["koz_unsound_clips"]) == 0.0
    assert con["margin"].min() > 0.0, (
        f"the certificate reads zero but the true margin is {con['margin'].min():+.4f} "
        "-- one of the two computations is wrong, and that is the finding"
    )
    # Measured +10.6424, clearing the body by 30.5243 m.
    assert con["margin"].min() > 5.0
    assert con["clearance"] > 10.0

    row = {
        "converged": bool(info["converged"]),
        "stop_label": "stationary",
        "certificate_violation": float(info["koz_violation_reference"]),
        "occlusion_violation": float(info["occlusion_violation_reference"]),
        "occlusion_planes_dropped": float(info["occlusion_planes_dropped"]),
        "koz_unsound_clips": float(info["koz_unsound_clips"]),
        "min_clearance": float(con["clearance"]),
        "total_slack": float(info["total_koz_slack_returned"]),
        "speed_cap_violation": float(info["speed_cap_violation"]),
    }
    assert figure_grade_failures(row) == [], figure_grade_failures(row)


# ---------------------------------------------------------------------------
# Timing is the decision variable -- the schedule graft
# ---------------------------------------------------------------------------


def test_the_retiming_is_what_saves_the_run_not_the_path(runs):
    """Graft each run's SPATIAL PATH onto the other's SCHEDULE and re-measure.

    This is the claim the paper makes about timing, and it is the one that would
    be worthless if the solver had simply flown around the shadow instead. The
    two grafts settle it in opposite directions:

      constrained PATH on the baseline SCHEDULE  ->  -2.1086, link lost
      baseline PATH on the constrained SCHEDULE  ->  +10.6198, link held

    The path is interchangeable and the schedule is not. Measured 2026-08-31; the
    constrained run deviates 0.0783 m laterally over a 200 m corridor, which is
    why the first graft fails: there is no sidestep to inherit.

    THREE NON-VACUITY GUARDS, because a graft is exactly the kind of measurement
    that can quietly stop being a measurement:

      1. the two schedules must genuinely differ, or both grafts are the run
         itself and both assertions hold trivially;
      2. the graft must produce no NaN and must still be MEASURING -- an
         all-`+inf` graft means no obstacle was ever active and the comparison
         below would be about nothing (`+inf` at a window boundary is defined and
         expected; NaN is not);
      3. the constrained path must NOT have sidestepped, or "the retiming saved
         it" is unsupported even when the graft happens to fail.

    FAILS IF: any guard trips, or the direction of either graft reverses.
    """
    con, base = runs["constrained"], runs["baseline"]

    # Guard 1 -- measured arrivals 55.7470 against 40.0000.
    arrival_con = float(con["info"]["arrival_time"])
    arrival_base = float(base["info"]["arrival_time"])
    assert abs(arrival_con - arrival_base) > 5.0, (
        f"the two schedules are {abs(arrival_con - arrival_base):.4f} s apart; "
        "the graft cannot distinguish path from timing when they nearly coincide"
    )
    assert arrival_con > arrival_base, "the constrained run should WAIT, not hurry"

    # Guard 2 -- the graft is computed, on the whole sampling.
    dim = runs["dim"]
    grafted = los_margin_at(
        con["fine"][:, :dim - 1], base["fine"][:, -1],
        runs["station"], runs["scenario"]["obstacles"],
    )
    assert len(grafted) == con["fine"].shape[0] == 4001
    # `+inf` is DEFINED here and is not a failure: `compute_los_margin` reports it
    # when no obstacle is active at that sample, and the first sample of this
    # scene lands on the first arc's window boundary. NaN is undefined and would
    # satisfy neither comparison below while being reported as neither, so it is
    # rejected outright; and a graft that degenerated to all-inf would be no
    # measurement at all, which the coverage floor catches.
    assert not np.isnan(grafted).any(), "the graft produced NaN margins"
    finite = float(np.isfinite(grafted).mean())
    assert finite > 0.99, (
        f"only {finite:.1%} of the graft samples have an active obstacle; the "
        "graft has stopped measuring anything"
    )

    # Guard 3 -- the path did not move, so it cannot be what saved the run.
    axis_y = float(runs["scenario"]["start"][1])
    lateral = float(np.max(np.abs(con["fine"][:, 1] - axis_y)))
    assert lateral < 1.0, (
        f"the constrained run sidestepped {lateral:.4f} m; the corridor allows it "
        "and the claim that TIMING is the mechanism no longer follows"
    )

    # The claim itself, both directions.
    assert _graft(runs, "constrained", "baseline") < 0.0, (
        "the constrained path flown on the baseline's schedule keeps line of "
        "sight -- then the retiming was decoration and the path did the work"
    )
    assert _graft(runs, "baseline", "constrained") > 0.0, (
        "the baseline path flown on the constrained schedule STILL loses the "
        "link -- then the schedule is not sufficient either, and the mechanism "
        "is something this test does not name"
    )
