"""A line-of-sight wall that CANNOT be built must never count as satisfied.

There is one keep-out zone — the obstacle and its shadow — and one row set. A
shadow wall is still impossible in one configuration: when the centreline passes
within `r_m` of the station, the body swallows its own light source, the shadow
becomes every position beyond it in every direction, and no supporting half-space
exists anywhere. The builder must count that as a DROPPED wall.

Reporting it as "out of reach" instead would be a check that cannot fail: out of
reach means *no row is needed*, so the row set goes silent and the certificate,
which sums the rows that exist, reports 0.0 — the strongest statement the
geometry can make ("the link is unrecoverably blocked") arriving as the strongest
statement of success. This file pins the refusal. Each test states what would
make it fail.

**The reviewer's original reproduction is now caught by a different mechanism**,
and the change is recorded in `test_the_reviewers_reproduction_is_still_refused`:
the old builder wrapped the occluder in a conservative containing ball and could
not aim a plane when the station fell inside it, so the run was refused by a
drop. The center surface has no such ball, so the wall IS built there — and it
reports a real violation instead. Refused either way, and the second way says
more.
"""

import numpy as np
import pytest

from spacetime_bezier.geometry import compute_los_margin
from spacetime_bezier.optimize import (
    figure_grade_failures,
    is_figure_grade,
    optimize_scenario,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import SCENARIO_MAP

bezier_opt = pytest.importorskip("bezier_opt")

# The reviewer's reproduction, verbatim in its geometry: a station off to one
# side, a small fast occluder crossing the sight line at t = 5, and a straight
# flight that has no reason to deviate. Before any of this existed it came back
# converged, every certificate 0.0, figure_grade True -- with a TRUE
# line-of-sight margin of -0.3999.
_SPEED = 8.0
_BLOCKER = {
    "pos0": [5.0 - _SPEED * 5.0, 0.0],
    "vel": [_SPEED, 0.0],
    "r": 0.4,
    "t_start": 0.0,
    "t_end": 10.0,
}
_STATION = [[5.0, -3.0]]
_START = [0.0, 4.0, 0.0]
_END = [10.0, 4.0, 10.0]

# A fat occluder whose path runs straight over the station: at t = 5 the body is
# centred on it, so the station is INSIDE the body and the shadow has no
# supporting half-space at all.
_SWALLOWER = {
    "pos0": [-5.0, 0.0],
    "vel": [2.0, 0.0],
    "r": 1.5,
    "t_start": 0.0,
    "t_end": 10.0,
}
_SWALLOWED_STATION = [[5.0, 0.0]]


def _run(obstacles, stations, n_seg=4, max_iter=60):
    P, info = optimize_spacetime(
        N=8,
        dim=3,
        p_start=_START,
        p_end=_END,
        obstacles=obstacles,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=1e-6,
        elastic_weight=1e5,
        stations=stations,
        verbose=False,
    )
    return np.asarray(P, dtype=float), info


def _row(info):
    return {
        "converged": bool(info.get("converged", 0.0)),
        "stop_label": "-",
        "certificate_violation": float(info["koz_violation_reference"]),
        "occlusion_violation": float(info["occlusion_violation_reference"]),
        "occlusion_planes_dropped": float(info["occlusion_planes_dropped"]),
        "min_clearance": 1.0,  # not what this file is testing
        "total_slack": float(info["total_koz_slack_returned"]),
    }


def _shadow_row_count(P, obstacles, stations, n_seg):
    """How many SHADOW rows the exact builder emitted at ``P``.

    One builder, one call: shadow rows are the ones carrying a station index.
    """
    from spacetime_bezier.geometry import obstacle_array_bundle

    ctrl, radii = obstacle_array_bundle(obstacles, np.asarray(P).shape[1] - 1)
    out = bezier_opt.spacetime_koz_rows_exact(
        p=np.ascontiguousarray(P, dtype=float),
        obstacle_ctrl=ctrl,
        obstacle_r=radii,
        n_seg=n_seg,
        stations=np.asarray(stations, dtype=float),
    )
    return int(np.sum(np.asarray(out[6]) >= 0))


def test_a_dropped_wall_makes_the_certificate_infinite_and_sinks_the_gate():
    """The station is inside the occluder, so no shadow wall exists.

    FAILS IF a generator that cannot produce a wall is reported as out of reach:
    the row set then goes silent, the certificate sums to 0.0, and this run --
    whose station spends part of the horizon inside a body -- comes back
    figure-grade. It also fails if the refusal leaks into the keep-out
    certificate, which is a different guarantee about a different set.
    """
    _, info = _run([_SWALLOWER], _SWALLOWED_STATION, max_iter=40)

    assert float(info["occlusion_planes_dropped"]) > 0.0
    assert not np.isfinite(float(info["occlusion_violation_reference"]))
    # The obstacle's own zone is a separate reading and must still certify.
    assert float(info["koz_violation_reference"]) == pytest.approx(0.0, abs=1e-9)

    row = _row(info)
    assert is_figure_grade(row) is False
    reasons = figure_grade_failures(row)
    assert any("could not be built" in r for r in reasons), reasons


def test_the_reviewers_reproduction_is_still_refused():
    """The original bad run, refused by the certificate rather than by a drop.

    The occluder crosses the sight line but never reaches the station, so the
    shadow wall IS buildable and nothing is dropped. What refuses the run now is
    the wall itself: the solver reports a real occlusion violation, and it agrees
    in sign with the independent line-of-sight oracle, which is the only pairing
    that can catch a lie.

    FAILS IF this returns a clean certificate beside a negative true margin --
    the exact silent-success this whole file exists to prevent.
    """
    P, info = _run([_BLOCKER], _STATION)

    _, margins = compute_los_margin(P, _STATION[0], [_BLOCKER], dim=3, n_eval=4001)
    true_min = float(margins.min())
    assert true_min < -0.1, "the fixture must actually lose the link"

    occlusion = float(info["occlusion_violation_reference"])
    assert occlusion > 1e-6, (
        f"true margin {true_min:.4f} is lost but the solver certified "
        f"{occlusion:.3e}"
    )
    assert is_figure_grade(_row(info)) is False


def test_a_partial_drop_also_sinks_the_gate():
    """Most walls emitted, one generator faulted -- what a row COUNT cannot see.

    A second, harmless static obstacle keeps the builder productive, so rows are
    emitted in quantity while one (obstacle, station) generator still cannot
    produce any. A guard that fired only when the whole block came back empty
    would pass this run.

    FAILS IF the drop count is derived from "were any rows emitted" rather than
    counted per generator.
    """
    obstacles = [
        _SWALLOWER,
        # Placed BETWEEN the station and the flight path so its shadow actually
        # reaches the trajectory; an occluder whose shadow points away from the
        # path is out of reach and emits nothing, which would make this the
        # total-drop case in disguise.
        {"pos0": [5.0, 2.0], "vel": [0.0, 0.0], "r": 0.3, "t_start": 0.0, "t_end": 10.0},
    ]
    P, info = _run(obstacles, _SWALLOWED_STATION, max_iter=40)

    assert _shadow_row_count(P, obstacles, _SWALLOWED_STATION, n_seg=4) > 0, (
        "the builder emitted no shadow rows at all, so this is the total-drop "
        "case and proves nothing about a partial one"
    )
    assert float(info["occlusion_planes_dropped"]) > 0.0
    assert not np.isfinite(float(info["occlusion_violation_reference"]))
    assert is_figure_grade(_row(info)) is False


def test_station_fence_drops_nothing_and_stays_figure_grade():
    """The scenario the refusal must NOT break.

    FAILS IF the drop counter over-reports -- e.g. counting generators that were
    out of range rather than generators whose wall failed -- because the paper's
    occlusion demo would stop being figure-grade.
    """
    # `sound_clip=True`: figure-grade requires the certificate to cover the whole
    # keep-out zone since 2026-08-31, and at the default clip this run returns
    # pairs it does not cover. That is a different condition than the drop
    # counter under test, and a control that fails for the wrong reason is not a
    # control.
    out = optimize_scenario(
        SCENARIO_MAP["station_fence"][0](), [(8, 8)], verbose=False, sound_clip=True
    )
    row = out["results"]["N8_seg8"]
    assert row["occlusion_planes_dropped"] == 0.0
    assert row["occlusion_violation"] == pytest.approx(0.0, abs=1e-9)
    assert row["figure_grade"] is True, row["figure_grade_reasons"]


def test_a_run_with_no_station_is_unaffected():
    """Rows without a station must stay exactly as they were.

    No station means no shadow generator and nothing that could be dropped, so
    the key is 0.0 and the gate condition is silent. FAILS IF the new condition
    defaults to NaN or to "missing", which would sink every scenario in the
    table.
    """
    from spacetime_bezier.scenarios import scenario_original

    # `sound_clip=True` for the same reason as above: `original` N8_seg4 returns
    # 4 uncovered pairs at the default clip, at no cost in clearance.
    out = optimize_scenario(
        scenario_original(), [(8, 4)], elastic_weight=100.0, verbose=False,
        sound_clip=True,
    )
    row = out["results"]["N8_seg4"]
    assert row["occlusion_planes_dropped"] == 0.0
    assert row["figure_grade"] is True, row["figure_grade_reasons"]
