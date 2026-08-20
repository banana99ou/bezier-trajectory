"""A line-of-sight plane that CANNOT be built must never count as satisfied.

`shadow_plane` returns `None` when the station lies inside the occluder piece's
inflated body: no half-space separates anything there, so no row can be emitted.
The builder used to skip such a window silently, `build_occlusion_rows` returned
`None` when nothing at all was emitted, and the certificate mapped that to 0.0 --
so the strongest possible statement the geometry can make ("line of sight is
unrecoverably blocked over this window") arrived as the strongest possible
statement of success ("occlusion certificate 0.000").

That is a check that cannot fail. The repair is to count the drops and report the
certificate as INFINITE whenever the count is nonzero, which is what these tests
pin. Each one states what would make it fail.
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
# flight that has no reason to deviate. Before the fix this came back converged,
# every certificate 0.0, figure_grade True -- with a TRUE line-of-sight margin of
# -0.3999.
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


def _run(obstacles, n_seg=4):
    P, info = optimize_spacetime(
        N=8,
        dim=3,
        p_start=_START,
        p_end=_END,
        obstacles=obstacles,
        n_seg=n_seg,
        max_iter=60,
        tol=1e-6,
        elastic_weight=1e5,
        stations=_STATION,
        verbose=False,
    )
    return np.asarray(P, dtype=float), info


def _row(P, info, obstacles):
    return {
        "converged": bool(info.get("converged", 0.0)),
        "stop_label": "-",
        "certificate_violation": float(info["koz_violation_reference"]),
        "occlusion_violation": float(info["occlusion_violation_reference"]),
        "occlusion_planes_dropped": float(info["occlusion_planes_dropped"]),
        "min_clearance": 1.0,  # not what this file is testing
        "total_slack": float(info["total_koz_slack_returned"]),
    }


def _exact_occlusion_row_count(P, obstacles, n_seg):
    """How many occlusion rows the EXACT builder actually emitted at ``P``."""
    out = bezier_opt.spacetime_occlusion_rows_exact(
        p=P,
        obstacle_pos0=np.array([o["pos0"] for o in obstacles], dtype=float),
        obstacle_vel=np.array([o["vel"] for o in obstacles], dtype=float),
        obstacle_r=np.array([o["r"] for o in obstacles], dtype=float),
        obstacle_t_start=np.array([o["t_start"] for o in obstacles], dtype=float),
        obstacle_t_end=np.array([o["t_end"] for o in obstacles], dtype=float),
        stations=np.array(_STATION, dtype=float),
        n_seg=n_seg,
    )
    return len(np.asarray(out[1]))


def test_a_dropped_plane_makes_the_certificate_infinite_and_sinks_the_gate():
    """The reviewer's end-to-end reproduction.

    FAILS IF a dropped shadow plane is ever folded back into a finite
    certificate: the run converges, the keep-out certificate is genuinely 0.0 and
    the elastic slack is genuinely negligible, so the ONLY thing standing between
    this trajectory and a paper figure is the occlusion refusal. Set
    `occlusion_violation_reference` back to the sum over the emitted rows and
    this test reports figure_grade True next to a sight margin of -0.4.
    """
    P, info = _run([_BLOCKER])

    # The run is otherwise clean -- which is exactly why the drop has to bite.
    assert bool(info["converged"]) is True
    assert float(info["koz_violation_reference"]) == pytest.approx(0.0, abs=1e-9)
    assert float(info["total_koz_slack_returned"]) < 1e-9

    assert float(info["occlusion_planes_dropped"]) > 0.0
    assert not np.isfinite(float(info["occlusion_violation_reference"]))

    # And the drop was hiding a real loss, not a formality.
    _, margins = compute_los_margin(P, _STATION[0], [_BLOCKER], dim=3, n_eval=4001)
    assert float(margins.min()) < -0.1

    row = _row(P, info, [_BLOCKER])
    assert is_figure_grade(row) is False
    reasons = figure_grade_failures(row)
    assert any("could not be built" in r for r in reasons), reasons


def test_a_partial_drop_also_sinks_the_gate():
    """Most planes emitted, one dropped -- the case a row COUNT cannot detect.

    A second, harmless static obstacle keeps the builder productive: at the
    returned iterate 63 of the 72 possible rows are emitted, and exactly one
    (segment, station, obstacle) window is dropped. A guard that fired only when
    the whole block came back empty -- i.e. one written against
    `build_occlusion_rows` returning `None` -- would pass this run.

    FAILS IF the drop count is derived from "were any rows emitted" rather than
    counted per window.
    """
    obstacles = [
        _BLOCKER,
        {"pos0": [2.0, -1.0], "vel": [0.0, 0.0], "r": 0.3, "t_start": 0.0, "t_end": 10.0},
    ]
    P, info = _run(obstacles)

    assert _exact_occlusion_row_count(P, obstacles, n_seg=4) > 0, (
        "the builder emitted nothing at all, so this is the total-drop case and "
        "proves nothing about a partial one"
    )
    assert float(info["occlusion_planes_dropped"]) > 0.0
    assert not np.isfinite(float(info["occlusion_violation_reference"]))
    assert is_figure_grade(_row(P, info, obstacles)) is False


def test_station_fence_drops_nothing_and_stays_figure_grade():
    """The scenario the refusal must NOT break.

    `station_fence` N8_seg8 is the occlusion demo: 60 planes, all of them
    buildable. FAILS IF the drop counter over-reports -- e.g. counting windows
    that were out of range rather than windows whose plane failed -- because the
    paper's only occlusion figure would stop being figure-grade.
    """
    out = optimize_scenario(SCENARIO_MAP["station_fence"][0](), [(8, 8)], verbose=False)
    row = out["results"]["N8_seg8"]
    assert row["occlusion_planes_dropped"] == 0.0
    assert row["occlusion_violation"] == pytest.approx(0.0, abs=1e-9)
    assert row["figure_grade"] is True, row["figure_grade_reasons"]


def test_a_run_with_no_station_is_unaffected():
    """Pre-B12 rows must stay exactly as they were.

    No station means no occlusion rows and nothing that could be dropped, so the
    key is 0.0 and the gate condition is silent. FAILS IF the new condition
    defaults to NaN or to "missing", which would sink every scenario in the
    table.
    """
    from spacetime_bezier.scenarios import scenario_original

    out = optimize_scenario(
        scenario_original(), [(8, 4)], elastic_weight=100.0, verbose=False
    )
    row = out["results"]["N8_seg4"]
    assert row["occlusion_planes_dropped"] == 0.0
    assert row["figure_grade"] is True, row["figure_grade_reasons"]
