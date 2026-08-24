"""The vis draws obstacles from its own JavaScript. This checks that code.

Nothing else does. `test_frontend.py` exercises the request/response path and
asserts on JSON, so the browser-side geometry -- where the obstacle is actually
turned into a picture -- has no coverage at all: the page could place every tube
in the wrong location and every test would still pass.

The check is two independent implementations of one quantity. Python's
``obstacle_positions_at`` and the page's ``obstacleCenter`` are written
separately, in different languages, and must agree; if they do not, one of them
is drawing a lie. This is the geometry-authenticity rule in CLAUDE.md applied to
the only code that was exempt from it.

Skipped when node is unavailable, which is a real gap and is reported as a skip
rather than silently passing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from spacetime_bezier.geometry import obstacle_positions_at
from spacetime_bezier.scenarios import SCENARIO_MAP

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "spacetime_bezier" / "static" / "frontend.html"

# Pulls the helper block out of the page and runs it against supplied cases.
DRIVER = r"""
import { readFileSync } from "node:fs";
const html = readFileSync(process.argv[2], "utf8");
const start = html.indexOf("function obstacleSpatialDim(o)");
const end = html.indexOf("function tubeTraces(");
if (start < 0 || end < 0) {
  console.error("MARKERS_MISSING");
  process.exit(2);
}
globalThis.RES = { solution: { t_lo: 0, t_hi: 10 } };
const mod = new Function(
  html.slice(start, end) +
    "\nreturn { obstacleCenter, obstacleWindow };"
)();
const spec = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out = {
  centers: spec.cases.map((c) =>
    mod.obstacleCenter({ control_points: c.control_points }, c.t)
  ),
  window: mod.obstacleWindow({ control_points: spec.window_case }).slice(0, 2),
};
console.log(JSON.stringify(out));
"""


def _run_js(spec: dict, tmp_path: Path) -> dict:
    driver = tmp_path / "driver.mjs"
    driver.write_text(DRIVER)
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec))
    proc = subprocess.run(
        ["node", str(driver), str(PAGE), str(spec_file)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode == 2:
        pytest.fail(
            "the vis's obstacle helpers could not be located in frontend.html -- "
            "if they were renamed, this test must be renamed with them, not deleted"
        )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_page_places_obstacles_where_the_solver_does(tmp_path):
    """Every scenario, every obstacle, across its own active window.

    `curve` carries a degree-3 obstacle, so this covers the case a
    constant-velocity reader would get wrong: an implementation that still did
    `pos0 + vel*t` would agree on the straight obstacles and diverge here.
    """
    cases = []
    for key in ("original", "curve", "station_fence", "door3d"):
        scenario = SCENARIO_MAP[key][0]()
        for obs in scenario["obstacles"][:4]:
            cps = np.asarray(obs["control_points"], dtype=float)
            for t in np.linspace(float(cps[0, -1]), float(cps[-1, -1]), 7):
                cases.append(
                    {
                        "control_points": cps.tolist(),
                        "t": float(t),
                        "expect": obstacle_positions_at(cps, np.array([t]))[0].tolist(),
                    }
                )
    assert cases, "no obstacles to check"
    degrees = {len(c["control_points"]) - 1 for c in cases}
    assert max(degrees) >= 2, (
        "no curved obstacle in the sample -- this test would then pass for a "
        "page that still assumed constant velocity, and prove nothing"
    )

    window_case = np.asarray(
        SCENARIO_MAP["station_fence"][0]()["obstacles"][-1]["control_points"], dtype=float
    )
    out = _run_js(
        {"cases": cases, "window_case": window_case.tolist()}, tmp_path
    )

    # Compared per case, not stacked: the scenarios mix two and three spatial
    # dimensions, and stacking would raise before it could compare anything.
    assert len(out["centers"]) == len(cases)
    worst = 0.0
    for got, case in zip(out["centers"], cases):
        g = np.asarray(got, dtype=float)
        w = np.asarray(case["expect"], dtype=float)
        assert g.shape == w.shape, f"page returned {g.shape}, solver has {w.shape}"
        worst = max(worst, float(np.max(np.abs(g - w))))
    assert worst < 1e-12, f"the page draws obstacles {worst} away from where the solver puts them"
    assert {len(c["expect"]) for c in cases} >= {2, 3}, (
        "all cases share one spatial dimension -- a page hardcoding 2D would pass"
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_page_reads_the_active_window_from_the_control_points(tmp_path):
    """The window is intrinsic now; a page defaulting to the plot range fails.

    The fixture is a fence piece that appears partway through the run, so a
    reader that fell back to the plotted horizon would report 0.0 and this test
    would catch it.
    """
    cps = np.asarray(
        SCENARIO_MAP["station_fence"][0]()["obstacles"][-1]["control_points"], dtype=float
    )
    t0 = float(cps[0, -1])
    assert t0 > 0.0, "fixture no longer starts late; it cannot detect a horizon fallback"

    out = _run_js({"cases": [], "window_case": cps.tolist()}, tmp_path)
    lo, hi = out["window"]
    assert abs(lo - max(t0, 0.0)) < 1e-12
    assert abs(hi - min(float(cps[-1, -1]), 10.0)) < 1e-12
