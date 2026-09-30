"""
`tools/bench.py` -- demo spec C sec. 2.8, each invariant stated as what makes it fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("bezier_opt")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import bench  # noqa: E402

from spacetime_bezier.optimize import optimize_scenario  # noqa: E402
from spacetime_bezier.scenarios import scenario_original  # noqa: E402


def _item(**kw):
    base = dict(mode="scenario", family=None, scenario_name="original", axis_value=None, seed=None,
                arm="ours", start="own", N=8, n_seg=4)
    base.update(kw)
    return base


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("name,N,n_seg", [("original", 8, 4), ("diverse", 8, 4)])
def test_the_bench_row_is_optimize_scenarios_row(name, N, n_seg):
    """The price of timing the solve outside `optimize_scenario` (spec sec. 2.3).
    FAILS IF the bench forwards a different knob -- trust radius, time bound, start
    weight -- or reads the final weight with a non-NaN default. `diverse` N8_seg4
    raises the weight once, so a default of 100 cannot pass there. (The prox weight
    is not a knob: the Rust loop discards it by design.) Ten fields and a control
    polygon bit for bit: it cannot pass vacuously."""
    from spacetime_bezier.scenarios import SCENARIO_MAP

    ref = optimize_scenario(SCENARIO_MAP[name][0](), [(N, n_seg)], verbose=False)["results"][f"N{N}_seg{n_seg}"]
    assert name != "diverse" or ref["weight_raises"] >= 1, "the raising case no longer raises"
    rec = bench.run_one(_item(scenario_name=name, N=N, n_seg=n_seg), repeats=1)
    np.testing.assert_array_equal(np.asarray(rec["control_points"]), np.asarray(ref["control_points"]))
    for k in ("iterations", "converged", "stop_reason", "certificate_violation", "total_slack",
              "elastic_weight", "weight_raises", "koz_unsound_clips", "stop_label"):
        assert rec[k] == ref[k], k
    assert rec["min_clearance_3000"] == ref["min_clearance"]


# Regression pin against the build of `b506f34` (SOLVER.md sec. Measurements, taken
# 2026-09-29 through optimize_scenario at n_eval=3000, start weight 100 escalating,
# walls at the trust region's cap). Constants lifted verbatim from that table. They
# are a pin on a specific build, not a law: when this goes red, re-run the full
# backbone pass and update the table AND these constants in one commit -- never
# loosen the tolerance.
BACKBONE_PINS = [
    ("original", 8, 4, +0.6204, 9, 0, 100.0, "stationary", True),
    ("fence3d", 8, 2, +0.1623, 9, 0, 100.0, "stationary", True),
    ("door3d", 8, 4, +0.9825, 8, 0, 100.0, "stationary", True),
]


@pytest.mark.parametrize("name,N,n_seg,clr,iters,raises,weight,stop,grade", BACKBONE_PINS)
def test_the_bench_regenerates_the_backbone_rows(name, N, n_seg, clr, iters, raises, weight, stop, grade):
    """FAILS IF any solver change moves these three configurations, or the bench
    forwards a knob the table's pass did not use (a leaked time weight, a floor-off
    clip)."""
    rec = bench.run_one(_item(scenario_name=name, N=N, n_seg=n_seg), repeats=1)
    assert rec["min_clearance_3000"] == pytest.approx(clr, abs=5e-5)
    assert rec["iterations"] == iters and rec["weight_raises"] == raises
    assert rec["elastic_weight"] == weight
    assert rec["stop_label"].split(" ")[0] == stop
    assert rec["figure_grade"] is grade


def test_verdict_and_reasons_agree_including_a_deliberate_failure(tmp_path):
    """FAILS IF a failing record has no reasons or a passing one has any. The
    deliberately failing run -- 16 aircraft, seed 0, measured 2026-09-09 and again
    2026-09-29 to collapse at -0.1918 -- keeps this from holding vacuously on an
    all-green pass. Also checks the dense clearance never exceeds the coarse one:
    denser sampling of a penetration detector must not find a LARGER minimum."""
    recs = [bench.run_one(_item(N=8, n_seg=4), repeats=1),
            bench.run_one(dict(mode="family", family="traffic", scenario_name=None, axis_value=16,
                               seed=0, arm="ours", start="own", N=8, n_seg=8), repeats=1)]
    for r in recs:
        assert r["figure_grade"] == (len(r["figure_grade_reasons"]) == 0), r["figure_grade_reasons"]
        assert r["min_clearance"] <= r["min_clearance_3000"] + 1e-6
    bad = recs[1]
    assert not bad["figure_grade"]
    joined = " | ".join(bad["figure_grade_reasons"])
    for needle in ("not converged", "hull certificate violated", "penetrates", "elastic slack"):
        assert needle in joined, joined


def test_resume_skips_the_same_run_and_reruns_a_different_build(tmp_path, monkeypatch):
    """Both directions. FAILS IF --resume re-runs a record already in the file, or
    skips one made by a different build -- the build is inside the identity key on
    purpose, and dropping it is the obvious optimisation that mixes two solvers."""
    out = tmp_path / "r.jsonl"
    argv = ["--out", str(out), "--scenarios", "original", "--configs", "4x4", "--repeats", "1", "--resume"]
    assert bench.main(argv) == 0
    assert len(_read(out)) == 1
    assert bench.main(argv) == 0
    assert len(_read(out)) == 1, "the identical run was re-run"

    prov = bench.provenance()
    for field, value in (("git_commit", "0000000"), ("extension_so_build_time", 1.0)):
        monkeypatch.setattr(bench, "_PROV_CACHE", {**prov, field: value})
        before = len(_read(out))
        assert bench.main(argv) == 0
        assert len(_read(out)) == before + 1, f"a record from another build ({field}) was taken as this build's"


def test_a_run_that_raises_is_still_a_record(tmp_path, monkeypatch):
    """FAILS IF a raising run leaves no line behind."""
    def boom(**kw):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(bench, "optimize_spacetime", boom)
    out = tmp_path / "e.jsonl"
    assert bench.main(["--out", str(out), "--scenarios", "original", "--configs", "4x4", "--repeats", "1"]) == 0
    recs = _read(out)
    assert len(recs) == 1
    assert recs[0]["error_type"] == "RuntimeError"
    assert recs[0]["figure_grade"] is False and recs[0]["figure_grade_reasons"]


def test_records_carry_no_nan_tokens_and_the_dry_run_is_the_real_plan(tmp_path, capsys):
    """FAILS IF a record contains NaN / Infinity (a later `json.load` would accept
    them and invert the gate), or the dry run lists a different plan from the one
    executed."""
    out = tmp_path / "n.jsonl"
    argv = ["--out", str(out), "--scenarios", "original,door3d", "--arms", "ours,clipfloor_off", "--repeats", "1"]
    assert bench.main(argv + ["--dry-run"]) == 0
    planned = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert bench.main(argv) == 0
    text = out.read_text()
    for token in ("NaN", "Infinity"):
        assert token not in text
    recs = _read(out)
    assert [(r["scenario_name"], r["arm"], r["N"], r["n_seg"]) for r in recs] == \
           [(p["scenario_name"], p["arm"], p["N"], p["n_seg"]) for p in planned]
