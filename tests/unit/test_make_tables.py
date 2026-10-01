"""
`tools/make_tables.py` on synthetic records -- demo spec C sec. 3.5, plus the
reliability table of SPEC.md sec. 4.7. No solver needed.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import make_tables as mt  # noqa: E402

PROV = dict(git_commit="abc1234", git_dirty=False, extension_build_time=1.0, extension_so_build_time=2.0,
            extension_stale=False, solve_seconds_min=0.10, solve_seconds_max=0.11, load_avg_before=0.2)


def _rec(name, N, n_seg, arm, *, grade=True, clr=0.5, cert=0.0, family=None, value=None, seed=None, **kw):
    r = dict(PROV, mode="family" if family else "scenario", family=family, scenario_name=name,
             axis_value=value, seed=seed, arm=arm, start="own", N=N, n_seg=n_seg,
             figure_grade=grade, figure_grade_reasons=[] if grade else ["hull certificate violated by 1.0e+00"],
             min_clearance=clr, certificate_violation=cert, koz_unsound_clips=0.0 if arm == "ours" else 3.0,
             iterations=10, weight_raises=0, elastic_weight=100.0, stop_label="stationary", solve_seconds=0.105,
             walls_seed=5, walls_returned=6, walls_bound=16, rows_returned=54, error_type=None)
    r.update(kw)
    r["identity"] = json.dumps([r["scenario_name"], r["seed"], r["arm"], r["N"], r["n_seg"], r["axis_value"]])
    return r


def _backbone():
    recs = []
    for name, N, s in (("original", 8, 4), ("curve", 8, 8), ("wall", 8, 16)):
        recs.append(_rec(name, N, s, "ours"))
        recs.append(_rec(name, N, s, "clipfloor_off", grade=False, clr=0.4))
    grid = [dict(mode="scenario", family=None, scenario_name=r["scenario_name"], axis_value=None, seed=None,
                 arm=r["arm"], start="own", N=r["N"], n_seg=r["n_seg"]) for r in recs]
    return recs, grid


def test_a_non_figure_grade_record_is_marked_and_a_missing_one_refused(tmp_path):
    """Mutation, both halves. FAILS IF a non-figure-grade primary cell renders
    without its dagger, its reasons verbatim and `no`; or a planned grid point
    with no record yields a table and a clean exit. The clean set must render with
    no dagger and no missing point first, so neither half can pass vacuously."""
    recs, grid = _backbone()
    text, _, missing = mt.build(recs, "backbone", "md", grid=grid)
    assert "†" not in text and not missing, text

    bad = copy.deepcopy(recs)
    bad[2]["figure_grade"] = False
    bad[2]["figure_grade_reasons"] = ["not converged (trust_collapse)", "elastic slack 2.0e+00 > 1e-06"]
    text, _, _ = mt.build(bad, "backbone", "md", grid=grid)
    row = next(line for line in text.splitlines() if line.startswith("| `curve`"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert "†" in cells[1] and cells[-2] == "no", row
    for reason in bad[2]["figure_grade_reasons"]:
        assert reason in text

    short = [r for r in recs if not (r["scenario_name"] == "wall" and r["arm"] == "ours")]
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in short))
    (tmp_path / "g.jsonl").write_text("\n".join(json.dumps(g) for g in grid))
    with pytest.raises(SystemExit, match="missing grid points"):
        mt.main(["--in", str(tmp_path / "r.jsonl"), "--table", "backbone", "--format", "md",
                 "--out", str(tmp_path / "t.md"), "--expect-grid", str(tmp_path / "g.jsonl")])
    assert "MISSING" in (tmp_path / "t.md").read_text()


@pytest.mark.parametrize("field,value", [("git_commit", "fff0000"), ("extension_build_time", 9.0),
                                         ("extension_so_build_time", 9.0), ("extension_stale", True)])
def test_check_refuses_a_table_from_two_builds_or_a_stale_one(field, value):
    """FAILS IF check() accepts records that disagree on the build, or one that
    describes a build older than the sources."""
    recs, _ = _backbone()
    mt.check(recs)
    recs[0][field] = value
    with pytest.raises(SystemExit, match="refused"):
        mt.check(recs)


def test_a_dirty_tree_is_refused_unless_allowed_and_then_marked():
    recs, grid = _backbone()
    for r in recs:
        r["git_dirty"] = True
    with pytest.raises(SystemExit, match="dirty"):
        mt.build(recs, "backbone", "md", grid=grid)
    text, _, _ = mt.build(recs, "backbone", "md", grid=grid, allow_dirty=True)
    assert "‡" in text and "dirty tree" in text


def _traffic(n_bad_false_cert=0):
    recs = []
    for fam in ("traffic", "traffic_curved"):
        for n in (2, 8):
            for seed in range(3):
                recs.append(_rec(f"{fam}_n{n:02d}", 8, 8, "ours", family=fam, value=n, seed=seed,
                                 grade=not (n == 8 and seed == 0), clr=0.3 if seed else -0.1,
                                 cert=0.0 if seed else 1.5))
    for i in range(n_bad_false_cert):
        recs[i].update(figure_grade=False, figure_grade_reasons=["not converged"], certificate_violation=0.0,
                       min_clearance=-0.2)
    return recs


def test_reliability_bins_are_exhaustive_and_a_false_certificate_stops_the_build():
    """SPEC.md sec. 4.7. FAILS IF a run lands in no bin or two, or a false
    certificate -- certified yet penetrating -- lets the table render."""
    recs = _traffic()
    text, _, _ = mt.build(recs, "reliability", "md")
    for r in recs:
        assert mt.bin_of(r) in ("solved", "safe, uncertified", "failed", "false certificate", "error")
    assert "| 8 | 3 | 2 | 0 | 1 | 0 | 0 |" in text, text
    with pytest.raises(SystemExit, match="FALSE CERTIFICATE"):
        mt.build(_traffic(n_bad_false_cert=1), "reliability", "md")


def test_aggregates_exclude_failures_and_say_so():
    """FAILS IF a median is taken over non-figure-grade runs, or an all-failed
    count renders a number instead of n/a."""
    recs = _traffic()
    for r in recs:
        if r["axis_value"] == 2 and r["family"] == "traffic":
            r.update(figure_grade=False, figure_grade_reasons=["not converged"])
    grid = [dict(mode="family", family=r["family"], scenario_name=r["scenario_name"], axis_value=r["axis_value"],
                 seed=r["seed"], arm="ours", start="own", N=8, n_seg=8) for r in recs]
    text, _, _ = mt.build(recs, "scaling", "md", grid=grid)
    assert "n/a (0/3 figure-grade)" in text
    assert "| 8 | 2/3 |" in text


def test_output_is_a_pure_function_of_the_records():
    """FAILS IF the same records give different bytes -- a generated file could not
    then be checked against its generator."""
    recs, grid = _backbone()
    a = mt.build(copy.deepcopy(recs), "backbone", "tex", grid=grid, sources=("x.jsonl",))
    b = mt.build(copy.deepcopy(recs), "backbone", "tex", grid=grid, sources=("x.jsonl",))
    assert a == b and a[1].count("\\newcommand") == 2


def test_a_sampling_disagreement_is_marked_not_hidden():
    """Spec C sec. 2.8. FAILS IF a record whose 20001-sample clearance exceeds its
    3000-sample one renders unmarked (measured on `wall` N10_seg24, 2026-09-29)."""
    recs, grid = _backbone()
    for r in recs:
        r["min_clearance_3000"] = r["min_clearance"]
    text, _, _ = mt.build(recs, "backbone", "md", grid=grid)
    assert "sampling disagreement" not in text
    recs[4]["min_clearance_3000"] = recs[4]["min_clearance"] - 3e-4
    text, _, _ = mt.build(recs, "backbone", "md", grid=grid)
    assert "sampling disagreement" in text and "‡" in text


def _bar_records():
    """traffic, counts 2 and 8, three seeds, from the own start; the one own failure
    (count 8, seed 0) solved from `wait`; every solved problem started at its answer."""
    own = _traffic()
    for r in own:
        r.update(walls_seed=5, rows_seed=45, walls_returned=6, rows_returned=54, walls_bound=16, sound_clip=True,
                 certificate_recheck_agrees=True, start_certified=False)
    wait = copy.deepcopy(own[3])                          # traffic n8 seed 0
    wait.update(start="wait", identity=wait["identity"] + "w", figure_grade=True, figure_grade_reasons=[],
                min_clearance=0.3, certificate_violation=0.0)
    answers = []
    for r in [x for x in own if x["figure_grade"]] + [wait]:
        a = copy.deepcopy(r)
        a.update(start="answer", identity=a["identity"] + "a", start_certified=True, answer_start=r["start"])
        answers.append(a)
    return own + [wait] + answers


def _bar_row(text, fam, timing="arrival pinned"):
    line = next(x for x in text.splitlines() if x.startswith(f"| `{fam}` | {timing}"))
    return [c.strip() for c in line.strip("|").split("|")]


def test_the_bar_holds_on_a_clean_set_and_counts_without_judging():
    """FAILS IF the clean set reads as broken, the counts mix starts, or a solved
    share is turned into a verdict -- the paper claims no success rate."""
    text, _, _ = mt.build(_bar_records(), "bar", "md")
    assert "BAR BROKEN" not in text and "**BROKEN**" not in text, text
    # problems 6; solved own 5; any start 6; no answer known 0; four zeros; refused 0
    assert _bar_row(text, "traffic")[2:] == ["6", "5", "6", "0", "0", "0", "0", "0", "0"], text
    assert "%" not in text.split("**The bar**")[0], "a share became a verdict"


@pytest.mark.parametrize("row,edit", [
    ("false certificates", dict(certificate_violation=0.0, min_clearance=-0.1)),
    ("false certificates", dict(occlusion_violation=0.0, occlusion_planes_dropped=0.0, los_margin=-0.05)),
    ("uncovered walls", dict(koz_unsound_clips=2.0)),
    ("size breaks", dict(rows_returned=55)),
    ("size breaks", dict(walls_returned=17, rows_returned=153)),
    ("certified starts lost", dict(figure_grade=False, figure_grade_reasons=["not converged"], min_clearance=0.2,
                                   certificate_violation=1.0)),
])
def test_each_bar_row_breaks_on_its_own(row, edit, tmp_path):
    """Mutation per row. FAILS IF one bad run of each kind leaves its row at zero,
    or the table still exits clean."""
    recs = _bar_records()
    target = next(r for r in recs if r["start"] == "answer")
    target.update(edit)
    text, _, _ = mt.build(recs, "bar", "md")
    assert f"BAR BROKEN, {row}" in text, text
    verdict = next(x for x in text.splitlines() if f"| {row} |" in x)
    assert "**BROKEN**" in verdict, verdict
    (tmp_path / "r.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    with pytest.raises(SystemExit, match="bar is broken"):
        mt.main(["--in", str(tmp_path / "r.jsonl"), "--table", "bar", "--format", "md", "--out", str(tmp_path / "b.md")])


def test_the_bar_refuses_when_the_certificate_recomputation_disagrees():
    """FAILS IF the bar renders although the recomputed certificate -- the one the
    start's certificate is read from -- disagreed with the solver's somewhere."""
    recs = _bar_records()
    recs[0]["certificate_recheck_agrees"] = False
    with pytest.raises(SystemExit, match="recomputed certificate disagrees"):
        mt.build(recs, "bar", "md")


def test_the_other_tables_still_count_one_start():
    """FAILS IF a table other than the bar counts the answer and wait runs too."""
    solo, _, _ = mt.build(_bar_records(), "reliability", "md")
    assert "| 8 | 3 |" in solo, "the reliability table counted several starts"
