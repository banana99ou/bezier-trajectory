#!/usr/bin/env python3
"""Tables from bench records: JSONL in, markdown or LaTeX out. Computes no solver
quantity -- it reads, filters, aggregates and formats.

Built to `paper/journal_1/demo_specs/C_traffic_and_backbone.md` Part 3, plus one
table the spec predates: `reliability`, the four outcome bins of
`paper/journal_1/SPEC.md` sec. 4.7 per family and count. Its one hard rule: a false
certificate anywhere stops the build.

    python3 tools/make_tables.py --in bench/backbone.jsonl --table backbone --format md \\
        --out tables/backbone.md --expect-grid bench/backbone.plan.jsonl

`--expect-grid` takes the bench's own `--dry-run` output: a grid point is exactly a
planned run, so "missing" means "planned and never recorded".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

CERT_TOL = 1e-6
GRID_KEYS = ("mode", "family", "scenario_name", "axis_value", "seed", "arm", "start", "N", "n_seg")
BUILD_KEYS = ("git_commit", "extension_build_time", "extension_so_build_time")
AXIS_LABEL = {"dimension": "Spatial dims (count)", "horizon": "Horizon (s)", "motion": "Motion (kind)"}


# ------------------------------------------------------------------- loading

def load(paths: list[str]) -> list[dict]:
    """Records from every file, in order. A duplicate identity is an error naming
    both copies: two records of one run means one of them is stale."""
    seen, out = {}, []
    for path in paths:
        for n, line in enumerate(Path(path).read_text().splitlines(), 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            ident = rec.get("identity")
            if ident in seen:
                raise SystemExit(f"duplicate identity {ident[:12]}: {seen[ident]} and {path}:{n}")
            seen[ident] = f"{path}:{n}"
            rec["_source"] = f"{path}:{n}"
            out.append(rec)
    return out


def grid_key(d: dict) -> tuple:
    return tuple(d.get(k) for k in GRID_KEYS)


def load_grid(path: str | None) -> list[dict] | None:
    if not path:
        return None
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------- check

class Marks:
    """Footnotes and row marks, numbered in order of first use."""

    def __init__(self):
        self.notes: list[str] = []

    def add(self, text: str) -> int:
        self.notes.append(text)
        return len(self.notes)


def check(records: list[dict], allow_dirty: bool = False, strict_timing: bool = False) -> list[str]:
    """Every reason these records may not form one table. Raises SystemExit when
    there is any; returns the timing warnings otherwise."""
    fails, warns = [], []
    for key in BUILD_KEYS:
        values = {r.get(key) for r in records}
        if len(values) > 1:
            fails.append(f"records disagree on {key}: {sorted(map(str, values))} -- a table from two builds is not a table")
    for r in records:
        where = r.get("_source", "?")
        if r.get("extension_stale"):
            fails.append(f"{where}: extension_stale -- the record describes a build older than the Rust sources")
        if r.get("git_dirty") and not allow_dirty:
            fails.append(f"{where}: taken on a dirty tree ({', '.join(r.get('git_dirty_paths') or []) or 'paths unknown'}); "
                         "pass --allow-dirty to render it marked")
        lo, hi = r.get("solve_seconds_min"), r.get("solve_seconds_max")
        if (lo and hi and hi / lo > 1.2) or (r.get("load_avg_before") or 0.0) > 1.0:
            warns.append(f"{where}: timing unreliable (spread {hi / lo if lo and hi else float('nan'):.2f}, "
                         f"load {r.get('load_avg_before')})")
    if strict_timing:
        fails += warns
    if fails:
        raise SystemExit("make_tables refused:\n  " + "\n  ".join(fails))
    return warns


def _sampling_flag(r: dict) -> bool:
    """Spec C sec. 2.8: denser sampling of a penetration detector must not report a
    LARGER minimum. When it does, that is a finding about the oracle, marked here."""
    a, b = r.get("min_clearance"), r.get("min_clearance_3000")
    return a is not None and b is not None and a > b + 1e-6


def _timing_flag(r: dict) -> bool:
    lo, hi = r.get("solve_seconds_min"), r.get("solve_seconds_max")
    return bool((lo and hi and hi / lo > 1.2) or (r.get("load_avg_before") or 0.0) > 1.0)


# ------------------------------------------------------------------ formatting

def _f(v, spec):
    return "—" if v is None else format(v, spec)


def _mark(r: dict, marks: Marks, allow_dirty: bool) -> str:
    """† for a non-figure-grade record (footnote carries its reasons verbatim),
    ‡ for a dirty tree or an unreliable timing."""
    tag = ""
    if r.get("error_type"):
        n = marks.add(f"{_who(r)}: error {r['error_type']}: {r.get('error_message')}")
        tag += f"†{n}"
    elif not r.get("figure_grade"):
        n = marks.add(f"{_who(r)}: not figure-grade: " + "; ".join(r.get("figure_grade_reasons") or []))
        tag += f"†{n}"
    if (allow_dirty and r.get("git_dirty")) or _timing_flag(r) or _sampling_flag(r):
        why = []
        if _sampling_flag(r):
            why.append(f"clearance sampling disagreement: 3000 samples give {r['min_clearance_3000']:+.4f}, "
                       f"20001 give {r['min_clearance']:+.4f} -- the grids are not nested, and the denser one "
                       "missed the closer approach")
        if allow_dirty and r.get("git_dirty"):
            why.append("dirty tree")
        if _timing_flag(r):
            why.append(f"timing spread/load (load {r.get('load_avg_before')})")
        n = marks.add(f"{_who(r)}: " + ", ".join(why))
        tag += f"‡{n}"
    return tag


def _who(r: dict) -> str:
    bits = [str(r.get("scenario_name")), f"N{r.get('N')}_seg{r.get('n_seg')}", f"arm {r.get('arm')}"]
    if r.get("seed") is not None:
        bits.append(f"seed {r['seed']}")
    if r.get("start") not in (None, "own"):
        bits.append(f"start {r['start']}")
    return " ".join(bits)


def _grade(r: dict | None) -> str:
    if r is None:
        return "MISSING"
    if r.get("error_type"):
        return f"error: {r['error_type']}"
    return "**yes**" if r.get("figure_grade") else "no"


def _unit(records: list[dict]) -> str:
    scales = {r.get("scale") for r in records}
    return "scene units" if scales <= {None} else f"scale {sorted(map(str, scales))}"


def render(header: list[str], rows: list[list[str]], fmt: str) -> str:
    if fmt == "md":
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%").replace("**", "")  # noqa: E731
    out = [r"\begin{tabular}{" + "l" * len(header) + "}", r"\hline",
           " & ".join(esc(h) for h in header) + r" \\", r"\hline"]
    out += [" & ".join(esc(c) for c in r) + r" \\" for r in rows]
    out += [r"\hline", r"\end{tabular}"]
    return "\n".join(out)


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# --------------------------------------------------------------------- tables

def table_backbone(records, primary, marks, allow_dirty, grid):
    by = {}
    for r in records:
        by.setdefault((r["scenario_name"], r["N"], r["n_seg"]), {})[r["arm"]] = r
    keys = list(by)
    if grid:
        for g in grid:
            k = (g["scenario_name"], g["N"], g["n_seg"])
            if k not in by:
                by[k] = {}
                keys.append(k)
    unit = _unit(records)
    header = ["Scenario", "Config", "Unsound walls (floor OFF, count)", f"Clearance ON ({unit})",
              f"Clearance OFF ({unit})", f"Cert ON ({unit})", "Iters ON", "Raises (x10 steps)", "Weight",
              "Stop", "Solve (s, median)", "Grade ON", "Grade OFF"]
    rows = []
    for k in keys:
        on, off = by[k].get(primary), by[k].get("clipfloor_off")
        if on is None:
            rows.append([f"`{k[0]}`", f"N{k[1]}_seg{k[2]}"] + ["MISSING"] * (len(header) - 2))
            continue
        err = bool(on.get("error_type"))
        tag = _mark(on, marks, allow_dirty)
        cell = lambda v, s: "—" if err else _f(v, s)  # noqa: E731
        rows.append([
            f"`{k[0]}`", f"N{k[1]}_seg{k[2]}{tag}",
            _f(off.get("koz_unsound_clips"), ".0f") if off else "MISSING",
            cell(on.get("min_clearance"), "+.4f"),
            _f(off.get("min_clearance"), "+.4f") if off else "MISSING",
            cell(on.get("certificate_violation"), ".2e"), cell(on.get("iterations"), "d"),
            cell(on.get("weight_raises"), "d"), cell(on.get("elastic_weight"), "g"),
            "—" if err else (on.get("stop_label") or "").split(" ")[0],
            cell(on.get("solve_seconds"), ".3f"), _grade(on), _grade(off).replace("**", ""),
        ])
    return header, rows


def _aggregate_rows(records, knob_label, marks, allow_dirty, grid, family):
    by = {}
    for r in records:
        by.setdefault(r.get("axis_value"), []).append(r)
    values = list(by)
    if grid:
        for g in grid:
            if g.get("family") == family and g.get("axis_value") not in by:
                by[g["axis_value"]] = []
                values.append(g["axis_value"])
    unit = _unit(records)
    header = [knob_label, "Seeds (fig-grade/total)", "Walls seed (median)", "Walls returned (median)",
              "Bound (seg x obs)", "Rows (median)", "Iters (median)", "Solve s: median [min, max]",
              f"Clearance ({unit}, median)", "Grade"]
    rows = []
    for v in values:
        rs = by[v]
        for r in rs:
            if (not r.get("figure_grade") or r.get("error_type") or _timing_flag(r) or _sampling_flag(r)
                    or (allow_dirty and r.get("git_dirty"))):
                _mark(r, marks, allow_dirty)
        ok = [r for r in rs if r.get("figure_grade")]
        kn = f"{len(ok)}/{len(rs)}"
        if not rs:
            rows.append([str(v)] + ["MISSING"] * (len(header) - 1))
            continue
        bounds = {r.get("walls_bound") for r in rs if r.get("walls_bound") is not None}
        if len(bounds) > 1:
            raise SystemExit(f"{family} {v}: walls_bound differs across seeds: {sorted(bounds)}")
        if not ok:
            rows.append([str(v), kn] + [f"n/a (0/{len(rs)} figure-grade)"] * (len(header) - 3) + [kn])
            continue
        solves = [r["solve_seconds"] for r in ok]
        rows.append([
            str(v), kn, _f(_median([r.get("walls_seed") for r in ok]), "g"),
            _f(_median([r.get("walls_returned") for r in ok]), "g"), _f(next(iter(bounds), None), "d"),
            _f(_median([r.get("rows_returned") for r in ok]), "g"), _f(_median([r.get("iterations") for r in ok]), "g"),
            f"{statistics.median(solves):.3f} [{min(solves):.3f}, {max(solves):.3f}]",
            _f(_median([r.get("min_clearance") for r in ok]), "+.4f"), kn,
        ])
    return header, rows


def table_scaling(records, primary, marks, allow_dirty, grid):
    blocks = []
    for fam in dict.fromkeys(r["family"] for r in records if r.get("family")):
        rs = [r for r in records if r.get("family") == fam and r["arm"] == primary]
        blocks.append((fam, *_aggregate_rows(rs, "Aircraft (count)", marks, allow_dirty, grid, fam)))
    return blocks


def table_axes(records, primary, marks, allow_dirty, grid):
    blocks = []
    for fam in ("dimension", "horizon", "motion"):
        rs = [r for r in records if r.get("family") == fam and r["arm"] == primary]
        if rs:
            blocks.append((fam, *_aggregate_rows(rs, AXIS_LABEL[fam], marks, allow_dirty, grid, fam)))
    return blocks


def table_coverage(records, primary, marks, allow_dirty, grid):
    by = {}
    for r in records:
        by.setdefault((r["scenario_name"], r.get("seed"), r["N"], r["n_seg"]), {})[r["arm"]] = r
    header = ["Scenario", "Config", "Unsound walls (floor OFF, count)", "Clearance cost of the floor (ON - OFF)",
              "Hole fired (floor OFF: certified yet clearance <= 0)"]
    rows, fired, total = [], 0, 0
    for (name, seed, N, s), arms in by.items():
        on, off = arms.get(primary), arms.get("clipfloor_off")
        if off is None:
            continue
        total += 1
        hole = (off.get("certificate_violation") is not None and off["certificate_violation"] <= CERT_TOL
                and off.get("min_clearance") is not None and off["min_clearance"] <= 0.0)
        fired += hole
        cost = (on["min_clearance"] - off["min_clearance"]
                if on and on.get("min_clearance") is not None and off.get("min_clearance") is not None else None)
        rows.append([f"`{name}`" + (f" s{seed}" if seed is not None else ""), f"N{N}_seg{s}",
                     _f(off.get("koz_unsound_clips"), ".0f"), _f(cost, "+.4f"), "yes" if hole else "no"])
    rows.append(["**all**", "", "", "", f"{fired}/{total}"])
    return header, rows, {"HoleFired": fired, "HoleTotal": total}


def bin_of(r: dict) -> str:
    """SPEC.md sec. 4.7: exactly one bin per run."""
    if r.get("error_type"):
        return "error"
    cert, clr = r.get("certificate_violation"), r.get("min_clearance")
    if cert is not None and clr is not None and cert <= CERT_TOL and clr <= 0.0:
        return "false certificate"
    if r.get("figure_grade"):
        return "solved"
    if clr is not None and clr > 0.0:
        return "safe, uncertified"
    return "failed"


def table_reliability(records, primary, marks, allow_dirty, grid):
    bins = ["solved", "safe, uncertified", "failed", "false certificate", "error"]
    blocks, false_certs = [], []
    for fam in dict.fromkeys(r["family"] for r in records if r.get("family")):
        rs = [r for r in records if r.get("family") == fam and r["arm"] == primary]
        by = {}
        for r in rs:
            by.setdefault(r.get("axis_value"), []).append(r)
        header = ["Count", "Runs"] + [b.capitalize() for b in bins] + ["Solved share"]
        rows = []
        for v, group in by.items():
            counts = {b: 0 for b in bins}
            for r in group:
                b = bin_of(r)
                counts[b] += 1
                if b == "false certificate":
                    false_certs.append(_who(r))
            rows.append([str(v), str(len(group))] + [str(counts[b]) for b in bins]
                        + [f"{counts['solved'] / len(group):.0%}"])
        blocks.append((fam, header, rows))
    if false_certs:
        raise SystemExit("FALSE CERTIFICATE -- a certified run penetrates; SPEC.md sec. 4.7 stops the pass:\n  "
                         + "\n  ".join(false_certs))
    return blocks


TABLES = {"backbone": table_backbone, "scaling": table_scaling, "axes": table_axes,
          "coverage": table_coverage, "reliability": table_reliability}


# ------------------------------------------------------------------------ main

def build(records, table, fmt, primary="ours", grid=None, allow_dirty=False, strict_timing=False,
          sources=()) -> tuple[str, str | None, list[str]]:
    """(table text, numbers.tex text or None, missing grid points). Pure: the same
    records give byte-identical output."""
    check(records, allow_dirty=allow_dirty, strict_timing=strict_timing)
    marks = Marks()
    result = TABLES[table](records, primary, marks, allow_dirty, grid)
    numbers = {}
    if table in ("backbone", "coverage"):
        header, rows = result[0], result[1]
        if table == "coverage":
            numbers = result[2]
        blocks = [(None, header, rows)]
    else:
        blocks = result
    missing = []
    if grid:
        have = {grid_key(r) for r in records}
        missing = [g for g in grid if grid_key(g) not in have]
    commits = sorted({str(r.get("git_commit")) for r in records})
    head = (f"GENERATED by tools/make_tables.py -- do not edit. inputs: {', '.join(sources) or '?'}; "
            f"git_commit: {', '.join(commits)}")
    parts = [f"<!-- {head} -->" if fmt == "md" else f"% {head}"]
    for name, header, rows in blocks:
        if name:
            parts.append(f"**{name}**" if fmt == "md" else f"% {name}")
        parts.append(render(header, rows, fmt))
    if marks.notes:
        if fmt == "md":
            parts.append("\n".join(f"{i}. {t}" for i, t in enumerate(marks.notes, 1)))
        else:
            parts.append("\\begin{footnotesize}\n" + "\n".join(f"{i}. {t}\\\\" for i, t in enumerate(marks.notes, 1))
                         + "\n\\end{footnotesize}")
    text = "\n\n".join(parts) + "\n"
    if table == "backbone":
        ok = sum(1 for r in records if r["arm"] == primary and r.get("figure_grade"))
        tot = sum(1 for r in records if r["arm"] == primary)
        numbers = {"BackboneFigureGrade": ok, "BackboneTotal": tot}
    if table == "scaling":
        solved = [r for r in records if r["arm"] == primary and r.get("figure_grade") and r.get("family") == "traffic"]
        if solved:
            top = max(solved, key=lambda r: (r["axis_value"], -r["solve_seconds"]))
            numbers = {"LargestAircraftSolved": top["axis_value"], "LargestAircraftSolveSeconds": f"{top['solve_seconds']:.3f}"}
    num = None
    if fmt == "tex":
        num = f"% {head}\n" + "".join(f"\\newcommand{{\\TBL{k}}}{{{v}}}\n" for k, v in sorted(numbers.items()))
    return text, num, [json.dumps(g, sort_keys=True) for g in missing]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--table", choices=sorted(TABLES), required=True)
    ap.add_argument("--format", choices=("md", "tex"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="ours")
    ap.add_argument("--expect-grid")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--strict-timing", action="store_true")
    args = ap.parse_args(argv)
    if args.table in ("backbone", "scaling") and not args.expect_grid:
        raise SystemExit(f"--expect-grid is required for `{args.table}`: a silently short table is the failure it prevents")
    records = load(args.inputs)
    text, num, missing = build(records, args.table, args.format, args.arm, load_grid(args.expect_grid),
                               args.allow_dirty, args.strict_timing, sources=args.inputs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    if num is not None:
        Path(str(args.out) + ".numbers.tex").write_text(num)
    if missing:
        raise SystemExit("missing grid points (planned, never recorded):\n  " + "\n  ".join(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
