#!/usr/bin/env python3
"""
Ablation of the normal-change term in the linearized obstacle rows.

Section 3.3 differentiates the supporting half-space through its normal: the
centroid rule re-aims n^(s) whenever the control points move, and the
subproblem's rows carry that rotation (`build_koz_constraints_linearized`). The
paper counts this as a contribution, so it needs a result, not only a
derivation: what happens when the subproblem keeps the normal fixed at the
reference instead?

Two solvers are built from the SAME working-tree source by the same compiler.
They differ in exactly one line, PATCH below: the "without" arm builds the
subproblem's obstacle rows with `build_koz_constraints`, the normal frozen at the
reference. Everything else is identical -- the merit still grades h on rows
rebuilt at the candidate, and the ratio test, mu, Delta_0 and the stopping rules
are untouched. The solver source in the repo is never modified: each arm is
copied to a temporary directory, built there and imported through PYTHONPATH.

Each arm then runs
  - the near-set Monte Carlo (tools/build_monte_carlo.py, unchanged), the
    population the paper's subdivision results rest on, and
  - the five table-2 scenarios at every Monte Carlo n_seg, with SCVX_TRACE on, so
    every rejected step is classified.

Checks, each with the outcome that would fail it:
  A. Each arm imported its own build: the extension hashes recorded by the Monte
     Carlo runs equal the hashes of the two builds, and differ from each other.
     Fails if PYTHONPATH did not take effect and both arms ran one binary.
  B. The "with" arm is the paper's solver: its scenario rows reproduce
     doc/results/paper_tables.csv (certified, iterations, stop, J to 1e-6).
  C. Where the obstacle constraint is inactive at the solution, the patched line
     cannot matter: both arms must agree on stop and iterations, and on J to
     1e-9 relative (the rows differ, so the QP's roundoff may). Fails if the
     patch changed anything besides the obstacle rows, or if no such row exists.
The Monte Carlo's own checks (baseline vs Table 3, rotation invariance,
determinism) are in each arm's summary.md; the "without" arm is expected to fail
the first two, since it is not the paper's solver.

Run:
    .venv/bin/python tools/build_normal_term_ablation.py --workers 12
    .venv/bin/python tools/build_normal_term_ablation.py --summarize-only
    .venv/bin/python tools/build_normal_term_ablation.py --n 8 --out /tmp/ablation_pilot
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_tables as BT  # noqa: E402
from tools import build_monte_carlo as BM  # noqa: E402
from tools.verify import harness_common as H  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "doc" / "results" / "normal_term_ablation"
ARMS = ("with", "without")

PATCH_FILE = "core/src/optimizer.rs"
PATCH_OLD = "constraints::build_koz_constraints_linearized("
PATCH_NEW = "constraints::build_koz_constraints("

GRID_FIELDS = ["arm", "scenario", "n_seg", "trust_radius_km", "certified", "stop",
               "iters", "objective", "margin_km", "hull_violation_km",
               "halfspace_slack_km", "rejects", "phantom_rejects"]


# --------------------------------------------------------------------------
# Builds
# --------------------------------------------------------------------------

def _so_digest(site):
    """Same digest as harness_common._ext_digest, taken from a build directory."""
    h = hashlib.sha256()
    for p in sorted((Path(site) / "bezier_opt").glob("*.so")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def build_arm(work, arm):
    """Copy rust_optimizer, patch it for the "without" arm, build and install it."""
    src = work / arm / "rust_optimizer"
    shutil.copytree(REPO_ROOT / "rust_optimizer", src,
                    ignore=shutil.ignore_patterns("target"))
    if arm == "without":
        path = src / PATCH_FILE
        text = path.read_text()
        # Exactly one call site builds the subproblem's rows on the trust path; a
        # second match would mean the patch reaches somewhere it was not meant to.
        assert text.count(PATCH_OLD) == 1, f"{PATCH_OLD!r} found {text.count(PATCH_OLD)} times"
        path.write_text(text.replace(PATCH_OLD, PATCH_NEW))
    wheels, site = work / arm / "wheels", work / arm / "site"
    env = {**os.environ, "CARGO_TARGET_DIR": str(work / arm / "target")}
    subprocess.run([str(Path(sys.executable).parent / "maturin"), "build", "--release",
                    "-i", sys.executable, "-o", str(wheels)],
                   cwd=src / "pybind", env=env, check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--quiet",
                    "--target", str(site), *map(str, wheels.glob("*.whl"))], check=True)
    return site


def arm_env(site, trace=False):
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(filter(None, [str(site), os.environ.get("PYTHONPATH")]))}
    if trace:
        env["SCVX_TRACE"] = "1"
    return env


# --------------------------------------------------------------------------
# Scenario grid (runs inside an arm's environment)
# --------------------------------------------------------------------------

def _traced_solve(sc, n_seg, r0):
    """Solve with the Rust trace on stderr captured; returns (P, info, trace rows)."""
    with tempfile.TemporaryFile(mode="w+") as tf:
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(tf.fileno(), 2)
        try:
            P, info = H.run_rust(sc, n_seg=n_seg, scp_trust_radius=r0)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        tf.seek(0)
        lines = [ln.strip().split(",") for ln in tf if ln.startswith("SCVXTRACE,")]
    head = lines[0]
    return P, info, [dict(zip(head, ln)) for ln in lines[1:]]


def grid_worker(arm, site, out_json):
    import bezier_opt
    assert Path(bezier_opt.__file__).resolve().is_relative_to(Path(site).resolve()), bezier_opt.__file__
    rows = []
    for name, _ in BT.T2_SCENARIOS:
        sc = H.make_scenario(name, N=BT.T2_DEGREE)
        r0 = float(BT.PAPER_R0.get(name, sc["r0"]))
        for n_seg in BM.NSEGS:
            P, info, trace = _traced_solve(sc, n_seg, r0)
            hull = float(info["final_hull_violation_km"])
            stop = int(info["scvx_stop_reason"])
            rejects = [t for t in trace if t["outcome"] == "reject"]
            # A phantom rejection: the candidate satisfies the rows it was solved
            # against, keeps every subdivided control point outside the sphere, and
            # is rejected only because the rows rebuilt AT the candidate -- with the
            # normal re-aimed -- report a violation.
            phantom = [t for t in rejects
                       if float(t["vlin_c"]) <= BT.CERT_TOL
                       and float(t["vtrue_c"]) > BT.CERT_TOL
                       and float(t["cpviol_c"]) <= BT.CERT_TOL]
            rows.append(dict(
                arm=arm, scenario=name, n_seg=n_seg, trust_radius_km=r0,
                certified=hull <= BT.CERT_TOL and stop in BT.PRINCIPLED_STOPS,
                stop=stop, iters=int(info["iterations"]), objective=float(info["cost"]),
                margin_km=float(info["min_radius"]) - sc["r_e"], hull_violation_km=hull,
                halfspace_slack_km=min(BM.halfspace_slacks(P, n_seg, sc["r_e"])),
                rejects=len(rejects), phantom_rejects=len(phantom)))
            r = rows[-1]
            print(f"  {arm:7s} {name:12s} n_seg {n_seg:2d}  stop {stop}  iters {r['iters']:4d}  "
                  f"rejects {r['rejects']:3d} (phantom {r['phantom_rejects']:3d})", flush=True)
    Path(out_json).write_text(json.dumps(rows))


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

def run(args):
    if OUT.exists():
        sys.exit(f"{OUT} exists. The two builds get new extension hashes every time, and "
                 "the Monte Carlo refuses to mix them; move it away to re-run, or pass "
                 "--summarize-only.")
    OUT.mkdir(parents=True)
    # Recorded at run time: the summary can be regenerated later from a different
    # tree, and must still name the one the builds came from. script_sha256 ties
    # the run to this file even before it is committed, as build_monte_carlo does.
    source = dict(commit=H._git("rev-parse", "HEAD", default="unknown"),
                  dirty=bool(H._git("status", "--porcelain", "--", *H._CODE_PATHS,
                                    "tools/build_monte_carlo.py", "tools/build_tables.py")),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    builds, grid = {}, []
    with tempfile.TemporaryDirectory(prefix="normal_term_ablation_") as tmp:
        work = Path(tmp)
        for arm in ARMS:
            site = build_arm(work, arm)
            builds[arm] = dict(ext=_so_digest(site), patched=(arm == "without"))
            print(f"{arm}: built extension {builds[arm]['ext']}", flush=True)
            out_json = work / f"grid_{arm}.json"
            subprocess.run([sys.executable, __file__, "--grid-worker", arm, str(site),
                            str(out_json)], env=arm_env(site, trace=True), check=True)
            grid += json.loads(out_json.read_text())
            subprocess.run([sys.executable, str(REPO_ROOT / "tools" / "build_monte_carlo.py"),
                            "--ranges", "near", "--n", str(args.n), "--workers", str(args.workers),
                            "--out", str(OUT / arm)], env=arm_env(site), check=True)
    builds["patch"] = dict(file=f"rust_optimizer/{PATCH_FILE}", old=PATCH_OLD, new=PATCH_NEW)
    builds["source"] = source
    (OUT / "builds.json").write_text(json.dumps(builds, indent=2) + "\n")
    H.write_csv(OUT / "grid.csv", grid, GRID_FIELDS)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def _grid():
    rows = []
    for r in BM.read_rows(OUT / "grid.csv"):
        t = dict(r)
        for k in ("n_seg", "stop", "iters", "rejects", "phantom_rejects"):
            t[k] = int(r[k])
        for k in ("trust_radius_km", "objective", "margin_km", "hull_violation_km",
                  "halfspace_slack_km"):
            t[k] = float(r[k])
        t["certified"] = r["certified"] == "True"
        rows.append(t)
    return {(r["arm"], r["scenario"], r["n_seg"]): r for r in rows}


def check_builds(builds, metas):
    ok = (builds["with"]["ext"] != builds["without"]["ext"]
          and all(metas[a]["ext"] == builds[a]["ext"] for a in ARMS))
    lines = [f"- `{a}`: built `{builds[a]['ext']}`, Monte Carlo imported `{metas[a]['ext']}`"
             for a in ARMS]
    return ok, lines


def check_paper(grid):
    ref = [r for r in BM.read_rows(REPO_ROOT / "doc/results/paper_tables.csv")
           if int(r["degree"]) == BT.T2_DEGREE
           and ("with", r["scenario"], int(r["n_seg"])) in grid]
    ok, lines = bool(ref), []
    for q in ref:
        r = grid[("with", q["scenario"], int(q["n_seg"]))]
        rel = abs(r["objective"] / float(q["objective"]) - 1)
        good = (r["certified"] == (q["certified"] == "True") and r["iters"] == int(q["iters"])
                and r["stop"] == int(q["stop"]) and rel < 1e-6)
        ok &= good
        lines.append(f"| {q['scenario']} | {q['n_seg']} | {'OK' if good else 'MISMATCH'} | "
                     f"{rel:.1e} | {r['iters']} vs {q['iters']} |")
    return ok, ["| scenario | n_seg | vs paper_tables.csv | J rel. diff | iterations |",
                "|:--|---:|:--|---:|:--|", *lines]


def check_inactive(grid):
    lines, n = [], 0
    ok = True
    for (arm, name, s), a in sorted(grid.items()):
        b = grid.get(("without", name, s))
        if arm != "with" or b is None:
            continue
        if min(a["halfspace_slack_km"], b["halfspace_slack_km"]) <= BM.ACTIVE_TOL_KM:
            continue
        n += 1
        rel = abs(b["objective"] / a["objective"] - 1)
        good = a["stop"] == b["stop"] and a["iters"] == b["iters"] and rel < 1e-9
        ok &= good
        lines.append(f"| {name} | {s} | {min(a['halfspace_slack_km'], b['halfspace_slack_km']):.1f} | "
                     f"{'OK' if good else 'DIFFERENT'} | {rel:.1e} |")
    return ok and n > 0, ["| scenario | n_seg | min half-space slack (km) | arms agree | J rel. diff |",
                          "|:--|---:|---:|:--|---:|", *lines]


def summarize():
    builds = json.loads((OUT / "builds.json").read_text())
    metas = {a: json.loads((OUT / a / "meta.json").read_text()) for a in ARMS}
    runs = {a: {(int(r["instance"]), int(r["n_seg"])): BM._typed(r)
                for r in BM.read_rows(OUT / a / "runs.csv")} for a in ARMS}
    grid = _grid()
    nsegs = metas["with"]["nsegs"]
    ids = sorted({i for (i, _), r in runs["with"].items() if r["kind"] == "sample"})
    feas = lambda r: r["hull_violation_km"] <= BT.CERT_TOL

    md = ["# Ablation: the normal-change term in the obstacle rows", "",
          "Two builds of the same source, differing in one line of "
          f"`{builds['patch']['file']}`: the \"without\" arm replaces `{PATCH_OLD}` with "
          f"`{PATCH_NEW}`, so the subproblem's obstacle rows keep the normal fixed at the "
          "reference. The merit, ratio test, mu, Delta_0 and stopping rules are identical. "
          "Produced by `tools/build_normal_term_ablation.py`; each arm's full Monte Carlo "
          "output is in `with/` and `without/`. The `without/` provenance line reports the "
          "repo source as clean because the patch was applied to a temporary copy; the "
          "extension hash is what identifies each build (`builds.json`, check A).", "",
          f"## Near-set Monte Carlo ({len(ids)} instances per n_seg)", "",
          "Converged = certified by the paper's definition (h <= 1e-6 km and a principled "
          "stop). Cells: median [5th, 95th percentile]; rates with 95% Wilson interval. "
          "The cost change pairs each instance across the arms, over instances whose final "
          "iterate satisfies the avoidance condition in both.", "",
          "| n_seg | converged, with (%) | converged, without (%) | iterations, with | "
          "iterations, without | without: stops | without, not converged but h <= 1e-6 | "
          "J without vs with (%) |",
          "|---:|:--|:--|:--|:--|:--|---:|:--|"]
    for s in nsegs:
        a = [runs["with"][(i, s)] for i in ids]
        b = [runs["without"][(i, s)] for i in ids]
        ca, cb = sum(r["certified"] for r in a), sum(r["certified"] for r in b)
        la, ha = BM.wilson(ca, len(a))
        lb, hb = BM.wilson(cb, len(b))
        stops = ", ".join(f"{BT._STOP_NAME.get(k, k)} {v}" for k, v in
                          sorted(collections.Counter(r["stop"] for r in b).items()))
        fails = [r for r in b if not r["certified"]]
        dj = [100 * (y["objective"] / x["objective"] - 1) for x, y in zip(a, b)
              if feas(x) and feas(y)]
        md.append(f"| {s} | {100*ca/len(a):.1f} [{la:.1f}, {ha:.1f}] | "
                  f"{100*cb/len(b):.1f} [{lb:.1f}, {hb:.1f}] | "
                  f"{BM.pct([r['iters'] for r in a], '{:.0f}')} | "
                  f"{BM.pct([r['iters'] for r in b], '{:.0f}')} | {stops} | "
                  f"{sum(feas(r) for r in fails)}/{len(fails)} | "
                  f"{BM.pct(dj)} (n={len(dj)}) |")

    md += ["", "## Table-2 scenarios", "",
           "A phantom rejection: the candidate satisfies the rows it was solved against and "
           "keeps every subdivided control point outside the sphere, but violates the rows "
           "rebuilt at itself with the normal re-aimed (tolerance 1e-6 km throughout).", "",
           "| scenario | n_seg | with: stop, iterations, rejects | without: stop, iterations, "
           "rejects (phantom) | J without vs with (%) |",
           "|:--|---:|:--|:--|---:|"]
    for name, _ in BT.T2_SCENARIOS:
        for s in nsegs:
            a, b = grid[("with", name, s)], grid[("without", name, s)]
            # A cost below the "with" arm's is no saving when the final iterate
            # violates the avoidance condition; show the violation instead.
            dj = (f"{100*(b['objective']/a['objective']-1):+.3f}" if feas(a) and feas(b)
                  else f"— (h = {b['hull_violation_km']:.3g} km)")
            md.append(f"| {name} | {s} | {BT._STOP_NAME[a['stop']]}, {a['iters']}, {a['rejects']} | "
                      f"{BT._STOP_NAME[b['stop']]}, {b['iters']}, {b['rejects']} "
                      f"({b['phantom_rejects']}) | {dj} |")
    rej = {arm: (sum(r["rejects"] for k, r in grid.items() if k[0] == arm),
                 sum(r["phantom_rejects"] for k, r in grid.items() if k[0] == arm)) for arm in ARMS}
    md += ["", "Rejected steps over the grid, total (phantom): "
           + "; ".join(f"{arm} {t} ({p})" for arm, (t, p) in rej.items()) + "."]

    ok_a, la_ = check_builds(builds, metas)
    ok_b, lb_ = check_paper(grid)
    ok_c, lc_ = check_inactive(grid)
    md += ["", f"## Check A -- each arm ran its own build: {'PASS' if ok_a else 'FAIL'}", "", *la_,
           "", f"## Check B -- the \"with\" arm reproduces the paper's tables: "
           f"{'PASS' if ok_b else 'FAIL'}", "", *lb_,
           "", f"## Check C -- where the obstacle constraint is inactive, the arms agree: "
           f"{'PASS' if ok_c else 'FAIL'}", "", *lc_]
    for a in ARMS:
        checks = [ln for ln in (OUT / a / "summary.md").read_text().splitlines()
                  if ln.startswith("## Check")]
        md += ["", f"Monte Carlo checks, `{a}` arm:", *[f"- {c[3:]}" for c in checks]]
    # harness_common.provenance() would stamp the extension THIS process imports,
    # which is neither arm's; name the source tree and both builds instead.
    src = builds["source"]
    md += ["", "---", "",
           f"_both arms built from commit `{src['commit']}`"
           + (" **with uncommitted changes in the code paths**" if src["dirty"] else ", source clean")
           + f"; this script sha256 `{src['script_sha256'][:12]}`; extensions: with "
           f"`{builds['with']['ext']}`, without `{builds['without']['ext']}`._"]
    H.write_text(OUT / "summary.md", "\n".join(md) + "\n")
    print((OUT / "summary.md").read_text())


def main():
    global OUT
    if len(sys.argv) == 5 and sys.argv[1] == "--grid-worker":
        grid_worker(*sys.argv[2:])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--n", type=int, default=1024, help="Monte Carlo instances")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    OUT = Path(args.out)
    if not args.summarize_only:
        run(args)
    summarize()


if __name__ == "__main__":
    main()
