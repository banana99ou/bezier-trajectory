#!/usr/bin/env python3
"""
Monte Carlo over perturbed baseline instances, solved at every subdivision count.

Professor's feedback of 2026-09-30 (papers/feedback/paper_draft_feedback_applied_
11c772e_PF.md, p.11): Table 3 and Figure 5 rest on ONE geometry, the 120 deg
baseline. This tool draws instances from a box around that baseline, solves each
one at every n_seg in NSEGS -- the same instances at every n_seg, so comparisons
between subdivision counts are paired within one geometry -- and reports per-n_seg
statistics.

Run:
    .venv/bin/python tools/build_monte_carlo.py                 # near set, 1024
    .venv/bin/python tools/build_monte_carlo.py --ranges wide --n 4096
    .venv/bin/python tools/build_monte_carlo.py --n 16 --out /tmp/mc_pilot
    .venv/bin/python tools/build_monte_carlo.py --summarize-only --out DIR

What is perturbed (RANGE_SETS) and why. `near` asks whether the baseline's
subdivision trends hold around it; `wide` asks where the method stops working
across the family of transfers the paper's scenarios belong to.
  lag           argument-of-latitude gap, the central angle when coplanar.
                near 110..130 deg: keeps the obstacle constraint active (it goes
                inactive below the ~100 deg free-fall angle at T = 1500 s).
                wide 100..170 deg: from that activation angle up to the paper's
                hardest scenario. The geometric central angle is recorded too.
  progress_alt  departure altitude. near 225..265 km, wide 195..295 km.
  iss_alt       arrival altitude. near 380..420 km, wide 350..450 km.
  dinc, draan   arrival-plane inclination / RAAN offsets. near -5..5 deg each;
                wide -20..20 and -15..15 deg, which contains the paper's plane-
                change scenario (+20, +15).
  iss_u         arrival argument of latitude, 0..360 deg in both sets. The
                baseline's 45 deg is arbitrary; it sets where the transfer sits
                in latitude, which is what J2 sees.
  Held fixed: inc = 51.64 deg, departure RAAN = 0 (J2 is symmetric about z, so a
  common RAAN is a pure rotation), R_KOZ, T, N = 7, n_lin = 100, tolerances --
  all as in the paper.

Initial trust-region size, per instance, by rule: Delta_0 = 2000 km, doubled
until it exceeds the iteration-1 repair distance (2-norm) -- the distance the
boundary conditions move p_1 and p_{N-1} in the first QP, which must fit inside
the box. This is the paper's own rule (2000 km, and 4000 km for the 170 deg
scenario whose repair is 2416 km), written down once and applied to every
instance; above the threshold the converged answer does not move
(harness_common, phase135/phase170 notes). Every near-set instance gets 2000 km.

Two check instances are solved but excluded from every statistic:
  baseline   the unperturbed phase120 geometry. It must reproduce the committed
             Table 3 rows in doc/results/paper_tables.csv; a mismatch means this
             tool is not solving the paper's problem.
  rotated    the baseline turned by ROT_DEG about Earth's axis. The problem is
             invariant under that rotation, so the converged solution must match
             the baseline. The trust region is an axis-aligned box, which is NOT
             rotation invariant, so the iteration count may legitimately differ;
             a different converged solution would not be legitimate.

Sampling: Latin hypercube (strength 1, no optimization, seeded) by default. Its
variance for any finite-variance output, the 0/1 certificate included, is at most
n/(n-1) times that of independent sampling (Owen, Monte Carlo theory, methods and
examples, Cor. 17.1), so the usual binomial and order-statistic error bars stay
valid. Scrambled Sobol buys little here: the outputs that matter are a pass/fail
indicator and tail percentiles, both discontinuous, and a single scrambled set
has no iid error bars. `--sampler sobol|random` remain for comparison.

Determinism check: the first RECHECK sampled instances are solved a second time
at every n_seg, and every deterministic output must match bit for bit.

Nothing is filtered silently: every (instance, n_seg) solve is one CSV row, a
solver exception is a row with `error` set, and the summary counts every stop
reason. Timing: the Rust solver is single-threaded, so `--workers` scales, but
parallel solves share caches and clocks, so the summary reports iterations, not
runtime; runs.csv keeps both clocks as diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_tables as BT  # noqa: E402
from tools.verify import harness_common as H  # noqa: E402
from orbital_docking.de_casteljau import segment_matrices_equal_params  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "doc" / "results"

NSEGS = (4, 8, 16, 32, 64)
DEGREE = 7
TRUST_BASE_KM = 2000.0
TRUST_RULE = "2000 km doubled until it exceeds the iteration-1 repair (2-norm)"
ROT_DEG = 37.0
RECHECK = 32
ACTIVE_TOL_KM = 1e-3   # a half-space row this close to equality is binding
DET_KEYS = ("certified", "koz_binds", "margin_km", "ctrl_cost_ms2", "objective",
            "iters", "stop", "hull_violation_km")

BASE = dict(lag=120.0, progress_alt=245.0, iss_alt=400.0,
            dinc=0.0, draan=0.0, iss_u=45.0, raan=0.0)
RANGE_SETS = {
    "near": {
        "lag": (110.0, 130.0),
        "progress_alt": (225.0, 265.0),
        "iss_alt": (380.0, 420.0),
        "dinc": (-5.0, 5.0),
        "draan": (-5.0, 5.0),
        "iss_u": (0.0, 360.0),
    },
    "wide": {
        "lag": (100.0, 170.0),
        "progress_alt": (195.0, 295.0),
        "iss_alt": (350.0, 450.0),
        "dinc": (-20.0, 20.0),
        "draan": (-15.0, 15.0),
        "iss_u": (0.0, 360.0),
    },
}
FIXED = dict(inc=51.64, r_e=6471.0, T=1500.0)
PARAMS = tuple(RANGE_SETS["near"])
assert all(tuple(r) == PARAMS for r in RANGE_SETS.values())

FIELDS = list(["instance", "kind", *PARAMS, "raan", "central_angle_deg",
           "repair_inf_km", "repair_2_km", "trust_radius_km",
           "n_seg", "certified", "dense_probe_ok", "disagree", "koz_binds",
           "tau_star", "halfspace_slack_km", "bind_slack_km", "margin_km", "l_seg_km", "pred_margin_km", "ratio",
           "ctrl_cost_ms2", "objective", "runtime_s", "cpu_s", "iters", "stop",
           "stop_name", "hull_violation_km", "error"])


# --------------------------------------------------------------------------
# Instances
# --------------------------------------------------------------------------

def sample_unit(sampler, n, seed):
    """(n, d) points in [0, 1)^d."""
    from scipy.stats import qmc
    d = len(PARAMS)
    if sampler == "sobol":
        m = int(round(math.log2(n)))
        if 2 ** m != n:
            sys.exit(f"--sampler sobol needs a power-of-two --n, got {n}")
        return qmc.Sobol(d, scramble=True, rng=seed).random_base2(m)
    if sampler == "lhs":
        return qmc.LatinHypercube(d, rng=seed, optimization=None).random(n)
    if sampler == "random":
        return np.random.default_rng(seed).random((n, d))
    sys.exit(f"unknown sampler {sampler!r}")


def instances(sampler, n, seed, ranges):
    """Check instances first (negative ids), then the n sampled ones."""
    out = [dict(instance=-2, kind="baseline", **BASE),
           dict(instance=-1, kind="rotated", **{**BASE, "raan": ROT_DEG})]
    u = sample_unit(sampler, n, seed)
    for i, row in enumerate(u):
        p = {k: lo + (hi - lo) * float(x) for (k, (lo, hi)), x in zip(ranges.items(), row)}
        out.append(dict(instance=i, kind="sample", raan=BASE["raan"], **p))
    return out


def scenario(inst, degree=DEGREE):
    """harness_common.make_scenario, generalized from named presets to parameters."""
    r0_km = H.R_E_EARTH + inst["progress_alt"]
    r1_km = H.R_E_EARTH + inst["iss_alt"]
    raan0 = inst["raan"]
    P_start, v0 = H.eci_from_circular(r0_km, FIXED["inc"], raan0,
                                      inst["iss_u"] - inst["lag"])
    P_end, v1 = H.eci_from_circular(r1_km, FIXED["inc"] + inst["dinc"],
                                    raan0 + inst["draan"], inst["iss_u"])
    P_init = np.asarray(H.generate_initial_control_points(degree, P_start, P_end), float)
    return dict(N=degree, P_start=P_start, P_end=P_end, v0=v0, v1=v1,
                r_e=FIXED["r_e"], T=FIXED["T"], P_init=P_init)


def trust_radius(repair_2_km):
    """TRUST_RULE: the first QP's box must contain the boundary-condition repair."""
    d = TRUST_BASE_KM
    while repair_2_km >= d:
        d *= 2.0
    return d


def central_angle(sc):
    """Angle at Earth's centre between the two endpoints (deg)."""
    a, b = sc["P_start"], sc["P_end"]
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def halfspace_slacks(P, n_seg, r_e):
    """Supporting-half-space slack of each sub-arc: min over its control points (km).

    Section 3.1's row, gamma = n^T (q - c_KOZ) - R_KOZ with n from the sub-arc
    centroid and c_KOZ at the origin. About zero means the obstacle constraint
    shapes the solution; tens of km means it never touched it. An interior
    closest approach alone does not say which: a transfer that never nears the
    obstacle also has its minimum radius between the endpoints.
    """
    out = []
    for A in segment_matrices_equal_params(P.shape[0] - 1, n_seg):
        Q = A @ P
        c = Q.mean(axis=0)
        out.append(float((Q @ (c / np.linalg.norm(c))).min() - r_e))
    return out


def repair_distance(sc):
    """(inf-norm, 2-norm) km the boundary conditions move p_1 and p_{N-1} at iteration 1.

    The trust region is |x_i - p_i| <= Delta per coordinate, so the first QP can
    only meet the velocity rows if the inf-norm figure is <= Delta_0. The 2-norm is
    the figure build_tables.py quotes; it bounds the inf-norm from above.
    """
    N, T, P = sc["N"], sc["T"], sc["P_init"]
    d1 = sc["P_start"] + (T / N) * sc["v0"] - P[1]
    d2 = sc["P_end"] - (T / N) * sc["v1"] - P[N - 1]
    return (float(max(np.abs(d1).max(), np.abs(d2).max())),
            float(max(np.linalg.norm(d1), np.linalg.norm(d2))))


# --------------------------------------------------------------------------
# One solve -- same metrics, same definitions as build_tables.measure
# --------------------------------------------------------------------------

def solve(task):
    inst, n_seg = task
    sc = scenario(inst)
    rep_inf, rep_2 = repair_distance(sc)
    row = {k: inst[k] for k in ("instance", "kind", *PARAMS, "raan")}
    delta0 = trust_radius(rep_2)
    row.update(central_angle_deg=central_angle(sc), repair_inf_km=rep_inf,
               repair_2_km=rep_2, trust_radius_km=delta0, n_seg=n_seg, error="")
    try:
        # CPU time of this process across the (single-threaded) solve: unlike the
        # wall clock it does not count time spent descheduled by other work.
        c0 = time.process_time()
        P, info = H.run_rust(sc, n_seg=n_seg, scp_trust_radius=delta0)
        cpu = time.process_time() - c0
    except Exception as e:  # recorded, never dropped
        row["error"] = f"{type(e).__name__}: {e}"[:300]
        return row
    hull = float(info["final_hull_violation_km"])
    stop = int(info["scvx_stop_reason"])
    certified = hull <= BT.CERT_TOL and stop in BT.PRINCIPLED_STOPS
    dense_ok = bool(info["feasible"])
    binds, tau_star = BT._koz_binds(P)
    s_bind = min(int(tau_star * n_seg), n_seg - 1)
    slacks = halfspace_slacks(P, n_seg, sc["r_e"])
    a, b = H.positions(P, np.array([s_bind / n_seg, (s_bind + 1) / n_seg]))
    l_seg = float(np.linalg.norm(b - a))
    pred = l_seg ** 2 / (8.0 * sc["r_e"])
    margin = float(info["min_radius"]) - sc["r_e"]
    row.update(
        certified=certified, dense_probe_ok=dense_ok, disagree=(certified != dense_ok),
        koz_binds=binds, tau_star=tau_star,
        halfspace_slack_km=min(slacks), bind_slack_km=slacks[s_bind],
        margin_km=margin, l_seg_km=l_seg,
        pred_margin_km=pred, ratio=(margin / pred) if binds else float("nan"),
        ctrl_cost_ms2=float(info["mean_control_accel_ms2"]),
        objective=float(info["cost"]), runtime_s=float(info["elapsed_time"]), cpu_s=cpu,
        iters=int(info["iterations"]), stop=stop,
        stop_name=BT._STOP_NAME.get(stop, "?"), hull_violation_km=hull)
    return row


# --------------------------------------------------------------------------
# Run bookkeeping: incremental CSV, resumable, refuses to mix configurations
# --------------------------------------------------------------------------

def run_meta(args):
    # workers is part of the configuration because it changes what the runtime
    # column measures: a resumed run must not mix serial and parallel timings.
    # script_sha256 ties the run to this exact file even before it is committed.
    return dict(sampler=args.sampler, n=args.n, seed=args.seed, nsegs=list(args.nsegs),
                workers=args.workers, range_set=args.ranges,
                degree=DEGREE, trust_rule=TRUST_RULE, rot_deg=ROT_DEG,
                base=BASE, ranges=RANGE_SETS[args.ranges], fixed=FIXED,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                commit=H._git("rev-parse", "HEAD", default="unknown"),
                dirty=bool(H._git("status", "--porcelain", "--", *H._CODE_PATHS,
                                  "tools/build_monte_carlo.py", "tools/build_tables.py")),
                ext=H._ext_digest())


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _repair_csv(path):
    """Drop rows a crash left incomplete; returns how many were dropped.

    The host may lose power mid-write, which can leave a truncated last line or,
    on ext4, trailing garbage. A row survives only if it has every field and
    parseable keys; everything else is re-solved on resume.
    """
    if not path.exists():
        return 0
    lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    if not lines or lines[0].split(",") != FIELDS:
        return 0
    keep = [lines[0]]
    for ln in lines[1:]:
        cells = next(csv.reader([ln]), [])
        try:
            ok = len(cells) == len(FIELDS) and int(cells[0]) >= -2 and int(cells[FIELDS.index("n_seg")]) > 0
        except ValueError:
            ok = False
        if ok:
            keep.append(ln)
    dropped = len(lines) - len(keep)
    if dropped:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("\n".join(keep) + "\n")
        tmp.replace(path)
    return dropped


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path, csv_path = out / "meta.json", out / "runs.csv"
    meta = json.loads(json.dumps(run_meta(args)))  # tuples -> lists, as stored
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old != meta:
            diff = sorted(k for k in meta if old.get(k) != meta.get(k))
            sys.exit(f"{out} holds a run with a different configuration ({', '.join(diff)}). "
                     "Use a new --out; mixing code versions or samples is not one result.")
    else:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    insts = instances(args.sampler, args.n, args.seed, RANGE_SETS[args.ranges])
    inst_path = out / "instances.csv"
    if not inst_path.exists():
        H.write_csv(inst_path, [{k: i[k] for k in ("instance", "kind", *PARAMS, "raan")}
                                for i in insts])
    _solve_into(csv_path, insts, args.nsegs, args.workers, "main pass")
    # Same instances again: every deterministic output must match bit for bit.
    recheck = [i for i in insts if i["kind"] == "sample"][:RECHECK]
    _solve_into(out / "recheck.csv", recheck, args.nsegs, args.workers, "determinism recheck")


def _solve_into(csv_path, insts, nsegs, workers, label):
    """Solve every (instance, n_seg) not already in csv_path, appending as it goes."""
    dropped = _repair_csv(csv_path)
    if dropped:
        print(f"{label}: dropped {dropped} incomplete row(s) left by an interrupted run", flush=True)
    done = {(int(r["instance"]), int(r["n_seg"])) for r in read_rows(csv_path)}
    tasks = [(inst, s) for inst in insts for s in nsegs if (inst["instance"], s) not in done]
    print(f"{label}: {len(done)} solves already in {csv_path.name}; {len(tasks)} to go, "
          f"{workers} worker(s)", flush=True)
    if not tasks:
        return
    new_file = not csv_path.exists()
    t0 = time.time()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            w.writeheader()
        if workers == 1:
            it = map(solve, tasks)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(workers)
            it = pool.imap_unordered(solve, tasks, chunksize=4)
        try:
            for k, row in enumerate(it, 1):
                w.writerow(row)
                f.flush()
                if k % 50 == 0 or k == len(tasks):
                    el = time.time() - t0
                    eta = el / k * (len(tasks) - k)
                    print(f"  {k}/{len(tasks)}  elapsed {el/60:.1f} min  eta {eta/60:.1f} min",
                          flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion, in percent."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (c - h), 100 * (c + h)


def pct(xs, fmt="{:.3g}"):
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    if xs.size == 0:
        return "—"
    p5, p50, p95 = np.percentile(xs, [5, 50, 95])
    return f"{fmt.format(p50)} [{fmt.format(p5)}, {fmt.format(p95)}]"


def _typed(r):
    t = dict(r)
    for k in ("instance", "n_seg", "iters", "stop"):
        t[k] = int(r[k]) if r.get(k) not in (None, "") else None
    for k in ("margin_km", "pred_margin_km", "ratio", "ctrl_cost_ms2", "objective",
              "runtime_s", "cpu_s", "hull_violation_km", "repair_inf_km",
              "central_angle_deg", "trust_radius_km", "halfspace_slack_km", "bind_slack_km",
              *PARAMS):
        t[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
    for k in ("certified", "dense_probe_ok", "disagree", "koz_binds"):
        t[k] = r.get(k) == "True"
    return t


def _pressed(r):
    """The sub-arc holding the closest approach is itself tight against its half-space.

    Section 3.1's estimate is built from that sub-arc's chord with the sub-arc
    lying on its supporting plane. When the closest approach falls in a sub-arc
    whose own constraint is slack -- a tight neighbour pushed the curve -- the
    estimate is computed from the wrong chord and does not apply.
    """
    s = r.get("bind_slack_km", float("nan"))
    return bool(np.isfinite(s) and s <= ACTIVE_TOL_KM)


def _active(r):
    """Some supporting half-space row is (numerically) tight at the solution."""
    s = r.get("halfspace_slack_km", float("nan"))
    return bool(np.isfinite(s) and s <= ACTIVE_TOL_KM)


def check_baseline(by):
    """Baseline rows against the committed Table 3 rows. Returns markdown lines."""
    ref = {int(r["n_seg"]): r for r in read_rows(REPO_ROOT / "doc/results/paper_tables.csv")
           if r["scenario"] == "phase120" and int(r["degree"]) == DEGREE}
    lines, ok = [], True
    for (inst, s), r in sorted(by.items()):
        if inst != -2 or s not in ref:
            continue
        q = ref[s]
        same = (r["certified"] == (q["certified"] == "True") and r["iters"] == int(q["iters"])
                and r["koz_binds"] == (q["koz_binds"] == "True"))
        rel = max(abs(r[k] / float(q[k]) - 1) for k in ("margin_km", "ctrl_cost_ms2", "objective"))
        # paper_tables.csv stores 7 significant digits, so the stored value is
        # itself rounded by up to 5e-7 relative; a real change of code or local
        # optimum moves these numbers far more than that.
        good = same and rel < 1e-6
        ok &= good
        lines.append(f"| {s} | {'OK' if good else 'MISMATCH'} | {rel:.1e} | "
                     f"{r['iters']} vs {q['iters']} |")
    head = ["| n_seg | vs Table 3 | max rel. diff (margin, cost, J) | iterations |",
            "|---:|:--|---:|:--|"]
    return ok, head + lines


def check_rotation(by, nsegs):
    lines, ok = [], True
    for s in nsegs:
        a, b = by.get((-2, s)), by.get((-1, s))
        if not a or not b:
            continue
        rel = max(abs(b[k] / a[k] - 1) for k in ("margin_km", "ctrl_cost_ms2", "objective"))
        good = a["certified"] == b["certified"] and rel < 1e-6
        ok &= good
        lines.append(f"| {s} | {'OK' if good else 'DIFFERENT'} | {rel:.1e} | "
                     f"{a['iters']} vs {b['iters']} |")
    head = ["| n_seg | rotated vs baseline | max rel. diff (margin, cost, J) | iterations |",
            "|---:|:--|---:|:--|"]
    return ok, head + lines


def summarize(args):
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    rows = [_typed(r) for r in read_rows(out / "runs.csv")]
    nsegs = meta["nsegs"]
    by = {(r["instance"], r["n_seg"]): r for r in rows}
    samples = [r for r in rows if r["kind"] == "sample"]
    ids = sorted({r["instance"] for r in samples})
    top = max(nsegs)

    md = [f"# Monte Carlo over perturbed baseline instances",
          "",
          f"- range set `{meta.get('range_set', 'near')}`, sampler `{meta['sampler']}`, "
          f"{len(ids)} of {meta['n']} instances present, seed {meta['seed']}, "
          f"degree {meta['degree']}",
          f"- Delta_0: {meta.get('trust_rule', 'fixed 2000 km')}; instances by Delta_0: "
          + ", ".join(f"{d:.0f} km: {c}" for d, c in sorted(collections.Counter(
              by[(i, nsegs[0])]["trust_radius_km"] for i in ids if (i, nsegs[0]) in by
              and np.isfinite(by[(i, nsegs[0])]["trust_radius_km"])).items())),
          "- ranges: " + ", ".join(f"{k} {lo:g}..{hi:g}" for k, (lo, hi) in meta["ranges"].items()),
          "- cells: median [5th, 95th percentile]; rates with 95% Wilson interval",
          "",
          "- constraint active: some half-space row within "
          f"{ACTIVE_TOL_KM:g} km of equality; interior: closest approach strictly "
          "between the endpoints; closest sub-arc pressed: the sub-arc holding the closest "
          "approach is itself within that tolerance. The clearance ratio uses certified, "
          "active, interior solves whose closest sub-arc is pressed -- where the section 3.1 "
          "estimate applies. Cost is the objective J; its change uses certified solves where "
          "the constraint is active.",
          "",
          "| n_seg | solves | errors | certified % [95% CI] | constraint active % | "
          "interior % | closest sub-arc pressed % | "
          f"measured/estimated clearance | J vs n_seg={top} (%) | iterations |",
          "|---:|---:|---:|:--|---:|---:|---:|:--|:--|:--|"]
    for s in nsegs:
        rs = [by[(i, s)] for i in ids if (i, s) in by]
        n = len(rs)
        err = sum(1 for r in rs if r["error"])
        cert = [r for r in rs if not r["error"] and r["certified"]]
        lo, hi = wilson(len(cert), n)
        act = [r for r in cert if _active(r)]
        inter = [r for r in cert if r["koz_binds"]]
        active = [r for r in act if r["koz_binds"]]
        pressed = [r for r in active if _pressed(r)]
        dc = []
        for i in ids:
            a, b = by.get((i, s)), by.get((i, top))
            # Only where the constraint shapes the solution at this n_seg: an
            # instance it never touches has the same cost at every n_seg.
            if a and b and a["certified"] and b["certified"] and _active(a):
                dc.append(100 * (a["objective"] / b["objective"] - 1))
        md.append(
            f"| {s} | {n} | {err} | {100*len(cert)/max(n,1):.1f} [{lo:.1f}, {hi:.1f}] | "
            f"{100*len(act)/max(len(cert),1):.1f} | {100*len(inter)/max(len(cert),1):.1f} | "
            f"{100*len(pressed)/max(len(active),1):.1f} | "
            f"{pct([r['ratio'] for r in pressed])} max {max([r['ratio'] for r in pressed], default=float('nan')):.3f} | "
            f"{pct(dc)} | {pct([r['iters'] for r in cert], '{:.0f}')} "
            f"max {max([r['iters'] for r in rs if r['iters'] is not None], default=0)}, "
            f">100: {sum(1 for r in rs if (r['iters'] or 0) > 100)} |")

    md += ["", "## Clearance ratio by whether the closest sub-arc is pressed", "",
           "| n_seg | pressed: n, ratio | not pressed: n, ratio |", "|---:|:--|:--|"]
    for s in nsegs:
        a = [by[(i, s)] for i in ids if (i, s) in by and by[(i, s)]["certified"]
             and _active(by[(i, s)]) and by[(i, s)]["koz_binds"]]
        y = [r["ratio"] for r in a if _pressed(r)]
        z = [r["ratio"] for r in a if not _pressed(r)]
        md.append(f"| {s} | {len(y)}, {pct(y)} | {len(z)}, {pct(z)} |")

    # Per-instance scaling exponent over n_seg >= 8, where the closest approach is
    # interior at every such n_seg; the section 3.1 estimate predicts -2.
    fit_s = [s for s in nsegs if s >= 8]
    slopes = []
    for i in ids:
        rs = [by.get((i, s)) for s in fit_s]
        if all(r and r["certified"] and r["koz_binds"] and _active(r) and r["margin_km"] > 0
               for r in rs):
            slopes.append(np.polyfit(np.log(fit_s), np.log([r["margin_km"] for r in rs]), 1)[0])
    md += ["", f"Clearance scaling exponent over n_seg {fit_s} (per instance, "
               f"{len(slopes)} instances): {pct(slopes)}; the arc estimate predicts -2."]

    # Certification against the input that sets difficulty most directly: where
    # in the range does the method stop working, if anywhere?
    edges = list(range(100, 181, 10))
    md += ["", "## Certified / solved, by geometric central angle", "",
           "| central angle (deg) | " + " | ".join(f"n_seg {s}" for s in nsegs) + " |",
           "|:--|" + "---:|" * len(nsegs)]
    for lo, hi in zip(edges[:-1], edges[1:]):
        cells = []
        for s in nsegs:
            rs = [by[(i, s)] for i in ids if (i, s) in by
                  and lo <= by[(i, s)]["central_angle_deg"] < hi]
            cells.append(f"{sum(1 for r in rs if r['certified'])}/{len(rs)}" if rs else "—")
        if any(c != "—" for c in cells):
            md.append(f"| {lo}–{hi} | " + " | ".join(cells) + " |")

    md += ["", "## Every non-certified solve, by stop reason", "",
           "| n_seg | stop reason | count |", "|---:|:--|---:|"]
    for s in nsegs:
        tally = {}
        for i in ids:
            r = by.get((i, s))
            if r and (r["error"] or not r["certified"]):
                key = r["error"].split(":")[0] if r["error"] else r["stop_name"]
                tally[key] = tally.get(key, 0) + 1
        for k, v in sorted(tally.items()):
            md.append(f"| {s} | {k} | {v} |")
    dis = sum(1 for r in samples if r["disagree"])
    md += ["", f"Certificate vs dense-probe disagreements: {dis}"]

    det = [_typed(r) for r in read_rows(out / "recheck.csv")]
    bad = [(r["instance"], r["n_seg"]) for r in det
           if (r["instance"], r["n_seg"]) not in by
           or any(r[k] != by[(r["instance"], r["n_seg"])][k] for k in DET_KEYS)]
    md += ["", f"## Check 3 -- re-solving {len({r['instance'] for r in det})} instances "
               f"gives identical outputs: {'PASS' if det and not bad else 'FAIL'}",
           "", f"{len(det)} repeated solves, {len(bad)} differing in any of {', '.join(DET_KEYS)}"
           + (f": {bad[:10]}" if bad else ".")]

    ok_b, lb = check_baseline(by)
    ok_r, lr = check_rotation(by, nsegs)
    md += ["", f"## Check 1 -- baseline reproduces Table 3: {'PASS' if ok_b else 'FAIL'}", "", *lb,
           "", f"## Check 2 -- rotation about Earth's axis leaves the solution unchanged: "
           f"{'PASS' if ok_r else 'FAIL'}", "", *lr]
    md += ["", f"This run used {meta['workers']} worker(s). No runtime is summarized: a "
           "solve's cost per iteration is set by n_seg and N, not by the geometry, so the "
           "runtime-versus-n_seg claim rests on the baseline timing (Table 3, min of 15 on "
           "the paper's machine) and this study reports iterations. runs.csv keeps the "
           "solver's wall clock `runtime_s` and the process CPU time `cpu_s` as diagnostics."]
    H.write_text(out / "summary.md", "\n".join(md) + "\n" + H.provenance())
    print((out / "summary.md").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--sampler", choices=("lhs", "random", "sobol"), default="lhs")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--nsegs", type=int, nargs="+", default=list(NSEGS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--ranges", choices=tuple(RANGE_SETS), default="near")
    ap.add_argument("--out", default=None,
                    help="default doc/results/monte_carlo_subdivision[_wide]")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.ranges == "near" else f"_{args.ranges}"
        args.out = str(RESULTS / f"monte_carlo_subdivision{suffix}")
    if not args.summarize_only:
        run(args)
    summarize(args)


if __name__ == "__main__":
    main()
