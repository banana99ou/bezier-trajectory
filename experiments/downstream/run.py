#!/usr/bin/env python3
"""Command line entry point for the downstream comparison experiment.

    python -m experiments.downstream.run selftest
    python -m experiments.downstream.run freeze  --out artifacts/downstream/cases.json
    python -m experiments.downstream.run run     --cases artifacts/downstream/cases.json \
                                                 --outdir artifacts/downstream/primary

The whole experiment is reproducible from the committed case file plus the
`run` command (design section 7).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .cases import REPO_ROOT, Thresholds, freeze, git_provenance, load
from .pipeline import ARMS, BezierConfig, RunConfig, run_arm

# Flat CSV schema. One row per (case, arm); every field of design section 7 is
# present, and no column derives from a quantity the design prohibits.
CSV_FIELDS = [
    # identity
    "case_id", "stratum", "arm",
    "h0", "delta_a", "delta_i", "T_normed", "e0", "ef", "u_max", "h_min",
    # outcome
    "outcome", "note", "return_status", "solve_succeeded", "converged_flag",
    "horizon_s", "pass1_T_f", "horizon_left_bound",
    # structure
    "n_peaks_stage1", "profile_class", "n_phases", "node_allocation",
    "phase_boundaries",
    # objective
    "objective_solver", "objective_grader", "pass1_cost", "cost_recon_rel_err",
    # grader
    "grader_verdict", "pos_err_km", "vel_err_km_s", "min_altitude_km",
    "max_thrust_km_s2", "thrust_cap_km_s2", "altitude_floor_km",
    "resolution_check_passed", "corruption_check_passed",
    # bridge
    "bezier_feasible", "bezier_termination", "bezier_iterations",
    "bridge_residual", "gravity_model_gap_rel", "bezier_min_radius",
    # timing
    "t_stage1_s", "t_structure_s", "t_pass2_s", "t_total_s", "stage1_attempts",
]


def _row(res, case) -> dict:
    g = res.grade or {}
    b = res.bezier_info or {}
    return {
        "case_id": res.case_id, "stratum": res.stratum, "arm": res.arm,
        "h0": case.h0, "delta_a": case.delta_a, "delta_i": case.delta_i,
        "T_normed": case.T_normed, "e0": case.e0, "ef": case.ef,
        "u_max": case.u_max, "h_min": case.h_min,
        "outcome": res.outcome, "note": res.note,
        "return_status": res.return_status,
        "solve_succeeded": res.solve_succeeded,
        "converged_flag": res.converged_flag,
        "horizon_s": res.horizon_s, "pass1_T_f": res.pass1_T_f,
        "horizon_left_bound": res.horizon_left_bound,
        "n_peaks_stage1": res.n_peaks_stage1,
        "profile_class": res.profile_class, "n_phases": res.n_phases,
        "node_allocation": json.dumps(res.node_allocation),
        "phase_boundaries": json.dumps(
            [[round(a, 6), round(b_, 6)] for a, b_ in res.phase_boundaries]),
        "objective_solver": res.objective_solver,
        "objective_grader": g.get("objective_grader"),
        "pass1_cost": res.pass1_cost,
        "cost_recon_rel_err": g.get("cost_recon_rel_err"),
        "grader_verdict": g.get("verdict"),
        "pos_err_km": g.get("pos_err_km"),
        "vel_err_km_s": g.get("vel_err_km_s"),
        "min_altitude_km": g.get("min_altitude_km"),
        "max_thrust_km_s2": g.get("max_thrust_km_s2"),
        "thrust_cap_km_s2": g.get("thrust_cap_km_s2"),
        "altitude_floor_km": g.get("altitude_floor_km"),
        "resolution_check_passed": (res.resolution_check or {}).get("passed"),
        "corruption_check_passed": (res.corruption_check or {}).get("passed"),
        "bezier_feasible": b.get("feasible"),
        "bezier_termination": b.get("termination_reason"),
        "bezier_iterations": b.get("iterations"),
        "bridge_residual": b.get("bridge_residual"),
        "gravity_model_gap_rel": b.get("gravity_model_gap_rel"),
        "bezier_min_radius": b.get("min_radius"),
        "t_stage1_s": res.t_stage1_s, "t_structure_s": res.t_structure_s,
        "t_pass2_s": res.t_pass2_s, "t_total_s": res.t_total_s,
        "stage1_attempts": res.stage1_attempts,
    }


@contextlib.contextmanager
def _muted(enabled: bool):
    """Silence the solver banners on the C-level file descriptors.

    IPOPT and CasADi write from C, so redirecting `sys.stdout` alone leaves the
    output in place; the file descriptor itself has to be swapped.
    """
    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def cmd_selftest(args) -> int:
    """The grader's own checks. Reported before any experimental number so a
    void oracle is visible up front."""
    from . import grader

    print("Grader self-test")
    print("  what would make this fail: the exact coasting arc scored as a")
    print("  miss, the corrupted arc accepted, the bridge invariant drifting")
    print("  off machine precision, or the resolution check failing to fire")
    print("  on a deliberately crippled integrator.")
    print()
    ok = grader.self_test(verbose=True)
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_freeze(args) -> int:
    out = Path(args.out)
    payload = freeze(
        out_path=out,
        n_stratum_b=args.n_stratum_b,
        seed=args.seed,
        thresholds=Thresholds(),
    )
    print(f"wrote {out}")
    print(f"  stratum A : {payload['stratum_a']['n_distinct']} distinct cases "
          f"from {payload['stratum_a']['n_rows']} rows")
    print(f"  stratum B : {payload['stratum_b']['n']} designed cases "
          f"(seed {payload['seed']})")
    print(f"  outcome columns read from the database: "
          f"{payload['stratum_a']['outcome_columns_read']}")
    print(f"  commit    : {payload['frozen_by']['commit'][:12]} "
          f"(dirty={payload['frozen_by']['dirty']})")
    return 0


def cmd_run(args) -> int:
    cases, thresholds, meta = load(Path(args.cases))
    if args.stratum:
        cases = [c for c in cases if c.stratum in set(args.stratum)]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [c for c in cases if c.case_id in wanted]
    if args.limit:
        cases = cases[:args.limit]
    arms = args.arms or list(ARMS)

    bez = BezierConfig()
    run_cfg = RunConfig(recovery=args.recovery, grade=not args.no_grade,
                        self_checks=args.self_checks)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = outdir / "results.csv", outdir / "results.json"

    prov = git_provenance()
    header = {
        "design_doc": "doc/dcm_downstream_experiment_design.md",
        "case_file": str(Path(args.cases)),
        "case_file_frozen_by": meta.get("frozen_by"),
        "produced_by": prov,
        "arms": arms,
        "bezier_config": asdict(bez),
        "run_config": asdict(run_cfg),
        "thresholds": asdict(thresholds),
        "n_cases": len(cases),
        "started_at_unix": time.time(),
    }
    if prov["dirty"]:
        print("WARNING: the working tree is dirty; these artifacts are not "
              "reproducible from the recorded commit alone.", file=sys.stderr)

    rows, records = [], []
    t0 = time.perf_counter()
    for i, case in enumerate(cases, 1):
        for arm in arms:
            t_arm = time.perf_counter()
            with _muted(not args.verbose):
                res = run_arm(arm, case, thresholds, bez=bez, run=run_cfg)
            rows.append(_row(res, case))
            records.append(asdict(res))
            print(f"[{i}/{len(cases)}] {case.case_id} {arm:6s} "
                  f"{res.outcome:19s} {time.perf_counter() - t_arm:6.1f}s "
                  f"{res.note[:70]}", flush=True)

            # Written after every arm so an interrupted run still leaves a
            # readable partial artifact.
            with csv_path.open("w", newline="\n") as fh:
                w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            json_path.write_text(json.dumps(
                {**header, "elapsed_s": time.perf_counter() - t0,
                 "results": records}, indent=2, default=str) + "\n",
                newline="\n")

    counts: dict = {}
    for r in rows:
        counts.setdefault(r["arm"], {}).setdefault(r["outcome"], 0)
        counts[r["arm"]][r["outcome"]] += 1
    print()
    print(f"{len(rows)} rows -> {csv_path}")
    for arm in arms:
        if arm in counts:
            print(f"  {arm:6s} " + "  ".join(
                f"{k}={v}" for k, v in sorted(counts[arm].items())))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.downstream.run")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="run the grader's own checks")
    s.set_defaults(func=cmd_selftest)

    s = sub.add_parser("freeze", help="write the frozen case file")
    s.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "downstream"
                                        / "cases.json"))
    s.add_argument("--n-stratum-b", type=int, default=100)
    s.add_argument("--seed", type=int, default=20260812)
    s.set_defaults(func=cmd_freeze)

    s = sub.add_parser("run", help="run the arms over a frozen case file")
    s.add_argument("--cases", required=True)
    s.add_argument("--outdir", required=True)
    s.add_argument("--arms", nargs="+", choices=list(ARMS))
    s.add_argument("--stratum", nargs="+", choices=["A", "B"])
    s.add_argument("--case-id", nargs="+")
    s.add_argument("--limit", type=int)
    s.add_argument("--recovery", action="store_true",
                   help="enable the retry ladder for EVERY arm (the secondary "
                        "'as shipped' run of design section 4.4)")
    s.add_argument("--self-checks", action="store_true",
                   help="run the resolution and corruption checks per case")
    s.add_argument("--no-grade", action="store_true")
    s.add_argument("--verbose", action="store_true",
                   help="let the solver banners through")
    s.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
