#!/usr/bin/env python3
"""The bench runner: one JSONL record per run, and nothing else.

Built to `paper/journal_1/demo_specs/C_traffic_and_backbone.md` Part 2, with the
adaptations below stated rather than made silently (2026-09-29):

* **Starting weight in `--scenarios` mode is `DEFAULT_INITIAL_ELASTIC_WEIGHT`
  (100, escalating in-loop)**, not `scenario_elastic_weight(name)`. The spec was
  written in the ladder era; the committed `SOLVER.md` §Measurements table and
  `optimize_scenario` both start every run at 100 now, and the backbone table has
  to regenerate THAT table.
* **`--starts`** is a record field, not a retry: `own` (the scene's own start),
  `straight`, or `wait` -- the starts the package already has. A run from a
  different start is a different record; nothing here picks the best of several.
* **The `realistic` arm is refused by name**: `families.py` has no `realistic`
  placement mode yet (spec §1.3). Asking for it fails at plan time.
* The gate row comes from `optimize.result_row`, the one mapping
  `optimize_scenario` also uses -- the bench times the solve itself (spec §2.3)
  but never re-implements the verdict.

The runner never prints a table and never rounds; `tools/make_tables.py` does.

    python3 tools/bench.py --out bench/backbone.jsonl --scenarios all --all-configs \\
        --arms ours,clipfloor_off
    python3 tools/bench.py --out bench/traffic.jsonl --family traffic \\
        --seeds 0,1,2,3,4 --arms ours
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import queue
import socket
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spacetime_bezier import frontend  # noqa: E402
from spacetime_bezier.families import FAMILY_MAP  # noqa: E402
from spacetime_bezier.geometry import (  # noqa: E402
    compute_min_clearance,
    normalize_obstacles,
    obstacle_array_bundle,
    obstacle_positions_at,
)
from spacetime_bezier.objective import build_initial_guess  # noqa: E402
from spacetime_bezier.optimize import (  # noqa: E402
    DEFAULT_INITIAL_ELASTIC_WEIGHT,
    DEFAULT_TRUST_RADIUS,
    _STOP_REASONS,
    optimize_spacetime,
    result_row,
)
from spacetime_bezier.scenarios import SCENARIO_MAP  # noqa: E402

RECORD_VERSION = 1

# Every knob `optimize_scenario` passes, and the `optimize_spacetime` defaults it
# relies on for the rest -- written out, so the record can state what was run.
BASE_SOLVE = dict(
    max_iter=200, tol=1e-6, scp_prox_weight=0.3, min_dt=0.1,
    escalate_elastic_weight=True, coord_lb=-20.0, coord_ub=20.0, time_ub_scale=1.5,
    sound_clip=True, v_max=None, time_weight=0.0, free_arrival_time=False,
)

ARMS = {
    "ours": {},
    "clipfloor_off": {"sound_clip": False},
    "snapshot": {},
    "free_arrival": {"free_arrival_time": True, "time_weight": 10.0, "v_max": 1.5},
}
REFUSED_ARMS = {
    "realistic": "families.py has no `realistic` placement mode (demo spec C sec. 1.3); "
                 "build it before asking for this arm",
}
STARTS = {"own": None, "straight": {"mode": "straight"}, "wait": {"mode": "wait", "power": 3}}

IDENTITY_KEYS = (
    "mode", "family", "scenario_name", "axis", "axis_value", "seed", "arm", "start", "N", "n_seg",
    "initial_elastic_weight", "max_iter", "tol", "scp_prox_weight", "trust_radius", "min_dt",
    "sound_clip", "escalate_elastic_weight", "v_max", "time_weight", "free_arrival_time",
    "repeats", "record_version", "git_commit", "extension_build_time", "extension_so_build_time",
)


# --------------------------------------------------------------------------- plan

def _parse_value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_configs(text: str | None):
    if not text:
        return None
    out = []
    for part in text.split(","):
        n, s = part.lower().split("x")
        out.append((int(n), int(s)))
    return out


def _takes_seed(gen) -> bool:
    return "seed" in inspect.signature(gen).parameters


def build_plan(args) -> list[dict]:
    """The exact run list a pass executes; `--dry-run` prints this and nothing else."""
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a in REFUSED_ARMS:
            raise SystemExit(f"arm `{a}` refused: {REFUSED_ARMS[a]}")
        if a not in ARMS:
            raise SystemExit(f"unknown arm `{a}`; known: {sorted(ARMS)}")
    starts = [s.strip() for s in args.starts.split(",") if s.strip()]
    for s in starts:
        if s not in STARTS:
            raise SystemExit(f"unknown start `{s}`; known: {sorted(STARTS)}")
    configs_override = _parse_configs(args.configs)
    plan = []
    if args.family:
        if args.family not in FAMILY_MAP:
            raise SystemExit(f"unknown family `{args.family}`; known: {sorted(FAMILY_MAP)}")
        gen, values, cfg = FAMILY_MAP[args.family]
        values = [_parse_value(v) for v in args.values.split(",")] if args.values else list(values)
        seeds = [int(s) for s in args.seeds.split(",")] if _takes_seed(gen) else [None]
        for v in values:
            for seed in seeds:
                for N, n_seg in configs_override or [cfg]:
                    for arm in arms:
                        for start in starts:
                            plan.append(dict(mode="family", family=args.family, scenario_name=None,
                                             axis_value=v, seed=seed, arm=arm, start=start, N=N, n_seg=n_seg))
    else:
        names = list(SCENARIO_MAP) if args.scenarios == "all" else [n.strip() for n in args.scenarios.split(",")]
        for name in names:
            if name not in SCENARIO_MAP:
                raise SystemExit(f"unknown scenario `{name}`; known: {sorted(SCENARIO_MAP)}")
            registered = SCENARIO_MAP[name][1]
            for N, n_seg in configs_override or (registered if args.all_configs else registered[:1]):
                for arm in arms:
                    for start in starts:
                        plan.append(dict(mode="scenario", family=None, scenario_name=name,
                                         axis_value=None, seed=None, arm=arm, start=start, N=N, n_seg=n_seg))
    return plan[: args.limit] if args.limit else plan


def build_scenario(item: dict) -> dict:
    if item["mode"] == "family":
        gen = FAMILY_MAP[item["family"]][0]
        sc = gen(item["axis_value"], seed=item["seed"]) if item["seed"] is not None else gen(item["axis_value"])
    else:
        sc = SCENARIO_MAP[item["scenario_name"]][0]()
    if STARTS[item["start"]] is not None:
        sc = dict(sc, init_curve=STARTS[item["start"]])
    return sc


def snapshot_obstacles(obstacles: list[dict], T: float) -> list[dict]:
    """Spec §1.9: every obstacle frozen at its t = 0 position, spanning the horizon."""
    out = []
    for o in normalize_obstacles(obstacles, T):
        p0 = obstacle_positions_at(o["control_points"], np.array([0.0]))[0]
        out.append({"control_points": [list(p0) + [0.0], list(p0) + [float(T)]],
                    "radius": float(o["radius"]), "name": o.get("name")})
    return out


# ----------------------------------------------------------------- provenance

_PROV_CACHE: dict | None = None


def provenance() -> dict:
    global _PROV_CACHE
    if _PROV_CACHE is None:
        import bezier_opt

        pkg = Path(bezier_opt.__file__).parent
        so = next((p for p in sorted(pkg.iterdir()) if p.suffix in (".so", ".pyd")), None)
        prov = frontend.provenance_payload()
        machine = platform.platform()
        if sys.platform == "darwin":
            try:
                brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       capture_output=True, text=True, timeout=5).stdout.strip()
                machine = f"{machine} {brand}".strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.splitlines()
        _PROV_CACHE = {
            "git_commit": prov.get("git_commit"),
            "git_dirty": prov.get("git_dirty"),
            "git_dirty_paths": [line[3:] for line in status],
            "extension_path": prov.get("extension_path"),
            "extension_build_time": frontend.extension_build_time(),
            "extension_so_path": str(so) if so else None,
            "extension_so_build_time": so.stat().st_mtime if so else None,
            "newest_rust_source_time": frontend.newest_rust_source_time(),
            "extension_stale": prov.get("extension_stale"),
            "python_version": platform.python_version(),
            "machine": machine or None,
            "cpu_count": os.cpu_count(),
            "hostname": socket.gethostname(),
        }
    return dict(_PROV_CACHE)


# -------------------------------------------------------------------- one run

def _walls_and_rows(P, obstacles, n_seg, trust, sound_clip, stations):
    import bezier_opt

    P = np.asarray(P, dtype=float)
    ctrl, radii = obstacle_array_bundle(obstacles, P.shape[1] - 1)
    st = None if stations is None else np.asarray(stations, dtype=float)
    res = bezier_opt.spacetime_koz_rows_exact(p=P, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
                                              trust_radius=float(trust), sound_clip=bool(sound_clip),
                                              stations=st)
    walls = set(zip(np.asarray(res[2]).tolist(), np.asarray(res[4]).tolist(),
                    np.asarray(res[5]).tolist(), np.asarray(res[6]).tolist()))
    return len(walls), int(len(res[1]))


def resolved_arguments(item: dict, sc: dict) -> dict:
    args = dict(BASE_SOLVE)
    args.update(ARMS[item["arm"]])
    args["trust_radius"] = float(sc.get("trust_radius", DEFAULT_TRUST_RADIUS))
    args["initial_elastic_weight"] = float(DEFAULT_INITIAL_ELASTIC_WEIGHT)
    return args


def _family_fields(sc: dict, item: dict) -> dict:
    fam = sc.get("family") or {}
    return {
        "family": item["family"],
        "axis": fam.get("axis"),
        "axis_value": item["axis_value"],
        "seed": item["seed"],
        "n_obs": fam.get("n_obs", fam.get("value") if fam.get("axis") == "obstacle_count" else None),
        "dim_spatial": fam.get("dim_spatial"),
        "T": sc.get("T"),
        "motion": fam.get("motion", fam.get("value") if fam.get("axis") == "motion" else None),
        "placement_mode": fam.get("placement_mode"),
        "schedule": fam.get("schedule"),
        "scale": fam.get("scale"),
    }


def run_one(item: dict, repeats: int = 3, warmup: bool = False) -> dict:
    """Execute one planned run and return its record. Never raises on a solver
    failure: a run that fails is a record (spec §2.5)."""
    rec = {"record_version": RECORD_VERSION, **provenance()}
    rec.update({k: item[k] for k in ("mode", "scenario_name", "arm", "start", "N", "n_seg")})
    rec["repeats"] = int(repeats)
    rec["error_type"] = rec["error_message"] = rec["error_traceback"] = None
    try:
        sc = build_scenario(item)
        rec.update(_family_fields(sc, item))
        rec["scenario_name"] = sc["name"]
        args = resolved_arguments(item, sc)
        rec.update(args)
        true_obs = sc["obstacles"]
        solve_obs = snapshot_obstacles(true_obs, sc["T"]) if item["arm"] == "snapshot" else true_obs
        dim = len(sc["start"])
        rec.update(dim=dim, n_obstacles=len(true_obs), init_curve=sc.get("init_curve"),
                   has_stations=sc.get("stations") is not None,
                   has_coord_bounds=sc.get("coord_bounds") is not None)
        kwargs = dict(
            N=item["N"], dim=dim, p_start=sc["start"], p_end=sc["end"], obstacles=solve_obs,
            n_seg=item["n_seg"], max_iter=args["max_iter"], tol=args["tol"],
            scp_prox_weight=args["scp_prox_weight"], scp_trust_radius=args["trust_radius"],
            elastic_weight=args["initial_elastic_weight"],
            escalate_elastic_weight=args["escalate_elastic_weight"], min_dt=args["min_dt"],
            coord_lb=args["coord_lb"], coord_ub=args["coord_ub"], time_ub_scale=args["time_ub_scale"],
            sound_clip=args["sound_clip"], v_max=args["v_max"], time_weight=args["time_weight"],
            free_arrival_time=args["free_arrival_time"], stations=sc.get("stations"),
            coord_bounds=sc.get("coord_bounds"), verbose=False, init_curve=sc.get("init_curve"),
        )
        rec["warmup_seconds"] = None
        if warmup:
            t = time.perf_counter()
            optimize_spacetime(**kwargs)
            rec["warmup_seconds"] = time.perf_counter() - t
        rec["load_avg_before"] = os.getloadavg()[0]
        wall, cpu = [], []
        for _ in range(repeats):
            t0, c0 = time.perf_counter(), time.process_time()
            P, info = optimize_spacetime(**kwargs)
            wall.append(time.perf_counter() - t0)
            cpu.append(time.process_time() - c0)
        rec["load_avg_after"] = os.getloadavg()[0]
        rec.update(solve_seconds=statistics.median(wall), solve_seconds_min=min(wall),
                   solve_seconds_max=max(wall), solve_seconds_all=wall,
                   solve_cpu_seconds=statistics.median(cpu), kept_repeat="last")

        t = time.perf_counter()
        clr3000 = compute_min_clearance(P, solve_obs, dim=dim, n_eval=3000)
        clr = compute_min_clearance(P, solve_obs, dim=dim, n_eval=20001)
        rec["clearance_seconds"] = time.perf_counter() - t
        rec["min_clearance_3000"] = clr3000
        rec["min_clearance"] = clr
        rec["clearance_sampling_delta"] = clr3000 - clr
        rec["min_clearance_vs_true"] = (compute_min_clearance(P, true_obs, dim=dim, n_eval=20001)
                                        if item["arm"] == "snapshot" else None)

        t = time.perf_counter()
        P0 = build_initial_guess(sc["start"], sc["end"], item["N"] + 1, init_curve=sc.get("init_curve"))
        rec["walls_seed"], rec["rows_seed"] = _walls_and_rows(P0, solve_obs, item["n_seg"], args["trust_radius"],
                                                              args["sound_clip"], sc.get("stations"))
        rec["walls_returned"], rec["rows_returned"] = _walls_and_rows(P, solve_obs, item["n_seg"],
                                                                      args["trust_radius"], args["sound_clip"],
                                                                      sc.get("stations"))
        rec["rows_seconds"] = time.perf_counter() - t
        rec["walls_bound"] = item["n_seg"] * len(solve_obs)

        g = lambda k, d=float("nan"): float(info.get(k, d))  # noqa: E731
        rec.update(
            converged=bool(info.get("converged", 0.0)), stop_reason=int(info.get("stop_reason", -1)),
            stop_label=_STOP_REASONS.get(int(info.get("stop_reason", -1)), "unknown"),
            iterations=int(info.get("iterations", -1)),
            accept_count=int(info.get("accept_count", 0)), reject_count=int(info.get("reject_count", 0)),
            null_step_count=int(info.get("null_step_count", 0)), bootstrap_count=int(info.get("bootstrap_count", 0)),
            certificate_violation=g("koz_violation_reference"),
            occlusion_violation=g("occlusion_violation_reference", 0.0),
            occlusion_planes_dropped=g("occlusion_planes_dropped", 0.0),
            koz_unsound_clips=g("koz_unsound_clips"), sound_clip_reported=bool(info.get("sound_clip", 0.0)),
            total_slack=g("total_koz_slack_returned"), total_slack_last_subproblem=g("total_koz_slack"),
            elastic_weight=g("final_elastic_weight"),
            weight_raises=int(info["weight_raises"]) if "weight_raises" in info else -1,
            max_koz_dual=g("max_koz_dual"), speed_cap_violation=g("speed_cap_violation", 0.0),
            arrival_time=g("arrival_time"), arrival_on_min_dt_floor=g("arrival_on_min_dt_floor", 0.0),
            time_ub_used=g("time_ub_used"), arrival_on_time_ub=g("arrival_on_time_ub", 0.0),
            final_delta_norm=g("final_delta_norm"), final_trust=g("final_trust"),
            max_trust=g("max_trust"), wall_reach_trust=g("wall_reach_trust"),
            rho_mean=g("rho_mean"), rho_min=g("rho_min"), rho_max=g("rho_max"), rho_samples=g("rho_samples"),
            returned_best_iterate=bool(info.get("returned_best_iterate", 0.0)),
            backend=info.get("backend"), solver_min_clearance=g("min_clearance"),
            control_points=np.asarray(P, dtype=float).tolist(),
        )
        gate = result_row(P, info, clr, item["N"], item["n_seg"], args["trust_radius"],
                          args["initial_elastic_weight"])
        gate.pop("control_points", None)
        rec["gate_row"] = gate
        rec["figure_grade"] = bool(gate["figure_grade"])
        rec["figure_grade_reasons"] = list(gate["figure_grade_reasons"])
    except Exception as exc:  # a failing run is a record; KeyboardInterrupt still stops the pass
        rec.update(error_type=type(exc).__name__, error_message=str(exc)[:2000],
                   error_traceback=traceback.format_exc()[-4000:], figure_grade=False,
                   figure_grade_reasons=[f"run raised {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"])
        rec.setdefault("family", item["family"])
        rec.setdefault("axis_value", item["axis_value"])
        rec.setdefault("seed", item["seed"])
    return finish(rec, item)


def finish(rec: dict, item: dict) -> dict:
    rec.setdefault("axis", None)
    for k in ("family", "axis_value", "seed"):
        rec.setdefault(k, item[k])
    for k in IDENTITY_KEYS:
        rec.setdefault(k, None)
    key = {k: rec.get(k) for k in IDENTITY_KEYS}
    rec["identity"] = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":"),
                                                default=str).encode()).hexdigest()
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    return rec


# ------------------------------------------------------------------- writing

def _clean(value):
    """JSON has no NaN: a non-finite float becomes null plus a `*_nonfinite` sibling."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(v, float) and not math.isfinite(v):
                out[k] = None
                out[f"{k}_nonfinite"] = "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
            else:
                out[k] = _clean(v)
        return out
    if isinstance(value, (list, tuple)):
        return [None if isinstance(v, float) and not math.isfinite(v) else _clean(v) for v in value]
    if isinstance(value, (np.floating,)):
        return _clean(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write_record(fh, rec: dict) -> None:
    fh.write(json.dumps(_clean(rec), allow_nan=False, sort_keys=True) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def existing_identities(path: Path) -> tuple[set, int]:
    ids, bad = set(), 0
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["identity"])
            except (ValueError, KeyError):
                bad += 1
    return ids, bad


def planned_identity(item: dict, repeats: int) -> str:
    """The identity a planned run WILL have, computed without solving -- what
    `--resume` compares against."""
    sc = build_scenario(item)
    rec = {"record_version": RECORD_VERSION, **provenance(), **resolved_arguments(item, sc)}
    rec.update({k: item[k] for k in ("mode", "arm", "start", "N", "n_seg")})
    rec["scenario_name"] = sc["name"]
    rec.update(_family_fields(sc, item))
    rec["repeats"] = int(repeats)
    return finish(rec, item)["identity"]


def _child(item, repeats, q):
    q.put(run_one(item, repeats, warmup=True))


def run_with_timeout(item: dict, repeats: int, timeout: float) -> dict:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(item, repeats, q))
    proc.start()
    try:
        rec = q.get(timeout=timeout)
    except queue.Empty:
        proc.kill()
        proc.join()
        rec = {"record_version": RECORD_VERSION, **provenance(),
               **{k: item[k] for k in ("mode", "scenario_name", "arm", "start", "N", "n_seg")},
               "repeats": int(repeats), "error_type": "timeout", "timeout_seconds": float(timeout),
               "error_message": f"no record within {timeout} s", "error_traceback": None,
               "figure_grade": False, "figure_grade_reasons": [f"run raised timeout: no record within {timeout} s"]}
        try:
            rec.update(resolved_arguments(item, build_scenario(item)))
        except Exception:  # noqa: BLE001 -- the timeout record must still be written
            pass
        return finish(rec, item)
    proc.join()
    return rec


# ---------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--family")
    src.add_argument("--scenarios", help="comma list, or `all`")
    ap.add_argument("--values")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--all-configs", action="store_true")
    ap.add_argument("--configs")
    ap.add_argument("--arms", default="ours")
    ap.add_argument("--starts", default="own")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    plan = build_plan(args)
    if args.dry_run:
        for item in plan:
            print(json.dumps(item, sort_keys=True))
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done, bad = existing_identities(out) if args.resume else (set(), 0)
    if bad:
        print(f"[bench] {bad} unparseable line(s) in {out}; they are not treated as done", file=sys.stderr)
    prov = provenance()
    print(f"[bench] {len(plan)} planned run(s); git {prov['git_commit']} dirty={prov['git_dirty']}; "
          f"load {os.getloadavg()[0]:.2f} -- close other work for timing", file=sys.stderr)
    missing = 0
    first = True
    with out.open("a") as fh:
        for i, item in enumerate(plan):
            if args.resume and planned_identity(item, args.repeats) in done:
                continue
            rec = (run_with_timeout(item, args.repeats, args.timeout) if args.timeout
                   else run_one(item, args.repeats, warmup=first))
            first = False
            if rec is None:
                missing += 1
                continue
            write_record(fh, rec)
            tag = "ok " if rec.get("figure_grade") else ("ERR" if rec.get("error_type") else "no ")
            print(f"[bench] {i + 1}/{len(plan)} {tag} {rec.get('scenario_name')} N{item['N']}_seg{item['n_seg']} "
                  f"arm={item['arm']} start={item['start']} seed={item['seed']}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
