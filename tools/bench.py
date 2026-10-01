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

Added 2026-10-01, for the bar as the paper claims it (`make_tables.py --table bar`):

* **The `free_arrival` arm's speed cap is per scene**: 1.5x the scene's own
  straight-line speed. One fixed cap cannot fit every scene -- `loiter` needs 1.67
  just to reach its end by the latest allowed arrival, so a cap of 1.5 fails it by
  construction.
* **`--starts answer --answers FILE...`** starts each problem AT a figure-grade
  trajectory some other run returned for that same problem. Problems with no
  answer are not planned. The arrival-time box is rescaled so the problem is the
  one the answer solved.
* **Every record carries the certificate of its start**, recomputed from the row
  builder at the solver's own wall reach, and the same recomputation at the
  returned trajectory with whether it agrees with the solver's -- the start's
  certificate is trusted only because the returned one matches.
* **Walls and rows are counted at the solver's wall reach** (`wall_reach_trust`).
  They were counted at the starting trust radius, which is not the set the solver
  builds since `b506f34`.

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
from orbital_docking.de_casteljau import segment_matrices_equal_params  # noqa: E402
from spacetime_bezier.geometry import (  # noqa: E402
    compute_los_margin,
    compute_min_clearance,
    normalize_obstacles,
    obstacle_array_bundle,
    obstacle_positions_at,
)
from spacetime_bezier.objective import build_initial_guess  # noqa: E402
from spacetime_bezier.optimize import (  # noqa: E402
    DEFAULT_INITIAL_ELASTIC_WEIGHT,
    DEFAULT_TRUST_RADIUS,
    FIGURE_GRADE_CERTIFICATE_TOL,
    _STOP_REASONS,
    optimize_spacetime,
    optimize_spacetime_from_control_points,
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
    # Arrival free, a time penalty, and a speed cap scaled to the scene: the cap is
    # this factor times the scene's straight-line speed (see `resolved_arguments`).
    "free_arrival": {"free_arrival_time": True, "time_weight": 10.0, "v_max_factor": 1.5},
}
REFUSED_ARMS = {
    "realistic": "families.py has no `realistic` placement mode (demo spec C sec. 1.3); "
                 "build it before asking for this arm",
}
STARTS = {"own": None, "straight": {"mode": "straight"}, "wait": {"mode": "wait", "power": 3},
          # The control points of a figure-grade answer to the same problem, from
          # `--answers` -- never an init_curve, so the scene is left untouched.
          "answer": None}
# Which answer a problem is started at when several runs solved it: this build's
# before another build's, then by the start that found it.
ANSWER_ORDER = ("own", "straight", "wait")

IDENTITY_KEYS = (
    "mode", "family", "scenario_name", "axis", "axis_value", "seed", "arm", "start", "N", "n_seg",
    "initial_elastic_weight", "max_iter", "tol", "scp_prox_weight", "trust_radius", "min_dt",
    "sound_clip", "escalate_elastic_weight", "v_max", "time_weight", "free_arrival_time",
    "repeats", "record_version", "git_commit", "extension_build_time", "extension_so_build_time",
    "answer_identity",
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


def cell_key(d: dict) -> tuple:
    """One problem -- scene or family member, arm, discretization; everything but
    the start. Plan items and records both carry it. A family record's
    `scenario_name` is the member's generated name, so it is left out there."""
    name = d.get("scenario_name") if d.get("mode") == "scenario" else None
    return (d.get("mode"), d.get("family"), name, d.get("axis_value"), d.get("seed"), d.get("arm"),
            d.get("N"), d.get("n_seg"))


def load_answers(paths: list[str]) -> dict:
    """The figure-grade trajectory each problem is started at, by `cell_key`. An
    answer run is never itself an answer, so a pass cannot feed on its own output."""
    prov = provenance()
    here = (prov["extension_so_path"], prov["extension_so_build_time"])
    best = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("figure_grade") or r.get("start") == "answer" or not r.get("control_points"):
                continue
            rank = ((r.get("extension_so_path"), r.get("extension_so_build_time")) != here,
                    ANSWER_ORDER.index(r["start"]) if r.get("start") in ANSWER_ORDER else len(ANSWER_ORDER))
            k = cell_key(r)
            if k not in best or rank < best[k][0]:
                best[k] = (rank, r)
    return {k: r for k, (_, r) in best.items()}


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
    if "answer" in starts:
        if not getattr(args, "answers", None):
            raise SystemExit("--starts answer needs --answers: the records whose trajectories to start at")
        answers = load_answers(args.answers)
        kept = []
        for item in plan:
            if item["start"] != "answer":
                kept.append(item)
                continue
            r = answers.get(cell_key(item))
            if r is None:
                continue  # unsolved by every run given; make_tables counts these
            kept.append(dict(item, answer_identity=r["identity"], answer_start=r["start"],
                             answer_extension=r.get("extension_so_path"),
                             answer_reach_policy=r.get("reach_policy_env"),
                             answer_control_points=r["control_points"]))
        plan = kept
    if getattr(args, "skip_solved", None):
        solved = load_answers(args.skip_solved)
        plan = [item for item in plan if cell_key(item) not in solved]
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
            # Read only by the scratch policy builds of the wall reach; the frozen
            # build ignores it. Recorded so a record from one says which policy ran.
            "reach_policy_env": os.environ.get("ST_REACH_POLICY"),
            "python_version": platform.python_version(),
            "machine": machine or None,
            "cpu_count": os.cpu_count(),
            "hostname": socket.gethostname(),
        }
    return dict(_PROV_CACHE)


# -------------------------------------------------------------------- one run

def rows_at(P, obstacles, n_seg, reach, sound_clip, stations) -> dict:
    """The walls the solver builds at `P` with wall reach `reach`, and the hull
    certificate they give -- the solver's own sum, recomputed: per row the
    shortfall of its control point below the wall, keep-out and shadow rows summed
    apart, infinite where a needed wall could not be built."""
    import bezier_opt

    P = np.asarray(P, dtype=float)
    dim = P.shape[1]
    ctrl, radii = obstacle_array_bundle(obstacles, dim - 1)
    st = None if stations is None else np.asarray(stations, dtype=float).reshape(-1, dim - 1)
    (normals, lbs, seg, cp, obs, comp, sta, _rho, _sound, dropped, dropped_shadow,
     unsound) = bezier_opt.spacetime_koz_rows_exact(p=P, obstacle_ctrl=ctrl, obstacle_r=radii, n_seg=n_seg,
                                                    trust_radius=float(reach), sound_clip=bool(sound_clip),
                                                    stations=st)
    normals = np.asarray(normals, dtype=float).reshape(-1, dim)
    seg, cp, sta = (np.asarray(a, dtype=int) for a in (seg, cp, sta))
    hulls = [np.asarray(a, dtype=float) @ P for a in segment_matrices_equal_params(P.shape[0] - 1, n_seg)]
    short = np.array([max(float(lb) - float(n @ hulls[s][c]), 0.0)
                      for n, lb, s, c in zip(normals, lbs, seg, cp)])
    keep_out = float(short[sta < 0].sum()) if len(short) else 0.0
    shadow = float(short[sta >= 0].sum()) if len(short) else 0.0
    walls = set(zip(seg.tolist(), np.asarray(obs).tolist(), np.asarray(comp).tolist(), sta.tolist()))
    return {"walls": len(walls), "rows": int(len(lbs)),
            "certificate": math.inf if int(dropped) > 0 else keep_out,
            "occlusion": math.inf if int(dropped_shadow) > 0 else shadow,
            "occlusion_dropped": int(dropped_shadow), "unsound_clips": int(unsound)}


def speed_cap_violation(P, v_max) -> float:
    """The solver's cone violation, recomputed: per control-polygon leg, how far the
    spatial step exceeds `v_max` times the time step, summed. Zero with no cap."""
    if v_max is None:
        return 0.0
    d = np.diff(np.asarray(P, dtype=float), axis=0)
    return float(np.maximum(np.linalg.norm(d[:, :-1], axis=1) - float(v_max) * d[:, -1], 0.0).sum())


def los_margin(P, stations, obstacles, dim) -> float | None:
    """The independent line-of-sight check (`compute_los_margin`), worst station."""
    if stations is None:
        return None
    return min(float(np.min(compute_los_margin(P, s, obstacles, dim=dim, n_eval=20001)[1])) for s in stations)


def _agrees(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def resolved_arguments(item: dict, sc: dict) -> dict:
    args = dict(BASE_SOLVE)
    args.update(ARMS[item["arm"]])
    if "v_max_factor" in args:
        gap = np.asarray(sc["end"][:-1], dtype=float) - np.asarray(sc["start"][:-1], dtype=float)
        args["v_max"] = float(args["v_max_factor"]) * float(np.linalg.norm(gap)) / float(sc["T"])
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
    rec.update({k: item.get(k) for k in ("answer_identity", "answer_start", "answer_extension",
                                         "answer_reach_policy")})
    rec["repeats"] = int(repeats)
    rec["error_type"] = rec["error_message"] = rec["error_traceback"] = None
    try:
        sc = build_scenario(item)
        rec.update(_family_fields(sc, item))
        rec["scenario_name"] = sc["name"]
        args = resolved_arguments(item, sc)
        if item["start"] == "answer":
            P_init = np.asarray(item["answer_control_points"], dtype=float)
            # The arrival box is P_init's last time times the scale, so an answer
            # that arrives late would widen it. Rescaled to the box every other
            # start gets -- latest arrival = scale x the scene's T -- so the problem
            # is the one the answer solved.
            args["time_ub_scale"] = args["time_ub_scale"] * float(sc["T"]) / float(P_init[-1, -1])
        else:
            P_init = build_initial_guess(sc["start"], sc["end"], item["N"] + 1, init_curve=sc.get("init_curve"))
        rec.update(args)
        true_obs = sc["obstacles"]
        solve_obs = snapshot_obstacles(true_obs, sc["T"]) if item["arm"] == "snapshot" else true_obs
        dim = len(sc["start"])
        stations = sc.get("stations")
        rec.update(dim=dim, n_obstacles=len(true_obs), init_curve=sc.get("init_curve"),
                   has_stations=stations is not None, n_stations=len(stations or []),
                   has_coord_bounds=sc.get("coord_bounds") is not None)
        common = dict(
            obstacles=solve_obs,
            n_seg=item["n_seg"], max_iter=args["max_iter"], tol=args["tol"],
            scp_prox_weight=args["scp_prox_weight"], scp_trust_radius=args["trust_radius"],
            elastic_weight=args["initial_elastic_weight"],
            escalate_elastic_weight=args["escalate_elastic_weight"], min_dt=args["min_dt"],
            coord_lb=args["coord_lb"], coord_ub=args["coord_ub"], time_ub_scale=args["time_ub_scale"],
            sound_clip=args["sound_clip"], v_max=args["v_max"], time_weight=args["time_weight"],
            free_arrival_time=args["free_arrival_time"], stations=stations,
            coord_bounds=sc.get("coord_bounds"), verbose=False,
        )
        kwargs = dict(N=item["N"], dim=dim, p_start=sc["start"], p_end=sc["end"],
                      init_curve=sc.get("init_curve"), **common)
        if item["start"] == "answer":
            def solve():
                return optimize_spacetime_from_control_points(P_init, **common)
        else:
            def solve():
                return optimize_spacetime(**kwargs)
        rec["warmup_seconds"] = None
        if warmup:
            t = time.perf_counter()
            solve()
            rec["warmup_seconds"] = time.perf_counter() - t
        rec["load_avg_before"] = os.getloadavg()[0]
        wall, cpu = [], []
        for _ in range(repeats):
            t0, c0 = time.perf_counter(), time.process_time()
            P, info = solve()
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
        # The reach the solver built its walls and its certificate with. A build
        # that does not report it (a scratch policy build) built them at the
        # starting radius or says so in `reach_policy_env`.
        reach = float(info.get("wall_reach_trust", args["trust_radius"]))
        at_start = rows_at(P_init, solve_obs, item["n_seg"], reach, args["sound_clip"], stations)
        at_end = rows_at(P, solve_obs, item["n_seg"], reach, args["sound_clip"], stations)
        rec["rows_seconds"] = time.perf_counter() - t
        rec["certificate_reach"] = reach
        rec["walls_seed"], rec["rows_seed"] = at_start["walls"], at_start["rows"]
        rec["walls_returned"], rec["rows_returned"] = at_end["walls"], at_end["rows"]
        # One wall per (segment, obstacle, approach) for the zone, and again per
        # station for its shadow: the bound counts one approach per pair.
        rec["walls_bound"] = item["n_seg"] * len(solve_obs) * (1 + len(stations or []))
        # The recomputation at the RETURNED trajectory must equal the solver's own
        # numbers -- that is what licenses the same recomputation at the start.
        rec["certificate_recheck"] = at_end["certificate"]
        rec["certificate_recheck_agrees"] = (
            _agrees(at_end["certificate"], float(info.get("koz_violation_reference", math.nan)))
            and _agrees(at_end["occlusion"], float(info.get("occlusion_violation_reference", 0.0)))
            and at_end["unsound_clips"] == int(info.get("koz_unsound_clips", -1))
            and _agrees(speed_cap_violation(P, args["v_max"]), float(info.get("speed_cap_violation", 0.0))))
        start_clr = compute_min_clearance(P_init, solve_obs, dim=dim, n_eval=20001)
        start_los = los_margin(P_init, stations, solve_obs, dim)
        start_speed = speed_cap_violation(P_init, args["v_max"])
        rec.update(start_certificate=at_start["certificate"], start_occlusion=at_start["occlusion"],
                   start_unsound_clips=at_start["unsound_clips"], start_min_clearance=start_clr,
                   start_los_margin=start_los, start_speed_cap_violation=start_speed,
                   los_margin=los_margin(P, stations, solve_obs, dim))
        # The start passes every gate condition a trajectory can pass without a
        # solve -- everything but convergence and slack.
        rec["start_certified"] = bool(
            at_start["certificate"] <= FIGURE_GRADE_CERTIFICATE_TOL
            and at_start["occlusion"] <= FIGURE_GRADE_CERTIFICATE_TOL
            and at_start["unsound_clips"] == 0 and start_clr > 0.0
            and start_speed <= FIGURE_GRADE_CERTIFICATE_TOL)

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
    rec["answer_identity"] = item.get("answer_identity")
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
               "answer_identity": item.get("answer_identity"),
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
    ap.add_argument("--answers", nargs="+", help="records whose figure-grade trajectories `--starts answer` uses")
    ap.add_argument("--skip-solved", nargs="+", help="plan only problems no figure-grade record in these files solves")
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
