"""Single-case DB-seeded DCM experiment driver.

This script implements the workflow described in `doc/dcm_db_experiment_note.md`
as closely as the current `orbit-transfer-analysis` codebase allows.

Target workflow:
    baseline:  DB seed -> DCM -> output
    proposed:  same DB seed -> upstream optimizer -> DCM -> output

Important scientific caveat
---------------------------
The local DB schema stores solved trajectories, not a validated "raw DB initial
guess" artifact. Therefore this script uses the stored `trajectory_file` as the
seed path. That makes the experiment *faithful to the same case and same
downstream DCM settings*, but the seed should be interpreted as the best
available repo-native proxy for the original DB initial path.

Default mode (`seeded_collocation`) is the strictest repo-native implementation:
    DB stored trajectory -> seeded collocation upstream -> seeded collocation DCM

Optional exploratory mode (`blade`) uses:
    same case params -> BLADE -> seeded collocation DCM

The BLADE mode is useful, but it does NOT satisfy the strict "same DB seed into
upstream optimizer" rule because the current BLADE wrapper does not accept an
external trajectory seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv
from orbit_transfer.benchmark.result import BenchmarkResult
from orbit_transfer.benchmark.solvers import BladeSolver, CollocationSolver
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.types import TransferConfig


DEFAULT_DB = _REPO_ROOT / "data" / "trajectories.duckdb"
DEFAULT_OUTDIR = _REPO_ROOT / "results" / "dcm_db_experiment"


@dataclass
class CaseSeed:
    row: dict[str, Any]
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    trajectory_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single DB-seeded DCM experiment.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB),
        help="DuckDB path (default: data/trajectories.duckdb)",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=None,
        help="Explicit trajectories.id to test. Preferred for reproducible reporting.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="first-converged",
        choices=["first-converged", "lowest-cost-converged"],
        help="Deterministic case selector when --case-id is omitted.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Directory for JSON/CSV outputs.",
    )
    parser.add_argument(
        "--upstream-mode",
        type=str,
        default="seeded_collocation",
        choices=["seeded_collocation", "blade"],
        help=(
            "Upstream optimizer. "
            "'seeded_collocation' is the strictest matched-seed mode available here; "
            "'blade' is exploratory."
        ),
    )
    parser.add_argument(
        "--upstream-l1-lambda",
        type=float,
        default=1e-3,
        help=(
            "Upstream regularization for seeded_collocation mode. "
            "Use a nonzero value to keep the upstream stage meaningfully distinct."
        ),
    )
    parser.add_argument("--blade-K", type=int, default=12, help="BLADE segments")
    parser.add_argument("--blade-n", type=int, default=2, help="BLADE segment degree")
    parser.add_argument("--blade-max-iter", type=int, default=50, help="BLADE SCP max iterations")
    parser.add_argument("--blade-tol-bc", type=float, default=1e-3, help="BLADE BC tolerance")
    parser.add_argument("--blade-l1-lambda", type=float, default=0.0, help="BLADE l1 regularization")
    return parser.parse_args()


def _table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return {row[1] for row in rows}


def _select_case_query(columns: set[str], args: argparse.Namespace) -> tuple[str, list[Any]]:
    params: list[Any] = []
    conditions = ["converged = TRUE", "trajectory_file IS NOT NULL"]
    if "method" in columns:
        conditions.append("COALESCE(method, 'collocation') = 'collocation'")

    query = "SELECT * FROM trajectories"
    if args.case_id is not None:
        query += " WHERE id = ?"
        params.append(args.case_id)
        return query, params

    query += " WHERE " + " AND ".join(conditions)
    if args.selection == "lowest-cost-converged":
        query += " ORDER BY cost ASC, id ASC"
    else:
        query += " ORDER BY id ASC"
    query += " LIMIT 1"
    return query, params


def _resolve_trajectory_path(db_path: Path, stored_path: str) -> Path:
    traj_path = Path(stored_path)
    if traj_path.is_absolute():
        return traj_path
    return (db_path.parent / traj_path).resolve()


def load_case_seed(db_path: Path, args: argparse.Namespace) -> CaseSeed:
    con = duckdb.connect(str(db_path), read_only=True)
    columns = _table_columns(con, "trajectories")
    query, params = _select_case_query(columns, args)
    row = con.execute(query, params).fetchone()
    desc = [item[0] for item in con.description]
    con.close()

    if row is None:
        raise RuntimeError("No suitable DB case was found for the requested selector.")

    data = dict(zip(desc, row))
    if not data.get("trajectory_file"):
        raise RuntimeError("Selected DB case does not include a trajectory_file seed.")

    traj_path = _resolve_trajectory_path(db_path, data["trajectory_file"])
    if not traj_path.exists():
        raise FileNotFoundError(f"Seed trajectory file not found: {traj_path}")

    npz = np.load(traj_path)
    return CaseSeed(
        row=data,
        t=np.asarray(npz["t"]),
        x=np.asarray(npz["x"]),
        u=np.asarray(npz["u"]),
        trajectory_path=traj_path,
    )


def build_config(row: dict[str, Any]) -> TransferConfig:
    return TransferConfig(
        h0=float(row["h0"]),
        delta_a=float(row["delta_a"]),
        delta_i=float(row["delta_i"]),
        T_max_normed=float(row["T_max_normed"]),
        e0=float(row.get("e0", 0.0)),
        ef=float(row.get("ef", 0.0)),
    )


def run_collocation_stage(
    config: TransferConfig,
    *,
    x_init: np.ndarray,
    u_init: np.ndarray,
    t_init: np.ndarray,
    l1_lambda: float,
) -> tuple[BenchmarkResult, float]:
    solver = CollocationSolver(x_init=x_init, u_init=u_init, t_init=t_init, l1_lambda=l1_lambda)
    t0 = time.perf_counter()
    result = solver.solve(config)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_upstream_stage(
    config: TransferConfig,
    seed: CaseSeed,
    args: argparse.Namespace,
) -> tuple[BenchmarkResult, float, dict[str, Any]]:
    if args.upstream_mode == "seeded_collocation":
        result, elapsed = run_collocation_stage(
            config,
            x_init=seed.x,
            u_init=seed.u,
            t_init=seed.t,
            l1_lambda=args.upstream_l1_lambda,
        )
        return result, elapsed, {
            "strict_same_seed_into_upstream": True,
            "exploratory_only": False,
            "upstream_mode_note": (
                "Repo-native matched-seed mode. Upstream stage uses the same DB seed "
                "and a distinct l1_lambda regularization."
            ),
        }

    solver = BladeSolver(
        K=args.blade_K,
        n=args.blade_n,
        max_iter=args.blade_max_iter,
        tol_bc=args.blade_tol_bc,
        l1_lambda=args.blade_l1_lambda,
    )
    t0 = time.perf_counter()
    result = solver.solve(config)
    elapsed = time.perf_counter() - t0
    return result, elapsed, {
        "strict_same_seed_into_upstream": False,
        "exploratory_only": True,
        "upstream_mode_note": (
            "Exploratory mode. BLADE uses the same case parameters, but the current "
            "wrapper does not accept the DB seed trajectory directly."
        ),
    }


def compute_boundary_errors(result: BenchmarkResult, config: TransferConfig) -> dict[str, float | None]:
    nu0 = result.extra.get("nu0")
    nuf = result.extra.get("nuf")
    if nu0 is None or nuf is None:
        return {
            "start_r_error_km": None,
            "start_v_error_kms": None,
            "end_r_error_km": None,
            "end_v_error_kms": None,
        }

    r0_ref, v0_ref = oe_to_rv((config.a0, config.e0, config.i0, 0.0, 0.0, float(nu0)), MU_EARTH)
    rf_ref, vf_ref = oe_to_rv((config.af, config.ef, config.if_, 0.0, 0.0, float(nuf)), MU_EARTH)
    return {
        "start_r_error_km": float(np.linalg.norm(result.x[:3, 0] - r0_ref)),
        "start_v_error_kms": float(np.linalg.norm(result.x[3:, 0] - v0_ref)),
        "end_r_error_km": float(np.linalg.norm(result.x[:3, -1] - rf_ref)),
        "end_v_error_kms": float(np.linalg.norm(result.x[3:, -1] - vf_ref)),
    }


def summarize_result(
    name: str,
    result: BenchmarkResult,
    solve_time_s: float,
    config: TransferConfig,
) -> dict[str, Any]:
    radius = np.linalg.norm(result.x[:3], axis=0) if result.x.size else np.array([np.nan])
    min_altitude_margin_km = float(np.min(radius) - (R_E + config.h_min))
    summary = {
        "stage": name,
        "converged": bool(result.converged),
        "solve_time_s": float(solve_time_s),
        "tof_s": float(result.metrics.get("tof", np.nan)),
        "tof_norm": float(result.metrics.get("tof_norm", np.nan)),
        "cost_l1": float(result.metrics.get("cost_l1", np.nan)),
        "cost_l2": float(result.metrics.get("cost_l2", np.nan)),
        "cost_l2_solver": float(result.metrics.get("cost_l2_solver", np.nan)),
        "dv_total": float(result.metrics.get("dv_total", np.nan)),
        "n_peaks": int(result.metrics.get("n_peaks", -1)) if result.converged else -1,
        "profile_class": result.metrics.get("profile_class"),
        "pass1_cost": result.extra.get("pass1_cost"),
        "T_f_solver": result.extra.get("T_f"),
        "nu0": result.extra.get("nu0"),
        "nuf": result.extra.get("nuf"),
        "min_altitude_margin_km": min_altitude_margin_km,
    }
    summary.update(compute_boundary_errors(result, config))
    return summary


def build_case_metadata(seed: CaseSeed, config: TransferConfig, args: argparse.Namespace) -> dict[str, Any]:
    row = seed.row
    return {
        "case_id": int(row["id"]),
        "selection_rule": args.selection if args.case_id is None else "explicit-case-id",
        "db_path": args.db,
        "trajectory_seed_path": str(seed.trajectory_path),
        "seed_source": "trajectory_file stored in trajectories table",
        "seed_is_raw_db_initial_guess": "unknown",
        "config": {
            "h0": config.h0,
            "delta_a": config.delta_a,
            "delta_i": config.delta_i,
            "T_max_normed": config.T_max_normed,
            "e0": config.e0,
            "ef": config.ef,
            "u_max": config.u_max,
            "h_min": config.h_min,
        },
    }


def compare_stage_metrics(baseline: dict[str, Any], proposed: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "solve_time_s",
        "cost_l1",
        "cost_l2",
        "cost_l2_solver",
        "tof_s",
        "tof_norm",
        "min_altitude_margin_km",
        "end_r_error_km",
        "end_v_error_kms",
    ]
    out: dict[str, Any] = {
        "baseline_converged": baseline["converged"],
        "proposed_converged": proposed["converged"],
    }
    for key in keys:
        b_val = baseline.get(key)
        p_val = proposed.get(key)
        if isinstance(b_val, (int, float)) and isinstance(p_val, (int, float)):
            out[f"{key}_delta_proposed_minus_baseline"] = float(p_val - b_val)
        else:
            out[f"{key}_delta_proposed_minus_baseline"] = None
    return out


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_trajectory_csv(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u_mag = np.linalg.norm(result.u, axis=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t[s]",
                "x[km]",
                "y[km]",
                "z[km]",
                "vx[km/s]",
                "vy[km/s]",
                "vz[km/s]",
                "ux[km/s2]",
                "uy[km/s2]",
                "uz[km/s2]",
                "u_mag[km/s2]",
            ]
        )
        for i in range(len(result.t)):
            writer.writerow(
                [
                    result.t[i],
                    result.x[0, i],
                    result.x[1, i],
                    result.x[2, i],
                    result.x[3, i],
                    result.x[4, i],
                    result.x[5, i],
                    result.u[0, i],
                    result.u[1, i],
                    result.u[2, i],
                    u_mag[i],
                ]
            )


def write_raw_trajectory_csv(path: Path, t: np.ndarray, x: np.ndarray, u: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u_mag = np.linalg.norm(u, axis=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t[s]",
                "x[km]",
                "y[km]",
                "z[km]",
                "vx[km/s]",
                "vy[km/s]",
                "vz[km/s]",
                "ux[km/s2]",
                "uy[km/s2]",
                "uz[km/s2]",
                "u_mag[km/s2]",
            ]
        )
        for i in range(len(t)):
            writer.writerow(
                [
                    t[i],
                    x[0, i],
                    x[1, i],
                    x[2, i],
                    x[3, i],
                    x[4, i],
                    x[5, i],
                    u[0, i],
                    u[1, i],
                    u[2, i],
                    u_mag[i],
                ]
            )


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    out_root = Path(args.outdir).expanduser().resolve()

    if args.upstream_mode == "seeded_collocation" and args.upstream_l1_lambda == 0.0:
        raise ValueError(
            "seeded_collocation mode with upstream_l1_lambda=0.0 makes the upstream stage "
            "effectively indistinguishable from the downstream DCM. Use a nonzero value."
        )

    seed = load_case_seed(db_path, args)
    config = build_config(seed.row)
    case_dir = out_root / f"case_{int(seed.row['id']):06d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DCM DB experiment")
    print("=" * 72)
    print(f"DB:         {db_path}")
    print(f"Case id:    {int(seed.row['id'])}")
    print(f"Seed file:  {seed.trajectory_path}")
    print(f"Mode:       {args.upstream_mode}")
    print(
        "Config:     "
        f"h0={config.h0:.0f} km, "
        f"delta_a={config.delta_a:.1f} km, "
        f"delta_i={config.delta_i:.2f} deg, "
        f"T_max/T0={config.T_max_normed:.3f}, "
        f"e0={config.e0:.4f}, ef={config.ef:.4f}"
    )
    print("-" * 72)

    baseline_result, baseline_time = run_collocation_stage(
        config,
        x_init=seed.x,
        u_init=seed.u,
        t_init=seed.t,
        l1_lambda=0.0,
    )
    baseline_summary = summarize_result("baseline_dcm", baseline_result, baseline_time, config)
    print(
        "Baseline:   "
        f"conv={baseline_summary['converged']} "
        f"cost_l2_solver={baseline_summary['cost_l2_solver']:.6e} "
        f"time={baseline_summary['solve_time_s']:.2f}s"
    )

    upstream_result, upstream_time, fairness = run_upstream_stage(config, seed, args)
    upstream_summary = summarize_result("upstream_optimizer", upstream_result, upstream_time, config)
    print(
        "Upstream:   "
        f"conv={upstream_summary['converged']} "
        f"cost_l2_solver={upstream_summary['cost_l2_solver']:.6e} "
        f"time={upstream_summary['solve_time_s']:.2f}s"
    )

    proposed_summary: dict[str, Any]
    proposed_result: BenchmarkResult | None
    if upstream_result.converged:
        proposed_result, proposed_time = run_collocation_stage(
            config,
            x_init=upstream_result.x,
            u_init=upstream_result.u,
            t_init=upstream_result.t,
            l1_lambda=0.0,
        )
        proposed_summary = summarize_result("proposed_dcm", proposed_result, proposed_time, config)
        print(
            "Proposed:   "
            f"conv={proposed_summary['converged']} "
            f"cost_l2_solver={proposed_summary['cost_l2_solver']:.6e} "
            f"time={proposed_summary['solve_time_s']:.2f}s"
        )
    else:
        proposed_result = None
        proposed_summary = {
            "stage": "proposed_dcm",
            "converged": False,
            "solve_time_s": 0.0,
            "skipped_reason": "upstream optimizer did not converge",
        }
        print("Proposed:   skipped because upstream optimizer did not converge")

    comparison = compare_stage_metrics(baseline_summary, proposed_summary)
    metadata = build_case_metadata(seed, config, args)
    metadata["fairness"] = {
        "same_case_entry": True,
        "same_downstream_solver": True,
        "same_downstream_settings": True,
        **fairness,
        "note": (
            "If the DB trajectory is not the original raw seed, interpret the result as "
            "a matched-case, matched-downstream rerun rather than a validated raw-seed study."
        ),
    }

    payload = {
        "metadata": metadata,
        "baseline_dcm": baseline_summary,
        "upstream_optimizer": upstream_summary,
        "proposed_dcm": proposed_summary,
        "comparison": comparison,
    }

    with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_summary_csv(
        case_dir / "stage_metrics.csv",
        [baseline_summary, upstream_summary, proposed_summary],
    )
    write_raw_trajectory_csv(case_dir / "db_seed_trajectory.csv", seed.t, seed.x, seed.u)
    write_trajectory_csv(case_dir / "baseline_dcm_trajectory.csv", baseline_result)
    write_trajectory_csv(case_dir / "upstream_optimizer_trajectory.csv", upstream_result)
    if proposed_result is not None:
        write_trajectory_csv(case_dir / "proposed_dcm_trajectory.csv", proposed_result)

    print("-" * 72)
    print(f"Output dir: {case_dir}")
    print(f"Summary:    {case_dir / 'summary.json'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
