#!/usr/bin/env python3
"""DCM downstream experiment: Bézier as Pass 1 replacement.

Compares two pipelines on the same TransferConfig cases:

  Baseline:  TransferConfig → Pass 1 (H-S) → peak detect → Pass 2 (LGL) → result
  Proposed:  TransferConfig → Bézier SCP → peak detect → Pass 2 (LGL) → result

The proposed pipeline replaces the expensive Hermite-Simpson Pass 1 with the
fast Bézier SCP optimizer for structural pre-analysis (peak detection + warm-start).

See doc/dcm_downstream_experiment_design.md for full design.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
OTA_ROOT = REPO_ROOT / "orbit-transfer-analysis"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(OTA_ROOT / "src"))

# ── imports: Bézier optimizer ───────────────────────────────────────────────
from orbital_docking.bezier import BezierCurve
from orbital_docking.optimization import optimize_orbital_docking

# ── imports: orbit-transfer-analysis (DCM) ──────────────────────────────────
from orbit_transfer.astrodynamics.kepler import kepler_propagate
from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv
from orbit_transfer.classification.classifier import classify_profile, determine_phase_structure
from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.collocation.interpolation import interpolate_pass1_to_pass2
from orbit_transfer.collocation.multiphase_lgl import MultiPhaseLGLCollocation
from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.dynamics.two_body import gravity_acceleration
from orbit_transfer.optimizer.two_pass import TwoPassOptimizer
from orbit_transfer.types import TransferConfig, TrajectoryResult

# ── defaults ────────────────────────────────────────────────────────────────
DEFAULT_DB = OTA_ROOT / "data" / "trajectories.duckdb"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "dcm_downstream_experiment"
DEFAULT_DEGREE = 6
DEFAULT_NSEG = 16
DEFAULT_SAMPLE_POINTS = 61  # matches H-S grid (30 segments → 61 points)


# ═══════════════════════════════════════════════════════════════════════════
# Bridge: Bézier optimizer → DCM warm-start
# ═══════════════════════════════════════════════════════════════════════════


def bezier_warm_start(
    config: TransferConfig,
    degree: int = DEFAULT_DEGREE,
    n_seg: int = DEFAULT_NSEG,
    n_samples: int = DEFAULT_SAMPLE_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Run Bézier SCP optimizer and sample the result into (t, x, u).

    Returns
    -------
    t_init : (M,) time array [s]
    x_init : (6, M) state [km, km/s]
    u_init : (3, M) thrust acceleration [km/s²]
    bezier_info : dict with optimizer diagnostics
    """
    T = config.T_max
    r_e = R_E + config.h_min

    # Boundary conditions from orbital elements (nu0=0, nuf=π)
    nu0 = 0.0
    nuf = np.pi
    oe_dep = (config.a0, config.e0, config.i0, 0.0, 0.0, nu0)
    oe_arr = (config.af, config.ef, config.if_, 0.0, 0.0, nuf)
    r0, v0 = oe_to_rv(oe_dep, MU_EARTH)
    rf, vf = oe_to_rv(oe_arr, MU_EARTH)

    # Orbit-aware initialization: propagate departure orbit forward.
    # All control points stay at orbital altitude (above KOZ).
    # The optimizer adjusts the curve to reach the arrival orbit.
    N = degree
    P_init = np.zeros((N + 1, 3))
    for j in range(N + 1):
        tau_j = j / N
        t_j = tau_j * T
        r_j, _ = kepler_propagate(r0, v0, t_j, MU_EARTH)
        P_init[j] = r_j

    # Bézier optimizer
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=r_e,
        v0=v0,
        v1=vf,
        transfer_time=T,
        verbose=False,
        use_cache=True,
    )

    # Sample the optimized curve
    curve = BezierCurve(P_opt)
    tau = np.linspace(0.0, 1.0, n_samples)
    t_init = tau * T

    x_init = np.zeros((6, n_samples))
    u_init = np.zeros((3, n_samples))

    for k, tk in enumerate(tau):
        r_k = curve.point(tk)
        # velocity: dr/dt = (1/T) dr/dτ
        v_k = curve.velocity(tk) / T
        # geometric acceleration: d²r/dt² = (1/T²) d²r/dτ²
        a_k = curve.acceleration(tk) / (T * T)
        # thrust = geometric acceleration - gravity
        g_k = gravity_acceleration(r_k, MU_EARTH)
        u_k = a_k - g_k

        x_init[:3, k] = r_k
        x_init[3:, k] = v_k
        u_init[:, k] = u_k

    bezier_info = {
        "converged": info.get("feasible", False),
        "iterations": int(info.get("iterations", -1)),
        "cost": float(info.get("cost", float("nan"))),
        "cost_true_energy": float(info.get("cost_true_energy", float("nan"))),
        "elapsed_time": float(info.get("elapsed_time", float("nan"))),
        "min_radius": float(info.get("min_radius", float("nan"))),
        "feasible": bool(info.get("feasible", False)),
        "termination_reason": info.get("termination_reason", "unknown"),
    }

    return t_init, x_init, u_init, bezier_info


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runners
# ═══════════════════════════════════════════════════════════════════════════


def run_baseline(config: TransferConfig) -> tuple[Any, float]:
    """Baseline: full two-pass (Pass 1 H-S → peak detect → Pass 2 LGL)."""
    optimizer = TwoPassOptimizer(config)
    t0 = time.perf_counter()
    result = optimizer.solve()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_proposed(
    config: TransferConfig,
    degree: int = DEFAULT_DEGREE,
    n_seg: int = DEFAULT_NSEG,
) -> tuple[Any, float, float, dict]:
    """Proposed: Bézier SCP → peak detect → Pass 2 LGL (skip Pass 1).

    Returns (result, pass2_time, bezier_time, bezier_info).
    """
    T = config.T_max
    n_dense = 300

    # Step 1: Bézier optimizer
    t0_bez = time.perf_counter()
    t_bez, x_bez, u_bez, bezier_info = bezier_warm_start(
        config, degree=degree, n_seg=n_seg, n_samples=n_dense,
    )
    bezier_time = time.perf_counter() - t0_bez

    if not bezier_info.get("feasible", False):
        # Bézier failed — return a non-converged result
        return TrajectoryResult(
            converged=False, cost=float("inf"),
            t=np.array([0.0, T]), x=np.zeros((6, 2)), u=np.zeros((3, 2)),
            nu0=0.0, nuf=np.pi, n_peaks=0, profile_class=0, T_f=T,
        ), 0.0, bezier_time, bezier_info

    # Step 2: Peak detection on Bézier thrust profile
    u_mag = np.linalg.norm(u_bez, axis=0)
    n_peaks, peak_times, peak_widths = detect_peaks(t_bez, u_mag, T)
    profile_class = classify_profile(n_peaks)
    phases = determine_phase_structure(peak_times, peak_widths, T)

    bezier_info["n_peaks"] = int(n_peaks)
    bezier_info["profile_class"] = int(profile_class)
    bezier_info["n_phases"] = len(phases)

    # Step 3: Interpolate Bézier trajectory to LGL nodes
    _, x_phases, u_phases = interpolate_pass1_to_pass2(
        t_bez, x_bez, u_bez, phases,
    )

    # Step 4: Pass 2 directly (skip Pass 1)
    nu0_guess = 0.0
    nuf_guess = np.pi
    lgl = MultiPhaseLGLCollocation(config, phases, T_fixed=T)
    t0_p2 = time.perf_counter()
    result = lgl.solve(
        x_phases=x_phases, u_phases=u_phases,
        nu0_guess=nu0_guess, nuf_guess=nuf_guess,
    )
    pass2_time = time.perf_counter() - t0_p2

    return result, pass2_time, bezier_time, bezier_info


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════


def summarize_result(
    name: str,
    result: Any,
    solve_time: float,
    config: TransferConfig,
) -> dict[str, Any]:
    """Extract a flat metrics dict from a TrajectoryResult."""
    if result.x.size == 0:
        return {"stage": name, "converged": False, "solve_time_s": solve_time}

    radius = np.linalg.norm(result.x[:3], axis=0)
    min_alt_margin = float(np.min(radius) - (R_E + config.h_min))
    u_mag = np.linalg.norm(result.u, axis=0)
    dv_total = float(np.trapezoid(u_mag, result.t)) if len(result.t) > 1 else float("nan")

    return {
        "stage": name,
        "converged": bool(result.converged),
        "solve_time_s": float(solve_time),
        "cost": float(result.cost),
        "dv_total_km_s": dv_total,
        "min_altitude_margin_km": min_alt_margin,
        "n_peaks": int(result.n_peaks),
        "profile_class": int(result.profile_class),
        "T_f": float(result.T_f),
        "nu0": float(result.nu0),
        "nuf": float(result.nuf),
    }


def compare_metrics(
    baseline: dict[str, Any],
    proposed: dict[str, Any],
) -> dict[str, Any]:
    """Compute deltas between the two pipelines."""
    delta_keys = ["solve_time_s", "cost", "dv_total_km_s", "min_altitude_margin_km"]
    out: dict[str, Any] = {
        "baseline_converged": baseline.get("converged"),
        "proposed_converged": proposed.get("converged"),
    }
    for key in delta_keys:
        bv = baseline.get(key)
        pv = proposed.get(key)
        if isinstance(bv, (int, float)) and isinstance(pv, (int, float)):
            out[f"{key}_delta"] = float(pv - bv)
        else:
            out[f"{key}_delta"] = None
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Case selection
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CaseRow:
    id: int
    config: TransferConfig


def load_cases(
    db_path: Path,
    case_id: int | None = None,
    limit: int | None = None,
    h0: float | None = None,
    max_ecc: float = 0.01,
    max_t_normed: float | None = None,
    converged_filter: str = "converged",
) -> list[CaseRow]:
    """Load collocation cases from the trajectory DB.

    converged_filter:
        "converged" — only rows where converged=TRUE (default)
        "failed"    — only rows where converged=FALSE
        "all"       — both
    """
    con = duckdb.connect(str(db_path), read_only=True)

    # check available columns
    cols = {row[1] for row in con.execute("PRAGMA table_info('trajectories')").fetchall()}

    conditions: list[str] = []
    if converged_filter == "converged":
        conditions.append("converged = TRUE")
    elif converged_filter == "failed":
        conditions.append("converged = FALSE")
    # "all" → no filter on converged
    params: list[Any] = []

    if "method" in cols:
        conditions.append("COALESCE(method, 'collocation') = 'collocation'")

    if case_id is not None:
        conditions.append("id = ?")
        params.append(case_id)
    else:
        if h0 is not None:
            conditions.append("h0 = ?")
            params.append(h0)
        if max_ecc < 1.0:
            conditions.append("COALESCE(e0, 0) <= ?")
            params.append(max_ecc)
            conditions.append("COALESCE(ef, 0) <= ?")
            params.append(max_ecc)
        if max_t_normed is not None:
            conditions.append("T_normed <= ?")
            params.append(max_t_normed)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT id, h0, delta_a, delta_i, T_normed, e0, ef FROM trajectories {where} ORDER BY id"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    rows = con.execute(query, params).fetchall()
    con.close()

    cases = []
    for row in rows:
        rid, rh0, rda, rdi, rtmax, re0, ref = row
        cases.append(CaseRow(
            id=int(rid),
            config=TransferConfig(
                h0=float(rh0),
                delta_a=float(rda),
                delta_i=float(rdi),
                T_max_normed=float(rtmax),
                e0=float(re0 or 0.0),
                ef=float(ref or 0.0),
            ),
        ))

    return cases


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DCM downstream experiment.")
    p.add_argument("--db", type=str, default=str(DEFAULT_DB))
    p.add_argument("--case-id", type=int, default=None,
                   help="Run a single case by DB id.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of cases to run.")
    p.add_argument("--h0", type=float, default=None,
                   help="Filter by initial altitude [km].")
    p.add_argument("--max-ecc", type=float, default=0.01,
                   help="Filter out cases with e0 or ef above this.")
    p.add_argument("--max-t-normed", type=float, default=None,
                   help="Filter out cases with T_normed above this.")
    p.add_argument("--converged", type=str, default="converged",
                   choices=["converged", "failed", "all"],
                   help="Case filter by DB convergence status.")
    p.add_argument("--degree", type=int, default=DEFAULT_DEGREE,
                   help="Bézier curve degree.")
    p.add_argument("--n-seg", type=int, default=DEFAULT_NSEG,
                   help="KOZ linearization segments.")
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser().resolve()
    out_root = Path(args.outdir).expanduser().resolve()

    cases = load_cases(
        db_path,
        case_id=args.case_id,
        limit=args.limit,
        h0=args.h0,
        max_ecc=args.max_ecc,
        max_t_normed=args.max_t_normed,
        converged_filter=args.converged,
    )

    if not cases:
        print("No cases found matching the filters.")
        return

    print(f"Running {len(cases)} case(s)  degree={args.degree}  n_seg={args.n_seg}")
    print("=" * 72)

    aggregate_rows: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        cfg = case.config
        print(
            f"[{i+1}/{len(cases)}] case_id={case.id}  "
            f"h0={cfg.h0:.0f}  da={cfg.delta_a:.1f}  di={cfg.delta_i:.2f}  "
            f"T/T0={cfg.T_max_normed:.3f}  e0={cfg.e0:.4f}  ef={cfg.ef:.4f}"
        )

        # ── baseline ──
        try:
            baseline_result, baseline_time = run_baseline(cfg)
            baseline_summary = summarize_result("baseline", baseline_result, baseline_time, cfg)
        except Exception as exc:
            print(f"  Baseline FAILED: {exc}")
            baseline_summary = {"stage": "baseline", "converged": False, "error": str(exc)}

        # ── proposed ──
        try:
            proposed_result, pass2_time, bezier_time, bezier_info = run_proposed(
                cfg, degree=args.degree, n_seg=args.n_seg,
            )
            proposed_summary = summarize_result("proposed", proposed_result, pass2_time, cfg)
            proposed_summary["bezier_time_s"] = bezier_time
            proposed_summary["pass2_time_s"] = pass2_time
            proposed_summary["total_time_s"] = bezier_time + pass2_time
            proposed_summary["bezier_n_peaks"] = bezier_info.get("n_peaks", -1)
            proposed_summary["bezier_profile_class"] = bezier_info.get("profile_class", -1)
        except Exception as exc:
            print(f"  Proposed FAILED: {exc}")
            proposed_summary = {"stage": "proposed", "converged": False, "error": str(exc)}
            bezier_info = {"error": str(exc)}

        # ── compare ──
        comparison = compare_metrics(baseline_summary, proposed_summary)

        # ── print summary ──
        bc = baseline_summary.get("converged", False)
        pc = proposed_summary.get("converged", False)
        b_cost = baseline_summary.get("cost", float("nan"))
        p_cost = proposed_summary.get("cost", float("nan"))
        b_time = baseline_summary.get("solve_time_s", float("nan"))
        p_time = proposed_summary.get("total_time_s", proposed_summary.get("solve_time_s", float("nan")))
        bz_peaks = bezier_info.get("n_peaks", "?")
        bz_time = proposed_summary.get("bezier_time_s", float("nan"))
        p2_time = proposed_summary.get("pass2_time_s", float("nan"))
        print(
            f"  Baseline:  conv={bc}  cost={b_cost:.6e}  time={b_time:.2f}s  "
            f"peaks={baseline_summary.get('n_peaks', '?')}\n"
            f"  Proposed:  conv={pc}  cost={p_cost:.6e}  time={p_time:.2f}s  "
            f"(bez={bz_time:.2f}s + p2={p2_time:.2f}s)  "
            f"bez_peaks={bz_peaks}"
        )

        # ── save per-case JSON ──
        case_payload = {
            "case_id": case.id,
            "config": {
                "h0": cfg.h0, "delta_a": cfg.delta_a, "delta_i": cfg.delta_i,
                "T_max_normed": cfg.T_max_normed, "e0": cfg.e0, "ef": cfg.ef,
            },
            "bezier_upstream": bezier_info,
            "baseline": baseline_summary,
            "proposed": proposed_summary,
            "comparison": comparison,
        }
        write_json(out_root / f"case_{case.id:06d}.json", case_payload)

        # ── aggregate row ──
        agg = {"case_id": case.id, "h0": cfg.h0, "delta_a": cfg.delta_a, "delta_i": cfg.delta_i}
        for k, v in baseline_summary.items():
            if k != "stage":
                agg[f"baseline_{k}"] = v
        for k, v in proposed_summary.items():
            if k != "stage":
                agg[f"proposed_{k}"] = v
        agg.update(comparison)
        aggregate_rows.append(agg)

    # ── write aggregate CSV ──
    write_aggregate_csv(out_root / "aggregate.csv", aggregate_rows)

    print("=" * 72)
    n_both = sum(1 for r in aggregate_rows if r.get("baseline_converged") and r.get("proposed_converged"))
    n_base_only = sum(1 for r in aggregate_rows if r.get("baseline_converged") and not r.get("proposed_converged"))
    n_prop_only = sum(1 for r in aggregate_rows if not r.get("baseline_converged") and r.get("proposed_converged"))
    print(f"Done. {len(aggregate_rows)} cases.  Both converged: {n_both}  Baseline only: {n_base_only}  Proposed only: {n_prop_only}")
    print(f"Results: {out_root}")


if __name__ == "__main__":
    main()
