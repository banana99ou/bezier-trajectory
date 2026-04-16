#!/usr/bin/env python3
"""
Run scenarios through the Rust optimizer and report results.

Primarily useful for verifying optimizer behavior across configs and
for dumping trace data for offline inspection.

Usage:
    python3 tools/compare_backends.py                    # run 'original' scenario
    python3 tools/compare_backends.py --scenario wall    # pick scenario
    python3 tools/compare_backends.py --all              # all scenarios
    python3 tools/compare_backends.py --json             # dump full JSON traces
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spacetime_bezier.optimize import (
    compute_min_clearance,
    create_spacetime_debug_stepper,
    optimize_spacetime,
)
from spacetime_bezier.scenarios import SCENARIO_MAP


# ── helpers ──────────────────────────────────────────────────────────────────

def _np_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(type(obj))


def _run_backend(scenario: dict, N: int, n_seg: int, max_iter: int, backend: str) -> dict:
    """Run one backend, return a summary dict. Crash loudly on failure."""
    try:
        P_opt, info = optimize_spacetime(
            N=N,
            dim=len(scenario["start"]),
            p_start=scenario["start"],
            p_end=scenario["end"],
            obstacles=scenario["obstacles"],
            n_seg=n_seg,
            max_iter=max_iter,
            tol=1e-6,
            scp_prox_weight=0.3,
            scp_trust_radius=0.0,
            min_dt=0.1,
            verbose=False,
            init_curve=scenario.get("init_curve"),
            backend=backend,
        )
    except RuntimeError as exc:
        return {"backend": backend, "error": str(exc)}

    clearance = compute_min_clearance(P_opt, scenario["obstacles"], dim=len(scenario["start"]), n_eval=3000)
    return {
        "backend": info["backend"],
        "control_points": P_opt,
        "info": info,
        "clearance": float(clearance),
        "feasible": bool(clearance > 0),
        "iterations": int(info.get("iterations", -1)),
    }


def _collect_stepper_stages(scenario: dict, N: int, n_seg: int, max_iter: int, backend: str) -> list[dict]:
    """Run the debug stepper, collect stage names and key data per frame."""
    try:
        stepper = create_spacetime_debug_stepper(
            N=N,
            dim=len(scenario["start"]),
            p_start=scenario["start"],
            p_end=scenario["end"],
            obstacles=scenario["obstacles"],
            n_seg=n_seg,
            max_iter=max_iter,
            tol=1e-6,
            scp_prox_weight=0.3,
            scp_trust_radius=0.0,
            min_dt=0.1,
            init_curve=scenario.get("init_curve"),
            backend=backend,
        )
    except (RuntimeError, ValueError) as exc:
        return [{"error": str(exc)}]

    stages = []
    try:
        while True:
            frame = stepper.next_frame()
            if frame is None:
                break

            entry = {
                "stage": frame.stage,
                "iteration": frame.iteration,
                "backend": frame.payload.get("backend") if isinstance(frame.payload, dict) else None,
            }
            payload = frame.payload if isinstance(frame.payload, dict) else {}

            if frame.stage == "init-guess":
                entry["clearance"] = payload.get("initial_clearance") or payload.get("metrics", {}).get("clearance")
            elif frame.stage == "koz-linearization":
                koz = payload.get("koz", {})
                entry["koz_row_count"] = koz.get("row_count", 0)
                koz_segments = koz.get("segments", [])
                entry["active_obstacles"] = sum(
                    len(seg.get("active_obstacles", [])) for seg in koz_segments
                )
            elif frame.stage == "supporting-surface-generation":
                koz = payload.get("koz", {})
                entry["koz_row_count"] = koz.get("row_count", 0)
                summary = koz.get("summary", {})
                entry["worst_margin"] = summary.get("worst_margin")
                normal = summary.get("worst_normal_with_time_coeff", [])
                entry["worst_time_coeff"] = float(normal[-1]) if normal else None
            elif frame.stage in {"solver-candidate", "solver-call"}:
                solver = payload.get("solver", {})
                entry["solver_status"] = solver.get("status") or solver.get("raw_status")
                entry["solver_name"] = solver.get("name", "trust-constr" if "nit" in solver else "unknown")
            elif frame.stage == "post-eval":
                metrics = payload.get("metrics", {})
                entry["delta"] = metrics.get("delta")
                entry["clearance"] = metrics.get("clearance")
            elif frame.stage == "finalize":
                entry["clearance"] = payload.get("final_clearance")
                entry["converged"] = payload.get("converged")

            stages.append(entry)
    except RuntimeError as exc:
        stages.append({"stage": "CRASH", "error": str(exc)})
    return stages


# ── diff logic ───────────────────────────────────────────────────────────────

def _diff_results(py_result: dict, rs_result: dict) -> list[str]:
    """Return a list of human-readable diff lines."""
    diffs = []

    if "error" in py_result:
        diffs.append(f"  PYTHON FAILED: {py_result['error']}")
    if "error" in rs_result:
        diffs.append(f"  RUST FAILED: {rs_result['error']}")
    if "error" in py_result or "error" in rs_result:
        return diffs

    # backend attribution
    if py_result["backend"] != "python":
        diffs.append(f"  !! Python result claims backend={py_result['backend']}")
    if rs_result["backend"] != "rust":
        diffs.append(f"  !! Rust result claims backend={rs_result['backend']}")

    # iterations
    py_iter = py_result["iterations"]
    rs_iter = rs_result["iterations"]
    if py_iter != rs_iter:
        diffs.append(f"  iterations: python={py_iter}, rust={rs_iter}")

    # clearance
    py_cl = py_result["clearance"]
    rs_cl = rs_result["clearance"]
    cl_diff = abs(py_cl - rs_cl)
    diffs.append(f"  clearance: python={py_cl:.6f}, rust={rs_cl:.6f}, diff={cl_diff:.6f}")
    if py_result["feasible"] != rs_result["feasible"]:
        diffs.append(f"  !! FEASIBILITY DISAGREES: python={py_result['feasible']}, rust={rs_result['feasible']}")

    # control point distance
    P_py = np.asarray(py_result["control_points"], dtype=float)
    P_rs = np.asarray(rs_result["control_points"], dtype=float)
    if P_py.shape != P_rs.shape:
        diffs.append(f"  !! SHAPE MISMATCH: python={P_py.shape}, rust={P_rs.shape}")
    else:
        cp_diff = np.linalg.norm(P_py - P_rs)
        max_cp_diff = float(np.max(np.abs(P_py - P_rs)))
        diffs.append(f"  control points: L2 diff={cp_diff:.6f}, max element diff={max_cp_diff:.6f}")

        # per-coordinate breakdown
        spatial_diff = np.linalg.norm(P_py[:, :-1] - P_rs[:, :-1])
        time_diff = np.linalg.norm(P_py[:, -1] - P_rs[:, -1])
        diffs.append(f"    spatial diff={spatial_diff:.6f}, time diff={time_diff:.6f}")

        # time monotonicity check
        py_dt = np.diff(P_py[:, -1])
        rs_dt = np.diff(P_rs[:, -1])
        py_mono = bool(np.all(py_dt >= 0.1 - 1e-9))
        rs_mono = bool(np.all(rs_dt >= 0.1 - 1e-9))
        if not py_mono:
            diffs.append(f"  !! PYTHON time monotonicity violated: min dt={py_dt.min():.6f}")
        if not rs_mono:
            diffs.append(f"  !! RUST time monotonicity violated: min dt={rs_dt.min():.6f}")

    return diffs


def _diff_stages(py_stages: list[dict], rs_stages: list[dict]) -> list[str]:
    """Diff the debug stepper stage sequences."""
    diffs = []

    if py_stages and "error" in py_stages[0]:
        diffs.append(f"  PYTHON stepper failed: {py_stages[0]['error']}")
    if rs_stages and "error" in rs_stages[0]:
        diffs.append(f"  RUST stepper failed: {rs_stages[0]['error']}")
    if (py_stages and "error" in py_stages[0]) or (rs_stages and "error" in rs_stages[0]):
        return diffs

    py_names = [s["stage"] for s in py_stages]
    rs_names = [s["stage"] for s in rs_stages]
    diffs.append(f"  stage count: python={len(py_names)}, rust={len(rs_names)}")

    # compare finalize frames
    py_final = next((s for s in py_stages if s["stage"] == "finalize"), None)
    rs_final = next((s for s in rs_stages if s["stage"] == "finalize"), None)
    if py_final and rs_final:
        py_cl = py_final.get("clearance")
        rs_cl = rs_final.get("clearance")
        if py_cl is not None and rs_cl is not None:
            diffs.append(f"  stepper final clearance: python={py_cl:.6f}, rust={rs_cl:.6f}")

    # compare KOZ row counts from first KOZ frame
    py_koz = next((s for s in py_stages if s["stage"] == "koz-linearization"), None)
    rs_koz = next((s for s in rs_stages if s["stage"] == "supporting-surface-generation"), None)
    if py_koz and rs_koz:
        diffs.append(
            f"  first KOZ rows: python={py_koz.get('koz_row_count', '?')}, "
            f"rust={rs_koz.get('koz_row_count', '?')}"
        )
        if rs_koz.get("worst_time_coeff") is not None:
            diffs.append(f"  rust worst time coefficient: {rs_koz['worst_time_coeff']:.6f}")

    # compare solver names
    py_solver = next((s for s in py_stages if s["stage"] == "solver-candidate"), None)
    rs_solver = next((s for s in rs_stages if s["stage"] == "solver-call"), None)
    if py_solver and rs_solver:
        diffs.append(
            f"  solver: python={py_solver.get('solver_name', '?')}, "
            f"rust={rs_solver.get('solver_name', '?')}"
        )

    return diffs


# ── main ─────────────────────────────────────────────────────────────────────

def run_comparison(scenario_name: str, dump_json: bool = False) -> dict:
    scenario_fn, configs = SCENARIO_MAP[scenario_name]
    scenario = scenario_fn()

    # use the smallest config for fast comparison
    N, n_seg = configs[0]
    max_iter = 30

    print(f"\n{'=' * 70}")
    print(f"  {scenario['title']} ({scenario_name})  |  N={N}, n_seg={n_seg}")
    print(f"{'=' * 70}")

    print("\n  Running Python backend...", end="", flush=True)
    py_result = _run_backend(scenario, N, n_seg, max_iter, "python")
    print(" done")

    print("  Running Rust backend...", end="", flush=True)
    rs_result = _run_backend(scenario, N, n_seg, max_iter, "rust")
    print(" done")

    print("\n--- Batch result diff ---")
    for line in _diff_results(py_result, rs_result):
        print(line)

    print("\n  Running Python stepper...", end="", flush=True)
    py_stages = _collect_stepper_stages(scenario, N, n_seg, max_iter, "python")
    print(" done")

    print("  Running Rust stepper...", end="", flush=True)
    rs_stages = _collect_stepper_stages(scenario, N, n_seg, max_iter, "rust")
    print(" done")

    print("\n--- Stepper diff ---")
    for line in _diff_stages(py_stages, rs_stages):
        print(line)

    result = {
        "scenario": scenario_name,
        "config": {"N": N, "n_seg": n_seg, "max_iter": max_iter},
        "python": {k: v for k, v in py_result.items() if k != "control_points" or dump_json},
        "rust": {k: v for k, v in rs_result.items() if k != "control_points" or dump_json},
        "python_stages": py_stages if dump_json else None,
        "rust_stages": rs_stages if dump_json else None,
    }

    if dump_json:
        out_path = Path(f"figures/backend_diff_{scenario_name}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=_np_to_json))
        print(f"\n  Full trace written to {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare Python vs Rust spacetime backends")
    parser.add_argument("--scenario", default="original", choices=list(SCENARIO_MAP.keys()))
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--json", action="store_true", help="Dump full JSON traces")
    args = parser.parse_args()

    scenarios = list(SCENARIO_MAP.keys()) if args.all else [args.scenario]

    for name in scenarios:
        run_comparison(name, dump_json=args.json)

    print()


if __name__ == "__main__":
    main()
