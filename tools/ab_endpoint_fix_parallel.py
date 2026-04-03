#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Any


SEG_COUNTS_DEFAULT = [2, 4, 8, 16, 32, 64]
ORDERS_DEFAULT = [2, 3, 4]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def patch_to_before_fix(opt_path: Path) -> None:
    txt = opt_path.read_text(encoding="utf-8")

    # Revert keep_feasible behavior
    txt = txt.replace(
        "bounds = Bounds(lb, ub, keep_feasible=True)",
        "bounds = Bounds(lb, ub)"
    )

    # Remove endpoint hard projection block
    txt = re.sub(
        r"\n\s*# Hard-project endpoint positions each outer iteration\.[\s\S]*?"
        r"P_new\[-1, :\] = P_init\[-1, :\]\n",
        "\n",
        txt,
        flags=re.MULTILINE,
    )

    opt_path.write_text(txt, encoding="utf-8")


def prepare_variants(repo: Path, work_root: Path) -> Tuple[Path, Path]:
    a_dir = work_root / "A_before_fix"
    b_dir = work_root / "B_after_fix"

    shutil.copytree(repo, a_dir)
    shutil.copytree(repo, b_dir)

    opt_a = a_dir / "orbital_docking" / "optimization.py"
    if not opt_a.exists():
        raise FileNotFoundError(f"Missing optimization.py in {opt_a}")
    patch_to_before_fix(opt_a)

    return a_dir, b_dir


def build_case_list(orders: List[int], seg_counts: List[int]) -> List[Tuple[int, int]]:
    return [(N, nseg) for N in orders for nseg in seg_counts]


def _worker_run_case(
    variant_path: str,
    variant_label: str,
    N: int,
    n_seg: int,
    max_iter: int,
    tol: float,
    objective: str,
) -> Dict[str, Any]:
    try:
        # Avoid BLAS oversubscription per process
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        vpath = Path(variant_path).resolve()
        sys.path.insert(0, str(vpath))

        import numpy as np
        from orbital_docking import constants, generate_initial_control_points
        from orbital_docking.optimization import optimize_orbital_docking

        # Progress-to-ISS inspired scenario (same as main script)
        PROGRESS_START_ALTITUDE_KM = 245.0
        ISS_TARGET_ALTITUDE_KM = 400.0
        INCLINATION_DEG = 51.64
        RAAN_DEG = 0.0
        ISS_U_DEG = 45.0
        PROGRESS_LAG_DEG = 30.0

        def _rotz(theta_rad: float):
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        def _rotx(theta_rad: float):
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

        def _eci_from_circular_elements(radius_km: float, inc_deg: float, raan_deg: float, u_deg: float):
            mu = constants.EARTH_MU_SCALED
            inc = np.deg2rad(inc_deg)
            raan = np.deg2rad(raan_deg)
            u = np.deg2rad(u_deg)
            r_pqw = np.array([radius_km * np.cos(u), radius_km * np.sin(u), 0.0])
            v_circ = np.sqrt(mu / radius_km)
            v_pqw = v_circ * np.array([-np.sin(u), np.cos(u), 0.0])
            q = _rotz(raan) @ _rotx(inc)
            return q @ r_pqw, q @ v_pqw

        iss_u_deg = ISS_U_DEG
        progress_u_deg = iss_u_deg - PROGRESS_LAG_DEG
        progress_radius_km = constants.EARTH_RADIUS_KM + PROGRESS_START_ALTITUDE_KM
        iss_radius_km = constants.EARTH_RADIUS_KM + ISS_TARGET_ALTITUDE_KM
        P_start, v0 = _eci_from_circular_elements(progress_radius_km, INCLINATION_DEG, RAAN_DEG, progress_u_deg)
        P_end, v1 = _eci_from_circular_elements(iss_radius_km, INCLINATION_DEG, RAAN_DEG, iss_u_deg)

        P_init = generate_initial_control_points(N, P_start, P_end)

        import time
        t0 = time.time()
        P_opt, info = optimize_orbital_docking(
            P_init,
            n_seg=n_seg,
            r_e=constants.KOZ_RADIUS,
            max_iter=max_iter,
            tol=tol,
            v0=v0,
            v1=v1,
            objective_mode=objective,
            verbose=False,
            use_cache=False,
        )
        wall = time.time() - t0

        d0 = float(np.linalg.norm(P_opt[0] - P_start))
        d1 = float(np.linalg.norm(P_opt[-1] - P_end))

        return {
            "variant": variant_label,
            "N": int(N),
            "n_seg": int(n_seg),
            "iterations": int(info.get("iterations", -1)),
            "feasible": bool(info.get("feasible", False)),
            "min_radius": float(info.get("min_radius", math.nan)),
            "cost_true_energy": float(info.get("cost_true_energy", info.get("cost", math.nan))),
            "dv_proxy_m_s": float(info.get("dv_proxy_m_s", math.nan)),
            "max_control_accel_ms2": float(info.get("max_control_accel_ms2", info.get("accel", math.nan))),
            "mean_control_accel_ms2": float(info.get("mean_control_accel_ms2", math.nan)),
            "elapsed_solver_s": float(info.get("elapsed_time", math.nan)),
            "elapsed_wall_s": float(wall),
            "endpoint_drift_start_km": d0,
            "endpoint_drift_end_km": d1,
            "error": "",
        }
    except Exception as e:
        return {
            "variant": variant_label,
            "N": int(N),
            "n_seg": int(n_seg),
            "iterations": -1,
            "feasible": False,
            "min_radius": math.nan,
            "cost_true_energy": math.nan,
            "dv_proxy_m_s": math.nan,
            "max_control_accel_ms2": math.nan,
            "mean_control_accel_ms2": math.nan,
            "elapsed_solver_s": math.nan,
            "elapsed_wall_s": math.nan,
            "endpoint_drift_start_km": math.nan,
            "endpoint_drift_end_km": math.nan,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }


def run_variant_parallel(
    variant_path: Path,
    variant_label: str,
    cases: List[Tuple[int, int]],
    max_iter: int,
    tol: float,
    objective: str,
    workers: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for N, n_seg in cases:
            futures.append(
                ex.submit(
                    _worker_run_case,
                    str(variant_path),
                    variant_label,
                    N,
                    n_seg,
                    max_iter,
                    tol,
                    objective,
                )
            )
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda r: (r["N"], r["n_seg"]))
    return rows


def write_outputs(
    out_dir: Path,
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    max_iter: int,
    tol: float,
    objective: str,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_csv = out_dir / f"ab_raw_{ts}.csv"
    cmp_csv = out_dir / f"ab_compare_{ts}.csv"
    report_md = out_dir / f"ab_report_{ts}.md"

    # Raw CSV
    raw_fields = [
        "variant", "N", "n_seg", "iterations", "feasible", "min_radius",
        "cost_true_energy", "dv_proxy_m_s", "max_control_accel_ms2",
        "mean_control_accel_ms2", "elapsed_solver_s", "elapsed_wall_s",
        "endpoint_drift_start_km", "endpoint_drift_end_km", "error"
    ]
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for r in rows_a + rows_b:
            w.writerow(r)

    idx_a = {(r["N"], r["n_seg"]): r for r in rows_a}
    idx_b = {(r["N"], r["n_seg"]): r for r in rows_b}
    keys = sorted(idx_a.keys())

    metrics = [
        "cost_true_energy",
        "dv_proxy_m_s",
        "max_control_accel_ms2",
        "mean_control_accel_ms2",
        "min_radius",
        "elapsed_solver_s",
        "elapsed_wall_s",
        "endpoint_drift_start_km",
        "endpoint_drift_end_km",
    ]

    def pct(a: float, b: float) -> float:
        # percent change from A -> B
        if not math.isfinite(a) or a == 0.0:
            return math.nan
        return (b - a) / a * 100.0

    cmp_fields = ["N", "n_seg", "metric", "A_before_fix", "B_after_fix", "delta_abs", "delta_pct"]
    cmp_rows = []
    for k in keys:
        ra, rb = idx_a[k], idx_b[k]
        for m in metrics:
            a = float(ra[m])
            b = float(rb[m])
            cmp_rows.append({
                "N": k[0],
                "n_seg": k[1],
                "metric": m,
                "A_before_fix": a,
                "B_after_fix": b,
                "delta_abs": b - a,
                "delta_pct": pct(a, b),
            })

    with cmp_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cmp_fields)
        w.writeheader()
        w.writerows(cmp_rows)

    # Markdown summary
    max_drift_a = max(
        max(float(r["endpoint_drift_start_km"]), float(r["endpoint_drift_end_km"]))
        for r in rows_a if not r["error"]
    )
    max_drift_b = max(
        max(float(r["endpoint_drift_start_km"]), float(r["endpoint_drift_end_km"]))
        for r in rows_b if not r["error"]
    )

    lines = []
    lines.append("# A/B Endpoint-Invariance Impact Report")
    lines.append(f"- Max iter: `{max_iter}`")
    lines.append(f"- Tol: `{tol}`")
    lines.append(f"- Objective: `{objective}`")
    lines.append("")
    lines.append("## Endpoint drift")
    lines.append(f"- A (before fix) max drift: `{max_drift_a:.6e}` km")
    lines.append(f"- B (after fix)  max drift: `{max_drift_b:.6e}` km")
    lines.append("")
    lines.append("## Per-case key metrics")
    lines.append("| N | n_seg | cost A | cost B | dv A | dv B | max_u A | max_u B | min_r A | min_r B |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in keys:
        ra, rb = idx_a[k], idx_b[k]
        lines.append(
            f"| {k[0]} | {k[1]} | "
            f"{float(ra['cost_true_energy']):.6e} | {float(rb['cost_true_energy']):.6e} | "
            f"{float(ra['dv_proxy_m_s']):.6e} | {float(rb['dv_proxy_m_s']):.6e} | "
            f"{float(ra['max_control_accel_ms2']):.6e} | {float(rb['max_control_accel_ms2']):.6e} | "
            f"{float(ra['min_radius']):.6e} | {float(rb['min_radius']):.6e} |"
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cmp_csv, report_md


def main():
    ap = argparse.ArgumentParser(description="Parallel A/B test for endpoint fix impact.")
    ap.add_argument("--repo", type=Path, default=Path.cwd(), help="Path to repo root")
    ap.add_argument("--outdir", type=Path, default=Path("artifacts/ab_tests_parallel"))
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--objective", type=str, default="dv", choices=["dv", "energy"])
    ap.add_argument("--orders", type=str, default="2,3,4")
    ap.add_argument("--seg-counts", type=str, default="2,4,8,16,32,64")
    ap.add_argument("--workers-total", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--concurrent-variants", action="store_true",
                    help="Run A and B at the same time (splits workers).")
    args = ap.parse_args()

    repo = args.repo.resolve()
    orders = parse_int_list(args.orders)
    seg_counts = parse_int_list(args.seg_counts)
    cases = build_case_list(orders, seg_counts)

    # Keep BLAS sane in parent as well
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    work_root = Path(tempfile.mkdtemp(prefix="ab_endpoint_fix_parallel_"))
    try:
        print(f"[INFO] Preparing variants in {work_root}")
        a_dir, b_dir = prepare_variants(repo, work_root)

        wt = max(1, int(args.workers_total))
        if args.concurrent_variants:
            w_a = max(1, wt // 2)
            w_b = max(1, wt - w_a)
            print(f"[INFO] Concurrent variants: workers A={w_a}, B={w_b}")

            with ThreadPoolExecutor(max_workers=2) as tx:
                fut_a = tx.submit(run_variant_parallel, a_dir, "A_before_fix", cases,
                                  args.max_iter, args.tol, args.objective, w_a)
                fut_b = tx.submit(run_variant_parallel, b_dir, "B_after_fix", cases,
                                  args.max_iter, args.tol, args.objective, w_b)
                rows_a = fut_a.result()
                rows_b = fut_b.result()
        else:
            print(f"[INFO] Sequential variants: workers per variant={wt}")
            rows_a = run_variant_parallel(a_dir, "A_before_fix", cases,
                                          args.max_iter, args.tol, args.objective, wt)
            rows_b = run_variant_parallel(b_dir, "B_after_fix", cases,
                                          args.max_iter, args.tol, args.objective, wt)

        cmp_csv, report_md = write_outputs(
            args.outdir.resolve(), rows_a, rows_b,
            args.max_iter, args.tol, args.objective
        )

        print("[DONE] A/B test complete.")
        print(f"[OUT] Compare CSV: {cmp_csv}")
        print(f"[OUT] Report MD : {report_md}")

    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()