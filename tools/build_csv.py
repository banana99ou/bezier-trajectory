#!/usr/bin/env python3
"""
Rebuild orbital_results_summary.csv and .json from the 120-deg cache files.

These are the correct cache files for the paper's demonstration scenario
(Progress-to-ISS, 120-deg phase lag, KOZ at Earth+100 km).

Usage:
    python tools/build_csv.py
"""

import sys, json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from orbital_docking import constants, generate_initial_control_points
from orbital_docking.cache import get_cache_key, get_cache_path, load_from_cache

# ── 120-deg scenario geometry (must match build_f3.py) ───────────────
INCLINATION_DEG = 51.64
RAAN_DEG = 0.0
ISS_U_DEG = 45.0
PROGRESS_LAG_DEG = 120.0


def _rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rotx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _eci(radius, inc_deg, raan_deg, u_deg):
    mu = constants.EARTH_MU_SCALED
    inc, raan, u = np.deg2rad(inc_deg), np.deg2rad(raan_deg), np.deg2rad(u_deg)
    r = np.array([radius * np.cos(u), radius * np.sin(u), 0.0])
    v = np.sqrt(mu / radius) * np.array([-np.sin(u), np.cos(u), 0.0])
    q = _rotz(raan) @ _rotx(inc)
    return q @ r, q @ v


def get_120deg_endpoints():
    """Return (P_start, v0, P_end, v1) for the 120-deg phase-lag scenario."""
    progress_r = constants.EARTH_RADIUS_KM + constants.PROGRESS_START_ALTITUDE_KM
    iss_r = constants.EARTH_RADIUS_KM + constants.ISS_TARGET_ALTITUDE_KM
    P_start, v0 = _eci(progress_r, INCLINATION_DEG, RAAN_DEG,
                        ISS_U_DEG - PROGRESS_LAG_DEG)
    P_end, v1 = _eci(iss_r, INCLINATION_DEG, RAAN_DEG, ISS_U_DEG)
    return P_start, v0, P_end, v1


def get_cache_path_120(N, n_seg, P_start, v0, P_end, v1):
    """Compute the cache path for a 120-deg run with given degree and seg count."""
    P_init = generate_initial_control_points(N, P_start, P_end)
    key = get_cache_key(
        P_init, n_seg,
        r_e=constants.KOZ_RADIUS,
        max_iter=10000, tol=1e-12, sample_count=100,
        v0=v0, v1=v1, a0=None, a1=None,
        objective="dv", scp_prox_weight=1e-6, scp_trust_radius=2000.0,
    )
    return get_cache_path(key, n_seg)


DEGREES = [6, 7, 8]
SEGS = [2, 4, 8, 16, 32, 64]

CSV_FIELDS = [
    "degree", "n_seg", "solve_success", "min_radius_km", "safety_margin_km",
    "dv_proxy_m_s", "max_control_accel_ms2", "mean_control_accel_ms2",
    "runtime_s", "outer_iterations", "termination_reason",
    "objective_mode", "velocity_bc_enforced", "tol", "max_iter",
    "solver_backend",
]


def main():
    P_start, v0, P_end, v1 = get_120deg_endpoints()
    rows = []

    for N in DEGREES:
        for n_seg in SEGS:
            path = get_cache_path_120(N, n_seg, P_start, v0, P_end, v1)
            if not path.exists():
                print(f"  MISSING: N={N} seg={n_seg} -> {path}")
                continue

            result = load_from_cache(path)
            if result is None:
                print(f"  LOAD FAIL: {path}")
                continue

            P_opt, info = result
            min_r = info["min_radius"]
            safety = min_r - constants.KOZ_RADIUS

            row = {
                "degree": N,
                "n_seg": n_seg,
                "solve_success": info["feasible"],
                "min_radius_km": min_r,
                "safety_margin_km": safety,
                "dv_proxy_m_s": info["dv_proxy_m_s"],
                "max_control_accel_ms2": info["max_control_accel_ms2"],
                "mean_control_accel_ms2": info["mean_control_accel_ms2"],
                "runtime_s": info["elapsed_time"],
                "outer_iterations": info["iterations"],
                "termination_reason": info["termination_reason"],
                "objective_mode": "dv",
                "velocity_bc_enforced": True,
                "tol": 1e-12,
                "max_iter": 10000,
                "solver_backend": "rust",
            }
            rows.append(row)
            status = "OK" if info["feasible"] else "INFEASIBLE"
            print(f"  {status}: N={N} seg={n_seg:2d}  dv={info['dv_proxy_m_s']:10.3f}  "
                  f"safety={safety:8.3f} km  rt={info['elapsed_time']:.1f}s")

    # Write CSV
    out_dir = ROOT / "artifacts" / "paper_artifacts"
    out_csv = out_dir / "orbital_results_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV -> {out_csv}  ({len(rows)} rows)")

    # Write JSON
    out_json = out_dir / "orbital_results_summary.json"
    doc = {
        "configuration": {
            "degrees": DEGREES,
            "segment_counts": SEGS,
            "objective": "dv",
            "max_iter": 10000,
            "tol": 1e-12,
            "scp_prox_weight": 1e-6,
            "scp_trust_radius": 2000.0,
            "enforce_prograde": True,
            "v0_enforced": True,
            "v1_enforced": True,
            "transfer_time_s": constants.TRANSFER_TIME_S,
            "koz_radius_km": constants.KOZ_RADIUS,
            "progress_lag_deg": PROGRESS_LAG_DEG,
            "iss_u_deg": ISS_U_DEG,
            "solver_backend": "rust",
        },
        "rows": rows,
    }
    with open(out_json, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    print(f"JSON -> {out_json}")


if __name__ == "__main__":
    main()
