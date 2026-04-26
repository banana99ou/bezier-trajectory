"""
KOZ altitude sweep for subdivision ablation study.

Runs N=7, all segment counts, across 12 KOZ altitudes to find
scenarios where the constraint is active and n_seg matters.
"""

import numpy as np
import pickle
import time
import sys
from pathlib import Path
from collections import defaultdict

from orbital_docking import constants
from orbital_docking.optimization import (
    optimize_all_segment_counts,
    generate_initial_control_points,
)

EARTH_RADIUS_KM = constants.EARTH_RADIUS_KM


def _rotz(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def eci_from_circular(radius_km, inc_deg, raan_deg, u_deg):
    mu = constants.EARTH_MU_SCALED
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    u = np.deg2rad(u_deg)
    r_pqw = np.array([radius_km * np.cos(u), radius_km * np.sin(u), 0.0])
    v_circ = np.sqrt(mu / radius_km)
    v_pqw = v_circ * np.array([-np.sin(u), np.cos(u), 0.0])
    q = _rotz(raan) @ _rotx(inc)
    return q @ r_pqw, q @ v_pqw


def main():
    N = 7
    PROGRESS_ALT = 245.0
    ISS_ALT = 400.0
    INC = 51.64
    RAAN = 0.0
    ISS_U = 45.0
    PROGRESS_LAG = 70.0

    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT

    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PROGRESS_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N, P_start, P_end)

    # 12 KOZ altitudes: from trivially inactive to aggressively tight
    koz_altitudes = [100, 150, 180, 200, 210, 220, 225, 230, 234, 237, 240, 243]

    segment_counts = [2, 4, 8, 16, 32, 64]
    MAX_ITER = 1000
    N_JOBS = 6

    all_results = {}

    for koz_alt in koz_altitudes:
        koz_r = EARTH_RADIUS_KM + koz_alt
        gap = progress_r - koz_r
        print(f"\n{'='*70}")
        print(f"KOZ altitude = {koz_alt} km  (r_koz = {koz_r:.1f} km, gap to Progress orbit = {gap:.1f} km)")
        print(f"{'='*70}")

        t0 = time.time()
        results = optimize_all_segment_counts(
            P_init,
            r_e=koz_r,
            segment_counts=segment_counts,
            max_iter=MAX_ITER,
            tol=1e-6,
            verbose=False,
            use_cache=True,
            ignore_existing_cache=False,
            v0=v0, v1=v1,
            objective="dv",
            scp_prox_weight=1e-6,
            scp_trust_radius=2000.0,
            enforce_prograde=True,
            n_jobs=N_JOBS,
        )
        elapsed = time.time() - t0

        print(f"\n  Completed in {elapsed:.1f}s")
        print(f"  {'n_seg':>5s}  {'cost':>14s}  {'dv(m/s)':>10s}  {'clearance':>10s}  {'iters':>6s}  feas  term")
        print(f"  {'-'*70}")

        row_data = []
        for n_seg, P_opt, info in results:
            cost = info.get("cost")
            dv = info.get("dv_proxy_m_s")
            min_r = info.get("min_radius")
            clearance = (min_r - koz_r) if min_r is not None else None
            feas = info.get("feasible")
            iters = info.get("iterations", -1)
            term = info.get("termination_reason", "?")

            cost_s = f"{cost:.6e}" if cost is not None else "N/A"
            dv_s = f"{dv:.2f}" if dv is not None else "N/A"
            cl_s = f"{clearance:.2f}" if clearance is not None else "N/A"
            f_s = "Y" if feas else "N"

            print(f"  {n_seg:5d}  {cost_s:>14s}  {dv_s:>10s}  {cl_s:>10s}  {iters:6d}  {f_s:>4s}  {term}")

            row_data.append({
                "koz_alt": koz_alt, "koz_r": koz_r, "n_seg": n_seg,
                "cost": cost, "dv": dv, "min_radius": min_r,
                "clearance": clearance, "feasible": feas,
                "iterations": iters, "termination": term,
            })

        all_results[koz_alt] = row_data

    # Save results
    out_path = Path("figures/koz_sweep_N7.pkl")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {out_path}")

    # Summary table: does n_seg matter?
    print(f"\n{'='*70}")
    print("SUMMARY: dv spread across n_seg (for n_seg >= 4, feasible only)")
    print(f"{'='*70}")
    print(f"  {'koz_alt':>7s}  {'gap':>5s}  {'min_dv':>10s}  {'max_dv':>10s}  {'spread%':>8s}  {'n_feas':>6s}")
    print(f"  {'-'*55}")

    for koz_alt in koz_altitudes:
        rows = all_results[koz_alt]
        feas_4plus = [r for r in rows if r["n_seg"] >= 4 and r["feasible"]]
        gap = progress_r - (EARTH_RADIUS_KM + koz_alt)
        if feas_4plus:
            dvs = [r["dv"] for r in feas_4plus]
            mn, mx = min(dvs), max(dvs)
            spread = (mx - mn) / mn * 100 if mn > 0 else float("inf")
            print(f"  {koz_alt:7d}  {gap:5.0f}  {mn:10.2f}  {mx:10.2f}  {spread:7.3f}%  {len(feas_4plus):6d}")
        else:
            print(f"  {koz_alt:7d}  {gap:5.0f}  {'---':>10s}  {'---':>10s}  {'---':>8s}  {0:6d}")


if __name__ == "__main__":
    main()
