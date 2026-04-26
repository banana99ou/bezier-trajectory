"""
Convergence diagnostic: run the representative case (N=7, n_seg=16)
with truncated max_iter values to verify that solution metrics plateau
well before the 10,000-iteration budget used in the paper.

Scenario matches paper_execution_state.md:
  120 deg phase lag, T=1500 s, r_e=6471 km, objective=dv
"""

import json
import numpy as np
from pathlib import Path

from orbital_docking import constants
from orbital_docking.optimization import (
    optimize_orbital_docking,
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
    N_SEG = 16
    PROGRESS_ALT = 245.0
    ISS_ALT = 400.0
    INC = 51.64
    RAAN = 0.0
    ISS_U = 45.0
    PROGRESS_LAG = 120.0

    KOZ_RADIUS = 6471.0
    TRANSFER_TIME = 1500.0

    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT

    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PROGRESS_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N, P_start, P_end)

    max_iter_values = [100, 500, 1000, 2000, 5000, 10000]
    results = []

    print(f"Convergence diagnostic: N={N}, n_seg={N_SEG}, 120 deg phase lag")
    print(f"{'max_iter':>10} {'dv_proxy (m/s)':>16} {'safety (km)':>14} "
          f"{'runtime (s)':>12} {'final_delta':>14} {'feasible':>10}")
    print("-" * 80)

    for mi in max_iter_values:
        P_opt, info = optimize_orbital_docking(
            P_init,
            n_seg=N_SEG,
            r_e=KOZ_RADIUS,
            max_iter=mi,
            tol=1e-12,
            v0=v0,
            v1=v1,
            sample_count=100,
            objective_mode="dv",
            scp_prox_weight=1e-6,
            scp_trust_radius=2000.0,
            transfer_time=TRANSFER_TIME,
            verbose=False,
            use_cache=False,
        )

        dv = info["dv_proxy_m_s"]
        safety = info["min_radius"] - KOZ_RADIUS
        runtime = info["elapsed_time"]
        delta = info.get("final_delta_norm", float("nan"))
        feasible = info.get("feasible", False)

        results.append({
            "max_iter": mi,
            "dv_proxy_m_s": dv,
            "safety_margin_km": safety,
            "runtime_s": runtime,
            "final_delta_norm": delta,
            "feasible": feasible,
        })

        print(f"{mi:>10} {dv:>16.3f} {safety:>14.3f} "
              f"{runtime:>12.3f} {delta:>14.2e} {str(feasible):>10}")

    # Summary: relative change from 10k baseline
    baseline = results[-1]
    print("\n--- Relative change vs 10,000-iteration baseline ---")
    print(f"{'max_iter':>10} {'dv_delta (%)':>14} {'safety_delta (%)':>18}")
    print("-" * 46)
    for r in results:
        dv_pct = (r["dv_proxy_m_s"] - baseline["dv_proxy_m_s"]) / baseline["dv_proxy_m_s"] * 100
        if baseline["safety_margin_km"] > 0:
            s_pct = (r["safety_margin_km"] - baseline["safety_margin_km"]) / baseline["safety_margin_km"] * 100
        else:
            s_pct = float("nan")
        print(f"{r['max_iter']:>10} {dv_pct:>+14.4f} {s_pct:>+18.4f}")

    # Save
    out_path = Path("artifacts/convergence_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
