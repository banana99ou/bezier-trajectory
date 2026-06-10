"""
Basin-check: do frozen and unfrozen converge to the same minimum given enough
iterations, or to genuinely different optima?

Strategy: run n_seg=16 (the paper-relevant case where the dv gap is largest) at
progressively larger max_iter, and compare:
  - final dv_proxy
  - final_delta_norm
  - L2 distance between P_opt arrays (frozen vs unfrozen)
  - L2 distance to the iter=1000 result (within-mode "still moving" check)

If frozen catches up to unfrozen given enough iters, the gap is a convergence-rate
story. If it plateaus at a higher dv with low final_delta, they're different minima.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from orbital_docking import constants
from orbital_docking.optimization import (
    optimize_orbital_docking,
    generate_initial_control_points,
)
from tools.probe_frozen_jacobian import (
    eci_from_circular,
    EARTH_RADIUS_KM,
    PROGRESS_ALT,
    ISS_ALT,
    INC,
    RAAN,
    ISS_U,
    PROGRESS_LAG,
    KOZ_RADIUS,
    TRANSFER_TIME,
    N_DEG,
)


N_SEG = 16
MAX_ITER_GRID = [1000, 3000, 10000, 30000]
TOL = 1e-12  # tight; rely on max_iter for stopping


def make_p_init():
    progress_r = EARTH_RADIUS_KM + PROGRESS_ALT
    iss_r = EARTH_RADIUS_KM + ISS_ALT
    P_start, v0 = eci_from_circular(progress_r, INC, RAAN, ISS_U - PROGRESS_LAG)
    P_end, v1 = eci_from_circular(iss_r, INC, RAAN, ISS_U)
    P_init = generate_initial_control_points(N_DEG, P_start, P_end)
    return P_init, v0, v1


def run(freeze, max_iter, P_init, v0, v1):
    t0 = time.time()
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=N_SEG,
        r_e=KOZ_RADIUS,
        max_iter=max_iter,
        tol=TOL,
        v0=v0,
        v1=v1,
        sample_count=100,
        objective_mode="dv",
        scp_prox_weight=1e-6,
        scp_trust_radius=2000.0,
        transfer_time=TRANSFER_TIME,
        freeze_gravity_jacobian=freeze,
        verbose=False,
        use_cache=False,
        ignore_existing_cache=True,
    )
    return P_opt, info, time.time() - t0


def main():
    P_init, v0, v1 = make_p_init()

    print(
        f"Basin check: n_seg={N_SEG}, tol={TOL}, scp_prox=1e-6, trust=2000 km"
    )
    print(
        f"{'max_iter':>10} {'mode':>10} {'dv (m/s)':>12} "
        f"{'min_r (km)':>12} {'final_delta':>14} {'wall (s)':>10} "
        f"{'||P_f-P_u||':>14} {'||P_curr-P_prev||':>20}"
    )
    print("-" * 110)

    prev = {False: None, True: None}  # P_opt from previous max_iter
    for mi in MAX_ITER_GRID:
        # Run unfrozen first, then frozen (so we can compute frozen-vs-unfrozen P_opt distance)
        P_u, info_u, wall_u = run(False, mi, P_init, v0, v1)
        P_f, info_f, wall_f = run(True, mi, P_init, v0, v1)
        diff_fu = float(np.linalg.norm(P_f - P_u))

        for label, P_curr, info, wall in (
            ("unfrozen", P_u, info_u, wall_u),
            ("frozen", P_f, info_f, wall_f),
        ):
            freeze = label == "frozen"
            if prev[freeze] is None:
                d_self = float("nan")
            else:
                d_self = float(np.linalg.norm(P_curr - prev[freeze]))

            d_fu = diff_fu if label == "frozen" else float("nan")  # only show once
            d_fu_str = f"{d_fu:>14.4f}" if d_fu == d_fu else " " * 14

            print(
                f"{mi:>10d} {label:>10s} {info['dv_proxy_m_s']:>12.3f} "
                f"{info['min_radius']:>12.3f} {info.get('final_delta_norm', 0.0):>14.3e} "
                f"{wall:>10.2f} {d_fu_str} {d_self:>20.4f}"
            )

            prev[freeze] = P_curr.copy()

        # Always print the cross-mode distance on the frozen row above; for
        # readability, also print the unfrozen-row diff explicitly:
        print(
            f"{'':>10} {'(frozen-unfrozen P_opt L2 distance, km)':>50}: {diff_fu:.4f}"
        )

    print()
    print("Interpretation:")
    print(
        "  - If unfrozen dv plateaus and frozen dv keeps decreasing toward it as "
        "max_iter grows, the gap is a convergence-rate story (same basin)."
    )
    print(
        "  - If frozen dv plateaus at a higher value with final_delta -> 0, "
        "they live in different basins."
    )
    print(
        "  - ||P_curr - P_prev|| measures how much each mode is still moving "
        "between max_iter levels — should shrink to ~0 when truly converged."
    )


if __name__ == "__main__":
    main()
