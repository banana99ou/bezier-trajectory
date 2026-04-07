"""
Orbital Docking Optimizer using Bézier Curves

This script runs the orbital docking optimization using the modular orbital_docking package.
It finds trajectories that minimize the difference between geometric acceleration and 
gravitational acceleration while satisfying Keep Out Zone (KOZ) constraints.

Key Features:
    - Bézier curve trajectory representation with configurable degree (current paper focus: N=5, 6, 7)
    - Iterative optimization with KOZ constraint linearization using segment subdivision
    - Support for velocity and acceleration boundary conditions
    - Caching system for optimization results to speed up repeated runs (enabled by default)
    - Comprehensive visualization tools for trajectory analysis
    - Performance analysis across different segment counts and curve orders
    - Command-line interface for cache control

Usage:
    Run the script with default settings (cache enabled):
        python Orbital_Docking_Optimizer.py
    
    Ignore existing cache and recompute (new results are still cached):
        python Orbital_Docking_Optimizer.py --no-cache
    
    Show help:
        python Orbital_Docking_Optimizer.py --help

Scenario (Progress-to-ISS inspired, simplified single-arc):
    - Chaser (Progress-like) at 245 km circularized parking orbit
    - Target (ISS-like) at 400 km circular orbit
    - Both at 51.64 deg inclination, same plane
    - Keep Out Zone (KOZ) at 100 km altitude
    - 30 deg phase lag

Optimization:
    The cost function minimizes a delta-v surrogate (control acceleration minus gravity)
    while satisfying KOZ constraints enforced through iterative linearization
    using De Casteljau subdivision.

Visualization:
    - 3D trajectory plots with Earth and KOZ spheres
    - Position, velocity, and acceleration profiles
    - Performance comparison across segment counts
    - Calculation time analysis for different curve orders
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import warnings
from pathlib import Path

# Import from the orbital_docking package
from orbital_docking import (
    constants,
    optimize_all_segment_counts,
    generate_initial_control_points,
    create_trajectory_comparison_figure,
    create_performance_figure,
    create_acceleration_figure,
    create_time_vs_order_figure,
    compute_profile_ylims
)

warnings.filterwarnings('ignore')

# --- helpers ---
def _summarize_segment_times(results, label: str):
    """
    Print per-n_seg compute times (from cached metadata) and return total time.
    The optimizer stores compute time in each result's info['elapsed_time'].
    """
    if results is None:
        return 0.0

    rows = []
    total = 0.0
    for n_seg, _P_opt, info in results:
        t = None
        if isinstance(info, dict):
            t = info.get("elapsed_time", None)
        try:
            t = float(t) if t is not None else float("nan")
        except Exception:
            t = float("nan")
        rows.append((int(n_seg), t))
        if np.isfinite(t) and t >= 0.0:
            total += t

    print(f"\n⏱️  Per-segment-count compute time for {label} (from cache metadata):")
    for n_seg, t in sorted(rows, key=lambda x: x[0]):
        t_str = f"{t:.2f}s" if np.isfinite(t) else "n/a"
        print(f"  - n_seg={n_seg:>2d}: {t_str}")
    print(f"  = Total (sum over n_seg): {total:.2f}s")
    return total


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Orbital Docking Optimizer using Bézier Curves",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing cache and force fresh optimization (new results are still cached)",
    )
    parser.add_argument(
        "-N",
        type=int,
        nargs="+",
        choices=[5, 6, 7],
        default=[5, 6, 7],
        help="List of Bézier curve degrees N to optimize (any of 5, 6, 7). "
        'Examples: "-N 5", "-N 5 6 7", "-N 5 7". Default: 5 6 7.',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print debug output",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively (blocks until windows are closed). Default: save figures only.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for optimization",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Outer SCP convergence tolerance on control-point update norm. Default: 1e-6.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for parallelizing over segment counts (n_seg). "
        "Use 1 for serial; use 0 or negative for auto.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="dv",
        choices=["dv", "energy"],
        help="Objective to optimize. "
        "'dv' uses an IRLS L1-style delta-v proxy; 'energy' uses the legacy L2 control-energy surrogate. "
        "Default: dv.",
    )
    parser.add_argument(
        "--scp-prox",
        type=float,
        default=1e-6,
        help="Outer SCP proximal weight (>=0). Adds (lambda/2)||P-P_prev||^2 per SCP iteration. "
        "Use small positive values (e.g., 1e-6 to 1e-3) for stabilization.",
    )
    parser.add_argument(
        "--scp-trust-radius",
        type=float,
        default=2000.0,
        help="Outer SCP trust radius in control-point vector 2-norm (km). "
        "If >0, clips each SCP step to this radius. Default: 2000.",
    )
    args = parser.parse_args()

    USE_CACHE = True
    IGNORE_EXISTING_CACHE = args.no_cache

    # Keep consistent with the baseline run behavior
    MAX_ITER = args.max_iter

    # Extract constants for convenience
    EARTH_RADIUS_KM = constants.EARTH_RADIUS_KM
    KOZ_RADIUS = constants.KOZ_RADIUS

    TOL = args.tol

    # Figure directory
    FIGURE_DIR = Path(__file__).parent / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Display cache configuration
    print("=" * 60)
    if IGNORE_EXISTING_CACHE:
        print("Cache: REFRESH MODE (read disabled, write enabled)")
    else:
        print("Cache: ENABLED (read + write)")
    print("=" * 60)
    print()

    # Define endpoints for all curve orders
    # Positions in km, scaled appropriately

    # Progress->ISS fast-rendezvous-inspired geometry (simplified):
    # - Use a circularized Progress-like start orbit based on the published
    #   245 km insertion-orbit apogee for Progress MS missions.
    # - Use an ISS-like 400 km circular target orbit in the same plane.
    # - Use a 30 deg phase lag as a reduced-order, single-arc proxy.
    PROGRESS_START_ALTITUDE_KM = 245.0
    ISS_TARGET_ALTITUDE_KM = 400.0
    INCLINATION_DEG = 51.64
    RAAN_DEG = 0.0
    ISS_U_DEG = 45.0
    PROGRESS_LAG_DEG = 30.0

    def _rotz(theta_rad: float) -> np.ndarray:
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _rotx(theta_rad: float) -> np.ndarray:
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def _eci_from_circular_elements(radius_km: float, inc_deg: float, raan_deg: float, u_deg: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (r_eci, v_eci) for circular orbit with given radius and argument of latitude.
        Velocity direction follows prograde motion in the same orbital plane.
        """
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
    progress_radius_km = EARTH_RADIUS_KM + PROGRESS_START_ALTITUDE_KM
    iss_radius_km = EARTH_RADIUS_KM + ISS_TARGET_ALTITUDE_KM

    P_start, v0 = _eci_from_circular_elements(
        progress_radius_km,
        INCLINATION_DEG,
        RAAN_DEG,
        progress_u_deg,
    )
    P_end, v1 = _eci_from_circular_elements(
        iss_radius_km,
        INCLINATION_DEG,
        RAAN_DEG,
        iss_u_deg,
    )

    print(f"Endpoints (km):")
    print(f"  Start (Progress): {P_start}")
    print(f"  End (ISS):      {P_end}")
    print(f"  i={INCLINATION_DEG:.2f} deg, RAAN={RAAN_DEG:.2f} deg")
    print(f"  Progress lag behind ISS: {PROGRESS_LAG_DEG:.2f} deg")
    print(f"\nKOZ radius: {KOZ_RADIUS:.1f} km")
    print(f"Transfer time T: {constants.TRANSFER_TIME_S:.1f} s")

    print(f"\nBoundary velocities (km/s):")
    print(f"  v0 (initial): {v0}")
    print(f"  v1 (final):   {v1}")
    print(f"  |v0| = {np.linalg.norm(v0):.3f} km/s (Progress circular speed)")
    print(f"  |v1| = {np.linalg.norm(v1):.3f} km/s (ISS circular speed)")

    def _order_label(N: int) -> str:
        suffix = "th"
        if N % 100 not in (11, 12, 13):
            if N % 10 == 1:
                suffix = "st"
            elif N % 10 == 2:
                suffix = "nd"
            elif N % 10 == 3:
                suffix = "rd"
        return f"{N}{suffix} Degree"

    # Generate initial control points for the requested curve orders
    p_init_by_order = {
        N: generate_initial_control_points(N, P_start, P_end)
        for N in args.N
    }

    print("\n" + "=" * 60)
    print("Initial control points for each curve order:")
    print("=" * 60)

    for N in sorted(args.N):
        P_init = p_init_by_order[N]
        print(f"\nN={N} ({_order_label(N)}, {P_init.shape[0]} control points):")
        for i, P in enumerate(P_init):
            print(f"  P{i}: {P}")

    segment_counts = [2, 4, 8, 16, 32, 64]
    results_by_order = {}
    total_time_by_order = {}

    for N in sorted(args.N):
        print(f"\n⚠️  Starting optimization for N={N} ({_order_label(N)})...")
        print("    This may take a while depending on max_iter and convergence\n")

        start_time = time.time()
        results = optimize_all_segment_counts(
            p_init_by_order[N],
            r_e=KOZ_RADIUS,
            segment_counts=segment_counts,
            max_iter=MAX_ITER,
            tol=TOL,
            verbose=args.verbose,
            debug=args.debug,
            use_cache=USE_CACHE,
            ignore_existing_cache=IGNORE_EXISTING_CACHE,
            objective=args.objective,
            scp_prox_weight=args.scp_prox,
            scp_trust_radius=args.scp_trust_radius,
            enforce_prograde=True,
            v0=v0,
            v1=v1,
            n_jobs=args.n_jobs,
        )
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total optimization time for N={N}: {elapsed_time:.2f} seconds")
        total_time_by_order[N] = _summarize_segment_times(results, f"N={N}")
        results_by_order[N] = results

    for N in sorted(args.N):
        print(f"\n📊 Creating visualizations for N={N} ({_order_label(N)})...")
        print("=" * 60)

        results = results_by_order[N]
        fig_comparison = create_trajectory_comparison_figure(
            p_init_by_order[N], KOZ_RADIUS, results, curve_order=N, v0=v0, v1=v1
        )
        fig_comparison.savefig(FIGURE_DIR / f"comparison_N{N}.png", dpi=300)

        fig_performance = create_performance_figure(results, curve_order=N)
        fig_performance.savefig(FIGURE_DIR / f"performance_N{N}.png", dpi=300)

        print(f"\nCreating acceleration profiles for N={N}...")
        pos_ylim, vel_ylim, acc_ylim = compute_profile_ylims(results, segment_counts)
        for seg_count in segment_counts:
            fig = create_acceleration_figure(
                results,
                segcount=seg_count,
                pos_ylim=pos_ylim,
                vel_ylim=vel_ylim,
                acc_ylim=acc_ylim,
                curve_order=N,
            )
            fig.savefig(FIGURE_DIR / f"accel_profiles_N{N}_seg{seg_count}.png", dpi=300)
            print(f"✓ Created profiles for {seg_count} segments")

        print(f"\n✅ N={N} ({_order_label(N)}) visualizations complete!")

    # CALCULATION TIME VS CURVE ORDER ANALYSIS
    print("\n" + "=" * 60)
    print("📈 Creating calculation time vs curve order figure...")
    print("=" * 60)

    calculation_times = {N: float(total_time_by_order[N]) for N in sorted(results_by_order)}

    optimization_results = {}
    for N, results in sorted(results_by_order.items()):
        P_opt, info = None, None
        for seg_count, P_opt_iter, info_iter in results:
            if seg_count == 64:
                P_opt, info = P_opt_iter, info_iter
                break
        if P_opt is None and len(results) > 0:
            P_opt, info = results[-1][1], results[-1][2]
        optimization_results[N] = (P_opt, info)

    fig_time_order = create_time_vs_order_figure(calculation_times, optimization_results)
    fig_time_order.savefig(FIGURE_DIR / "time_vs_order.png", dpi=300)
    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("\n✅ Calculation time vs curve order figure complete!")
    print(f"\nSummary:")
    for N in sorted(results_by_order):
        print(f"  N={N} ({_order_label(N)}): {float(total_time_by_order[N]):.2f} seconds")


if __name__ == "__main__":
    main()
