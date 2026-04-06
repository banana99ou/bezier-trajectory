"""
Orbital Docking Optimizer using Bézier Curves

This script runs the orbital docking optimization using the modular orbital_docking package.
It finds trajectories that minimize the difference between geometric acceleration and 
gravitational acceleration while satisfying Keep Out Zone (KOZ) constraints.

Key Features:
    - Bézier curve trajectory representation with configurable degree (N=2, 3, 4)
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
    create_multi_order_performance_figure,
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
        choices=[2, 3, 4],
        default=[2],
        help="List of Bézier curve degrees N to optimize (any of 2, 3, 4). "
        'Examples: "-N 2", "-N 2 3 4", "-N 2 4". Default: 2.',
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

    TOL = 1e-12

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

    # Generate initial control points for different curve orders
    # N=2: Quadratic (3 control points)
    # N=3: Cubic (4 control points)
    # N=4: 4th degree (5 control points)

    # Generate initial control points for different curve orders
    P_init_N2 = generate_initial_control_points(2, P_start, P_end)
    P_init_N3 = generate_initial_control_points(3, P_start, P_end)
    P_init_N4 = generate_initial_control_points(4, P_start, P_end)

    print("\n" + "=" * 60)
    print("Initial control points for each curve order:")
    print("=" * 60)

    print(f"\nN=2 (Quadratic, {P_init_N2.shape[0]} control points):")
    for i, P in enumerate(P_init_N2):
        print(f"  P{i}: {P}")

    print(f"\nN=3 (Cubic, {P_init_N3.shape[0]} control points):")
    for i, P in enumerate(P_init_N3):
        print(f"  P{i}: {P}")

    print(f"\nN=4 (4th Degree, {P_init_N4.shape[0]} control points):")
    for i, P in enumerate(P_init_N4):
        print(f"  P{i}: {P}")

    # OPTIMIZATION for N=2 (Quadratic curve)
    # Measure calculation time for performance analysis

    if 2 in args.N:
        print("\n⚠️  Starting optimization for N=2 (Quadratic curve)...")
        print("    This may take a while depending on max_iter and convergence\n")

        segment_counts = [2, 4, 8, 16, 32, 64]
        start_time_N2 = time.time()
        results_N2 = optimize_all_segment_counts(
            P_init_N2,
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
            # Quadratic Bézier (N=2) has only one interior control point (P1),
            # so enforcing both v0 and v1 generally overconstrains the problem.
            # Keep endpoint positions only for the baseline run.
            v0=None,
            v1=None,
            n_jobs=args.n_jobs,
        )
        elapsed_time_N2 = time.time() - start_time_N2
        print(f"\n⏱️  Total optimization time for N=2: {elapsed_time_N2:.2f} seconds")
        total_compute_time_N2 = _summarize_segment_times(results_N2, "N=2")


    # OPTIMIZATION for N=3 (Cubic curve)
    # Measure calculation time for performance analysis

    if 3 in args.N:
        print("\n⚠️  Starting optimization for N=3 (Cubic curve)...")
        print("    This may take a while depending on max_iter and convergence\n")

        segment_counts = [2, 4, 8, 16, 32, 64]
        start_time_N3 = time.time()
        results_N3 = optimize_all_segment_counts(
            P_init_N3,
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
            v0=None,
            v1=None,
            n_jobs=args.n_jobs,
        )
        elapsed_time_N3 = time.time() - start_time_N3
        print(f"\n⏱️  Total optimization time for N=3: {elapsed_time_N3:.2f} seconds")
        total_compute_time_N3 = _summarize_segment_times(results_N3, "N=3")


    # OPTIMIZATION for N=4 (4th degree curve)
    # Measure calculation time for performance analysis

    if 4 in args.N:
        print("\n⚠️  Starting optimization for N=4 (4th degree curve)...")
        print("    This may take a while depending on max_iter and convergence\n")

        segment_counts = [2, 4, 8, 16, 32, 64]
        start_time_N4 = time.time()
        results_N4 = optimize_all_segment_counts(
            P_init_N4,
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
            v0=None,
            v1=None,
            n_jobs=args.n_jobs,
        )
        elapsed_time_N4 = time.time() - start_time_N4
        print(f"\n⏱️  Total optimization time for N=4: {elapsed_time_N4:.2f} seconds")
        total_compute_time_N4 = _summarize_segment_times(results_N4, "N=4")

    # VISUALIZATION: N=2 (Quadratic) Results
    if 2 in args.N:
        print("\n📊 Creating visualizations for N=2 (Quadratic curve)...")
        print("=" * 60)

        fig_comparison_N2 = create_trajectory_comparison_figure(
            P_init_N2, KOZ_RADIUS, results_N2, curve_order=2, v0=v0, v1=v1
        )
        fig_comparison_N2.savefig(FIGURE_DIR / "comparison_N2.png", dpi=300)

        fig_performance_N2 = create_performance_figure(results_N2, curve_order=2)
        fig_performance_N2.savefig(FIGURE_DIR / "performance_N2.png", dpi=300)

        print("\nCreating acceleration profiles for N=2...")
        accel_figures_N2 = {}
        _segcounts = [2, 4, 8, 16, 32, 64]
        pos_ylim, vel_ylim, acc_ylim = compute_profile_ylims(results_N2, _segcounts)
        for seg_count in _segcounts:
            fig = create_acceleration_figure(
                results_N2,
                segcount=seg_count,
                pos_ylim=pos_ylim,
                vel_ylim=vel_ylim,
                acc_ylim=acc_ylim,
                curve_order=2,
            )
            accel_figures_N2[seg_count] = fig
            fig.savefig(FIGURE_DIR / f"accel_profiles_N2_seg{seg_count}.png", dpi=300)
            print(f"✓ Created profiles for {seg_count} segments")

        print("\n✅ N=2 (Quadratic) visualizations complete!")

    if 3 in args.N:
        print("\n📊 Creating visualizations for N=3 (Cubic curve)...")
        print("=" * 60)
        fig_comparison_N3 = create_trajectory_comparison_figure(
            P_init_N3, KOZ_RADIUS, results_N3, curve_order=3, v0=v0, v1=v1
        )
        fig_comparison_N3.savefig(FIGURE_DIR / "comparison_N3.png", dpi=300)
        fig_performance_N3 = create_performance_figure(results_N3, curve_order=3)
        fig_performance_N3.savefig(FIGURE_DIR / "performance_N3.png", dpi=300)

        print("\nCreating acceleration profiles for N=3...")
        accel_figures_N3 = {}
        _segcounts = [2, 4, 8, 16, 32, 64]
        pos_ylim, vel_ylim, acc_ylim = compute_profile_ylims(results_N3, _segcounts)
        for seg_count in _segcounts:
            fig = create_acceleration_figure(
                results_N3,
                segcount=seg_count,
                pos_ylim=pos_ylim,
                vel_ylim=vel_ylim,
                acc_ylim=acc_ylim,
                curve_order=3,
            )
            accel_figures_N3[seg_count] = fig
            fig.savefig(FIGURE_DIR / f"accel_profiles_N3_seg{seg_count}.png", dpi=300)
            print(f"✓ Created profiles for {seg_count} segments")

        print("\n✅ N=3 (Cubic) visualizations complete!")

    if 4 in args.N:
        print("\n📊 Creating visualizations for N=4 (4th degree curve)...")
        print("=" * 60)
        fig_comparison_N4 = create_trajectory_comparison_figure(
            P_init_N4, KOZ_RADIUS, results_N4, curve_order=4, v0=v0, v1=v1
        )
        fig_comparison_N4.savefig(FIGURE_DIR / "comparison_N4.png", dpi=300)
        fig_performance_N4 = create_performance_figure(results_N4, curve_order=4)
        fig_performance_N4.savefig(FIGURE_DIR / "performance_N4.png", dpi=300)

        print("\nCreating acceleration profiles for N=4...")
        accel_figures_N4 = {}
        _segcounts = [2, 4, 8, 16, 32, 64]
        pos_ylim, vel_ylim, acc_ylim = compute_profile_ylims(results_N4, _segcounts)
        for seg_count in _segcounts:
            fig = create_acceleration_figure(
                results_N4,
                segcount=seg_count,
                pos_ylim=pos_ylim,
                vel_ylim=vel_ylim,
                acc_ylim=acc_ylim,
                curve_order=4,
            )
            accel_figures_N4[seg_count] = fig
            fig.savefig(FIGURE_DIR / f"accel_profiles_N4_seg{seg_count}.png", dpi=300)
            print(f"✓ Created profiles for {seg_count} segments")

        print("\n✅ N=4 (4th Degree) visualizations complete!")

    # CALCULATION TIME VS CURVE ORDER ANALYSIS
    print("\n" + "=" * 60)
    print("📈 Creating calculation time vs curve order figure...")
    print("=" * 60)

    calculation_times = {}
    if 2 in args.N:
        calculation_times[2] = (
            float(total_compute_time_N2) if "total_compute_time_N2" in globals() else elapsed_time_N2
        )
    if 3 in args.N:
        calculation_times[3] = (
            float(total_compute_time_N3) if "total_compute_time_N3" in globals() else elapsed_time_N3
        )
    if 4 in args.N:
        calculation_times[4] = (
            float(total_compute_time_N4) if "total_compute_time_N4" in globals() else elapsed_time_N4
        )

    optimization_results = {}
    for N in [2, 3, 4]:
        if N not in args.N:
            continue
        if N == 2:
            results = results_N2
        elif N == 3:
            results = results_N3
        elif N == 4:
            results = results_N4
        else:
            continue

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

    # Combined performance figure across all requested curve orders.
    if len(args.N) > 1:
        results_by_order = {}
        if 2 in args.N:
            results_by_order[2] = results_N2
        if 3 in args.N:
            results_by_order[3] = results_N3
        if 4 in args.N:
            results_by_order[4] = results_N4
        if results_by_order:
            fig_multi_perf = create_multi_order_performance_figure(results_by_order)
            fig_multi_perf.savefig(FIGURE_DIR / "performance_multi_order.png", dpi=300)

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print("\n✅ Calculation time vs curve order figure complete!")
    print(f"\nSummary:")
    if 2 in args.N:
        tN2 = float(total_compute_time_N2) if "total_compute_time_N2" in globals() else elapsed_time_N2
        print(f"  N=2 (Quadratic): {tN2:.2f} seconds")
    if 3 in args.N:
        tN3 = float(total_compute_time_N3) if "total_compute_time_N3" in globals() else elapsed_time_N3
        print(f"  N=3 (Cubic):      {tN3:.2f} seconds")
    if 4 in args.N:
        tN4 = float(total_compute_time_N4) if "total_compute_time_N4" in globals() else elapsed_time_N4
        print(f"  N=4 (4th Degree): {tN4:.2f} seconds")


if __name__ == "__main__":
    main()
