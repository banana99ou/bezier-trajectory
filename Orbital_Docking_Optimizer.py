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
    
    Disable caching:
        python Orbital_Docking_Optimizer.py --no-cache
    
    Show help:
        python Orbital_Docking_Optimizer.py --help

Scenario:
    - Chaser satellite at 300 km altitude
    - Target satellite (ISS) at 423 km altitude
    - Keep Out Zone (KOZ) at 100 km altitude
    - 90-degree angular separation between chaser and target

Optimization:
    The cost function minimizes the L² norm of the difference between geometric
    acceleration (from Bézier curve) and gravitational acceleration (from two-body dynamics).
    KOZ constraints are enforced through iterative linearization using De Casteljau subdivision.

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

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Orbital Docking Optimizer using Bézier Curves',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '--no-cache',
    action='store_true',
    help='Disable caching of optimization results (cache is enabled by default)'
)
args = parser.parse_args()

USE_CACHE = not args.no_cache

# Extract constants for convenience
ISS_ALTITUDE_KM = constants.ISS_ALTITUDE_KM
CHASER_ALTITUDE_KM = constants.CHASER_ALTITUDE_KM
KOZ_ALTITUDE_KM = constants.KOZ_ALTITUDE_KM
EARTH_RADIUS_KM = constants.EARTH_RADIUS_KM
KOZ_RADIUS = constants.KOZ_RADIUS
ISS_RADIUS = constants.ISS_RADIUS
CHASER_RADIUS = constants.CHASER_RADIUS

# Figure directory
FIGURE_DIR = Path(__file__).parent / "figure" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Display cache configuration
print("=" * 60)
print(f"Cache: {'ENABLED' if USE_CACHE else 'DISABLED'}")
print("=" * 60)
print()

# Define endpoints for all curve orders
# Positions in km, scaled appropriately

# Start: chaser position (at 300km altitude)
theta_start = -np.pi / 4  # 45 degrees
P_start = np.array([
    CHASER_RADIUS * np.cos(theta_start),
    CHASER_RADIUS * np.sin(theta_start),
    0.0
])

# End: ISS position (at 423km altitude)  
theta_end = np.pi / 4  # -45 degrees
P_end = np.array([
    ISS_RADIUS * np.cos(theta_end),
    ISS_RADIUS * np.sin(theta_end),
    0.0
])

print(f"Endpoints (km):")
print(f"  Start (chaser): {P_start}")
print(f"  End (ISS):      {P_end}")
print(f"\nKOZ radius: {KOZ_RADIUS:.1f} km")

# Generate initial control points for different curve orders
# N=2: Quadratic (3 control points)
# N=3: Cubic (4 control points)
# N=4: 4th degree (5 control points)

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

print("\n⚠️  Starting optimization for N=2 (Quadratic curve)...")
print("    This may take a while depending on max_iter and convergence\n")

# Run optimization for all segment counts with N=2
segment_counts = [2, 4, 8, 16, 32, 64]

# Start timing
start_time_N2 = time.time()

# Run optimizations for all segment counts
results_N2 = optimize_all_segment_counts(
    P_init_N2, 
    r_e=KOZ_RADIUS,
    segment_counts=segment_counts,
    max_iter=120,
    tol=1e-3,
    verbose=True,
    use_cache=USE_CACHE
)

# Calculate total time
elapsed_time_N2 = time.time() - start_time_N2

print(f"\n⏱️  Total optimization time for N=2: {elapsed_time_N2:.2f} seconds")


# OPTIMIZATION for N=3 (Cubic curve)
# Measure calculation time for performance analysis

print("\n⚠️  Starting optimization for N=3 (Cubic curve)...")
print("    This may take a while depending on max_iter and convergence\n")

# Run optimization for all segment counts with N=3
segment_counts = [2, 4, 8, 16, 32, 64]

# Start timing
start_time_N3 = time.time()

# Run optimizations for all segment counts
results_N3 = optimize_all_segment_counts(
    P_init_N3, 
    r_e=KOZ_RADIUS,
    segment_counts=segment_counts,
    max_iter=120,
    tol=1e-3,
    verbose=True,
    use_cache=USE_CACHE
)

# Calculate total time
elapsed_time_N3 = time.time() - start_time_N3

print(f"\n⏱️  Total optimization time for N=3: {elapsed_time_N3:.2f} seconds")


# OPTIMIZATION for N=4 (4th degree curve)
# Measure calculation time for performance analysis

print("\n⚠️  Starting optimization for N=4 (4th degree curve)...")
print("    This may take a while depending on max_iter and convergence\n")

# Run optimization for all segment counts with N=4
segment_counts = [2, 4, 8, 16, 32, 64]

# Start timing
start_time_N4 = time.time()

# Run optimizations for all segment counts
results_N4 = optimize_all_segment_counts(
    P_init_N4, 
    r_e=KOZ_RADIUS,
    segment_counts=segment_counts,
    max_iter=120,
    tol=1e-3,
    verbose=True,
    use_cache=USE_CACHE
)

# Calculate total time
elapsed_time_N4 = time.time() - start_time_N4

print(f"\n⏱️  Total optimization time for N=4: {elapsed_time_N4:.2f} seconds")



# VISUALIZATION: N=2 (Quadratic) Results
print("\n📊 Creating visualizations for N=2 (Quadratic curve)...")
print("=" * 60)

# Create and display the 2×3 comparison figure for N=2
fig_comparison_N2 = create_trajectory_comparison_figure(P_init_N2, KOZ_RADIUS, results_N2)
fig_comparison_N2.savefig(FIGURE_DIR / "comparison_N2.png", dpi=300)

# Create and display the performance figure for N=2
fig_performance_N2 = create_performance_figure(results_N2)
fig_performance_N2.savefig(FIGURE_DIR / "performance_N2.png", dpi=300)

# Create acceleration figures for each segment count (N=2)
print("\nCreating acceleration profiles for N=2...")
accel_figures_N2 = {}
# Compute shared y-limits across all segment counts for N=2
_segcounts = [2, 4, 8, 16, 32, 64]
pos_ylim, vel_ylim, acc_ylim = compute_profile_ylims(results_N2, _segcounts)
for seg_count in [2, 4, 8, 16, 32, 64]:
    fig = create_acceleration_figure(results_N2, segcount=seg_count, pos_ylim=pos_ylim, vel_ylim=vel_ylim, acc_ylim=acc_ylim)
    accel_figures_N2[seg_count] = fig
    fig.savefig(FIGURE_DIR / f"accel_profiles_N2_seg{seg_count}.png", dpi=300)
    print(f"✓ Created profiles for {seg_count} segments")


print("\n✅ N=2 (Quadratic) visualizations complete!")

# Create and save path comparison figures for N=3 and N=4
fig_comparison_N3 = create_trajectory_comparison_figure(P_init_N3, KOZ_RADIUS, results_N3)
fig_comparison_N3.savefig(FIGURE_DIR / "comparison_N3.png", dpi=300)

fig_comparison_N4 = create_trajectory_comparison_figure(P_init_N4, KOZ_RADIUS, results_N4)
fig_comparison_N4.savefig(FIGURE_DIR / "comparison_N4.png", dpi=300)

# Create and save performance figures for N=3 and N=4
fig_performance_N3 = create_performance_figure(results_N3)
fig_performance_N3.savefig(FIGURE_DIR / "performance_N3.png", dpi=300)

fig_performance_N4 = create_performance_figure(results_N4)
fig_performance_N4.savefig(FIGURE_DIR / "performance_N4.png", dpi=300)

# CALCULATION TIME VS CURVE ORDER ANALYSIS
print("\n" + "=" * 60)
print("📈 Creating calculation time vs curve order figure...")
print("=" * 60)

# Prepare data for time vs order figure
calculation_times = {
    2: elapsed_time_N2,
    3: elapsed_time_N3,
    4: elapsed_time_N4
}

# For optimization results, we'll use the best result (typically 64 segments)
# Extract the best result for each curve order
optimization_results = {}
for N in [2, 3, 4]:
    if N == 2:
        results = results_N2
    elif N == 3:
        results = results_N3
    else:  # N == 4
        results = results_N4

    # Find the result with 64 segments (or use the last one)
    P_opt, info = None, None
    for seg_count, P_opt_iter, info_iter in results:
        if seg_count == 64:
            P_opt, info = P_opt_iter, info_iter
            break

    # Fallback to last result
    if P_opt is None and len(results) > 0:
        P_opt, info = results[-1][1], results[-1][2]

    optimization_results[N] = (P_opt, info)

# Create and display the time vs order figure
fig_time_order = create_time_vs_order_figure(calculation_times, optimization_results)
fig_time_order.savefig(FIGURE_DIR / "time_vs_order.png", dpi=300)
plt.show()

print("\n✅ Calculation time vs curve order figure complete!")
print(f"\nSummary:")
print(f"  N=2 (Quadratic): {elapsed_time_N2:.2f} seconds")
print(f"  N=3 (Cubic):      {elapsed_time_N3:.2f} seconds")
print(f"  N=4 (4th Degree): {elapsed_time_N4:.2f} seconds")
