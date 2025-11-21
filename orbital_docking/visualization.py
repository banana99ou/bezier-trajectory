"""
Visualization functions for orbital docking trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .bezier import BezierCurve
from .de_casteljau import segment_matrices_equal_params
from .constants import EARTH_RADIUS_KM, EARTH_MU_SCALED
from .utils import format_number


def add_wire_sphere(ax, radius=3.0, center=(0.0, 0.0, 0.0), color='gray', alpha=0.25, resolution=40):
    """
    Add a wireframe sphere to a 3D plot.

    Args:
        ax: 3D matplotlib axes
        radius: Sphere radius (in km)
        center: Sphere center coordinates (in km)
        color: Sphere color
        alpha: Transparency (0-1)
        resolution: Number of grid points for sphere generation
    """
    cx, cy, cz = center
    # Generate spherical coordinates
    u = np.linspace(0, 2*np.pi, resolution)  # Azimuthal angle
    v = np.linspace(0, np.pi, resolution)    # Polar angle
    uu, vv = np.meshgrid(u, v)

    # Convert to Cartesian coordinates
    x = cx + radius * np.cos(uu) * np.sin(vv)
    y = cy + radius * np.sin(uu) * np.sin(vv)
    z = cz + radius * np.cos(vv)

    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)


def add_earth_sphere(ax, radius=EARTH_RADIUS_KM, center=(0.0, 0.0, 0.0), color='blue', alpha=0.3):
    """
    Add Earth as a wireframe sphere for orbital context.

    Args:
        ax: 3D matplotlib axes
        radius: Earth radius in km
        center: Earth center coordinates
        color: Earth color
        alpha: Transparency
    """
    add_wire_sphere(ax, radius=radius, center=center, color=color, alpha=alpha, resolution=20)


def set_axes_equal_around(ax, center=(0,0,0), radius=1.0, pad=0.05):
    """
    Set 3D axes to equal aspect ratio around a specified center and radius.

    Args:
        ax: 3D matplotlib axes
        center: Center point for the view (in km)
        radius: Radius around center to include (in km)
        pad: Additional padding factor
    """
    cx, cy, cz = center

    # Get current axis limits
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()

    # Expand limits to include the specified sphere
    x0 = min(x0, cx - radius); x1 = max(x1, cx + radius)
    y0 = min(y0, cy - radius); y1 = max(y1, cy + radius)
    z0 = min(z0, cz - radius); z1 = max(z1, cz + radius)

    # Set equal aspect ratio
    max_range = max(x1 - x0, y1 - y0, z1 - z0)
    half = 0.5 * max_range * (1 + pad)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect((1, 1, 1))  # Equal aspect ratio


def set_isometric(ax, elev=35.264, azim=45.0, ortho=True):
    """
    Set isometric view for 3D plot.

    Args:
        ax: 3D matplotlib axes
        elev: Elevation angle in degrees
        azim: Azimuth angle in degrees
        ortho: Whether to use orthogonal projection
    """
    ax.view_init(elev=elev, azim=azim)
    try:
        if ortho:
            ax.set_proj_type('ortho')  # Requires matplotlib ≥3.2
    except Exception:
        pass  # Fallback for older matplotlib versions


def beautify_3d_axes(ax, show_ticks=True, show_grid=True):
    """
    Apply paper-friendly styling to 3D axes.

    Args:
        ax: 3D matplotlib axes
        show_ticks: Whether to show axis ticks and labels
        show_grid: Whether to show grid lines
    """
    ax.grid(show_grid)
    if show_ticks:
        ax.tick_params(axis='both', which='major', labelsize=8, pad=2)
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Style the axis panes (background planes)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1, 1, 1, 1))  # White background
            axis.pane.set_edgecolor("0.85")        # Light gray edges
        except Exception:
            pass  # Fallback for older matplotlib versions


def plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', show_seg_ctrl_if=8, lw=2.0):
    """
    Plot Bézier curve with color-coded segments and control polygon.

    Args:
        ax: 3D matplotlib axes
        P_opt: Optimized control points
        n_seg: Number of segments
        cmap_name: Colormap name for segment colors
        show_seg_ctrl_if: Show segment control polygons if n_seg <= this value
        lw: Line width for the curve
    """
    curve = BezierCurve(P_opt)
    N = curve.degree
    A_list = segment_matrices_equal_params(N, n_seg)

    # Plot whole-curve control polygon in black
    ax.plot(P_opt[:,0], P_opt[:,1], P_opt[:,2], 'k.-', lw=1.2, ms=4)

    # Generate colors for segments using colormap
    base_colors = ['#E74C3C', '#3498DB', '#F39C12']  # (red, blue, orange)
    colors = [base_colors[i % 3] for i in range(len(A_list))]

    # Draw each segment with its color
    ts = np.linspace(0, 1, 180)  # Parameter values for smooth curves
    lines, cols = [], []
    for i, Ai in enumerate(A_list):
        Qi = Ai @ P_opt  # Control points for segment i
        seg = BezierCurve(Qi)
        Pseg = np.array([seg.point(t) for t in ts])
        lines.append(np.column_stack((Pseg[:,0], Pseg[:,1], Pseg[:,2])))
        cols.append(colors[i])

        # Show segment control polygons only for small numbers of segments
        if n_seg <= show_seg_ctrl_if:
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=5)
        if n_seg > show_seg_ctrl_if:
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=3)

    # Create 3D line collection for efficient rendering
    lc = Line3DCollection(lines, colors=cols, linewidths=lw, alpha=0.95)
    ax.add_collection3d(lc)


def create_trajectory_comparison_figure(P_init, r_e, results):
    """
    Create 2×3 layout showing trajectories with different segment counts.
    Uses same aesthetics as Orbital_Docking_Optimizer.py.
    All 6 panels share the same zoom level - zooming one zooms all.

    Args:
        P_init: Initial control points
        r_e: KOZ radius
        results: List of (n_seg, P_opt, info) tuples

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)

    # Create 2×3 subplot layout
    axes = []
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        axes.append(ax)

    # Plot trajectories for different segment counts
    segment_counts = [2, 4, 8, 16, 32, 64]

    # Calculate shared view radius (more zoomed in)
    # Find the maximum extent across all trajectories
    all_points = [P_init]
    for _, P_opt, _ in results:
        all_points.append(P_opt)
    all_points_array = np.vstack(all_points)
    max_extent = np.linalg.norm(all_points_array, axis=1).max()
    # More zoomed in: use smaller radius multiplier
    view_radius = max(max_extent * 0.6, EARTH_RADIUS_KM * 0.8)

    for i, (n_seg, P_opt, info) in enumerate(results):
        ax = axes[i]

        # Add Earth (smaller blue sphere, more transparent)
        add_earth_sphere(ax, radius=EARTH_RADIUS_KM * 0.7, color='blue', alpha=1)

        # Add safety zone (KOZ - smaller red sphere)
        add_wire_sphere(ax, radius=r_e, color='red', alpha=0.2, resolution=15)

        # Plot optimized trajectory (thicker lines with color-coded segments)
        plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', lw=3.0)

        # Add start and end markers (larger markers)
        ax.scatter(P_init[0,0], P_init[0,1], P_init[0,2], 
                  color='green', s=120, label='Chaser', zorder=10)
        ax.scatter(P_init[-1,0], P_init[-1,1], P_init[-1,2], 
                  color='orange', s=120, label='Target', zorder=10)
        ax.legend(fontsize=8)

        # Professional styling with shared zoom level
        set_axes_equal_around(ax, center=(0,0,0), radius=view_radius, pad=0.05)
        set_isometric(ax, elev=90, azim=0)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)

        # Title with acceleration and feasibility
        accel_ms2 = info.get('accel', 0.0) / 1e3
        accel_str = format_number(accel_ms2, '.1f')
        feasible = bool(info.get('feasible', False))
        status = 'Feasible' if feasible else 'Infeasible'
        title_color = 'black' if feasible else 'red'
        ax.set_title(f'{n_seg} Segments — {status}\nAccel: {accel_str} m/s²',
                     fontsize=10, pad=10, color=title_color)

    # Link all axes to share zoom - when one zooms, all zoom together
    # Use a flag to prevent infinite recursion
    _syncing = False
    
    def sync_limits(ax_source):
        """Sync limits from source axis to all other axes."""
        nonlocal _syncing
        if _syncing:
            return
        
        _syncing = True
        try:
            xlim = ax_source.get_xlim3d()
            ylim = ax_source.get_ylim3d()
            zlim = ax_source.get_zlim3d()
            
            # Update all other axes
            for ax_target in axes:
                if ax_target is not ax_source:
                    # Temporarily disconnect callbacks to avoid recursion
                    ax_target.set_xlim3d(xlim, emit=False)
                    ax_target.set_ylim3d(ylim, emit=False)
                    ax_target.set_zlim3d(zlim, emit=False)
        finally:
            _syncing = False
    
    # Connect the callback to each axis for all three dimensions
    for ax in axes:
        # Use lambda with default argument to capture ax correctly
        ax.callbacks.connect('xlim_changed', lambda event, ax=ax: sync_limits(ax))
        ax.callbacks.connect('ylim_changed', lambda event, ax=ax: sync_limits(ax))
        ax.callbacks.connect('zlim_changed', lambda event, ax=ax: sync_limits(ax))

    return fig


def compute_profile_ylims(results, segcounts):
    """
    Compute shared y-limits for position, velocity, and acceleration panels
    across the specified segment counts.
    Returns: (pos_ylim, vel_ylim, acc_ylim)
    """
    pos_min, pos_max = np.inf, -np.inf
    vel_min, vel_max = np.inf, -np.inf
    acc_max = 0.0  # acceleration lower bound enforced as 0
    ts = np.linspace(0.0, 1.0, 300)

    seg_set = set(segcounts)
    for seg_count, P_opt, info in results:
        if seg_count not in seg_set:
            continue
        if P_opt is None:
            continue

        curve = BezierCurve(P_opt)
        positions = np.array([curve.point(t) for t in ts])
        velocities = np.array([curve.velocity(t) for t in ts]) * 1e-3
        geom_accel_vec = np.array([curve.acceleration(t) for t in ts]) * 1e-6
        # Gravitational acceleration vectors (consistent with cost function: negative toward origin)
        grav_accel_vec = []
        for t in ts:
            pos = curve.point(t)
            r = np.linalg.norm(pos)
            if r > 1e-6:
                a_grav = -EARTH_MU_SCALED / r**2 * (pos / r) * 1e3
            else:
                a_grav = np.zeros_like(pos)
            grav_accel_vec.append(a_grav)
        grav_accel_vec = np.array(grav_accel_vec)
        # Total acceleration magnitude used by cost (difference of vectors)
        diff_accel_mag_kms2 = np.linalg.norm(geom_accel_vec - grav_accel_vec, axis=1)

        # Update mins/maxes
        pos_min = min(pos_min, positions.min())
        pos_max = max(pos_max, positions.max())
        vel_min = min(vel_min, velocities.min())
        vel_max = max(vel_max, velocities.max())
        acc_max = max(acc_max, diff_accel_mag_kms2.max())

    # Add small padding
    def pad_limits(lo, hi, pad_ratio=0.05):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return None
        if hi == lo:
            delta = 1.0 if hi == 0 else abs(hi) * pad_ratio
            return (lo - delta, hi + delta)
        delta = (hi - lo) * pad_ratio
        return (lo - delta, hi + delta)

    pos_ylim = pad_limits(pos_min, pos_max*1.3)
    vel_ylim = pad_limits(vel_min, vel_max*1.3)
    acc_ylim = (0.0, acc_max * 1.8 if acc_max > 0 else 1.0)
    return pos_ylim, vel_ylim, acc_ylim


def create_performance_figure(results):
    """
    Create performance figure showing acceleration vs segment count.

    Args:
        results: List of (n_seg, P_opt, info) tuples

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)

    # Extract data
    segment_counts = [n_seg for n_seg, _, _ in results]
    accelerations = [info['accel'] * 1e-3 for _, _, info in results]  # Convert km/s² to m/s²

    # Create acceleration performance graph
    ax.plot(segment_counts, accelerations, 'bo-', linewidth=3, markersize=10)
    ax.set_xlabel('Number of Segments', fontsize=14)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=14)
    ax.set_title('Performance Improvement with More Segments', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)  # Log scale for segment counts

    # Add data point labels
    for i, (seg, accel) in enumerate(zip(segment_counts, accelerations)):
        accel_str = format_number(accel, '.3f')
        ax.annotate(accel_str, (seg, accel), textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    return fig


def create_acceleration_figure(results, segcount=64, pos_ylim=None, vel_ylim=None, acc_ylim=None):
    """
    Create 3x1 layout showing xyz position, xyz velocity, and acceleration profiles.
    Adapted for notebook's BezierCurve class.

    Args:
        results: List of (n_seg, P_opt, info) tuples
        segcount: Segment count to plot (default: 64)

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f'Position, Velocity, and Acceleration Profiles for {segcount} Segments', fontsize=16)

    # Create 3x1 subplot layout
    ax1 = fig.add_subplot(3, 1, 1)  # Position plot
    ax2 = fig.add_subplot(3, 1, 2)  # Velocity plot
    ax3 = fig.add_subplot(3, 1, 3)  # Acceleration plot

    # Find the result for the specified segment count
    P_opt, info = None, None
    for seg_count, P_opt_iter, info_iter in results:
        if seg_count == segcount:
            P_opt, info = P_opt_iter, info_iter
            break

    # Fallback to last result if specified segment count not found
    if P_opt is None and len(results) > 0:
        P_opt, info = results[-1][1], results[-1][2]

    if P_opt is None:
        return fig

    # Create Bezier curve
    curve = BezierCurve(P_opt)
    ts = np.linspace(0.0, 1.0, 300)

    # Sample positions, velocities, and accelerations
    positions = np.array([curve.point(t) for t in ts])
    velocities = np.array([curve.velocity(t) for t in ts]) * 1e-3

    # Calculate magnitudes
    position_magnitudes = np.linalg.norm(positions, axis=1)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)

    # Calculate acceleration components (consistent with cost function)
    # Geometric acceleration vectors from curve
    geom_accelerations_vec = np.array([curve.acceleration(t) for t in ts]) * 1e-3
    geom_accel_mag = np.linalg.norm(geom_accelerations_vec, axis=1) 
    # Gravitational acceleration vectors (negative toward origin)
    grav_accelerations_vec = []
    for t in ts:
        pos = curve.point(t)
        r = np.linalg.norm(pos)
        if r > 1e-6:
            a_grav = -EARTH_MU_SCALED / r**2 * (pos / r) * 1e3
        else:
            a_grav = np.zeros_like(pos)
        grav_accelerations_vec.append(a_grav)
    grav_accelerations_vec = np.array(grav_accelerations_vec)
    grav_accel_mag = np.linalg.norm(grav_accelerations_vec, axis=1)
    # Total acceleration magnitude used by the cost: ||a_geom - a_grav||
    total_accel_mag_kms2 = np.linalg.norm(geom_accelerations_vec - grav_accelerations_vec, axis=1)

    # Plot 1: XYZ Position with total magnitude
    ax1.plot(ts, positions[:, 0], 'r-', linewidth=2.0, label='X', alpha=0.7)
    ax1.plot(ts, positions[:, 1], 'g-', linewidth=2.0, label='Y', alpha=0.7)
    ax1.plot(ts, positions[:, 2], 'b-', linewidth=2.0, label='Z', alpha=0.7)
    ax1.plot(ts, position_magnitudes, color='black', linewidth=2.5, label='||Position|| (Total)', linestyle='--')
    ax1.set_xlabel('Parameter τ', fontsize=12)
    ax1.set_ylabel('Position (km)', fontsize=12)
    ax1.set_title('Position Components (XYZ) and Total Magnitude', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    if pos_ylim is not None:
        ax1.set_ylim(pos_ylim)

    # Plot 2: XYZ Velocity with total magnitude
    ax2.plot(ts, velocities[:, 0], 'r-', linewidth=2.0, label='Vx', alpha=0.7)
    ax2.plot(ts, velocities[:, 1], 'g-', linewidth=2.0, label='Vy', alpha=0.7)
    ax2.plot(ts, velocities[:, 2], 'b-', linewidth=2.0, label='Vz', alpha=0.7)
    ax2.plot(ts, velocity_magnitudes, color='black', linewidth=2.5, label='||Velocity|| (Total)', linestyle='--')
    ax2.set_xlabel('Parameter τ', fontsize=12)
    ax2.set_ylabel('Velocity (km/s)', fontsize=12)
    ax2.set_title('Velocity Components (XYZ) and Total Magnitude', fontsize=14, pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    if vel_ylim is not None:
        ax2.set_ylim(vel_ylim)

    # Plot 3: Acceleration components and total
    ax3.plot(ts, geom_accel_mag, 'orange', linewidth=2.0, label='Geometric Acceleration', alpha=0.7)
    ax3.plot(ts, grav_accel_mag, 'cyan', linewidth=2.0, label='Gravitational Acceleration', alpha=0.7)
    ax3.plot(ts, total_accel_mag_kms2, 'purple', linewidth=2.5, label='Total Acceleration', linestyle='-')
    ax3.set_xlabel('Parameter τ', fontsize=12)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=12)
    accel_total_str = format_number(info["accel"]/1e3, '.1f')
    ax3.set_title(f'Acceleration Components (Accumulated: {accel_total_str} m/s²)', 
                 fontsize=14, pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_ylim(acc_ylim)

    # Add acceleration statistics
    max_accel = np.max(total_accel_mag_kms2)
    avg_accel = np.mean(total_accel_mag_kms2)
    max_str = format_number(max_accel, '.1f')
    avg_str = format_number(avg_accel, '.1f')
    ax3.text(0.02, 0.98, f'Max: {max_str} m/s²\nAvg: {avg_str} m/s²', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    return fig


def create_time_vs_order_figure(calculation_times, optimization_results):
    """
    Create figure showing calculation time vs curve order.

    Args:
        calculation_times: Dict mapping curve order N to time in seconds
        optimization_results: Dict mapping curve order N to (P_opt, info) tuple

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    ax = fig.add_subplot(111)

    # Extract data
    orders = sorted(calculation_times.keys())
    times = []
    for N in orders:
        t = calculation_times.get(N, 0.0)
        if (t is None or t == 0.0) and N in optimization_results:
            _, info = optimization_results[N]
            t = float(info.get('elapsed_time', 0.0)) if info is not None else 0.0
        times.append(t)

    # Create bar plot
    bars = ax.bar(orders, times, color=['#3498DB', '#E74C3C', '#F39C12'], alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (order, time_val) in enumerate(zip(orders, times)):
        time_str = format_number(time_val, '.2f')
        ax.text(order, time_val, f'{time_str}s', 
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Curve Order (N)', fontsize=14)
    ax.set_ylabel('Calculation Time (seconds)', fontsize=14)
    ax.set_title('Optimization Time vs Bézier Curve Order', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(orders)
    ax.set_xticklabels([f'Quadratic (N={N})' if N == 2 else 
                        f'Cubic (N={N})' if N == 3 else 
                        f'4th Degree (N={N})' for N in orders])

    # Add acceleration info as text
    accel_info = []
    for N in orders:
        _, info = optimization_results[N]
        accel_ms2 = info['accel'] / 1e3  # Convert to m/s²
        accel_str = format_number(accel_ms2, '.1f')
        accel_info.append(f'N={N}: {accel_str} m/s²')

    info_text = '\n'.join(accel_info)

    return fig

