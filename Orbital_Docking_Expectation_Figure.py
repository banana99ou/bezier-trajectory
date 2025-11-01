"""
Orbital Docking Scenario Expectation Figure

This module creates a visualization that sets expectations for the orbital docking
optimization code. It shows the initial setup without any trajectory curves,
displaying the spatial relationships between chaser satellite, target satellite,
Earth, and Keep-Out Zone (KOZ).

The figure uses a 1x2 layout:
- Left plot: Overall view showing all elements
- Right plot: Zoomed-in view perpendicular to the plane formed by chaser, target, and Earth center
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ─────────────────────────────────────────────────────────────────────────────
# Orbital parameters for realistic scaling
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS = 6.371  # Earth radius in scaled units (1000 km)
ISS_ALTITUDE = 0.408  # ISS altitude in scaled units (408 km)
ORBITAL_RADIUS = EARTH_RADIUS + ISS_ALTITUDE
SCALE_FACTOR = 1e6  # 1 unit = 1000 km (for realistic distances)

# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities
# ─────────────────────────────────────────────────────────────────────────────
def add_wire_sphere(ax, radius=3.0, center=(0.0, 0.0, 0.0), color='gray', alpha=0.25, resolution=40):
    """
    Add a wireframe sphere to a 3D plot.
    
    Args:
        ax: 3D matplotlib axes
        radius: Sphere radius
        center: Sphere center coordinates
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

def add_earth_sphere(ax, radius=EARTH_RADIUS, center=(0.0, 0.0, 0.0), color='blue', alpha=0.3):
    """
    Add Earth as a wireframe sphere for orbital context.
    
    Args:
        ax: 3D matplotlib axes
        radius: Earth radius in scaled units
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
        center: Center point for the view
        radius: Radius around center to include
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

def create_orbital_docking_scenario():
    """
    Create the orbital docking scenario setup.
    
    Returns:
        tuple: (chaser_pos, target_pos, r_e) - positions and safety radius
    """
    # Realistic orbital docking positions (scaled units: 1 unit = 1000 km)
    chaser_pos = np.array([40.0, -70.0, 0.0])   # Chaser spacecraft
    target_pos = np.array([-50.0, 70.0, 30.0])   # Target spacecraft
    
    # Safety zone radius (50,000 km - realistic for orbital docking)
    r_e = 50.0
    
    return chaser_pos, target_pos, r_e

def create_expectation_figure():
    """
    Create the expectation figure with 1x2 layout showing orbital docking setup.
    """
    # Get scenario data
    chaser_pos, target_pos, r_e = create_orbital_docking_scenario()
    
    # Create figure with 1x2 layout
    fig = plt.figure(figsize=(16, 8))
    
    # Left plot: Overall 3D view
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Add Earth (large blue sphere)
    add_earth_sphere(ax1, radius=EARTH_RADIUS, color='blue', alpha=0.3)
    
    # Add Keep-Out Zone (KOZ) - smaller red sphere
    add_wire_sphere(ax1, radius=r_e, color='red', alpha=0.2, resolution=15)
    
    # Plot chaser and target satellites with circle markers
    ax1.scatter(chaser_pos[0], chaser_pos[1], chaser_pos[2], 
               color='green', s=300, label='Chaser Satellite', marker='o')
    ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
               color='orange', s=300, label='Target Satellite', marker='o')
    
    # Add light grey dotted line between chaser and target
    ax1.plot([chaser_pos[0], target_pos[0]], 
             [chaser_pos[1], target_pos[1]], 
             [chaser_pos[2], target_pos[2]], 
             ':', color='lightgrey', alpha=0.8, linewidth=2)
    
    # Add coordinate labels for satellites
    ax1.text(chaser_pos[0], chaser_pos[1], chaser_pos[2] + 15, 
             f'Chaser\n({chaser_pos[0]:.0f}, {chaser_pos[1]:.0f}, {chaser_pos[2]:.0f})', 
             fontsize=10, ha='center', va='bottom', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax1.text(target_pos[0], target_pos[1], target_pos[2] + 15, 
             f'Target\n({target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f})', 
             fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Set up overall view
    set_axes_equal_around(ax1, center=(0,0,0), radius=100, pad=0.1)
    set_isometric(ax1, elev=20, azim=45)
    beautify_3d_axes(ax1, show_ticks=True, show_grid=True)
    # ax1.set_title('Overall View: Orbital Docking Scenario', fontsize=14, pad=20)
    ax1.legend(fontsize=10, loc='upper right', markerscale=0.5)
    
    # Right plot: 2D simplified view
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Set up 2D plot with basic axis limits (legacy version)
    ax2.set_xlim(-80, 80)
    ax2.set_ylim(-80, 80)
    ax2.set_aspect('equal')
    ax2.axis('off')  # Remove all axis lines and ticks
    
    # Draw blue solid half circle at the bottom (Earth) - scaled up
    earth_center_x, earth_center_y = 0, -100
    earth_radius = 90
    theta = np.linspace(0, np.pi, 100)
    earth_x = earth_center_x + earth_radius * np.cos(theta)
    earth_y = earth_center_y + earth_radius * np.sin(theta)
    ax2.fill(earth_x, earth_y, color='blue', alpha=0.7)
    
    # Draw red circle around Earth (KOZ) - scaled up
    koz_radius = earth_radius + 30
    koz_circle = plt.Circle((earth_center_x, earth_center_y), koz_radius, 
                           fill=False, color='red', linewidth=3, linestyle='--')
    ax2.add_patch(koz_circle)
    
    # Add KOZ label with altitude
    ax2.text(40, 0, 
             f'KOZ\nAltitude: {r_e*1000:.0f} km', 
             fontsize=12, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Position chaser at top left (moved down and left) and target at top right (moved up and right)
    chaser_2d_x, chaser_2d_y = -70, 10
    target_2d_x, target_2d_y = 60, 20

    # Add light grey dotted line between chaser and target (crossing KOZ)
    ax2.plot([chaser_2d_x, target_2d_x], [chaser_2d_y, target_2d_y], 
             ':', color='lightgrey', alpha=0.8, linewidth=2)

    # Plot satellites with circle markers
    ax2.scatter(chaser_2d_x, chaser_2d_y, s=200, color='green', 
               marker='o', label='Chaser Satellite')
    ax2.scatter(target_2d_x, target_2d_y, s=200, color='orange', 
               marker='o', label='Target Satellite')
    
    # Add coordinate labels for satellites (matching plot 1's coordinates)
    ax2.text(chaser_2d_x, chaser_2d_y + 10, 
             f'Chaser\n({chaser_pos[0]:.0f}, {chaser_pos[1]:.0f}, {chaser_pos[2]:.0f})', 
             fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax2.text(target_2d_x, target_2d_y + 10, 
             f'Target\n({target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f})', 
             fontsize=10, ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    
    # ax2.set_title('Simplified 2D View: Orbital Docking Scenario', fontsize=14, pad=20)
    ax2.legend(fontsize=10, loc='upper right', markerscale=0.5)
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1, wspace=0.2)
    
    return fig

def main():
    """
    Generate the expectation figure for orbital docking scenario.
    """
    print("Creating orbital docking expectation figure...")
    
    # Create the expectation figure
    fig = create_expectation_figure()
    
    # Save the figure
    fig.savefig("orbital_docking_expectation.png", dpi=300, bbox_inches="tight", facecolor='white')
    
    print("Figure saved as: orbital_docking_expectation.png")
    print("This figure shows the initial setup of the orbital docking scenario:")
    print("- Chaser satellite (green triangle)")
    print("- Target satellite (orange square)")  
    print("- Earth (blue sphere)")
    print("- Keep-Out Zone/KOZ (red wireframe sphere)")
    print("- Left plot: Overall view")
    print("- Right plot: Perpendicular view to chaser-target-Earth plane")
    
    # Show the figure
    plt.show()
    
    return fig

if __name__ == "__main__":
    main()
