# Three figures: Straight path (fig1), Straight path with segmentation (fig2), Bézier curve (fig3)
import numpy as np
import matplotlib.pyplot as plt

def bezier_quadratic(P0, P1, P2, t):
    t = np.asarray(t)
    return ((1 - t) ** 2)[:, None] * P0 + (2 * (1 - t) * t)[:, None] * P1 + (t ** 2)[:, None] * P2

def segment_points(P0, P1, P2, M):
    ts = np.linspace(0, 1, M + 1)
    return ts, bezier_quadratic(P0, P1, P2, ts)

def is_point_inside_sphere(point, center, radius):
    """Check if a point is inside the sphere"""
    distance = np.linalg.norm(point - center)
    return distance <= radius

def create_segmentation_lines(points, center, radius, line_length=0.3):
    """Create perpendicular segmentation lines with colors based on sphere intersection"""
    lines = []
    colors = []
    
    for point in points:
        # Calculate direction vector from sphere center to point
        direction = point - center
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            # Normalize direction
            direction_unit = direction / direction_norm
            
            # Create perpendicular line (simplified - using cross product with z-axis)
            perp_vector = np.cross(direction_unit, np.array([0, 0, 1]))
            if np.linalg.norm(perp_vector) < 0.1:  # If parallel to z-axis, use x-axis
                perp_vector = np.cross(direction_unit, np.array([1, 0, 0]))
            perp_vector = perp_vector / np.linalg.norm(perp_vector)
            
            # Create line endpoints
            line_start = point - perp_vector * line_length / 2
            line_end = point + perp_vector * line_length / 2
            
            lines.append([line_start, line_end])
            
            # Determine color based on whether point is inside sphere
            if is_point_inside_sphere(point, center, radius):
                colors.append('red')
            else:
                colors.append('green')
    
    return lines, colors

def plot_sphere(ax, center, radius):
    """Plot the sphere wireframe"""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 26)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.3, rstride=2, cstride=2, color="white", alpha=0.95)

def setup_plot():
    """Setup common plot parameters"""
    fig = plt.figure(figsize=(6, 6), facecolor='#1a1a1a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a1a')
    ax.set_axis_off()
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=0, azim=90)
    zoom_factor = 0.5
    rng = 10.0 * zoom_factor
    ax.set_xlim(-rng, rng)
    ax.set_ylim(-rng, rng)
    ax.set_zlim(-rng, rng)
    return fig, ax

# Sphere and path settings
sphere_center = np.array([-3.0, 3.0, -3.0])
sphere_radius = 4.0
P0 = np.array([-1.0, 7.0, -4.0])  # Start point
P1 = np.array([3.0, 5.0, -1.0])   # Control point
P2 = np.array([0.0, 1.0, 0.0])    # End point

# Create straight line path (same start and end as Bézier curve)
straight_path = np.array([P0, P2])
M = 7  # Number of segments

# Create all figures in a single window with subplots for synchronized zoom
print("Creating all three figures with synchronized zoom...")

# Create a single figure with 3 subplots
fig = plt.figure(figsize=(18, 6), facecolor='#1a1a1a')

# FIGURE 1: Straight path (no control polygon, no segmentation)
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_facecolor('#1a1a1a')
ax1.set_axis_off()
ax1.grid(False)
ax1.set_box_aspect([1, 1, 1])
ax1.view_init(elev=0, azim=90)
zoom_factor = 0.5
rng = 10.0 * zoom_factor
ax1.set_xlim(-rng, rng)
ax1.set_ylim(-rng, rng)
ax1.set_zlim(-rng, rng)

plot_sphere(ax1, sphere_center, sphere_radius)
ax1.plot(straight_path[:,0], straight_path[:,1], straight_path[:,2], linewidth=2.4, color="#f0f0f0")
ax1.set_title("Figure 1: Straight Path", color='white', fontsize=12)

# FIGURE 2: Straight path with segmentation lines
ax2 = fig.add_subplot(132, projection='3d')
ax2.set_facecolor('#1a1a1a')
ax2.set_axis_off()
ax2.grid(False)
ax2.set_box_aspect([1, 1, 1])
ax2.view_init(elev=0, azim=90)
ax2.set_xlim(-rng, rng)
ax2.set_ylim(-rng, rng)
ax2.set_zlim(-rng, rng)

plot_sphere(ax2, sphere_center, sphere_radius)

# Create straight line with segmentation points
t_seg_straight = np.linspace(0, 1, M + 1)
seg_pts_straight = np.array([P0 + t * (P2 - P0) for t in t_seg_straight])

# Plot straight path
ax2.plot(straight_path[:,0], straight_path[:,1], straight_path[:,2], linewidth=2.4, color="#f0f0f0", linestyle='--')

# Create and plot segmentation dots with arrows pointing away from sphere center
seg_lines, seg_colors = create_segmentation_lines(seg_pts_straight, sphere_center, sphere_radius)
arrow_length = 0.8  # Fixed arrow length

for i, (point, color) in enumerate(zip(seg_pts_straight, seg_colors)):
    # Plot the dot
    ax2.scatter(point[0], point[1], point[2], s=40, color=color, depthshade=False)
    
    # Calculate direction vector from sphere center to point
    direction = point - sphere_center
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm > 0:
        # Normalize direction and scale to arrow length
        direction_unit = direction / direction_norm
        arrow_end = point + direction_unit * arrow_length
        
        # Plot arrow with arrowhead using quiver
        ax2.quiver(point[0], point[1], point[2], 
                   direction_unit[0], direction_unit[1], direction_unit[2],
                   length=arrow_length, color=color, linewidth=2, arrow_length_ratio=0.3)

ax2.set_title("Figure 2: Straight Path with Segmentation", color='white', fontsize=12)

# FIGURE 3: Bézier curve with control polygon and dots (original)
ax3 = fig.add_subplot(133, projection='3d')
ax3.set_facecolor('#1a1a1a')
ax3.set_axis_off()
ax3.grid(False)
ax3.set_box_aspect([1, 1, 1])
ax3.view_init(elev=0, azim=90)
ax3.set_xlim(-rng, rng)
ax3.set_ylim(-rng, rng)
ax3.set_zlim(-rng, rng)

plot_sphere(ax3, sphere_center, sphere_radius)

# Bézier curve
t_dense = np.linspace(0, 1, 900)
curve = bezier_quadratic(P0, P1, P2, t_dense)
ax3.plot(curve[:,0], curve[:,1], curve[:,2], linewidth=2.4, color="#f0f0f0")

# Segmentation points
t_seg, seg_pts = segment_points(P0, P1, P2, M)
ax3.scatter(seg_pts[:,0], seg_pts[:,1], seg_pts[:,2], s=20, color="#e0e0e0", depthshade=False)

# Control polygon
ctrl = np.vstack([P0, P1, P2])
ax3.plot(ctrl[:,0], ctrl[:,1], ctrl[:,2], linestyle='--', linewidth=1.2, color="#d0d0d0", marker='o', markersize=4)

ax3.set_title("Figure 3: Bézier Curve", color='white', fontsize=12)

# Synchronize zoom across all subplots
def sync_zoom(event):
    if event.inaxes in [ax1, ax2, ax3]:
        # Get the limits from the axis that was zoomed
        source_ax = event.inaxes
        xlim = source_ax.get_xlim()
        ylim = source_ax.get_ylim()
        zlim = source_ax.get_zlim()
        
        # Apply the same limits to all other axes
        for ax in [ax1, ax2, ax3]:
            if ax != source_ax:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
        fig.canvas.draw()

# Connect the zoom event to all subplots
fig.canvas.mpl_connect('button_release_event', sync_zoom)

plt.tight_layout(pad=0)
plt.show()