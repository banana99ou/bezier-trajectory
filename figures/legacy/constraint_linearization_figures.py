"""
Figure generator used to illustrate how nonlinear constraints are converted 
to linear supporting halfspace constraints.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Circle
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint, Bounds

def bezier_quadratic(P0, P1, P2, t):
    t = np.asarray(t)
    return ((1 - t) ** 2)[:, None] * P0 + (2 * (1 - t) * t)[:, None] * P1 + (t ** 2)[:, None] * P2

def segment_points(P0, P1, P2, M):
    ts = np.linspace(0, 1, M + 1)
    return ts, bezier_quadratic(P0, P1, P2, ts)

def de_casteljau_split(P0, P1, P2, t):
    """Split quadratic Bézier curve at parameter t using de Casteljau's algorithm"""
    P01 = (1 - t) * P0 + t * P1
    P12 = (1 - t) * P1 + t * P2
    P012 = (1 - t) * P01 + t * P12
    return np.array([P0, P01, P012]), np.array([P012, P12, P2])

def get_segment_control_points(P0, P1, P2, M):
    """Get control points for each segment of the Bézier curve"""
    segments = []
    t_seg = np.linspace(0, 1, M + 1)
    
    for i in range(M):
        t_start = t_seg[i]
        t_end = t_seg[i + 1]
        
        if i == 0:
            left, right = de_casteljau_split(P0, P1, P2, t_end)
            seg_ctrl = left
        elif i == M - 1:
            left, right = de_casteljau_split(P0, P1, P2, t_start)
            seg_ctrl = right
        else:
            left1, right1 = de_casteljau_split(P0, P1, P2, t_start)
            t_scaled = (t_end - t_start) / (1 - t_start)
            left2, right2 = de_casteljau_split(right1[0], right1[1], right1[2], t_scaled)
            seg_ctrl = left2
        
        segments.append(seg_ctrl)
    
    return segments, t_seg

def plot_constraint_plane(ax, point_on_plane, normal, plane_size=3.0, color='orange', alpha=0.3):
    """Plot a constraint plane (half-space)"""
    # Create a grid for the plane
    # Find two perpendicular vectors in the plane
    # Use cross product to find a vector perpendicular to normal
    if abs(normal[2]) < 0.9:
        v1 = np.cross(normal, np.array([0, 0, 1]))
    else:
        v1 = np.cross(normal, np.array([1, 0, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Create grid
    u = np.linspace(-plane_size, plane_size, 20)
    v = np.linspace(-plane_size, plane_size, 20)
    U, V = np.meshgrid(u, v)
    
    # Generate plane points
    X = point_on_plane[0] + U * v1[0] + V * v2[0]
    Y = point_on_plane[1] + U * v1[1] + V * v2[1]
    Z = point_on_plane[2] + U * v1[2] + V * v2[2]
    
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, antialiased=True)

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

def plot_sphere_outline(ax, center, radius, view_normal):
    """Plot a single circle outline of the sphere aligned with the current view normal."""
    n = view_normal / (np.linalg.norm(view_normal) + 1e-12)
    # Find orthonormal basis (u, v) spanning the view plane
    if abs(n[2]) < 0.9:
        u = np.cross(n, np.array([0, 0, 1.0]))
    else:
        u = np.cross(n, np.array([1.0, 0, 0]))
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    t = np.linspace(0, 2 * np.pi, 400)
    pts = center[None, :] + radius * (np.cos(t)[:, None] * u[None, :] + np.sin(t)[:, None] * v[None, :])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="white", linewidth=0.8, alpha=0.9)

def plot_sphere_wireframe(ax, center, radius):
    """Plot a sphere wireframe (used only for Plot 3)."""
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 26)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, linewidth=0.6, rstride=2, cstride=2, color="white", alpha=0.65)

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
sphere_center = np.array([0.0, 0.0, 0.0])
sphere_radius = 4.0
P0 = np.array([-4.0, 4.0, 0.0])  # Start point
P1 = np.array([0.0, 5.0, 0.0])   # Control point
P2 = np.array([4.0, 2.0, 0.0])    # End point

# Calculate normal vector of the control point plane
def calculate_plane_normal(P0, P1, P2):
    """Calculate the normal vector of the plane defined by three points"""
    v1 = P1 - P0
    v2 = P2 - P0
    normal = np.cross(v1, v2)
    # Normalize the normal vector
    norm_mag = np.linalg.norm(normal)
    if norm_mag > 0:
        normal = normal / norm_mag
    return normal

def normal_to_view_angles(normal):
    """Convert a normal vector to elevation and azimuth angles for matplotlib view_init"""
    # To view perpendicular to the plane, we want to look along the normal vector
    # Elevation: angle from horizontal plane (-90 to 90)
    # Azimuth: angle in horizontal plane (0 to 360)
    nx, ny, nz = normal
    
    # Calculate horizontal distance in xy-plane
    xy_dist = np.sqrt(nx**2 + ny**2)
    
    # Calculate elevation: angle from horizontal (positive = looking down)
    # Use atan2 to handle all quadrants correctly
    elev = np.degrees(np.arctan2(nz, xy_dist))
    
    # Calculate azimuth: angle in horizontal plane from x-axis
    azim = np.degrees(np.arctan2(ny, nx))
    
    return elev, azim

# Calculate view angles perpendicular to control point plane
plane_normal = calculate_plane_normal(P0, P1, P2)
view_elev, view_azim = normal_to_view_angles(plane_normal)

# Create straight line path (same start and end as Bézier curve)
straight_path = np.array([P0, P2])
M = 7  # Number of segments

# Plot 2 zoom/visibility controls
plot2_zoom_pad = 0.99  # Increase to zoom out, decrease to zoom in

def plot_3d_figures():
    # Create all figures in a single window with subplots for synchronized zoom
    print("Creating all three figures with synchronized zoom...")
    
    # Calculate center point for focusing the view
    all_points = np.vstack([P0, P1, P2, sphere_center])
    scene_center = np.mean(all_points, axis=0)
    
    # Create a single figure with 3 subplots
    fig = plt.figure(figsize=(18, 6), facecolor='#1a1a1a')
    
    # Prepare segmentation points for plot 1 and 2
    t_seg_straight = np.linspace(0, 1, M + 1)
    seg_pts_straight = np.array([P0 + t * (P2 - P0) for t in t_seg_straight])
    seg_lines, seg_colors = create_segmentation_lines(seg_pts_straight, sphere_center, sphere_radius)
    arrow_length = 0.8  # Fixed arrow length
    
    # Get Bézier curve points for range calculation
    t_dense_temp = np.linspace(0, 1, 900)
    curve_points = bezier_quadratic(P0, P1, P2, t_dense_temp)
    segment_ctrls_temp, _ = get_segment_control_points(P0, P1, P2, M)
    seg_centers_temp = np.array([np.mean(seg_ctrl, axis=0) for seg_ctrl in segment_ctrls_temp])
    
    # Calculate appropriate range to focus on the scene
    all_points_extended = np.vstack([all_points, seg_pts_straight, curve_points, seg_centers_temp])
    max_range = np.max(np.abs(all_points_extended - scene_center))
    rng = max_range * 1.2  # Add 20% padding
    
    # FIGURE 1: Straight path with segmentation; highlight middle segment with single constraint plane (copied style)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor('#1a1a1a')
    ax1.set_axis_off()
    ax1.grid(False)
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=view_elev, azim=view_azim)
    ax1.set_xlim(scene_center[0] - rng, scene_center[0] + rng)
    ax1.set_ylim(scene_center[1] - rng, scene_center[1] + rng)
    ax1.set_zlim(scene_center[2] - rng, scene_center[2] + rng)
    
    plot_sphere_outline(ax1, sphere_center, sphere_radius, plane_normal)
    ax1.plot(straight_path[:,0], straight_path[:,1], straight_path[:,2], linewidth=2.0, color="#b0b0b0", linestyle='--')
    for point, color in zip(seg_pts_straight, seg_colors):
        ax1.scatter(point[0], point[1], point[2], s=30, color=color, depthshade=False, alpha=0.9)
    
    # Highlight the middle segment
    mid_idx = M // 2
    pA = seg_pts_straight[mid_idx]
    pB = seg_pts_straight[mid_idx + 1]
    ax1.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], linewidth=2.0, color="#ffd400")
    
    # Single constraint plane for the highlighted segment
    seg_center_ax1 = 0.5 * (pA + pB)
    direction_ax1 = seg_center_ax1 - sphere_center
    dn_ax1 = np.linalg.norm(direction_ax1)
    if dn_ax1 > 0:
        dunit_ax1 = direction_ax1 / dn_ax1
        plane_point_ax1 = sphere_center + dunit_ax1 * sphere_radius
        # Bright, noticeable plane color
        plot_constraint_plane(ax1, plane_point_ax1, dunit_ax1, plane_size=2.2, color='#ffd400', alpha=0.28)
        # Guide line from KOZ center to plane point
        ax1.plot([sphere_center[0], plane_point_ax1[0]],
                 [sphere_center[1], plane_point_ax1[1]],
                 [sphere_center[2], plane_point_ax1[2]],
                 color='#ffd400', linewidth=1.8, linestyle=':')
    
    # FIGURE 2: Middle Bézier segment with control polygon and single constraint plane
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_facecolor('#1a1a1a')
    ax2.set_axis_off()
    ax2.grid(False)
    ax2.set_box_aspect([1, 1, 1])
    ax2.view_init(elev=view_elev, azim=view_azim)
    ax2.set_xlim(scene_center[0] - rng, scene_center[0] + rng)
    ax2.set_ylim(scene_center[1] - rng, scene_center[1] + rng)
    ax2.set_zlim(scene_center[2] - rng, scene_center[2] + rng)
    
    plot_sphere_outline(ax2, sphere_center, sphere_radius, plane_normal)
    
    ## Compute middle Bézier segment range
    segment_ctrls, t_seg = get_segment_control_points(P0, P1, P2, M)
    mid_idx = M // 2
    t_start = t_seg[mid_idx]
    t_end = t_seg[mid_idx + 1]
    
    # Plot entire curve faintly for context
    t_dense2 = np.linspace(0, 1, 600)
    curve_full = bezier_quadratic(P0, P1, P2, t_dense2)
    ax2.plot(curve_full[:,0], curve_full[:,1], curve_full[:,2], linewidth=1.2, color="#888888", linestyle='--', alpha=0.25)
    
    # Sample and draw only the middle Bézier segment
    t_local = np.linspace(t_start, t_end, 200)
    curve_mid = bezier_quadratic(P0, P1, P2, t_local)
    ax2.plot(curve_mid[:,0], curve_mid[:,1], curve_mid[:,2], linewidth=2.0, color="#ffd400")
    
    # Control polygon for the middle segment (high-contrast)
    seg_ctrl = segment_ctrls[mid_idx]
    ax2.plot(
        seg_ctrl[:,0], seg_ctrl[:,1], seg_ctrl[:,2],
        linestyle='--', linewidth=1.8, color="#00e5ff", marker='o', markersize=5,
        zorder=6
    )
    poly = Poly3DCollection([seg_ctrl], facecolors="#00e5ff", edgecolors="none", alpha=0.15, zorder=5)
    ax2.add_collection3d(poly)
    
    # Single constraint plane using segment control polygon centroid
    seg_center = np.mean(seg_ctrl, axis=0)
    direction = seg_center - sphere_center
    dn = np.linalg.norm(direction)
    if dn > 0:
        dunit = direction / dn
        plane_point = sphere_center + dunit * sphere_radius
        plot_constraint_plane(ax2, plane_point, dunit, plane_size=2.2, color='#ffd400', alpha=0.18)
        ax2.plot([sphere_center[0], plane_point[0]],
                 [sphere_center[1], plane_point[1]],
                 [sphere_center[2], plane_point[2]],
                 color='#ffd400', linewidth=1.8, linestyle=':')
    
    # Zoom axes around the highlighted segment and its control polygon
    focus_points = np.vstack([curve_mid, seg_ctrl])
    focus_center = np.mean(focus_points, axis=0)
    focus_extent = np.max(np.abs(focus_points - focus_center), axis=0)
    pad = plot2_zoom_pad * np.max(focus_extent) + 1e-6
    ax2.set_xlim(focus_center[0] - (focus_extent[0] + pad), focus_center[0] + (focus_extent[0] + pad))
    ax2.set_ylim(focus_center[1] - (focus_extent[1] + pad), focus_center[1] + (focus_extent[1] + pad))
    ax2.set_zlim(focus_center[2] - (focus_extent[2] + pad), focus_center[2] + (focus_extent[2] + pad))
    
    # FIGURE 3: Bézier curve with control polygon and dots (original)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor('#1a1a1a')
    ax3.set_axis_off()
    ax3.grid(False)
    ax3.set_box_aspect([1, 1, 1])
    ax3.view_init(elev=view_elev, azim=view_azim)
    ax3.set_xlim(scene_center[0] - rng, scene_center[0] + rng)
    ax3.set_ylim(scene_center[1] - rng, scene_center[1] + rng)
    ax3.set_zlim(scene_center[2] - rng, scene_center[2] + rng)
    
    plot_sphere_wireframe(ax3, sphere_center, sphere_radius)
    
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

# =========================
# 2D KOZ linearization plot
# =========================

def signed_distance_to_halfspace(point, support_point, normal):
    return float(np.dot(point - support_point, normal))

def adjust_ctrl_polygon_to_halfspace(ctrl_pts, support_point, normal, margin):
    adjusted = ctrl_pts.copy()
    for i in range(3):
        sd = signed_distance_to_halfspace(adjusted[i], support_point, normal)
        if sd < margin:
            boost = 1.0
            if i == 1:
                boost = 1.4  # emphasize curvature by pushing control point further
            adjusted[i] = adjusted[i] + (margin - sd) * boost * normal
    return adjusted

def eval_bezier_from_ctrl(ctrl_pts, t):
    return bezier_quadratic(ctrl_pts[0], ctrl_pts[1], ctrl_pts[2], t)

def segment_violates_circle(ctrl_pts, center, radius):
    ts = np.linspace(0.0, 1.0, 120)
    pts = eval_bezier_from_ctrl(ctrl_pts, ts)
    d = np.linalg.norm(pts - center[None, :], axis=1)
    return np.any(d < radius - 1e-6)

def compute_support(center, radius, ctrl_pts):
    centroid = np.mean(ctrl_pts, axis=0)
    vec = centroid - center
    nrm = np.linalg.norm(vec)
    if nrm < 1e-12:
        n = np.array([1.0, 0.0])
    else:
        n = vec / nrm
    support_point = center + radius * n
    return n, support_point

def plot_support_line(ax, support_point, normal, length, **kwargs):
    t = np.array([-normal[1], normal[0]])  # 2D perpendicular
    p1 = support_point - t * length
    p2 = support_point + t * length
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)

def plot_koz_linearization_2d():
    # Use 2D projection of the existing control points and KOZ
    center = sphere_center[:2]
    radius = sphere_radius
    P0_2d = P0[:2]
    P1_2d = P1[:2]
    P2_2d = P2[:2]
    
    # Fewer segments for clarity
    M_local = 5
    segment_ctrls, t_segs = get_segment_control_points(P0_2d, P1_2d, P2_2d, M_local)
    t_dense = np.linspace(0, 1, 600)
    curve_full = bezier_quadratic(P0_2d, P1_2d, P2_2d, t_dense)
    seg_points = bezier_quadratic(P0_2d, P1_2d, P2_2d, t_segs)
    
    violating_indices = []
    for idx, ctrl in enumerate(segment_ctrls):
        if segment_violates_circle(ctrl, center, radius):
            violating_indices.append(idx)
    
    # Figure setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
    for ax in (ax1, ax2):
        ax.set_aspect('equal', adjustable='box')
        ax.set_facecolor('white')
        ax.grid(False)
        ax.tick_params(colors='#444444')
    
    # Determine plot bounds
    all_pts = np.vstack([curve_full, seg_points, np.vstack(segment_ctrls)])
    min_xy = np.min(np.vstack([all_pts, center[None, :] + radius]), axis=0)
    max_xy = np.max(np.vstack([all_pts, center[None, :] - radius]), axis=0)
    cx, cy = center
    pad = radius * 0.8
    ax_xmin = min(cx - radius - pad, min_xy[0] - pad)
    ax_xmax = max(cx + radius + pad, max_xy[0] + pad)
    ax_ymin = min(cy - radius - pad, min_xy[1] - pad)
    ax_ymax = max(cy + radius + pad, max_xy[1] + pad)
    
    # -----------------------
    # Panel 1: original curve
    # -----------------------
    ax1.set_title("Original curve with KOZ violations", color='#111111')
    # KOZ circle (filled)
    circ1 = Circle((center[0], center[1]), radius, edgecolor='red', facecolor='none', lw=2.0)
    ax1.add_patch(circ1)
    
    # Full curve (segmented)
    # First draw all segments in dark gray, then overlay violating segments
    for i, ctrl in enumerate(segment_ctrls):
        ts = np.linspace(0, 1, 80)
        pts = eval_bezier_from_ctrl(ctrl, ts)
        ax1.plot(pts[:, 0], pts[:, 1], color='#222222', linewidth=1.8)
    
    # Mark segmentation points
    ax1.scatter(seg_points[:, 0], seg_points[:, 1], s=25, color='black', zorder=3)
    
    # Highlight violating segments and draw arrows from center to centroid
    for i in violating_indices:
        ctrl = segment_ctrls[i]
        ts = np.linspace(0, 1, 120)
        pts = eval_bezier_from_ctrl(ctrl, ts)
        # Less noticeable color for segments 3 and 5 (1-based indexing)
        seg_color = '#ff7b00' if (i + 1) in (3, 5) else '#0066ff'
        ax1.plot(pts[:, 0], pts[:, 1], color=seg_color, linewidth=2.6)
        centroid = np.mean(ctrl, axis=0)
        vec = centroid - center
        nrm = np.linalg.norm(vec) + 1e-12
        u = vec / nrm
        arrow_len = radius * 0.7
        ax1.arrow(center[0], center[1], u[0] * arrow_len, u[1] * arrow_len,
                  width=0.0, head_width=radius * 0.12, head_length=radius * 0.18,
                  length_includes_head=True, color=seg_color, alpha=0.9)
    
    # Global control polygon
    ctrl_global = np.vstack([P0_2d, P1_2d, P2_2d])
    ax1.plot(ctrl_global[:, 0], ctrl_global[:, 1], linestyle='--', color='#333333', linewidth=1.6, marker='o', markersize=5)
    
    ax1.set_xlim(ax_xmin, ax_xmax)
    ax1.set_ylim(ax_ymin, ax_ymax)
    
    # -----------------------------------------------
    # Panel 2: supporting half-spaces and corrections
    # -----------------------------------------------
    ax2.set_title("Supporting half-spaces and corrected segments", color='#111111')
    # KOZ circle outline only
    circ2 = Circle((center[0], center[1]), radius, edgecolor='red', facecolor='none', lw=2.0)
    ax2.add_patch(circ2)
    
    # ================
    # New curve (2D) via KOZ-constrained optimization reusing optimizer structure
    # ================
    # Minimal quadratic Bézier helper with acceleration like in optimizer
    class BezierCurveQ2:
        def __init__(self, control_points):
            P = np.array(control_points, dtype=float)
            if P.shape != (3, 2):
                raise ValueError("BezierCurveQ2 expects control_points shape (3, 2)")
            self.P = P
        def point(self, tau):
            P0, P1, P2 = self.P
            one_t = 1.0 - tau
            return (one_t*one_t)[:, None] * P0 + (2*one_t*tau)[:, None] * P1 + (tau*tau)[:, None] * P2 if np.ndim(tau) else \
                   (one_t*one_t) * P0 + (2*one_t*tau) * P1 + (tau*tau) * P2
        def acceleration(self, tau):
            # Constant second derivative for quadratic Bézier
            P0, P1, P2 = self.P
            return 2.0 * (P0 - 2.0 * P1 + P2)
    # Cost copied in spirit from Orbital_Docking_Optimizer.py (gravity disabled)
    def _compute_cost_only(P_flat, Np1, dim, n_samples=60):
        P = P_flat.reshape(Np1, dim)
        curve = BezierCurveQ2(P)
        ts = np.linspace(0, 1, n_samples)
        cost = 0.0
        dtau = 1.0 / (n_samples - 1)
        for tau in ts:
            a_geom = curve.acceleration(tau)
            # No gravity term in this local 2D illustration
            norm_diff = np.linalg.norm(a_geom)
            cost += (norm_diff**2) * dtau
        return cost, cost  # return cost for both to mimic API (cost, accel)
    def cost_function_gradient_hessian(P_flat, Np1, dim, n_samples=60, compute_grad=True):
        cost, _ = _compute_cost_only(P_flat, Np1, dim, n_samples)
        if not compute_grad:
            return cost, None, None
        # Finite-difference gradient
        eps = 1e-6
        grad = np.zeros(Np1 * dim)
        for i in range(Np1 * dim):
            P_pert = P_flat.copy()
            P_pert[i] += eps
            cost_pert, _ = _compute_cost_only(P_pert, Np1, dim, n_samples)
            grad[i] = (cost_pert - cost) / eps
        # Simple diagonal regularization as Hessian approx
        hess = np.eye(Np1 * dim) * 1e-6
        return cost, grad, hess
    # De Casteljau-based segmentation (copied approach)
    def de_casteljau_split_1d(N, tau, basis_index):
        w = np.zeros(N+1)
        w[basis_index] = 1.0
        left = [w[0]]
        right = [w[-1]]
        W = w.copy()
        for _ in range(1, N+1):
            W = (1 - tau) * W[:-1] + tau * W[1:]
            left.append(W[0])
            right.append(W[-1])
        L = np.array(left); R = np.array(right[::-1])
        return L, R
    def de_casteljau_split_matrices(N, tau):
        S_left = np.zeros((N+1, N+1))
        S_right = np.zeros((N+1, N+1))
        for j in range(N+1):
            L, R = de_casteljau_split_1d(N, tau, j)
            S_left[:, j] = L
            S_right[:, j] = R
        return S_left, S_right
    def segment_matrices_equal_params(N, n_seg):
        if n_seg < 1:
            raise ValueError("n_seg must be >= 1")
        if n_seg == 1:
            return [np.eye(N+1)]
        mats = []
        remainder = np.eye(N+1)
        for k in range(n_seg, 1, -1):
            tau = 1.0 / k
            S_L, S_R = de_casteljau_split_matrices(N, tau)
            mats.append(S_L @ remainder)
            remainder = S_R @ remainder
        mats.append(remainder)
        return mats
    # KOZ linear constraints builder in 2D (supporting half-space, center-anchored)
    def build_koz_constraints_2d(A_list, P, r_e, c_center, safety=0.0):
        Np1 = A_list[0].shape[1]
        rows, lbs = [], []
        dim = 2
        for Ai in A_list:
            Qi = Ai @ P  # (3,2)
            ci = Qi.mean(axis=0)
            Nj = ci - c_center
            Nj_norm = np.linalg.norm(Nj)
            if Nj_norm < 1e-12:
                continue
            nj = Nj / Nj_norm
            # Center-anchored tangent half-space: n^T x >= n^T c_center + r_e (+ safety)
            rhs = float(np.dot(nj, c_center)) + float(r_e) + float(safety)
            for k in range(Np1):
                row = np.zeros(Np1 * dim)
                for j in range(Np1):
                    coeff = Ai[k, j]
                    start = j * dim
                    row[start:start+dim] += coeff * nj
                rows.append(row)
                lbs.append(rhs)
        if len(rows) == 0:
            A_const = np.zeros((1, Np1 * 2))
            lb_const = np.array([-np.inf])
            ub_const = np.array([np.inf])
        else:
            A_const = np.vstack(rows)
            lb_const = np.array(lbs)
            ub_const = np.full_like(lb_const, np.inf)
        return LinearConstraint(A_const, lb_const, ub_const)
    # Optimizer loop (trust-constr) similar to Orbital_Docking_Optimizer.optimize_orbital_docking
    def optimize_quadratic_2d(P_init, n_seg=5, r_e=1.0, c_center=np.zeros(2), max_outer=20, tol=1e-8, safety=0.0):
        P = P_init.copy()
        Np1, dim = P.shape  # (3,2)
        N = Np1 - 1         # 2
        A_list = segment_matrices_equal_params(N, n_seg)
        # bounds: fix endpoints
        x0 = P.reshape(-1)
        lb = np.full_like(x0, -np.inf)
        ub = np.full_like(x0, np.inf)
        lb[:dim] = ub[:dim] = x0[:dim]
        lb[-dim:] = ub[-dim:] = x0[-dim:]
        bounds = Bounds(lb, ub)
        for _ in range(max_outer):
            koz_constraint = build_koz_constraints_2d(A_list, P, r_e, c_center, safety=safety)
            def objective(x):
                cost, _ = _compute_cost_only(x, Np1, dim, n_samples=80)
                return cost
            def gradient(x):
                _, grad, _ = cost_function_gradient_hessian(x, Np1, dim, n_samples=80, compute_grad=True)
                return grad
            res = minimize(
                objective,
                P.reshape(-1),
                method='trust-constr',
                jac=gradient,
                constraints=[koz_constraint],
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-10, 'xtol': 1e-12, 'barrier_tol': 1e-12, 'disp': False}
            )
            P_new = res.x.reshape(Np1, dim)
            delta = np.linalg.norm(P_new - P)
            P = P_new
            if delta < tol:
                break
        return P
    # Run optimization using original control points as initial guess
    P_init = np.vstack([P0_2d, P1_2d, P2_2d])
    safety_margin = radius * 0.05
    P_new = optimize_quadratic_2d(P_init, n_seg=M_local, r_e=radius, c_center=center, max_outer=20, tol=1e-8, safety=safety_margin)
    # Build segment control points for the new curve
    A_list_new = segment_matrices_equal_params(2, M_local)
    segment_ctrls_new = [Ai @ P_new for Ai in A_list_new]
    # Draw entire new curve in blue and highlight one segment
    if len(violating_indices) > 0:
        target_idx = violating_indices[len(violating_indices) // 2]
    else:
        target_idx = M_local // 2
    # Draw new curve segments (entire curve as grey dashed), then highlight target in blue
    ts = np.linspace(0, 1, 160)
    for j, ctrl in enumerate(segment_ctrls_new):
        pts_corr = eval_bezier_from_ctrl(ctrl, ts)
        ax2.plot(pts_corr[:, 0], pts_corr[:, 1], color='#777777', linewidth=1.8, linestyle='--', alpha=0.9)
    # Highlight the selected segment on top in blue
    pts_corr_target = eval_bezier_from_ctrl(segment_ctrls_new[target_idx], ts)
    ax2.plot(pts_corr_target[:, 0], pts_corr_target[:, 1], color='#0066ff', linewidth=2.8)
    # Show adjusted control polygon for the selected segment of the NEW curve
    seg_ctrl_new_target = segment_ctrls_new[target_idx]
    ax2.plot(seg_ctrl_new_target[:, 0], seg_ctrl_new_target[:, 1],
             linestyle='--', color='#0066ff', linewidth=1.8, marker='o', markersize=5)
    # Draw supporting half-space line and arrow using NEW segment (consistent with constraints)
    n_sup, support_pt = compute_support(center, radius, seg_ctrl_new_target)
    line_len = (ax_xmax - ax_xmin + ax_ymax - ax_ymin) * 0.25
    plot_support_line(ax2, support_pt, n_sup, line_len, color='#111111', linewidth=1.8)
    arrow_len2 = radius * 0.6
    ax2.arrow(support_pt[0], support_pt[1], n_sup[0] * arrow_len2, n_sup[1] * arrow_len2,
              width=0.0, head_width=radius * 0.12, head_length=radius * 0.18,
              length_includes_head=True, color='#111111', alpha=0.9)
    # Show global control polygon for the NEW curve
    ax2.plot(P_new[:, 0], P_new[:, 1], linestyle='--', color='#222222', linewidth=1.6, marker='o', markersize=4)
    # Show segmentation points for the NEW curve
    seg_points_new = bezier_quadratic(P_new[0], P_new[1], P_new[2], t_segs)
    ax2.scatter(seg_points_new[:, 0], seg_points_new[:, 1], s=22, color='black', zorder=3)
    
    ax2.set_xlim(ax_xmin, ax_xmax)
    ax2.set_ylim(ax_ymin, ax_ymax)
    
    # Sync zoom between both panels
    def sync_zoom_2d(event):
        if event.inaxes in [ax1, ax2]:
            source_ax = event.inaxes
            xlim = source_ax.get_xlim()
            ylim = source_ax.get_ylim()
            for ax in [ax1, ax2]:
                if ax is not source_ax:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_release_event', sync_zoom_2d)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Default to the 2D KOZ linearization figure
    plot_koz_linearization_2d()