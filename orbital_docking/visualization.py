"""
Visualization functions for orbital docking trajectories.
"""

import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .bezier import BezierCurve
from .de_casteljau import segment_matrices_equal_params
from .constants import EARTH_RADIUS_KM, EARTH_MU_SCALED, EARTH_J2, TRANSFER_TIME_S
from .utils import format_number

PROGRESS_LABEL = "Progress"
ISS_LABEL = "ISS"
PROGRESS_MARKER = "^"   # cargo-vehicle triangle marker
ISS_MARKER = "P"     # station-like filled-plus marker

# Earth 3D model (NASA): under repo assets/models, used when available
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EARTH_GLB_PATH = os.path.join(_REPO_ROOT, "assets", "models", "earth_nasa.glb")
DEBUG_LOG_PATH = os.path.join(_REPO_ROOT, ".cursor", "debug-90cb20.log")
DEBUG_SESSION_ID = "90cb20"
_EARTH_MESH_CACHE = None


def _debug_log(run_id, hypothesis_id, location, message, data):
    """Append one NDJSON debug log line for this debug session."""
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
        timestamp = int(time.time() * 1000)
        payload = {
            "sessionId": DEBUG_SESSION_ID,
            "id": f"{run_id}:{hypothesis_id}:{timestamp}",
            "timestamp": timestamp,
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass


def _infer_texture_up_axis(mesh, verts) -> dict | None:
    """
    Infer which world axis corresponds to the texture's north-south direction.
    """
    visual = getattr(mesh, "visual", None)
    if visual is None or not hasattr(visual, "uv"):
        return None

    uv = np.asarray(visual.uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] != verts.shape[0]:
        return None

    # GLTF/GLB UVs use v=1 at the top of the image; larger v means farther north
    # for this Earth atlas, so the previous sign convention inverted the poles.
    northness = uv[:, 1] - 0.5
    if np.std(northness) < 1e-12:
        return None

    norms = np.linalg.norm(verts, axis=1)
    unit_verts = np.divide(
        verts,
        norms[:, None],
        out=np.zeros_like(verts),
        where=norms[:, None] > 1e-12,
    )
    correlations = {}
    for idx, axis_name in enumerate(("x", "y", "z")):
        coord = unit_verts[:, idx]
        if np.std(coord) < 1e-12:
            correlations[axis_name] = 0.0
        else:
            correlations[axis_name] = float(np.corrcoef(coord, northness)[0, 1])

    dominant_axis = max(correlations, key=lambda k: abs(correlations[k]))
    return {
        "dominant_axis": dominant_axis,
        "north_positive": bool(correlations[dominant_axis] > 0.0),
        "correlations": correlations,
    }


def _safe_set_window_title(fig, title: str) -> None:
    """
    Best-effort: set GUI window title for a Matplotlib figure.
    No-op for non-interactive backends (e.g., inline notebooks, Agg).
    """
    if fig is None or not title:
        return
    try:
        manager = getattr(getattr(fig, "canvas", None), "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(title)
    except Exception:
        pass


def _sample_texture_face_colors(mesh, faces, run_id=None) -> np.ndarray | None:
    """
    Convert a textured trimesh visual into per-face RGBA colors for Matplotlib.

    Matplotlib's Poly3DCollection cannot render GLB/PBR materials directly, so we
    sample the texture at each triangle's centroid UV and use that as the face color.
    """
    visual = getattr(mesh, "visual", None)
    if visual is None or not hasattr(visual, "uv"):
        return None

    material = getattr(visual, "material", None)
    texture = getattr(material, "baseColorTexture", None) if material is not None else None
    if texture is None:
        return None

    try:
        uv = np.asarray(visual.uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] == 0:
            return None

        rgba = np.asarray(texture.convert("RGBA"), dtype=float) / 255.0
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            return None

        h, w = rgba.shape[:2]
        face_uv = uv[faces].mean(axis=1)
        u = np.mod(face_uv[:, 0], 1.0)
        v = np.clip(face_uv[:, 1], 0.0, 1.0)
        x = np.clip(np.rint(u * (w - 1)).astype(int), 0, w - 1)
        y = np.clip(np.rint((1.0 - v) * (h - 1)).astype(int), 0, h - 1)
        face_colors = rgba[y, x]

        base_factor = getattr(material, "baseColorFactor", None)
        if base_factor is not None:
            factor = np.asarray(base_factor, dtype=float)
            if factor.shape == (4,):
                if factor.max() > 1.0:
                    factor = factor / 255.0
                face_colors = np.clip(face_colors * factor, 0.0, 1.0)

        if run_id is not None:
            unique_face_colors = int(
                np.unique((face_colors[:, :3] * 255).astype(np.uint8), axis=0).shape[0]
            )
            # region agent log
            _debug_log(
                run_id,
                "H2",
                "orbital_docking/visualization.py:_sample_texture_face_colors",
                "Sampled Earth texture into per-face colors",
                {
                    "texture_size_px": [int(w), int(h)],
                    "mesh_faces": int(faces.shape[0]),
                    "mesh_vertices": int(uv.shape[0]),
                    "unique_face_colors": unique_face_colors,
                    "sampling_mode": "triangle_centroid_uv",
                },
            )
            # endregion

        return face_colors
    except Exception as exc:
        print(f"[Earth mesh] Could not sample GLB texture; continuing without textured colors. Error: {exc!r}")
        return None


def accel_two_body_km_s2(r_km: np.ndarray) -> np.ndarray:
    """Two-body gravity acceleration in km/s^2."""
    r = np.asarray(r_km, dtype=float)
    rn = np.linalg.norm(r)
    if rn < 1e-12:
        return np.zeros(3)
    return (-EARTH_MU_SCALED / (rn**3)) * r


def accel_j2_km_s2(r_km: np.ndarray) -> np.ndarray:
    """
    J2 perturbation acceleration in km/s^2.
    Simplified model: Earth symmetry axis aligned with ECI Z.
    """
    r = np.asarray(r_km, dtype=float)
    x, y, z = r
    r2 = float(x*x + y*y + z*z)
    rn = np.sqrt(r2)
    if rn < 1e-12:
        return np.zeros(3)
    z2 = z*z
    r5 = rn**5
    factor = 1.5 * EARTH_J2 * EARTH_MU_SCALED * (EARTH_RADIUS_KM**2) / r5
    k = 5.0 * z2 / r2
    ax = factor * x * (k - 1.0)
    ay = factor * y * (k - 1.0)
    az = factor * z * (k - 3.0)
    return np.array([ax, ay, az], dtype=float)


def accel_gravity_total_km_s2(r_km: np.ndarray) -> np.ndarray:
    """Two-body + J2 gravity acceleration in km/s^2."""
    return accel_two_body_km_s2(r_km) + accel_j2_km_s2(r_km)


def control_effort_metrics(info: dict):
    """
    Convert optimizer cost into objective-consistent physical metrics.

    Returns:
        (rms_control_accel_m_s2, l2_effort_m2_s3, dv_proxy_m_s)
        - RMS control accel: sqrt(cost_true_energy) converted to m/s^2 (only meaningful for 'energy' objective).
        - Energy surrogate over physical time: cost_true_energy * T (m^2/s^3).
        - Δv proxy: info['dv_proxy_m_s'] if available (m/s).
    """
    if info is None:
        return None, None, None

    J_tau = info.get('cost_true_energy', info.get('cost', None))
    if J_tau is None:
        return None, None, None

    J_tau = float(J_tau)
    if not np.isfinite(J_tau) or J_tau < 0.0:
        return None, None, None

    T = float(info.get('T_transfer_s', TRANSFER_TIME_S))
    rms_control_accel_m_s2 = np.sqrt(J_tau) * 1e3
    l2_effort_m2_s3 = J_tau * T * 1e6
    dv_proxy_m_s = info.get("dv_proxy_m_s", None)
    try:
        dv_proxy_m_s = float(dv_proxy_m_s) if dv_proxy_m_s is not None else None
    except Exception:
        dv_proxy_m_s = None
    return float(rms_control_accel_m_s2), float(l2_effort_m2_s3), dv_proxy_m_s


def _load_earth_mesh(radius_km=EARTH_RADIUS_KM, center=(0.0, 0.0, 0.0), run_id=None):
    """
    Load NASA Earth mesh from assets/models/earth_nasa.glb, scale to radius_km, center at center.
    Returns (vertices, faces) or None if file missing or trimesh unavailable.
    """
    global _EARTH_MESH_CACHE
    if _EARTH_MESH_CACHE is not None:
        unit_verts, faces, face_colors, orientation, rotation_label = _EARTH_MESH_CACHE
        verts = unit_verts * float(radius_km) + np.asarray(center, dtype=float)
        if run_id is not None:
            # region agent log
            _debug_log(
                run_id,
                "H5",
                "orbital_docking/visualization.py:_load_earth_mesh",
                "Using cached UV-mapped Earth mesh",
                {
                    "render_mode": "cached_uv_mesh",
                    "cache_hit": True,
                    "mesh_faces": int(faces.shape[0]),
                    "rotation_applied": rotation_label,
                    "dominant_up_axis": None if orientation is None else orientation["dominant_axis"],
                    "north_positive": None if orientation is None else orientation["north_positive"],
                },
            )
            # endregion
        return verts, faces, face_colors

    if not os.path.isfile(EARTH_GLB_PATH):
        print(f"[Earth mesh] GLB file not found at {EARTH_GLB_PATH}")
        return None
    try:
        import trimesh
    except ImportError:
        print("[Earth mesh] trimesh not installed; falling back to wireframe Earth.")
        return None
    try:
        loaded = trimesh.load(EARTH_GLB_PATH)
        if loaded is None:
            print("[Earth mesh] trimesh.load returned None; falling back to wireframe Earth.")
            return None
        if hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
            mesh = loaded
        elif hasattr(loaded, "geometry"):
            meshes = list(loaded.geometry.values())
            if not meshes:
                print("[Earth mesh] Scene.geometry is empty; falling back to wireframe Earth.")
                return None
            mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        else:
            print("[Earth mesh] Unexpected GLB type without vertices or geometry; falling back.")
            return None

        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)
        if verts.size == 0 or faces.size == 0:
            print("[Earth mesh] Loaded mesh has no vertices or faces; falling back.")
            return None

        # Optional per-face colors from the GLB, if present.
        face_colors = None
        visual = getattr(mesh, "visual", None)
        try:
            if visual is not None and hasattr(visual, "face_colors"):
                fc = np.asarray(visual.face_colors)
                if fc.ndim == 2 and fc.shape[0] == faces.shape[0]:
                    # Normalize to 0–1 if stored as 0–255.
                    face_colors = fc.astype(float)
                    if face_colors.max() > 1.0:
                        face_colors /= 255.0
        except Exception as exc:
            print(f"[Earth mesh] Could not read face colors from GLB visual; continuing without colors. Error: {exc!r}")
            face_colors = None

        if face_colors is None:
            face_colors = _sample_texture_face_colors(mesh, faces, run_id=run_id)

        # Center and scale geometry to requested Earth radius.
        verts = verts - verts.mean(axis=0)
        orientation = _infer_texture_up_axis(mesh, verts)
        rotation = np.eye(3, dtype=float)
        rotation_label = "identity"
        if orientation is not None and orientation["dominant_axis"] == "y":
            if orientation["north_positive"]:
                rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)
                rotation_label = "rotate_x_plus_90"
            else:
                rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float)
                rotation_label = "rotate_x_minus_90"
        verts = verts @ rotation.T
        r_max = np.linalg.norm(verts, axis=1).max()
        if r_max < 1e-12:
            print("[Earth mesh] Loaded mesh has near-zero radius; falling back.")
            return None
        unit_verts = verts / r_max
        _EARTH_MESH_CACHE = (unit_verts, faces, face_colors, orientation, rotation_label)
        verts = unit_verts * float(radius_km) + np.asarray(center, dtype=float)
        if run_id is not None:
            orientation = _infer_texture_up_axis(mesh, verts - np.asarray(center, dtype=float))
            # region agent log
            _debug_log(
                run_id,
                "H1",
                "orbital_docking/visualization.py:_load_earth_mesh",
                "Inferred Earth texture orientation",
                {
                    "face_colors_available": bool(face_colors is not None),
                    "dominant_up_axis": None if orientation is None else orientation["dominant_axis"],
                    "north_positive": None if orientation is None else orientation["north_positive"],
                    "axis_correlations": None if orientation is None else orientation["correlations"],
                    "rotation_applied": rotation_label,
                },
            )
            # endregion
            # region agent log
            _debug_log(
                run_id,
                "H5",
                "orbital_docking/visualization.py:_load_earth_mesh",
                "Prepared Earth mesh from GLB UV atlas",
                {
                    "render_mode": "cached_uv_mesh",
                    "cache_hit": False,
                    "mesh_faces": int(faces.shape[0]),
                    "mesh_vertices": int(verts.shape[0]),
                    "rotation_applied": rotation_label,
                },
            )
            # endregion
        return verts, faces, face_colors
    except Exception as exc:
        print(f"[Earth mesh] Exception while loading GLB; falling back. Error: {exc!r}")
        return None


def add_earth_mesh(ax, radius=EARTH_RADIUS_KM, center=(0.0, 0.0, 0.0), color='#2E86AB', alpha=0.95):
    """
    Add Earth as a 3D mesh from assets/models/earth_nasa.glb when available.
    Falls back to wireframe sphere if model is missing or trimesh unavailable.
    """
    run_id = getattr(getattr(ax, "figure", None), "_debug_run_id", "earth-standalone")
    data = _load_earth_mesh(radius_km=radius, center=center, run_id=run_id)
    if data is None:
        print("[Earth mesh] Falling back to wireframe Earth in add_earth_mesh().")
        add_wire_sphere(ax, radius=radius, center=center, color=color, alpha=alpha, resolution=20)
        return

    # Backward-compatible unpacking: handle (verts, faces) or (verts, faces, face_colors).
    if len(data) == 3:
        verts, faces, face_colors = data
    else:
        verts, faces = data
        face_colors = None

    polygons = [verts[f] for f in faces]

    # If the GLB provided per-face colors, use them so continents/oceans show up.
    if face_colors is not None:
        fc = np.asarray(face_colors, dtype=float)
        # Ensure alpha channel exists and respects requested alpha.
        if fc.shape[1] == 4:
            fc = fc.copy()
            fc[:, 3] = alpha
        elif fc.shape[1] == 3:
            alpha_col = np.full((fc.shape[0], 1), alpha, dtype=float)
            fc = np.hstack([fc, alpha_col])
        coll = Poly3DCollection(
            polygons,
            facecolors=fc,
            edgecolors="none",
            shade=False,
        )
    else:
        coll = Poly3DCollection(
            polygons,
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            shade=False,
        )

    ax.add_collection3d(coll)
    ax._earth_debug_mesh_faces = int(faces.shape[0])


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
    Add Earth for orbital context: 3D mesh from assets/models/earth_nasa.glb when available,
    otherwise a wireframe sphere.

    Args:
        ax: 3D matplotlib axes
        radius: Earth radius in km
        center: Earth center coordinates
        color: Earth color
        alpha: Transparency
    """
    add_earth_mesh(ax, radius=radius, center=center, color=color, alpha=alpha)


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


def create_trajectory_comparison_figure(P_init, r_e, results, curve_order=None, window_title=None,
                                       v0=None, v1=None):
    """
    Create 2×3 layout showing trajectories with different segment counts.
    Uses same aesthetics as Orbital_Docking_Optimizer.py.
    All 6 panels share the same zoom level - zooming one zooms all.

    Args:
        P_init: Initial control points
        r_e: KOZ radius
        results: List of (n_seg, P_opt, info) tuples
        v0: Optional initial velocity vector (km/s) at P_init[0]
        v1: Optional final velocity vector (km/s) at P_init[-1]

    Returns:
        matplotlib Figure object
    """
    build_started = time.perf_counter()
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig._debug_run_id = f"earth-debug-{int(time.time() * 1000)}"
    if window_title is None and curve_order is not None:
        window_title = f"Orbital Docking — Trajectory Comparison (N={curve_order})"
    _safe_set_window_title(fig, window_title)

    # Create 2×3 subplot layout
    axes = []
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        axes.append(ax)

    manager = getattr(fig.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None) if manager is not None else None
    # region agent log
    _debug_log(
        fig._debug_run_id,
        "H3",
        "orbital_docking/visualization.py:create_trajectory_comparison_figure",
        "Figure interactivity backend state",
        {
            "backend": str(plt.get_backend()),
            "interactive": bool(plt.isinteractive()),
            "canvas_class": type(fig.canvas).__name__,
            "manager_class": None if manager is None else type(manager).__name__,
            "toolbar_class": None if toolbar is None else type(toolbar).__name__,
            "axes_count": int(len(axes)),
        },
    )
    # endregion

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

        # Add Earth at physical scale for consistent geometry interpretation.
        add_earth_sphere(ax, radius=EARTH_RADIUS_KM, color='blue', alpha=1)

        # Add safety zone (KOZ - smaller red sphere)
        add_wire_sphere(ax, radius=r_e, color='red', alpha=0.2, resolution=15)

        # Plot optimized trajectory (thicker lines with color-coded segments)
        plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', lw=3.0)

        p0 = P_init[0]
        p1 = P_init[-1]
        ax.scatter(
            p0[0], p0[1], p0[2],
            color='green', s=140, marker=PROGRESS_MARKER, label=PROGRESS_LABEL, zorder=10
        )
        ax.scatter(
            p1[0], p1[1], p1[2],
            color='orange', s=140, marker=ISS_MARKER, label=ISS_LABEL, zorder=10
        )

        # Draw endpoint velocity vectors if provided.
        # We scale the arrows so they are visually legible while preserving direction.
        if v0 is not None or v1 is not None:
            base_radius = float(np.linalg.norm(p0))
            mags = []
            if v0 is not None:
                mags.append(float(np.linalg.norm(v0)))
            if v1 is not None:
                mags.append(float(np.linalg.norm(v1)))
            vmax = max(mags) if mags else 0.0
            # Scale so the largest arrow is ~30% of the orbital radius
            scale = (1 * base_radius / vmax) if vmax > 1e-9 else 0.0

            if v0 is not None and scale > 0.0:
                dv0 = np.asarray(v0, dtype=float) * scale
                ax.quiver(
                    p0[0], p0[1], p0[2],
                    dv0[0], dv0[1], dv0[2],
                    color='cyan', linewidth=1.5, arrow_length_ratio=0.08,
                    label='v0'
                )
            if v1 is not None and scale > 0.0:
                dv1 = np.asarray(v1, dtype=float) * scale
                ax.quiver(
                    p1[0], p1[1], p1[2],
                    dv1[0], dv1[1], dv1[2],
                    color='magenta', linewidth=1.5, arrow_length_ratio=0.08,
                    label='v1'
                )

        # Build legend entries (vehicle markers + optional velocity arrows).
        ax.legend(fontsize=8)

        # Professional styling with shared zoom level
        set_axes_equal_around(ax, center=(0,0,0), radius=view_radius, pad=0.05)
        # Match baseline viewing angle used for paper figures
        set_isometric(ax, elev=20, azim=-45)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)

        # Title with optimizer-consistent control-effort metric and termination reason.
        rms_control_accel_m_s2, _l2, dv_proxy_m_s = control_effort_metrics(info)
        accel_str = "n/a" if rms_control_accel_m_s2 is None else format_number(rms_control_accel_m_s2, '.2f')
        dv_str = "n/a" if dv_proxy_m_s is None else format_number(dv_proxy_m_s, '.1f')
        feasible = bool(info.get('feasible', False))
        status = 'Feasible' if feasible else 'Infeasible'
        title_color = 'black' if feasible else 'red'

        term = str(info.get("termination_reason", "") or "").strip().lower()
        if term == "converged_delta_below_tol":
            term_label = "converged (ΔP < tol)"
        elif term == "stopped_max_iter":
            term_label = "stopped (max_iter)"
        elif term == "not_run":
            term_label = "not run"
        elif term:
            term_label = term
        else:
            term_label = "termination: n/a"

        ax.set_title(
            f'{n_seg} Segments — {status}\n'
            f'Δv proxy: {dv_str} m/s | RMS u: {accel_str} m/s²\n'
            f'{term_label}',
            fontsize=10,
            pad=10,
            color=title_color,
        )

    # Link all axes to share zoom - when one zooms, all zoom together
    # Use a flag to prevent infinite recursion
    _syncing = False
    sync_debug_state = {"count": 0}
    
    def sync_limits(ax_source):
        """Sync limits from source axis to all other axes."""
        nonlocal _syncing
        if _syncing:
            return

        started = time.perf_counter()
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
        if sync_debug_state["count"] < 6:
            sync_debug_state["count"] += 1
            # region agent log
            _debug_log(
                fig._debug_run_id,
                "H7",
                "orbital_docking/visualization.py:create_trajectory_comparison_figure",
                "Synced axis limits across comparison panels",
                {
                    "sync_index": int(sync_debug_state["count"]),
                    "source_axis_index": int(axes.index(ax_source)),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "xlim": [float(xlim[0]), float(xlim[1])],
                    "ylim": [float(ylim[0]), float(ylim[1])],
                    "zlim": [float(zlim[0]), float(zlim[1])],
                },
            )
            # endregion
    
    # Connect the callback to each axis for all three dimensions
    for ax in axes:
        # Use lambda with default argument to capture ax correctly
        ax.callbacks.connect('xlim_changed', lambda event, ax=ax: sync_limits(ax))
        ax.callbacks.connect('ylim_changed', lambda event, ax=ax: sync_limits(ax))
        ax.callbacks.connect('zlim_changed', lambda event, ax=ax: sync_limits(ax))

    total_earth_faces = int(sum(getattr(ax, "_earth_debug_mesh_faces", 0) for ax in axes))
    build_ms = int((time.perf_counter() - build_started) * 1000)
    # region agent log
    _debug_log(
        fig._debug_run_id,
        "H4",
        "orbital_docking/visualization.py:create_trajectory_comparison_figure",
        "Figure build complete",
        {
            "build_ms": build_ms,
            "total_earth_faces": total_earth_faces,
            "sync_callback_count": int(len(axes) * 3),
            "earth_render_mode": "cached_uv_mesh",
            "view_radius_km": float(view_radius),
        },
    )
    # endregion

    ui_debug_state = {"draws": 0, "mouse": 0}

    def _axis_index(event):
        return None if getattr(event, "inaxes", None) is None else axes.index(event.inaxes) if event.inaxes in axes else "other"

    def _log_mouse_event(event):
        if ui_debug_state["mouse"] >= 4:
            return
        ui_debug_state["mouse"] += 1
        step = getattr(event, "step", None)
        # region agent log
        _debug_log(
            fig._debug_run_id,
            "H3",
            "orbital_docking/visualization.py:create_trajectory_comparison_figure",
            "Mouse event received by Matplotlib canvas",
            {
                "event_name": str(getattr(event, "name", "unknown")),
                "axis_index": _axis_index(event),
                "button": None if getattr(event, "button", None) is None else str(event.button),
                "step": None if step is None else float(step),
                "xdata": None if getattr(event, "xdata", None) is None else float(event.xdata),
                "ydata": None if getattr(event, "ydata", None) is None else float(event.ydata),
            },
        )
        # endregion

    def _log_draw_event(_event):
        if ui_debug_state["draws"] >= 4:
            return
        ui_debug_state["draws"] += 1
        # region agent log
        _debug_log(
            fig._debug_run_id,
            "H4",
            "orbital_docking/visualization.py:create_trajectory_comparison_figure",
            "Draw event completed",
            {
                "draw_index": int(ui_debug_state["draws"]),
                "elapsed_since_build_ms": int((time.perf_counter() - build_started) * 1000),
            },
        )
        # endregion

    fig.canvas.mpl_connect("button_press_event", _log_mouse_event)
    fig.canvas.mpl_connect("scroll_event", _log_mouse_event)
    fig.canvas.mpl_connect("draw_event", _log_draw_event)

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
        # Convert tau-derivatives to physical time derivatives using fixed transfer time T.
        #   v(t) = (1/T) r'(tau)
        #   a_geom(t) = (1/T^2) r''(tau)
        T = float(TRANSFER_TIME_S)
        velocities_km_s = np.array([curve.velocity(t) for t in ts]) / T
        a_geom_km_s2 = np.array([curve.acceleration(t) for t in ts]) / (T**2)

        # Gravity (two-body + J2) along the curve
        a_grav_km_s2 = np.array([accel_gravity_total_km_s2(curve.point(t)) for t in ts])

        # Control acceleration magnitude (m/s^2): ||a_geom - a_grav||
        a_u_m_s2 = np.linalg.norm(a_geom_km_s2 - a_grav_km_s2, axis=1) * 1e3

        # Update mins/maxes
        pos_min = min(pos_min, positions.min())
        pos_max = max(pos_max, positions.max())
        vel_min = min(vel_min, velocities_km_s.min())
        vel_max = max(vel_max, velocities_km_s.max())
        acc_max = max(acc_max, float(np.max(a_u_m_s2)))

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


def create_performance_figure(results, curve_order=None, window_title=None):
    """
    Create performance figure showing acceleration vs segment count.

    Args:
        results: List of (n_seg, P_opt, info) tuples

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    if window_title is None and curve_order is not None:
        window_title = f"Orbital Docking — Performance (N={curve_order})"
    _safe_set_window_title(fig, window_title)
    ax = fig.add_subplot(111)

    # Extract delta-v proxy (preferred) and RMS u as a secondary diagnostic.
    segment_counts = []
    dv_proxies = []

    for n_seg, P_opt, info in results:
        if P_opt is None or info is None:
            continue
        _rms_u, _l2, dv_proxy_m_s = control_effort_metrics(info)
        if dv_proxy_m_s is None:
            continue

        segment_counts.append(n_seg)
        dv_proxies.append(dv_proxy_m_s)

    # Create performance graph
    ax.plot(segment_counts, dv_proxies, 'bo-', linewidth=3, markersize=10)
    ax.set_xlabel('Number of Segments', fontsize=14)
    ax.set_ylabel('Δv proxy (m/s)', fontsize=14)
    ax.set_title('Δv proxy vs Segment Count', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)  # Log scale for segment counts

    # Add data point labels
    for i, (seg, dv) in enumerate(zip(segment_counts, dv_proxies)):
        dv_str = format_number(dv, '.2f')
        ax.annotate(dv_str, (seg, dv), textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    return fig


def create_multi_order_performance_figure(results_by_order, window_title=None):
    """
    Create a single performance figure showing Δv proxy vs segment count
    for multiple curve orders simultaneously.

    Args:
        results_by_order: Dict mapping curve order N -> list of (n_seg, P_opt, info)
    """
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    if window_title is None:
        window_title = "Orbital Docking — Performance (multi-order)"
    _safe_set_window_title(fig, window_title)
    ax = fig.add_subplot(111)

    colors = {
        2: "#3498DB",
        3: "#E74C3C",
        4: "#F39C12",
    }

    for N, results in sorted(results_by_order.items()):
        segment_counts = []
        dv_proxies = []
        for n_seg, P_opt, info in results:
            if P_opt is None or info is None:
                continue
            _rms_u, _l2, dv_proxy_m_s = control_effort_metrics(info)
            if dv_proxy_m_s is None:
                continue
            segment_counts.append(n_seg)
            dv_proxies.append(dv_proxy_m_s)

        if not segment_counts:
            continue

        c = colors.get(N, None)
        label = f"N={N}"
        ax.plot(
            segment_counts,
            dv_proxies,
            marker="o",
            linewidth=3,
            markersize=9,
            label=label,
            color=c,
        )

        # Annotate points for this order.
        for seg, dv in zip(segment_counts, dv_proxies):
            dv_str = format_number(dv, ".2f")
            ax.annotate(
                dv_str,
                (seg, dv),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8),
            )

    ax.set_xlabel("Number of Segments", fontsize=14)
    ax.set_ylabel("Δv proxy (m/s)", fontsize=14)
    ax.set_title("Δv proxy vs Segment Count (multi-order)", fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.legend(title="Curve order", fontsize=11)

    return fig


def create_acceleration_figure(
    results,
    segcount=64,
    pos_ylim=None,
    vel_ylim=None,
    acc_ylim=None,
    toggle_accel_legend=True,
    curve_order=None,
    window_title=None,
):
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
    if curve_order is None:
        fig.suptitle(f'Position, Velocity, and Acceleration Profiles for {segcount} Segments', fontsize=16)
    else:
        fig.suptitle(
            f'Position, Velocity, and Acceleration Profiles (N={curve_order}) for {segcount} Segments',
            fontsize=16,
        )

    if window_title is None:
        if curve_order is None:
            window_title = f"Orbital Docking — Profiles ({segcount} seg)"
        else:
            window_title = f"Orbital Docking — Profiles (N={curve_order}, {segcount} seg)"
    _safe_set_window_title(fig, window_title)

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

    # Sample positions, velocities, and accelerations (unit-consistent)
    # curve.velocity(tau) is dr/dtau [km]; convert to km/s via v = (1/T) dr/dtau
    positions = np.array([curve.point(t) for t in ts])
    T = float(TRANSFER_TIME_S)
    velocities = np.array([curve.velocity(t) for t in ts]) / T

    # Calculate magnitudes
    position_magnitudes = np.linalg.norm(positions, axis=1)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)

    # Accelerations:
    # - curve.acceleration(tau) is d^2r/dtau^2 [km]; convert to km/s^2 via a = (1/T^2) d^2r/dtau^2
    # - gravity is computed in km/s^2; we convert magnitudes to m/s^2 for plotting
    a_geom_km_s2 = np.array([curve.acceleration(t) for t in ts]) / (T**2)
    a_grav_km_s2 = np.array([accel_gravity_total_km_s2(curve.point(t)) for t in ts])

    geom_accel_mag = np.linalg.norm(a_geom_km_s2, axis=1) * 1e3
    grav_accel_mag = np.linalg.norm(a_grav_km_s2, axis=1) * 1e3
    total_accel_mag_kms2 = np.linalg.norm(a_geom_km_s2 - a_grav_km_s2, axis=1) * 1e3

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
    geom_line, = ax3.plot(
        ts, geom_accel_mag, 'orange', linewidth=2.0, label='Inertial Accel (from curve)', alpha=0.7
    )
    grav_line, = ax3.plot(
        ts, grav_accel_mag, 'cyan', linewidth=2.0, label='Gravity + J2', alpha=0.7
    )
    total_line, = ax3.plot(
        ts, total_accel_mag_kms2, 'purple', linewidth=2.5, label='Control Accel  ||a_geom - a_grav||', linestyle='-'
    )
    ax3.set_xlabel('Parameter τ', fontsize=12)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=12)

    # Summary metrics aligned with optimizer objective.
    rms_control_accel_m_s2, l2_effort_m2_s3, dv_proxy_m_s = control_effort_metrics(info)
    if rms_control_accel_m_s2 is not None:
        rms_str = format_number(rms_control_accel_m_s2, '.2f')
        if dv_proxy_m_s is not None:
            dv_str = format_number(dv_proxy_m_s, '.1f')
            ax3.set_title(
                f'Acceleration Components (Δv proxy: {dv_str} m/s | RMS u: {rms_str} m/s²)',
                fontsize=14,
                pad=15,
            )
        else:
            ax3.set_title(f'Acceleration Components (RMS Control Accel: {rms_str} m/s²)',
                     fontsize=14, pad=15)
    else:
        ax3.set_title('Acceleration Components', fontsize=14, pad=15)
    ax3.grid(True, alpha=0.3)
    accel_legend = ax3.legend(fontsize=11)
    if acc_ylim is not None:
        ax3.set_ylim(acc_ylim)

    # Optional: make the acceleration legend clickable (debugging aid).
    # Clicking a legend entry toggles the corresponding curve visibility.
    if toggle_accel_legend and accel_legend is not None:
        orig_lines = [geom_line, grav_line, total_line]

        # Matplotlib draws separate artists in the legend; we map those back to the original lines.
        legend_lines = list(accel_legend.get_lines())
        legend_texts = list(accel_legend.get_texts())

        line_by_legend_artist = {}
        for leg_line, orig_line in zip(legend_lines, orig_lines):
            line_by_legend_artist[leg_line] = orig_line
        for leg_text, orig_line in zip(legend_texts, orig_lines):
            line_by_legend_artist[leg_text] = orig_line

        # Make legend artists pickable.
        for artist in line_by_legend_artist.keys():
            try:
                artist.set_picker(True)
            except Exception:
                pass
            try:
                artist.set_pickradius(5)
            except Exception:
                pass

        def _on_pick(event):
            artist = event.artist
            if artist not in line_by_legend_artist:
                return

            orig = line_by_legend_artist[artist]
            visible = not orig.get_visible()
            orig.set_visible(visible)

            # Dim both the legend line and its text when hidden (if they exist).
            dim_alpha = 0.25
            for leg_line, orig_line in zip(legend_lines, orig_lines):
                if orig_line is orig:
                    leg_line.set_alpha(1.0 if visible else dim_alpha)
                    break
            for leg_text, orig_line in zip(legend_texts, orig_lines):
                if orig_line is orig:
                    leg_text.set_alpha(1.0 if visible else dim_alpha)
                    break

            # Redraw (works for interactive backends; harmless for static saves).
            try:
                event.canvas.draw_idle()
            except Exception:
                pass

        # Store connection id on the figure to keep it alive and avoid duplicate hooks
        # if this function is called repeatedly in the same session.
        cid_attr = "_accel_legend_toggle_cid"
        old_cid = getattr(fig, cid_attr, None)
        if old_cid is not None:
            try:
                fig.canvas.mpl_disconnect(old_cid)
            except Exception:
                pass
        try:
            setattr(fig, cid_attr, fig.canvas.mpl_connect('pick_event', _on_pick))
        except Exception:
            pass

    # Add objective-consistent control-effort statistics.
    if rms_control_accel_m_s2 is not None and l2_effort_m2_s3 is not None:
        rms_str = format_number(rms_control_accel_m_s2, '.2f')
        if dv_proxy_m_s is not None:
            dv_str = format_number(dv_proxy_m_s, '.1f')
            stats_text = f'Δv proxy: {dv_str} m/s\nRMS accel: {rms_str} m/s²'
        else:
            stats_text = f'Δv proxy: n/a\nRMS accel: {rms_str} m/s²'
    else:
        stats_text = 'Δv proxy: n/a\nRMS accel: n/a'
    ax3.text(0.02, 0.98, stats_text,
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
        # Prefer the persisted optimization compute time (cached metadata) when available.
        # This avoids near-zero bars when the caller measured wall-time during a cache-hit run.
        t_info = None
        if N in optimization_results:
            _, info = optimization_results[N]
            if info is not None:
                try:
                    t_info = float(info.get('elapsed_time', 0.0))
                except Exception:
                    t_info = None

        t_calc = calculation_times.get(N, 0.0)
        if t_info is not None and np.isfinite(t_info) and t_info > 0.0:
            t = t_info
        else:
            t = 0.0 if t_calc is None else float(t_calc)
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

    # Add optimizer-consistent control-effort info as text.
    accel_info = []
    for N in orders:
        P_opt, info = optimization_results[N]
        if P_opt is None or info is None:
            continue
        rms_control_accel_m_s2, l2_effort_m2_s3, dv_proxy_m_s = control_effort_metrics(info)
        if rms_control_accel_m_s2 is None or l2_effort_m2_s3 is None:
            continue
        accel_str = format_number(rms_control_accel_m_s2, '.2f')
        if dv_proxy_m_s is not None:
            dv_str = format_number(dv_proxy_m_s, '.1f')
            accel_info.append(f'N={N}: Δv={dv_str} m/s, RMS={accel_str} m/s²')
        else:
            accel_info.append(f'N={N}: Δv=n/a, RMS={accel_str} m/s²')

    info_text = '\n'.join(accel_info)
    if info_text:
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        )

    return fig

