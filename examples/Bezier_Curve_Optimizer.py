"""
Bézier Trajectory Optimization with Sphere Avoidance

This module implements optimization of Bézier curves to avoid obstacles (spheres)
using segment-based linearization and iterative optimization. The main approach
splits a Bézier curve into segments and applies half-space constraints to ensure
the curve stays outside a given sphere.

Key components:
- BezierCurve: Core Bézier curve evaluation
- De Casteljau splitting: Matrix-based curve subdivision
- Linear constraint building: Half-space constraint formulation
- Optimization: SLSQP-based iterative optimization
- Visualization: 3D plotting utilities
"""

import numpy as np
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ─────────────────────────────────────────────────────────────────────────────
# Bézier curve core
# ─────────────────────────────────────────────────────────────────────────────
class BezierCurve:
    """
    A Bézier curve defined by control points.
    
    A Bézier curve of degree N is defined by N+1 control points P_0, P_1, ..., P_N.
    The curve is evaluated using the Bernstein polynomial basis:
    
    B(t) = Σ(i=0 to N) C(N,i) * t^i * (1-t)^(N-i) * P_i
    
    Attributes:
        control_points (np.ndarray): Control points of shape (N+1, dim)
        degree (int): Degree of the Bézier curve (N)
        dimension (int): Spatial dimension of the curve
    """
    
    def __init__(self, control_points: np.ndarray):
        """
        Initialize a Bézier curve with given control points.
        
        Args:
            control_points: Array of shape (N+1, dim) where N is the degree
                          and dim is the spatial dimension
        
        Raises:
            ValueError: If control_points is not 2D array
        """
        P = np.array(control_points, dtype=float)
        if P.ndim != 2:
            raise ValueError("control_points must be (N+1, dim)")
        self.control_points = P
        self.degree = P.shape[0] - 1
        self.dimension = P.shape[1]

    def point(self, tau: float) -> np.ndarray:
        """
        Evaluate the Bézier curve at parameter tau.
        
        Uses the Bernstein polynomial basis to compute the curve point:
        B(tau) = Σ(i=0 to N) C(N,i) * tau^i * (1-tau)^(N-i) * P_i
        
        Args:
            tau: Parameter value in [0, 1]
            
        Returns:
            Point on the curve as array of shape (dim,)
        """
        N, d = self.degree, self.dimension
        out = np.zeros(d, dtype=float)
        # Evaluate using Bernstein basis polynomials
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * self.control_points[i]
        return out

    def sample(self, num=200):
        """
        Sample the Bézier curve at evenly spaced parameter values.
        
        Args:
            num: Number of sample points (default: 200)
            
        Returns:
            tuple: (parameter_values, curve_points) where
                   - parameter_values: Array of tau values in [0, 1]
                   - curve_points: Array of shape (num, dim) with curve points
        """
        ts = np.linspace(0.0, 1.0, num)
        P = np.array([self.point(t) for t in ts])
        return ts, P

# ─────────────────────────────────────────────────────────────────────────────
# De Casteljau subdivision as matrix operations
# These functions implement curve subdivision using linear transformations
# on the original control points, enabling efficient segment-based optimization
# ─────────────────────────────────────────────────────────────────────────────
def de_casteljau_split_1d(N, tau, basis_index):
    """
    Compute De Casteljau subdivision coefficients for a single basis vector.
    
    Given the j-th basis vector e_j and split parameter tau, computes the
    coefficients that express e_j in terms of the left and right sub-curves
    after subdivision at tau.
    
    Args:
        N: Degree of the Bézier curve
        tau: Split parameter in [0, 1]
        basis_index: Index j of the basis vector e_j
        
    Returns:
        tuple: (left_coeffs, right_coeffs) where each is an array of length N+1
               representing the coefficients for the left and right sub-curves
    """
    # Start with j-th basis vector: w[j] = 1, others = 0
    w = np.zeros(N+1, dtype=float); 
    w[basis_index] = 1.0 # w for weights of basis vector; make N+1 sized array and set j-th element to 1
    left = [w[0]]   # First coefficient for left sub-curve
    right = [w[-1]] # First coefficient for right sub-curve
    W = w.copy()
    
    # Apply De Casteljau algorithm: linear interpolation at each level
    for _ in range(1, N+1):
        W = (1 - tau) * W[:-1] + tau * W[1:]
        left.append(W[0])   # Leftmost coefficient at this level
        right.append(W[-1]) # Rightmost coefficient at this level
    
    L = np.array(left)      # (N+1,) coefficients for left sub-curve
    R = np.array(right[::-1]) # (N+1,) coefficients for right sub-curve (reversed)
    return L, R

def de_casteljau_split_matrices(N, tau):
    """
    Compute subdivision matrices for De Casteljau splitting.
    
    Returns matrices S_left and S_right such that:
    - Left sub-curve control points: L = S_left @ P
    - Right sub-curve control points: R = S_right @ P
    
    where P is the original control point vector.
    
    Args:
        N: Degree of the Bézier curve
        tau: Split parameter in [0, 1]
        
    Returns:
        tuple: (S_left, S_right) where each is a (N+1, N+1) matrix
    """
    S_left  = np.zeros((N+1, N+1), dtype=float)
    S_right = np.zeros((N+1, N+1), dtype=float)
    
    # Build matrices column by column using basis vector coefficients
    for j in range(N+1):
        L, R = de_casteljau_split_1d(N, tau, j)
        S_left[:,  j] = L  # j-th column of S_left
        S_right[:, j] = R  # j-th column of S_right
    return S_left, S_right

def segment_matrices_equal_params(N, n_seg):
    """
    Generate subdivision matrices for equal-parameter segment splitting.
    
    Splits a Bézier curve into n_seg segments using equal parameter intervals.
    Each segment i has control points Qi = Ai @ P where P are the original
    control points and Ai is the transformation matrix for segment i.
    
    The splitting uses repeated subdivision at tau = 1/k for k = n_seg, n_seg-1, ..., 2.
    
    Args:
        N: Degree of the Bézier curve
        n_seg: Number of segments to create
        
    Returns:
        list: List of (N+1, N+1) matrices Ai, one per segment
              
    Raises:
        ValueError: If n_seg < 1
    """
    if n_seg < 1:
        raise ValueError("n_seg must be >= 1")
    if n_seg == 1:
        return [np.eye(N+1)]  # No subdivision needed
    
    mats = []
    remainder = np.eye(N+1)  # Start with identity (no transformation)
    
    # Iteratively split the remaining portion
    for k in range(n_seg, 1, -1):
        tau = 1.0 / k  # Split parameter for this iteration
        S_L, S_R = de_casteljau_split_matrices(N, tau)
        mats.append(S_L @ remainder)  # Left portion becomes a segment
        remainder = S_R @ remainder   # Right portion continues
    mats.append(remainder)  # Final remainder is the last segment
    return mats  # list of (N+1,N+1) matrices

# ─────────────────────────────────────────────────────────────────────────────
# Half-space constraint formulation for sphere avoidance
# Implements linear constraints of the form: n_i^T (A_i @ P)_k >= r_e
# where n_i is the normal vector, A_i is the segment matrix, and r_e is the radius
# ─────────────────────────────────────────────────────────────────────────────
def build_linear_constraint_matrix(A_list, n_list, dim, r_e):
    """
    Build linear constraints for sphere avoidance across all segments.
    
    Creates constraints of the form n_i^T (A_i @ P)_k >= r_e for each segment i
    and control point k, ensuring the curve stays outside the sphere of radius r_e.
    
    Args:
        A_list: List of segment transformation matrices
        n_list: List of normal vectors for each segment
        dim: Spatial dimension of the control points
        r_e: Sphere radius (constraint threshold)
        
    Returns:
        LinearConstraint: Constraint object for scipy.optimize
    """
    Np1 = A_list[0].shape[1]  # Number of control points (N+1)
    rows, lbs = [], []
    
    # Build constraint for each segment and each control point
    for Ai, n in zip(A_list, n_list):
        for k in range(Np1):  # For each control point in segment i
            row = np.zeros(Np1 * dim)  # Flattened control point vector
            for j in range(Np1):  # For each original control point
                coeff = Ai[k, j]  # Transformation coefficient
                start = j * dim   # Starting index in flattened vector
                row[start:start+dim] += coeff * n  # Add contribution to constraint
            rows.append(row)
            lbs.append(r_e)  # Lower bound: distance >= r_e
    
    A = np.vstack(rows)  # Constraint matrix
    lb = np.array(lbs)   # Lower bounds
    ub = np.full_like(lb, np.inf, dtype=float)  # Upper bounds (unconstrained)
    return LinearConstraint(A, lb, ub)

def curvature_H(Np1, dim, lam=0.0):
    """
    Compute curvature regularization matrix for smoothing.
    
    Returns the Hessian matrix H = 2*lam*(I_dim ⊗ D^T D) where D is the
    second-difference operator. This penalizes high curvature in the control
    polygon, encouraging smoother curves.
    
    Args:
        Np1: Number of control points (N+1)
        dim: Spatial dimension
        lam: Regularization parameter (0 = no smoothing)
        
    Returns:
        np.ndarray: (Np1*dim, Np1*dim) regularization matrix
    """
    if lam <= 0: 
        return np.zeros((Np1*dim, Np1*dim))  # No regularization
    
    # Second-difference operator D: approximates second derivative
    D = np.zeros((Np1-2, Np1))
    for r in range(Np1-2):
        D[r, r:r+3] = [1, -2, 1]  # [P_{i-1} - 2*P_i + P_{i+1}]
    
    DtD = D.T @ D  # D^T D: penalty for second differences
    H = np.kron(np.eye(dim), 2*lam*DtD)  # Kronecker product for all dimensions
    return H

def optimize_bezier_outside_sphere(P_init, n_seg=8, r_e=6.0, 
                                   max_iter=20, tol=1e-6, lam_smooth=0.0, 
                                   verbose=False, keep_last_normal=True):
    """
    Optimize Bézier curve to avoid sphere obstacle using iterative optimization.
    
    Uses a fixed-point iteration approach where each iteration:
    1. Computes normal vectors for each segment
    2. Builds linear constraints for sphere avoidance
    3. Solves optimization problem with SLSQP
    
    The endpoints P0 and PN are kept fixed during optimization.
    
    Args:
        P_init: Initial control points of shape (N+1, dim)
        n_seg: Number of segments for constraint evaluation
        r_e: Sphere radius to avoid
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for parameter changes
        lam_smooth: Curvature regularization parameter
        verbose: Whether to print iteration progress
        keep_last_normal: Whether to reuse previous normals when undefined
        
    Returns:
        tuple: (optimized_control_points, info_dict) where info contains:
               - iter: Number of iterations performed
               - feasible: Whether final solution satisfies constraints
               - min_radius: Minimum distance from curve to origin
    """
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1
    A_list = segment_matrices_equal_params(N, n_seg)

    # Set up regularization matrix for curvature smoothing
    Hs = curvature_H(Np1, dim, lam=lam_smooth)
    x0 = P.reshape(-1)  # Flatten control points for optimization

    # Set up bounds: fix endpoints, allow interior points to move
    lb = x0.copy(); ub = x0.copy()
    lb[:] = -np.inf; ub[:] =  np.inf  # Default: unconstrained
    lb[:dim] = ub[:dim] = x0[:dim]       # Lock first control point P0
    lb[-dim:] = ub[-dim:] = x0[-dim:]    # Lock last control point PN
    bounds = Bounds(lb, ub)

    # Initialize iteration state
    normals_prev = None  # Store previous normals for fallback
    info = {"iter": 0, "feasible": False, "min_radius": None}

    def obj(x):
        """Objective function: minimize deviation from initial + curvature penalty."""
        dx = x - x0  # Deviation from initial control points
        return 0.5 * dx @ dx + 0.5 * (x @ (Hs @ x))  # L2 penalty + curvature

    def grad(x):
        """Gradient of the objective function."""
        return (x - x0) + (Hs @ x)  # Gradient of L2 + curvature terms

    for it in range(1, max_iter+1):
        P_now = P
        normals = []
        eps = 1e-12  # Small value to avoid division by zero
        
        # Compute normal vectors for each segment
        for Ai in A_list:
            Qi = Ai @ P_now  # Control points of segment i
            ci = Qi.mean(axis=0)  # Centroid of segment i
            
            # Normalize centroid to get outward normal (if not too close to origin)
            if np.linalg.norm(ci) > 1e-9:
                n = ci / (np.linalg.norm(ci) + eps)
            else:
                n = None  # Undefined normal (too close to origin)
            
            # Fallback strategies for undefined normals
            if n is None and normals_prev is not None and keep_last_normal:
                n = normals_prev[len(normals)]  # Reuse previous normal
            if n is None:
                # Use direction from start to end as fallback
                v = P_now[-1] - P_now[0]
                n = v / (np.linalg.norm(v) + eps)
            normals.append(n)

        # Build linear constraints for current normals
        lin_con = build_linear_constraint_matrix(A_list, normals, dim, r_e)

        # Solve optimization problem with SLSQP
        res = minimize(obj, P_now.reshape(-1), method="SLSQP",
                       jac=grad, constraints=[lin_con], bounds=bounds,
                       options=dict(maxiter=200, ftol=1e-12, disp=False))
        x_new = res.x
        P_new = x_new.reshape(Np1, dim)

        # Check convergence
        delta = np.linalg.norm(P_new - P)
        P = P_new
        normals_prev = normals  # Store for next iteration
        
        if verbose:
            print(f"[iter {it}] Δ={delta:.3e}  success={res.success}  status={res.status}")

        if delta < tol:
            break  # Converged

    # Final feasibility check by dense sampling of the curve
    curve = BezierCurve(P)
    ts = np.linspace(0, 1, 1000)  # Dense parameter sampling
    pts = np.array([curve.point(t) for t in ts])
    radii = np.linalg.norm(pts, axis=1)  # Distance from origin for each point
    min_r = float(np.min(radii))
    
    # Update info with final results
    info.update({"iter": it, "feasible": bool(min_r >= r_e - 1e-6), "min_radius": min_r})
    return P, info

# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities for 3D plotting and analysis
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
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(0.0)] if len(A_list) == 1 else [cmap(i/(len(A_list)-1)) for i in range(len(A_list))]

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
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=3)

    # Create 3D line collection for efficient rendering
    lc = Line3DCollection(lines, colors=cols, linewidths=lw, alpha=0.95)
    ax.add_collection3d(lc)

def plot_initial_guess(ax, P_init, linestyle=':', color='0.5', lw=1.6):
    """
    Plot the initial curve as a reference overlay.
    
    Args:
        ax: 3D matplotlib axes
        P_init: Initial control points
        linestyle: Line style for the overlay
        color: Color for the overlay
        lw: Line width
    """
    init_curve = BezierCurve(P_init)
    _, Ppts = init_curve.sample(300)  # Dense sampling for smooth curve
    ax.plot(Ppts[:,0], Ppts[:,1], Ppts[:,2], linestyle=linestyle, color=color, lw=lw)

# ─────────────────────────────────────────────────────────────────────────────
# Demonstration: Multi-segment optimization with visualization
# Creates a 2×3 subplot showing optimization results for different segment counts
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Demonstration of Bézier curve optimization with sphere avoidance.
    
    Creates a 2×3 subplot showing optimization results for different numbers
    of segments, demonstrating how more segments lead to better constraint
    satisfaction and smoother curves.
    """
    np.random.seed(0)  # For reproducible results
    r_e = 6.0  # Sphere radius to avoid

    # Define a quadratic Bézier curve (degree 2) in 3D
    P0 = np.array([ 4.0, -7.0,  0.0])  # Start point
    P2 = np.array([ -5.0,  7.0,  3.0])  # End point
    P1 = np.array([  -1.0,   -1.0,  2.0])  # Middle control point (slightly away from origin)
    P_init = np.vstack([P0, P1, P2])  # (3,3) control points for degree-2 curve

    # Test different numbers of segments
    split_list = [2, 4, 8, 16, 32, 64]
    results = []
    
    print("Optimizing Bézier curves with different segment counts...")
    for n_seg in split_list:
        P_opt, info = optimize_bezier_outside_sphere(
            P_init, n_seg=n_seg, r_e=r_e, max_iter=120, tol=1e-8,
            lam_smooth=1e-6, verbose=False
        )
        results.append((n_seg, P_opt, info))
        print(f"n_seg={n_seg:>2d} | iter={info['iter']:>3d} | min_radius={info['min_radius']:.3f} | feasible={info['feasible']}")

    # Create 2×3 subplot layout for visualization
    fig, axes = plt.subplots(2, 3, subplot_kw={'projection': '3d'}, figsize=(7.0, 6.4))
    axes = axes.ravel()

    for i, (n_seg, P_opt, info) in enumerate(results):
        ax = axes[i]
        add_wire_sphere(ax, radius=r_e, color='#f38d0e', alpha=0.22, resolution=28)
        plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', show_seg_ctrl_if=8, lw=2.1)

        # Overlay initial guess only on the first subplot (2 segments)
        if i == 0 and n_seg == 2:
            plot_initial_guess(ax, P_init, linestyle=':', color='0.5', lw=1.8)

        # Set up the view
        set_axes_equal_around(ax, center=(0,0,0), radius=r_e*1.25, pad=0.04)
        set_isometric(ax, elev=35.264, azim=45.0, ortho=True)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)

        ax.set_title(f"{n_seg} segs   |   min |p| = {info['min_radius']:.2f}", fontsize=9, pad=3)
        ax.set_xlabel('X', labelpad=4, fontsize=9)
        ax.set_ylabel('Y', labelpad=4, fontsize=9)
        ax.set_zlabel('Z', labelpad=4, fontsize=9)

    # Adjust layout for publication-quality figure
    fig.subplots_adjust(left=0.03, right=0.98, top=0.94, bottom=0.04, wspace=0.02, hspace=0.02)

    # Uncomment to save high-resolution figure
    # plt.savefig("bezier_outside_sphere_2x3.png", dpi=450, bbox_inches="tight", pad_inches=0.02)

    plt.show()
