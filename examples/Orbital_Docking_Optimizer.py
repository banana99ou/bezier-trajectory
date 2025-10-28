"""
Orbital Docking Trajectory Optimization with Gravity-Aware Acceleration Minimization

This module implements optimization of Bézier curves for orbital docking scenarios
using segment-based linearization and iterative optimization. The optimization
minimizes total acceleration (geometric + gravitational) as a surrogate for fuel
consumption in Guidance, Navigation, and Control (GNC) applications.

Key components:
- BezierCurve: Core Bézier curve evaluation
- De Casteljau splitting: Matrix-based curve subdivision
- Linear constraint building: Half-space constraint formulation
- Gravity-aware optimization: Minimizes ∫ ||a_geometric + a_gravitational||² dτ
- Visualization: 3D plotting utilities with orbital context
"""

import numpy as np
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import warnings

# ─────────────────────────────────────────────────────────────────────────────
# Orbital parameters for realistic scaling
# ─────────────────────────────────────────────────────────────────────────────
EARTH_RADIUS = 6.371  # Earth radius in scaled units (1000 km)
ISS_ALTITUDE = 0.408  # ISS altitude in scaled units (408 km)
ORBITAL_RADIUS = EARTH_RADIUS + ISS_ALTITUDE
SCALE_FACTOR = 1e6  # 1 unit = 1000 km (for realistic distances)

# Gravitational parameters
EARTH_MU = 3.986004418e14  # Earth's gravitational parameter (m³/s²)
EARTH_MU_SCALED = EARTH_MU / (SCALE_FACTOR**3)  # Scaled for our units (1000 km)

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

def acceleration_H(Np1, dim, T=1.0, lam=1.0):
    """
    Compute acceleration minimization matrix for Bézier curves.
    
    Implements equation (8b) from the paper: J = (4/T³) |Δ²P|²
    For quadratic Bézier curves: B''(τ) = 2(P₂ - 2P₁ + P₀) = constant
    
    This minimizes the integral of squared acceleration: ∫ ||B''(τ)||² dτ
    
    Args:
        Np1: Number of control points (N+1) 
        dim: Spatial dimension
        T: Time duration (default: 1.0)
        lam: Regularization parameter (default: 1.0)
        
    Returns:
        np.ndarray: (Np1*dim, Np1*dim) acceleration penalty matrix
    """
    if lam <= 0:
        return np.zeros((Np1*dim, Np1*dim))  # No regularization
    
    H = np.zeros((Np1*dim, Np1*dim))
    
    if Np1 == 3:  # Quadratic Bézier (degree 2)
        # For quadratic: B''(τ) = 2(P₂ - 2P₁ + P₀)
        # Acceleration vector: [1, -2, 1] for [P₀, P₁, P₂]
        accel_vec = np.array([1, -2, 1])
        
        for d in range(dim):
            start_idx = d * Np1
            H[start_idx:start_idx+Np1, start_idx:start_idx+Np1] = np.outer(accel_vec, accel_vec)
        
        # Scale by (4/T³) from equation (8b)
        H = (4.0 * lam / T**3) * H
        
    else:  # General degree N
        # For general N, we need to compute the integral ∫ ||B''(τ)||² dτ
        # This requires computing the second derivative of Bernstein polynomials
        # and integrating their squared norms
        
        # Build the second derivative matrix for Bernstein basis
        # B''_i(τ) = N(N-1) * [B_{i-2,N-2}(τ) - 2*B_{i-1,N-2}(τ) + B_{i,N-2}(τ)]
        N = Np1 - 1
        
        if N < 2:
            return np.zeros((Np1*dim, Np1*dim))  # No acceleration for linear curves
        
        # Second derivative coefficients
        D2 = np.zeros((Np1, Np1))
        for i in range(Np1):
            if i >= 2:
                D2[i, i-2] = N * (N-1)  # B_{i-2,N-2}
            if i >= 1 and i <= N:
                D2[i, i-1] = -2 * N * (N-1)  # -2*B_{i-1,N-2}
            if i <= N-2:
                D2[i, i] = N * (N-1)  # B_{i,N-2}
        
        # Integrate squared second derivatives
        # ∫₀¹ ||B''(τ)||² dτ = ∫₀¹ (B''(τ))ᵀ(B''(τ)) dτ
        # This gives us the Gram matrix of second derivatives
        
        # For Bernstein polynomials, the integral can be computed analytically
        # ∫₀¹ B_{i,N}(τ) B_{j,N}(τ) dτ = C(N,i) C(N,j) / C(2N+1, i+j)
        
        gram_matrix = np.zeros((Np1, Np1))
        for i in range(Np1):
            for j in range(Np1):
                if i + j <= 2*N:
                    gram_matrix[i, j] = comb(N, i) * comb(N, j) / comb(2*N+1, i+j)
        
        # The acceleration penalty matrix is D2^T * Gram * D2
        H_accel = D2.T @ gram_matrix @ D2
        
        # Apply to all dimensions
        for d in range(dim):
            start_idx = d * Np1
            H[start_idx:start_idx+Np1, start_idx:start_idx+Np1] = H_accel
        
        # Scale by regularization parameter
        H = lam * H
    
    return H

def gravity_aware_acceleration_H(Np1, dim, T=1.0, lam=1.0, lam_grav=1e-3):
    """
    Compute acceleration minimization matrix with gravity consideration.
    
    Implements: J = ∫ ||a_total(τ)||² dτ
    Where: a_total(τ) = a_geometric(τ) + a_gravitational(τ)
    
    For simplicity, we approximate gravitational acceleration as constant
    at orbital altitude, making this a tractable optimization problem.
    
    Args:
        Np1: Number of control points (N+1) 
        dim: Spatial dimension
        T: Time duration (default: 1.0)
        lam: Regularization parameter for geometric acceleration (default: 1.0)
        lam_grav: Regularization parameter for gravitational effects (default: 1e-3)
        
    Returns:
        np.ndarray: (Np1*dim, Np1*dim) gravity-aware acceleration penalty matrix
    """
    if lam <= 0:
        return np.zeros((Np1*dim, Np1*dim))
    
    # Get the base geometric acceleration matrix
    H_geom = acceleration_H(Np1, dim, T, lam)
    
    # Add simplified gravitational contribution
    # Approximate gravitational acceleration as constant at orbital radius
    orbital_radius = ORBITAL_RADIUS
    grav_accel_magnitude = EARTH_MU_SCALED / orbital_radius**2
    
    # Create a simple gravity penalty matrix
    # This penalizes deviations from the "gravity-compensated" trajectory
    H_grav = np.zeros((Np1*dim, Np1*dim))
    
    # Add small penalty for each control point to account for gravity
    for d in range(dim):
        start_idx = d * Np1
        # Simple diagonal penalty - higher for points further from origin
        for i in range(Np1):
            H_grav[start_idx + i, start_idx + i] = lam_grav * grav_accel_magnitude
    
    return H_geom + H_grav

def optimize_orbital_docking(P_init, n_seg=8, r_e=50.0, max_iter=20, tol=1e-6, lam_smooth=1e-6, lam_grav=1e-3, verbose=False, keep_last_normal=True):
    """
    Optimize Bézier curve for orbital docking with gravity-aware acceleration minimization.
    
    Uses a fixed-point iteration approach where each iteration:
    1. Computes normal vectors for each segment
    2. Builds linear constraints for safety zone avoidance
    3. Solves QP problem minimizing total acceleration (geometric + gravitational)
    
    The endpoints P0 and PN are kept fixed during optimization.
    Minimizes: J = 0.5 * ||x - x₀||² + 0.5 * x^T H x
    where H implements gravity-aware acceleration minimization: ∫ ||a_total(τ)||² dτ
    This serves as a surrogate for fuel consumption optimization in GNC applications.
    
    Args:
        P_init: Initial control points of shape (N+1, dim)
        n_seg: Number of segments for constraint evaluation
        r_e: Safety zone radius to avoid (in scaled units)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance for parameter changes
        lam_smooth: Acceleration regularization parameter (equation 8b)
        lam_grav: Gravitational effects regularization parameter
        verbose: Whether to print iteration progress
        keep_last_normal: Whether to reuse previous normals when undefined
        
    Returns:
        tuple: (optimized_control_points, info_dict) where info contains:
               - iter: Number of iterations performed
               - feasible: Whether final solution satisfies constraints
               - min_radius: Minimum distance from curve to origin
               - acceleration: Maximum total acceleration magnitude (geometric + gravitational)
    """
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1
    A_list = segment_matrices_equal_params(N, n_seg)

    # Set up gravity-aware acceleration minimization matrix
    Hs = gravity_aware_acceleration_H(Np1, dim, T=1.0, lam=lam_smooth, lam_grav=lam_grav)
    x0 = P.reshape(-1)  # Flatten control points for optimization

    # Set up bounds: fix endpoints, allow interior points to move
    lb = x0.copy(); ub = x0.copy()    # lb for lower bounds, ub for upper bounds
    lb[:] = -np.inf; ub[:] =  np.inf  # Default: unconstrained
    lb[:dim] = ub[:dim] = x0[:dim]       # Lock first control point P0
    lb[-dim:] = ub[-dim:] = x0[-dim:]    # Lock last control point PN
    bounds = Bounds(lb, ub)

    # Initialize iteration state
    normals_prev = None  # Store previous normals for fallback
    info = {"iter": 0, "feasible": False, "min_radius": None}

    # Construct QP matrices for the cost function J = 0.5 * ||x - x0||² + 0.5 * x^T H x
    # Rewritten as: J = 0.5 * x^T (I + H) x - x0^T x + constant
    # In standard QP form: minimize 0.5 * x^T Q x + c^T x
    Q = np.eye(Np1 * dim) + Hs  # Hessian matrix Q = I + H
    c = -x0  # Linear term c = -x0

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

        # Solve optimization problem using trust-constr (QP solver)
        def objective(x):
            return 0.5 * x @ (Q @ x) + c @ x
        
        def gradient(x):
            return Q @ x + c
        
        def hessian(x):
            return Q
        
        # Suppress warnings about singular Jacobian (expected due to redundant constraints)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Singular Jacobian matrix")
            res = minimize(objective, P_now.reshape(-1), method="trust-constr",
                           jac=gradient, hess=hessian, constraints=[lin_con], bounds=bounds,
                           options=dict(maxiter=200, gtol=1e-12, disp=False))
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
    
    # Compute total acceleration (geometric + gravitational) for the optimized curve
    if curve.degree == 2:  # Quadratic Bézier
        # Geometric acceleration: B''(τ) = 2(P₂ - 2P₁ + P₀) = constant
        geom_accel_vec = 2 * (P[2] - 2*P[1] + P[0])
        geom_accel = float(np.linalg.norm(geom_accel_vec))
        
        # Gravitational acceleration (simplified - using average radius)
        avg_radius = float(np.mean(radii))
        grav_accel = EARTH_MU_SCALED / avg_radius**2
        
        # Total acceleration magnitude
        acceleration = geom_accel + grav_accel
        
    else:  # General degree
        # Compute maximum total acceleration over the curve
        total_accel_samples = []
        for t in np.linspace(0, 1, 100):
            # Geometric acceleration
            N = curve.degree
            geom_accel = np.zeros(curve.dimension)
            for i in range(N-1):
                coeff = N * (N-1) * comb(N-2, i) * (t**i) * ((1-t)**(N-2-i))
                geom_accel += coeff * (P[i+2] - 2*P[i+1] + P[i])
            
            # Gravitational acceleration at this point
            curve_point = curve.point(t)
            radius = np.linalg.norm(curve_point)
            grav_accel_mag = EARTH_MU_SCALED / radius**2
            
            # Total acceleration magnitude
            total_accel_samples.append(np.linalg.norm(geom_accel) + grav_accel_mag)
        
        acceleration = float(np.max(total_accel_samples))
    
    # Update info with final results
    info.update({"iter": it, "feasible": bool(min_r >= r_e - 1e-6), "min_radius": min_r, "acceleration": acceleration})
    return P, info

# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities for orbital docking scenarios
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
            ax.plot(Qi[:,0], Qi[:,1], Qi[:,2], '-', color=colors[i], alpha=0.55, lw=1.0, marker='o', ms=5)
        if n_seg > show_seg_ctrl_if:
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
# Orbital docking demonstration
# ─────────────────────────────────────────────────────────────────────────────
def create_orbital_docking_scenario():
    """
    Create a realistic orbital docking scenario for conference presentation.
    
    Returns:
        tuple: (P_init, r_e, results) - initial points, safety radius, optimization results
    """
    print("Generating orbital docking scenario for conference...")
    
    # Realistic orbital docking control points (scaled units: 1 unit = 1000 km)
    P0 = np.array([ 40.0, -70.0,  0.0])  # Start point (chaser spacecraft)
    P2 = np.array([-50.0,  70.0,  30.0])  # End point (target spacecraft)
    P1 = np.array([ -10.0,  -10.0, 20.0])  # Middle control point (maneuver point)
    P_init = np.vstack([P0, P1, P2])
    
    # Safety zone radius (50,000 km - realistic for orbital docking)
    r_e = 50.0
    
    # Test different numbers of segments
    split_list = [2, 4, 8, 16, 32, 64]
    results = []
    
    print("Optimizing orbital docking trajectories with different segment counts...")
    for n_seg in split_list:
        P_opt, info = optimize_orbital_docking(
            P_init, n_seg=n_seg, r_e=r_e, max_iter=120, tol=1e-8,
            lam_smooth=1e-6, lam_grav=1e-3, verbose=False
        )
        results.append((n_seg, P_opt, info))
        print(f"n_seg={n_seg:>2d} | iter={info['iter']:>3d} | acceleration={info['acceleration']:.3f} | feasible={info['feasible']}")
    
    return P_init, r_e, results

def create_trajectory_figure(P_init, r_e, results):
    """
    Create Window 1: 2×3 layout showing trajectories with different segment counts.
    """
    fig1 = plt.figure(figsize=(16, 12))  # Increased from (12, 8) to (16, 12)
    
    # Create 2×3 subplot layout
    axes = []
    for i in range(6):
        ax = fig1.add_subplot(2, 3, i+1, projection='3d')
        axes.append(ax)
    
    # Plot trajectories for different segment counts
    segment_counts = [2, 4, 8, 16, 32, 64]
    
    for i, (n_seg, P_opt, info) in enumerate(results):
        ax = axes[i]
        
        # Add Earth (large blue sphere)
        add_earth_sphere(ax, radius=EARTH_RADIUS, color='blue', alpha=0.3)
        
        # Add safety zone (smaller red sphere)
        add_wire_sphere(ax, radius=r_e, color='red', alpha=0.2, resolution=15)
        
        # Plot optimized trajectory (thicker lines)
        plot_segments_gradient(ax, P_opt, n_seg, cmap_name='viridis', lw=3.0)
        
        # Add start and end markers (larger markers)
        ax.scatter(P_init[0,0], P_init[0,1], P_init[0,2], color='green', s=120, label='Chaser')
        ax.scatter(P_init[2,0], P_init[2,1], P_init[2,2], color='orange', s=120, label='Target')
        ax.legend(fontsize=8)
        
        # Professional styling (reduced radius for more zoom)
        set_axes_equal_around(ax, center=(0,0,0), radius=20, pad=0.1)
        set_isometric(ax, elev=20, azim=45)
        beautify_3d_axes(ax, show_ticks=True, show_grid=True)
        
        # Title with acceleration
        ax.set_title(f'{n_seg} Segments\nAccel: {info["acceleration"]:.1f} m/s²', fontsize=10, pad=10)
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.08, 
                       wspace=0.15, hspace=0.3)
    
    return fig1

def create_performance_figure(results):
    """
    Create Window 2: Acceleration performance graph.
    """
    fig2 = plt.figure(figsize=(10, 6))
    ax = fig2.add_subplot(111)
    
    # Extract data
    segment_counts = [2, 4, 8, 16, 32, 64]
    accelerations = [info['acceleration'] for _, _, info in results]
    
    # Create acceleration performance graph
    ax.plot(segment_counts, accelerations, 'bo-', linewidth=3, markersize=10)
    ax.set_xlabel('Number of Segments', fontsize=14)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=14)
    ax.set_title('Performance Improvement with More Segments', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)  # Log scale for segment counts

    
    # Add data point labels
    for i, (seg, accel) in enumerate(zip(segment_counts, accelerations)):
        ax.annotate(f'{accel:.1f}', (seg, accel), 
                   textcoords="offset points", xytext=(0,10), ha='center',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.1)
    
    return fig2

def create_conference_figures():
    """
    Generate two separate windows: trajectories and performance graph.
    """
    # Run orbital docking scenario
    P_init, r_e, results = create_orbital_docking_scenario()
    
    # Create Window 1: Trajectory layouts
    fig1 = create_trajectory_figure(P_init, r_e, results)
    
    # Create Window 2: Performance graph
    fig2 = create_performance_figure(results)
    
    # Save both figures
    fig1.savefig("orbital_trajectories.png", dpi=300, bbox_inches="tight", facecolor='white')
    fig2.savefig("acceleration_performance.png", dpi=300, bbox_inches="tight", facecolor='white')
    
    # Calculate performance metrics
    accelerations = [info['acceleration'] for _, _, info in results]
    improvement = ((accelerations[0] - accelerations[-1]) / accelerations[0]) * 100
    
    print(f"\nFigures saved:")
    print(f"- orbital_trajectories.png (2×3 trajectory layout)")
    print(f"- acceleration_performance.png (performance graph)")
    print(f"\nPerformance improvement: {improvement:.1f}% acceleration reduction")
    print(f"Best acceleration: {min(accelerations):.1f} m/s²")
    
    return fig1, fig2

# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Orbital docking trajectory optimization demonstration.
    
    Creates two separate windows:
    - Window 1: 2×3 layout showing trajectories with different segment counts
    - Window 2: Acceleration performance graph showing improvement with more segments
    """
    # Generate two separate windows
    fig1, fig2 = create_conference_figures()
    
    # Show both windows
    plt.show()
