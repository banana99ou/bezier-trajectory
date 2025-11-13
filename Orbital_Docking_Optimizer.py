#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.special import comb
from scipy.optimize import minimize, LinearConstraint, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from pathlib import Path
import warnings
import hashlib
import pickle
import os
import time
warnings.filterwarnings('ignore')

# Orbital parameters (from Project_Spec.md)
ISS_ALTITUDE_KM = 423  # ISS orbit altitude in km AMSL
CHASER_ALTITUDE_KM = 300  # Chaser altitude in km AMSL
KOZ_ALTITUDE_KM = 100  # Keep Out Zone altitude in km AMSL

EARTH_RADIUS_KM = 6371.0  # Earth radius in km
KOZ_RADIUS = EARTH_RADIUS_KM + KOZ_ALTITUDE_KM  # KOZ radius from Earth center
ISS_RADIUS = EARTH_RADIUS_KM + ISS_ALTITUDE_KM
CHASER_RADIUS = EARTH_RADIUS_KM + CHASER_ALTITUDE_KM

# Gravitational parameters
EARTH_MU = 3.986004418e14  # m³/s²

# Scaling: use km as base unit
SCALE_FACTOR = 1e3  # 1 unit = 1 km
EARTH_MU_SCALED = EARTH_MU / (SCALE_FACTOR**3)  # Scaled for km

USE_CACHE = True


def configure_custom_font(font_filename="NanumSquareR.otf"):
    """
    Configure Matplotlib to use a custom font if available.
    Uses fallback font for math symbols to fix negative sign rendering.
    """
    font_path = Path(__file__).resolve().parent / font_filename
    if not font_path.is_file():
        return

    try:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        
        # Use font with fallback to DejaVu Sans for better symbol rendering
        # This ensures minus signs and other symbols render correctly
        plt.rcParams["font.family"] = [font_name, "DejaVu Sans", "sans-serif"]
        
        # Use DejaVu Sans for math symbols to fix negative sign rendering
        plt.rcParams["mathtext.fontset"] = "dejavusans"
        plt.rcParams["mathtext.default"] = "regular"
        
        # Enable Unicode minus sign for better rendering in tick labels
        # This makes matplotlib use U+2212 (proper minus) instead of U+002D (hyphen-minus)
        plt.rcParams["axes.unicode_minus"] = True
        
        # The format_number() helper function ensures all manually formatted numbers
        # also use the proper Unicode minus sign (U+2212)
    except Exception:
        # Silently ignore font configuration errors to avoid breaking plots
        pass

def format_number(value, format_spec='.1f'):
    """
    Format a number with proper Unicode minus sign for better rendering.
    
    Args:
        value: Numeric value to format
        format_spec: Format specification (e.g., '.1f', '.2f')
    
    Returns:
        str: Formatted string with proper minus sign
    """
    if isinstance(value, (int, float)):
        if value < 0:
            # Use Unicode minus sign (U+2212) instead of hyphen-minus (U+002D)
            return '−' + format(abs(value), format_spec)
        else:
            return format(value, format_spec)
    return str(value)


configure_custom_font()

# ─────────────────────────────────────────────────────────────────────────────
# Caching utilities for optimization results
# ─────────────────────────────────────────────────────────────────────────────
FIGURE_DIR = Path(__file__).parent / "figure"
FIGURE_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_VERSION = "1.0"  # Increment to invalidate old caches

def get_cache_key(P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1):
    """
    Generate a deterministic cache key from optimization parameters.
    
    Args:
        P_init: Initial control points
        n_seg: Number of segments
        r_e: KOZ radius
        max_iter: Maximum iterations
        tol: Convergence tolerance
        sample_count: Number of samples for cost evaluation
        v0, v1: Velocity boundary conditions
        a0, a1: Acceleration boundary conditions
    
    Returns:
        str: Cache key (hex digest)
    """
    # Create a deterministic hash from all parameters
    hash_input = {
        'P_init': P_init.tobytes() if isinstance(P_init, np.ndarray) else str(P_init),
        'n_seg': n_seg,
        'r_e': float(r_e),
        'max_iter': max_iter,
        'tol': float(tol),
        'sample_count': sample_count,
        'v0': v0.tobytes() if v0 is not None and isinstance(v0, np.ndarray) else str(v0),
        'v1': v1.tobytes() if v1 is not None and isinstance(v1, np.ndarray) else str(v1),
        'a0': a0.tobytes() if a0 is not None and isinstance(a0, np.ndarray) else str(a0),
        'a1': a1.tobytes() if a1 is not None and isinstance(a1, np.ndarray) else str(a1),
        'version': CACHE_VERSION
    }
    
    # Convert to string and hash
    hash_str = str(sorted(hash_input.items()))
    return hashlib.md5(hash_str.encode()).hexdigest()

def get_cache_path(cache_key, n_seg):
    """
    Get the cache file path for a given cache key and segment count.
    
    Args:
        cache_key: Cache key (hex digest)
        n_seg: Number of segments
    
    Returns:
        Path: Path to cache file
    """
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"opt_{cache_key[:8]}_nseg{n_seg}.pkl"

def load_from_cache(cache_path):
    """
    Load optimization result from cache.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        tuple: (P_opt, info) if cache exists and is valid, None otherwise
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Validate cache structure
        if isinstance(data, dict) and 'P_opt' in data and 'info' in data:
            return data['P_opt'], data['info']
        else:
            # Old format compatibility
            return data
    except Exception as e:
        # If cache is corrupted, delete it
        try:
            cache_path.unlink()
        except:
            pass
        return None

def save_to_cache(cache_path, P_opt, info):
    """
    Save optimization result to cache.
    
    Args:
        cache_path: Path to cache file
        P_opt: Optimized control points
        info: Info dictionary
    """
    try:
        data = {
            'P_opt': P_opt,
            'info': info,
            'version': CACHE_VERSION
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        # Silently fail if cache write fails
        pass

def clear_cache(cache_key=None):
    """
    Clear cache files. If cache_key is provided, only clear that specific cache.
    
    Args:
        cache_key: Optional cache key to clear specific cache, or None to clear all
    """
    if not CACHE_DIR.exists():
        return
    
    if cache_key is None:
        # Clear all cache files
        for cache_file in CACHE_DIR.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass
    else:
        # Clear specific cache files
        for cache_file in CACHE_DIR.glob(f"opt_{cache_key[:8]}*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass


# In[47]:


def get_D_matrix(N):
    """
    Compute derivative matrix D for Bézier curve of degree N.

    From Project_Spec.md:
    [D]_i,j = N × { -1 if j=i, 1 if j=i+1, 0 otherwise }

    Args:
        N: Degree of Bézier curve

    Returns:
        D: (N, N+1) matrix
    """
    D = np.zeros((N, N+1))
    for i in range(N):
        D[i, i] = -N
        D[i, i+1] = N
    return D

def get_E_matrix(N):
    """
    Compute elevation matrix E for Bézier curve of degree N.
    Elevates degree from N to N+1.

    From Project_Spec.md equation:
    E_{N→N+1} with specific structure

    Args:
        N: Original degree

    Returns:
        E: (N+2, N+1) matrix
    """
    E = np.zeros((N+2, N+1))

    # First row: i=1, j=1 → 1
    E[0, 0] = 1.0

    # Last row: i=N+2, j=N+1 → 1
    E[N+1, N] = 1.0

    # Middle rows: 2 ≤ i ≤ N+1 (1-indexed) → rows 1 to N (0-indexed)
    for row_idx in range(1, N+1):  # 0-indexed row: 1 to N
        i_1idx = row_idx + 1  # Convert to 1-indexed for formula
        # j = i-1 (1-indexed) → col = row_idx - 1 (0-indexed)
        E[row_idx, row_idx - 1] = (N + 2 - i_1idx) / (N + 1)
        # j = i (1-indexed) → col = row_idx (0-indexed)
        E[row_idx, row_idx] = (i_1idx - 1) / (N + 1)

    return E


# In[48]:


class BezierCurve:
    """
    Bézier curve implementation using D/E matrices for derivatives.
    """

    def __init__(self, control_points):
        P = np.array(control_points, dtype=float)
        if P.ndim != 2: # cus np.array makes control_points a 2D matrix
            raise ValueError("control_points must be (N+1, dim)")
        self.control_points = P
        self.degree = P.shape[0] - 1 # = N
        self.dimension = P.shape[1] # P.shape = (N+1, dim)

        # Precompute D and E matrices
        self.D = get_D_matrix(self.degree)
        if self.degree > 0:
            self.E = get_E_matrix(self.degree - 1)  # E elevates from N-1 to N

    def point(self, tau):
        """Evaluate curve at parameter tau using Bernstein basis."""
        N, d = self.degree, self.dimension
        out = np.zeros(d)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * self.control_points[i]
        return out

    def velocity_control_points(self):
        """
        Compute velocity control points using V = EDP.
        Returns control points for velocity curve (degree N, N+1 control points).
        """
        if self.degree == 0:
            return np.zeros((1, self.dimension))

        # V = E @ D @ P
        # P is (N+1, dim), D is (N, N+1), E is (N+1, N)
        # Result: (N+1, dim)
        V_ctrl = self.E @ self.D @ self.control_points
        return V_ctrl

    def acceleration_control_points(self):
        """
        Compute acceleration control points using A = EDEDP.
        Returns control points for acceleration curve.
        """
        # Acceleration control points using EDEDP as instructed.
        if self.degree < 2:
            return np.zeros(((self.degree+1), self.dimension))
        # Use EDEDP directly per the analytic/formal instructions.
        A_ctrl = self.E @ self.D @ self.E @ self.D @ self.control_points
        # (Resulting shape: (N+1, d);
        return A_ctrl

    def velocity(self, tau):
        """Evaluate velocity at parameter tau."""
        if self.degree == 0:
            return np.zeros(self.dimension)
        V_ctrl = self.velocity_control_points()
        # Velocity curve has degree N (same as original)
        N = self.degree
        out = np.zeros(self.dimension)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * V_ctrl[i]
        return out

    def acceleration(self, tau):
        """Evaluate acceleration at parameter tau."""
        if self.degree < 2:
            return np.zeros(((self.degree+1), self.dimension))
        A_ctrl = self.acceleration_control_points()
        # Acceleration curve has degree N due to the E matrix
        N_accel = self.degree
        out = np.zeros(self.dimension)
        for i in range(N_accel + 1):
            b = comb(N_accel, i) * (tau ** i) * ((1 - tau) ** (N_accel - i))
            out += b * A_ctrl[i]
        return out


# In[49]:


def de_casteljau_split_1d(N, tau, basis_index):
    """
    Compute De Casteljau subdivision coefficients for a single basis vector.
    """
    w = np.zeros(N+1)
    w[basis_index] = 1.0
    left = [w[0]]
    right = [w[-1]]
    W = w.copy()

    for _ in range(1, N+1):
        W = (1 - tau) * W[:-1] + tau * W[1:]
        left.append(W[0])
        right.append(W[-1])

    L = np.array(left)
    R = np.array(right[::-1])
    return L, R

def de_casteljau_split_matrices(N, tau):
    """Compute subdivision matrices S_left and S_right."""
    S_left = np.zeros((N+1, N+1))
    S_right = np.zeros((N+1, N+1))

    for j in range(N+1):
        L, R = de_casteljau_split_1d(N, tau, j)
        S_left[:, j] = L
        S_right[:, j] = R
    return S_left, S_right

def segment_matrices_equal_params(N, n_seg):
    """
    Generate segment matrices for equal-parameter splitting.
    Returns list of (N+1, N+1) matrices, one per segment.
    """
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


# In[50]:


def build_koz_constraints(A_list, P, r_e, dim=3, c_KOZ=None):
    """
    Build KOZ (Keep Out Zone) linear constraints for all segments.

    For each segment j:
    1. Compute CG (centroid) of control points: Qi = Ai @ P
    2. Generate unit vector nj from KOZ center (c_KOZ) to CG
    3. Create half-space constraint: nj^T @ Qi >= r_e

    Args:
        A_list: List of segment transformation matrices
        P: Control points (N+1, dim)
        r_e: KOZ radius
        dim: Spatial dimension
        c_KOZ: KOZ center, shape (dim,) or None (if None, uses origin and warns)

    Returns:
        LinearConstraint object
    """
    Np1 = A_list[0].shape[1]
    rows, lbs = [], []

    if c_KOZ is None:
        import warnings
        warnings.warn("c_KOZ (KOZ center) not specified; defaulting to origin.")
        c_KOZ = np.zeros(dim)

    for Ai in A_list:
        Qi = Ai @ P  # Control points of segment i
        ci = Qi.mean(axis=0)  # Centroid (CG)

        # Unit vector from KOZ center to CG
        Nj = ci - c_KOZ
        Nj_norm = np.linalg.norm(Nj)
        if Nj_norm < 1e-12:
            # If CG coincides with KOZ center, skip this segment (degenerate case)
            continue

        nj = Nj / Nj_norm  # Unit normal vector

        # Create constraint for each control point in segment
        for k in range(Np1):
            row = np.zeros(Np1 * dim)
            for j in range(Np1):
                coeff = Ai[k, j]
                start = j * dim
                row[start:start+dim] += coeff * nj
            rows.append(row)
            lbs.append(r_e)

    if len(rows) == 0:
        # No constraints generated
        A_const = np.zeros((1, Np1 * dim))
        lb_const = np.array([-np.inf])
        ub_const = np.array([np.inf])
    else:
        A_const = np.vstack(rows)
        lb_const = np.array(lbs)
        ub_const = np.full_like(lb_const, np.inf)

    return LinearConstraint(A_const, lb_const, ub_const)

def build_boundary_constraints(P_init, v0=None, v1=None, a0=None, a1=None, dim=3):
    """
    Build boundary condition equality constraints.

    Args:
        P_init: Initial control points (N+1, dim)
        v0: Initial velocity (optional, shape (dim,))
        v1: Final velocity (optional, shape (dim,))
        a0: Initial acceleration (optional, shape (dim,))
        a1: Final acceleration (optional, shape (dim,))
        dim: Spatial dimension

    Returns:
        list of LinearConstraint objects
    """
    Np1 = P_init.shape[0]
    N = Np1 - 1

    constraints = []

    # Position constraints: p(0) = P0, p(1) = PN
    # These are handled via bounds, not equality constraints
    # (we'll set bounds separately)

    # Velocity constraints
    if v0 is not None:
        # v(0) = N(P1 - P0)
        # N*P1 - N*P0 = v0
        row = np.zeros(Np1 * dim)
        row[0:dim] = -N  # -N * P0
        row[dim:2*dim] = N  # N * P1
        constraints.append(LinearConstraint(row.reshape(1, -1), v0, v0))

    if v1 is not None:
        # v(1) = N(PN - PN-1)
        # N*PN - N*PN-1 = v1
        row = np.zeros(Np1 * dim)
        row[-2*dim:-dim] = -N  # -N * P_{N-1}
        row[-dim:] = N  # N * PN
        constraints.append(LinearConstraint(row.reshape(1, -1), v1, v1))

    # Acceleration constraints
    if a0 is not None and N >= 2:
        # a(0) = N(N-1)(P2 - 2P1 + P0)
        row = np.zeros(Np1 * dim)
        row[0:dim] = N*(N-1)  # N(N-1) * P0
        row[dim:2*dim] = -2*N*(N-1)  # -2N(N-1) * P1
        row[2*dim:3*dim] = N*(N-1)  # N(N-1) * P2
        constraints.append(LinearConstraint(row.reshape(1, -1), a0, a0))

    if a1 is not None and N >= 2:
        # a(1) = N(N-1)(PN - 2PN-1 + PN-2)
        row = np.zeros(Np1 * dim)
        row[-3*dim:-2*dim] = N*(N-1)  # N(N-1) * P_{N-2}
        row[-2*dim:-dim] = -2*N*(N-1)  # -2N(N-1) * P_{N-1}
        row[-dim:] = N*(N-1)  # N(N-1) * PN
        constraints.append(LinearConstraint(row.reshape(1, -1), a1, a1))

    return constraints


# In[51]:


def _compute_cost_only(P_flat, Np1, dim, n_samples=50):
    """Helper function to compute cost only (no recursion)."""
    P = P_flat.reshape(Np1, dim)
    curve = BezierCurve(P)
    ts = np.linspace(0, 1, n_samples)

    cost = 0.0
    accel = 0.0
    dtau = 1.0 / (n_samples - 1)

    for tau in ts:
        # Geometric acceleration at each point on the curve
        a_geom = curve.acceleration(tau)

        # Gravitational acceleration at each point on the curve
        pos = curve.point(tau)
        r = np.linalg.norm(pos)

        # Direction toward origin (negative of position unit vector)
        if r > 1e-6:
            a_grav = -EARTH_MU_SCALED / r**2 * (pos / r)
        else:
            a_grav = np.zeros_like(pos)

        # Cost: ||a_geom - a_grav||²
        diff = a_geom - a_grav
        norm_diff = np.linalg.norm(diff)
        cost += norm_diff**2 * dtau
        accel += norm_diff * dtau

    return cost, accel, a_geom, a_grav

def cost_function_gradient_hessian(P_flat, Np1, dim, n_samples=50, compute_grad=True):
    """
    Compute cost function, gradient, and Hessian approximation.
    Returns: (cost, gradient, hessian)

    Args:
        compute_grad: If False, only compute cost (for efficiency when gradient not needed)
    """
    cost, _, _, _ = _compute_cost_only(P_flat, Np1, dim, n_samples)

    if not compute_grad:
        return cost, None, None

    # Compute gradient using finite differences
    eps = 1e-6
    grad = np.zeros(Np1 * dim)
    for i in range(Np1 * dim):
        P_pert = P_flat.copy()
        P_pert[i] += eps
        cost_pert, _, _, _ = _compute_cost_only(P_pert, Np1, dim, n_samples)
        grad[i] = (cost_pert - cost) / eps

    # Simple diagonal Hessian approximation (regularization)
    hess = np.eye(Np1 * dim) * 1e-6

    return cost, grad, hess


# In[52]:


def optimize_orbital_docking(
    P_init, 
    n_seg=8, 
    r_e=KOZ_RADIUS, 
    max_iter=20,
    tol=1e-6,
    v0=None,
    v1=None,
    a0=None,
    a1=None,
    sample_count=100,
    verbose=True,
    use_cache=True
):
    """
    Optimize Bézier curve for orbital docking.

    Args:
        P_init: Initial control points (N+1, dim)
        n_seg: Number of segments for KOZ linearization
        r_e: KOZ radius
        max_iter: Max optimization iterations
        tol: Convergence tolerance
        v0, v1: Velocity boundary conditions
        a0, a1: Acceleration boundary conditions
        sample_count: Number of samples for cost evaluation
        verbose: Print progress
        use_cache: Whether to use cache (default: True)

    Returns:
        (P_opt, info_dict)
    """
    t0 = time.time()
    # Check cache first
    if use_cache:
        cache_key = get_cache_key(P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1)
        cache_path = get_cache_path(cache_key, n_seg)
        cached_result = load_from_cache(cache_path)
        
        if cached_result is not None:
            P_opt, info = cached_result
            if verbose:
                print(f"[Cache hit] Loaded result for n_seg={n_seg}")
            return P_opt, info
        elif verbose:
            print(f"[Cache miss] Running optimization for n_seg={n_seg}...")
    
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1

    # Get segment matrices
    A_list = segment_matrices_equal_params(N, n_seg)

    # Set up bounds: fix endpoints for position constraints
    x0 = P.reshape(-1)
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[:dim] = ub[:dim] = x0[:dim]  # Lock P0
    lb[-dim:] = ub[-dim:] = x0[-dim:]  # Lock PN
    bounds = Bounds(lb, ub)

    # Build boundary constraints
    boundary_constraints = build_boundary_constraints(P, v0, v1, a0, a1, dim)

    # Store results
    info = {"iterations": 0, "feasible": False, "cost": np.inf}

    # Iterative optimization: update KOZ constraints based on current solution
    for it in range(1, max_iter + 1):
        # Build KOZ constraints based on current control points
        koz_constraint = build_koz_constraints(A_list, P, r_e, dim)

        # Combine all constraints
        all_constraints = [koz_constraint] + boundary_constraints

        # Objective function
        def objective(x):
            cost, _, _, _ = _compute_cost_only(x, Np1, dim, n_samples=sample_count)
            return cost

        # Gradient - compute gradient separately
        def gradient(x):
            _, grad, _ = cost_function_gradient_hessian(x, Np1, dim, compute_grad=True, n_samples=sample_count)
            return grad

        # Solve
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = minimize(
                objective,
                P.reshape(-1),
                method='trust-constr',
                jac=gradient,
                constraints=all_constraints,
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
            )

        P_new = res.x.reshape(Np1, dim)
        delta = np.linalg.norm(P_new - P)
        P = P_new

        if verbose:
            print(f"Iter {it}: cost={res.fun:.6e}, delta={delta:.6e}")

        if delta < tol:
            break

    # Final feasibility check
    curve = BezierCurve(P)
    ts_check = np.linspace(0, 1, 1000)
    pts = np.array([curve.point(t) for t in ts_check])
    radii = np.linalg.norm(pts, axis=1)
    min_radius = float(np.min(radii))

    _, accel, _, _ = _compute_cost_only(P.reshape(-1), Np1, dim, n_samples=sample_count)

    info.update({
        "iterations": it,
        "feasible": min_radius >= r_e - 1e-6,
        "min_radius": min_radius,
        "cost": res.fun,
        "accel": accel,
        "elapsed_time": time.time() - t0
    })

    # Save to cache
    if use_cache:
        cache_key = get_cache_key(P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1)
        cache_path = get_cache_path(cache_key, n_seg)
        save_to_cache(cache_path, P, info)
        if verbose:
            print(f"[Cache saved] Result saved for n_seg={n_seg}")

    return P, info


# In[53]:


# Professional visualization helpers
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
    base_colors = ['#E74C3C', '#3498DB', '#F39C12'] # (red, blue, orange)
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

def optimize_all_segment_counts(P_init, r_e=KOZ_RADIUS, segment_counts=[2, 4, 8, 16, 32, 64], 
                                max_iter=120, tol=1e-8, verbose=True, use_cache=True):
    """
    Run optimization for multiple segment counts and return results.
    Uses caching to speed up repeated runs.

    Args:
        P_init: Initial control points (N+1, dim)
        r_e: KOZ radius
        segment_counts: List of segment counts to test
        max_iter: Maximum iterations per optimization
        tol: Convergence tolerance
        verbose: Print progress
        use_cache: Whether to use cache (default: True)

    Returns:
        List of tuples: [(n_seg, P_opt, info), ...] where info includes acceleration
    """
    results = []
    cache_hits = 0
    cache_misses = 0

    print(f"Optimizing trajectories for segment counts: {segment_counts}")
    if use_cache:
        print("Cache: ENABLED")
    else:
        print("Cache: DISABLED")
    print("=" * 60)

    for n_seg in segment_counts:
        if verbose:
            print(f"\nProcessing n_seg={n_seg}...")

        # Check if cached before running
        was_cached = False
        if use_cache:
            cache_key = get_cache_key(P_init, n_seg, r_e, max_iter, tol, 100, None, None, None, None)
            cache_path = get_cache_path(cache_key, n_seg)
            if cache_path.exists():
                was_cached = True
                cache_hits += 1
            else:
                cache_misses += 1

        # Run optimization (will use cache internally if enabled)
        P_opt, info = optimize_orbital_docking(
            P_init, 
            n_seg=n_seg, 
            r_e=r_e, 
            max_iter=max_iter, 
            tol=tol,
            sample_count=100,
            verbose=False,
            use_cache=use_cache
        )

        results.append((n_seg, P_opt, info))

        if verbose:
            cache_status = "[CACHED]" if was_cached else "[COMPUTED]"
            print(f"  ✓ n_seg={n_seg:>2d} {cache_status} | iter={info['iterations']:>3d} | "
                  f"acceleration={info['accel']:.3f} | feasible={info['feasible']}")

    print("\n" + "=" * 60)
    print("All optimizations complete!")
    if use_cache and verbose:
        print(f"Cache statistics: {cache_hits} hits, {cache_misses} misses")

    return results

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
        # set_isometric(ax, elev=20, azim=45)
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
    # accelerations = [info['accel'] for _, _, info in results]

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
    # print(f'positions: {positions}')
    # print(f'velocities: {velocities}')

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
    # Always start acceleration panel from 0; allow shared ylim if provided
    # if acc_ylim is not None:
    ax3.set_ylim(acc_ylim)
    # ax3.set_ylim(max(0.0, acc_ylim[0]), acc_ylim[1])
    # else:
    #     ax3.set_ylim(0, max(total_accel_mag_kms2) * 1.1)

    # Add acceleration statistics
    max_accel = np.max(total_accel_mag_kms2)
    avg_accel = np.mean(total_accel_mag_kms2)
    max_str = format_number(max_accel, '.1f')
    avg_str = format_number(avg_accel, '.1f')
    ax3.text(0.02, 0.98, f'Max: {max_str} m/s²\nAvg: {avg_str} m/s²', 
            transform=ax3.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    return fig

def generate_initial_control_points(degree, P_start, P_end):
    """
    Generate initial control points for a Bézier curve of given degree.
    Points are evenly spaced along the straight line from P_start to P_end.

    Args:
        degree: Degree of Bézier curve (N = 2, 3, 4, ...)
        P_start: Start point (chaser position)
        P_end: End point (ISS position)

    Returns:
        np.ndarray: Control points of shape (N+1, dim)
    """
    N = degree
    num_points = N + 1

    # Generate evenly spaced parameter values along the line
    # For N=2: [0, 0.5, 1.0] → 3 points
    # For N=3: [0, 1/3, 2/3, 1.0] → 4 points
    # For N=4: [0, 1/4, 1/2, 3/4, 1.0] → 5 points
    if N == 1:
        t_values = np.array([0.0, 1.0])
    else:
        t_values = np.linspace(0.0, 1.0, num_points)

    # Interpolate control points along the straight line
    control_points = []
    for t in t_values:
        P_t = P_start + t * (P_end - P_start)
        control_points.append(P_t)

    return np.vstack(control_points)

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


# # Secnario setup
# Chaser satelite @ \
# Target satelite @ ISS altitude \
# with 90 degree sepration

# In[54]:


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


# In[55]:


# OPTIMIZATION for N=2 (Quadratic curve)
# Measure calculation time for performance analysis
import time

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

