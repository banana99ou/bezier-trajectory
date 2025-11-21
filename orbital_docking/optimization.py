"""
Optimization functions for orbital docking trajectories.
"""

import numpy as np
import time
import warnings
from scipy.optimize import minimize, Bounds

from .bezier import BezierCurve
from .de_casteljau import segment_matrices_equal_params
from .constraints import build_koz_constraints, build_boundary_constraints
from .cache import get_cache_key, get_cache_path, load_from_cache, save_to_cache
from .constants import EARTH_MU_SCALED


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


def optimize_orbital_docking(
    P_init, 
    n_seg=8, 
    r_e=None, 
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
        r_e: KOZ radius (if None, uses default from constants)
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
    from .constants import KOZ_RADIUS
    if r_e is None:
        r_e = KOZ_RADIUS
    
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


def optimize_all_segment_counts(P_init, r_e=None, segment_counts=[2, 4, 8, 16, 32, 64], 
                                max_iter=120, tol=1e-8, verbose=True, use_cache=True):
    """
    Run optimization for multiple segment counts and return results.
    Uses caching to speed up repeated runs.

    Args:
        P_init: Initial control points (N+1, dim)
        r_e: KOZ radius (if None, uses default from constants)
        segment_counts: List of segment counts to test
        max_iter: Maximum iterations per optimization
        tol: Convergence tolerance
        verbose: Print progress
        use_cache: Whether to use cache (default: True)

    Returns:
        List of tuples: [(n_seg, P_opt, info), ...] where info includes acceleration
    """
    from .constants import KOZ_RADIUS
    from .cache import get_cache_key, get_cache_path
    
    if r_e is None:
        r_e = KOZ_RADIUS
    
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

