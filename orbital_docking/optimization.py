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


def _has_required_info_keys(info: dict) -> bool:
    """Return True only if cached info includes required schema keys."""
    if not isinstance(info, dict):
        return False
    required = (
        "iterations",
        "feasible",
        "min_radius",
        "cost",
        "cost_no_const",
        "cost_true_energy",
        "max_control_accel_ms2",
        "mean_control_accel_ms2",
        "elapsed_time",
    )
    return all(k in info for k in required)


def _bernstein_basis(N: int, tau: float) -> np.ndarray:
    """
    Bernstein basis weights for degree N at tau.
    Returns shape (N+1,).
    """
    # Use the curve evaluator for numerical stability/consistency via comb in bezier.py
    from scipy.special import comb
    i = np.arange(N + 1)
    return comb(N, i) * (tau ** i) * ((1.0 - tau) ** (N - i))


def _accel_two_body(r_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Two-body gravitational acceleration in km/s^2."""
    r = np.asarray(r_km, dtype=float)
    rn = np.linalg.norm(r)
    return (-mu_km3_s2 / (rn**3)) * r


def _accel_j2(r_km: np.ndarray, mu_km3_s2: float, r_e_km: float, j2: float) -> np.ndarray:
    """
    J2 perturbation acceleration in km/s^2 (simplified: Earth symmetry axis aligned with ECI Z).
    """
    r = np.asarray(r_km, dtype=float)
    x, y, z = r
    r2 = float(x*x + y*y + z*z)
    rn = np.sqrt(r2)
    if rn < 1e-12:
        return np.zeros(3)

    z2 = z*z
    r5 = rn**5
    factor = 1.5 * j2 * mu_km3_s2 * (r_e_km**2) / r5
    k = 5.0 * z2 / r2
    ax = factor * x * (k - 1.0)
    ay = factor * y * (k - 1.0)
    az = factor * z * (k - 3.0)
    return np.array([ax, ay, az], dtype=float)


def _accel_total(r_km: np.ndarray, mu_km3_s2: float, r_e_km: float, j2: float) -> np.ndarray:
    """Two-body + J2 total gravitational acceleration in km/s^2."""
    return _accel_two_body(r_km, mu_km3_s2) + _accel_j2(r_km, mu_km3_s2, r_e_km, j2)


def _jacobian_numeric(f, r0: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """
    Central-difference Jacobian of f: R^3 -> R^3 at r0.
    h is in km (default 1e-3 km = 1 m).
    """
    r0 = np.asarray(r0, dtype=float)
    J = np.zeros((3, 3), dtype=float)
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = h
        fp = f(r0 + dr)
        fm = f(r0 - dr)
        J[:, i] = (fp - fm) / (2.0 * h)
    return J


def _build_ctrl_accel_quadratic(P_ref: np.ndarray, T: float, sample_count: int):
    """
    Build quadratic objective for control acceleration energy with gravity+J2
    linearized about P_ref, without Bernstein sampling in the optimizer core.

    Matrix-only structure:
      1) Exact geometric acceleration-energy term from precomputed G_tilde
         (derived from E/D matrices):
             J_geom = (1/T^4) * x^T (G_tilde ⊗ I_3) x

      2) Gravity/J2 linearization term built on De Casteljau segment centroids:
             r_i(x)      = R_i x
             a_geom_i(x) = A_i x
             g_i(x) ≈ J_i r_i(x) + c_i
         and approximate:
             J ≈ J_geom + Σ_i w_i ||a_geom_i - g_i||^2

    This avoids Bernstein basis evaluation in the optimizer and keeps all mappings
    linear in control points via D/E and De Casteljau segment matrices.

    Returns:
        H, f, c for objective: 0.5 x^T H x + f^T x + c
        where c is the constant term (>= 0) that makes the value a true least-squares energy.
        (and some diagnostics dict)
    """
    from . import constants

    Np1, dim = P_ref.shape
    if dim != 3:
        raise ValueError("This dynamics cost currently assumes dim=3 (ECI).")
    N = Np1 - 1

    # Linear mapping from control points to acceleration control points:
    # A_ctrl = L @ P, where L = E D E D (shape (N+1, N+1))
    curve_ref = BezierCurve(P_ref)
    if curve_ref.degree < 2:
        raise ValueError("Control-acceleration cost requires Bézier degree >= 2.")
    L = curve_ref.E @ curve_ref.D @ curve_ref.E @ curve_ref.D  # (N+1, N+1)

    n = Np1 * dim
    Q = np.zeros((n, n), dtype=float)
    q = np.zeros(n, dtype=float)
    c_const = 0.0

    mu = constants.EARTH_MU_SCALED
    r_e = constants.EARTH_RADIUS_KM
    j2 = constants.EARTH_J2

    # 1) Exact geometric term via G_tilde (no sampling):
    #    x^T (G_tilde ⊗ I) x scaled by 1/T^4
    if curve_ref.G_tilde is not None:
        Q += (1.0 / (T**4)) * np.kron(curve_ref.G_tilde, np.eye(dim))

    # 2) Gravity/J2 linearization via De Casteljau segment centroids.
    n_lin_seg = max(1, int(sample_count))
    A_seg_list = segment_matrices_equal_params(N, n_lin_seg)
    w_seg = 1.0 / float(n_lin_seg)
    x_ref = P_ref.reshape(-1)

    for Aseg in A_seg_list:
        # Segment centroid linear map in control-point space:
        #   c_i = mean(Aseg @ P) = (w_row @ P), w_row shape (N+1,)
        w_row = Aseg.mean(axis=0)

        # r_i(x) = R_i x
        R_i = np.kron(w_row, np.eye(dim))  # (3, n)

        # a_geom_i(x) = A_i x with a_ctrl = (1/T^2) * (w_row @ L) @ P
        a_row = (w_row @ L)
        A_i = (1.0 / (T**2)) * np.kron(a_row, np.eye(dim))  # (3, n)

        # Reference point for gravity/J2 linearization
        r_ref = (R_i @ x_ref).reshape(3)
        g_ref = _accel_total(r_ref, mu, r_e, j2)
        J_i = _jacobian_numeric(lambda rr: _accel_total(rr, mu, r_e, j2), r_ref)

        # g_i(x) ≈ J_i R_i x + c_i
        c_i = g_ref - (J_i @ r_ref)
        B_i = J_i @ R_i

        # Add ||A_i x - (B_i x + c_i)||^2 contributions:
        # = x^T[(A-B)^T(A-B)]x - 2 c^T(A-B)x + c^T c
        M_i = A_i - B_i
        Q += w_seg * (M_i.T @ M_i)
        q += w_seg * (-M_i.T @ c_i)
        c_const += w_seg * float(c_i @ c_i)

    H = 2.0 * Q
    f = 2.0 * q
    diag = {"sample_count": sample_count, "n_lin_seg": n_lin_seg, "T": float(T)}
    return H, f, c_const, diag


def _compute_cost_only(P_flat, Np1, dim, n_samples=50):
    """
    Helper function to compute cost only (no recursion).

    Matches the monolithic optimizer behavior on `revert-3750d57`:
    - Cost uses the quadratic form based on G_tilde (no gravity term)
    - Returns cost=accel=a_geom for backward-compatible plotting/reporting
    """
    P = P_flat.reshape(Np1, dim)
    curve = BezierCurve(P)

    if curve.G_tilde is None:
        return 0.0, 0.0, 0.0, None

    # Quadratic form: tr(P^T G_tilde P) = sum(P * (G_tilde @ P))
    GP = curve.G_tilde @ P
    a_geom = float(np.sum(P * GP))
    cost = a_geom
    accel = a_geom
    return cost, accel, a_geom, None


def cost_function_gradient_hessian(P_flat, Np1, dim, n_samples=50, compute_grad=True):
    """
    Compute cost function, gradient, and Hessian approximation.
    Returns: (cost, gradient, hessian)

    Args:
        compute_grad: If False, only compute cost (for efficiency when gradient not needed)
    """
    # Analytic gradient/Hessian for the quadratic form:
    #   J(P) = tr(P^T G_tilde P)
    #   dJ/dP = 2 G_tilde P
    #   H(vec(P)) = 2 kron(I_dim, G_tilde)
    P = P_flat.reshape(Np1, dim)
    curve = BezierCurve(P)

    if curve.G_tilde is None:
        cost = 0.0
        grad = np.zeros(Np1 * dim)
        hess = np.zeros((Np1 * dim, Np1 * dim))
        return cost, grad, hess

    GP = curve.G_tilde @ P
    cost = float(np.sum(P * GP))
    grad_P = 2.0 * GP
    grad = grad_P.reshape(-1)
    hess = 2.0 * np.kron(np.eye(dim), curve.G_tilde)
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
    debug=False,
    use_cache=True,
    ignore_existing_cache=False
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
        ignore_existing_cache: If True, skip cache reads but still write cache

    Returns:
        (P_opt, info_dict)
    """
    from .constants import KOZ_RADIUS, TRANSFER_TIME_S
    if r_e is None:
        r_e = KOZ_RADIUS
    
    t0 = time.time()
    # Check cache first (optional bypass for forced fresh computation)
    if use_cache and not ignore_existing_cache:
        cache_key = get_cache_key(P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1)
        cache_path = get_cache_path(cache_key, n_seg)
        cached_result = load_from_cache(cache_path)
        
        if cached_result is not None:
            P_opt, info = cached_result
            if _has_required_info_keys(info):
                if verbose:
                    print(f"[Cache hit] Loaded result for n_seg={n_seg}")
                return P_opt, info
            if verbose:
                print(f"[Cache stale] Ignoring outdated result for n_seg={n_seg}")
        elif verbose:
            print(f"[Cache miss] Running optimization for n_seg={n_seg}...")
    
    P = P_init.copy()
    Np1, dim = P.shape
    N = Np1 - 1

    # Get segment matrices
    t_seg_start = time.time()
    A_list = segment_matrices_equal_params(N, n_seg)
    t_seg_end = time.time()

    # Set up bounds: fix endpoints for position constraints
    t_bounds_start = time.time()
    x0 = P.reshape(-1)
    lb = np.full_like(x0, -np.inf)
    ub = np.full_like(x0, np.inf)
    lb[:dim] = ub[:dim] = x0[:dim]  # Lock P0
    lb[-dim:] = ub[-dim:] = x0[-dim:]  # Lock PN
    bounds = Bounds(lb, ub)
    t_bounds_end = time.time()

    # Fixed time-scaling (arbitrary baseline)
    T = float(TRANSFER_TIME_S)

    # Build boundary constraints
    t_bc_start = time.time()
    boundary_constraints = build_boundary_constraints(P, v0, v1, a0, a1, dim, T=T)
    t_bc_end = time.time()

    # Store results
    info = {"iterations": 0, "feasible": False, "cost": np.inf}

    # Iterative optimization (SCP-style):
    # - update KOZ supporting half-spaces based on current solution
    # - update linearized gravity+J2 objective based on current solution
    for it in range(1, max_iter + 1):
        t_iter_start = time.time()
        # Build KOZ constraints based on current control points
        t_koz_start = time.time()
        koz_constraint = build_koz_constraints(A_list, P, r_e, dim)
        t_koz_end = time.time()

        # Combine all constraints
        all_constraints = [koz_constraint] + boundary_constraints

        # Build quadratic objective for control acceleration with gravity+J2 linearized about P
        H, f, c_const, _diag = _build_ctrl_accel_quadratic(P, T=T, sample_count=sample_count)

        def objective(x):
            return 0.5 * float(x @ (H @ x)) + float(f @ x)

        def gradient(x):
            return (H @ x) + f

        def hessian(_x):
            return H

        # Solve
        t_opt_start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res = minimize(
                objective,
                P.reshape(-1),
                method='trust-constr',
                jac=gradient,
                hess=hessian,
                constraints=all_constraints,
                bounds=bounds,
                options={'maxiter': 50, 'gtol': 1e-8, 'disp': False}
            )
        t_opt_end = time.time()

        P_new = res.x.reshape(Np1, dim)
        delta = np.linalg.norm(P_new - P)
        P = P_new

        t_iter_end = time.time()
        if debug:
            # Note: res.fun is the constant-dropped quadratic objective; it may be negative.
            cost_true_iter = float(res.fun) + float(c_const)
            print(f"N_seg={n_seg}, Iter {it}: cost_no_const={res.fun:.6e}, cost_true={cost_true_iter:.6e}, delta={delta:.6e}")
            print(f"Time taken for iteration {it}: {t_iter_end - t_iter_start:.2f} seconds")
            # print(f"Time taken for koz constraints: {t_koz_end - t_koz_start:.2f} seconds")
            # print(f"Time taken for boundary constraints: {t_bc_end - t_bc_start:.2f} seconds")
            # print(f"Time taken for segment matrices: {t_seg_end - t_seg_start:.2f} seconds")
            # print(f"Time taken for bounds: {t_bounds_end - t_bounds_start:.2f} seconds")
            print(f"Time taken for optimization: {t_opt_end - t_opt_start:.2f} seconds")

        if delta < tol:
            break

    # Final feasibility check
    curve = BezierCurve(P)
    ts_check = np.linspace(0, 1, 1000)
    pts = np.array([curve.point(t) for t in ts_check])
    radii = np.linalg.norm(pts, axis=1)
    min_radius = float(np.min(radii))

    # Compute rigorous objective values at the final solution.
    # - cost_no_const: quadratic part used by the solver (can be negative)
    # - cost_true_energy: full least-squares energy (>= 0) including constant term
    x_final = P.reshape(-1)
    Hf, ff, cf, _ = _build_ctrl_accel_quadratic(P, T=T, sample_count=sample_count)
    cost_no_const = 0.5 * float(x_final @ (Hf @ x_final)) + float(ff @ x_final)
    cost_true_energy = cost_no_const + float(cf)

    # Physically interpretable trajectory metrics (using the same fixed T and gravity model)
    ts_metrics = np.linspace(0.0, 1.0, 300)
    a_geom_km_s2 = np.array([curve.acceleration(t) for t in ts_metrics]) / (T**2)
    from .constants import EARTH_RADIUS_KM, EARTH_J2, EARTH_MU_SCALED
    a_grav_km_s2 = np.array([_accel_total(curve.point(t), EARTH_MU_SCALED, EARTH_RADIUS_KM, EARTH_J2) for t in ts_metrics])
    a_u_m_s2 = np.linalg.norm(a_geom_km_s2 - a_grav_km_s2, axis=1) * 1e3
    max_control_accel_ms2 = float(np.max(a_u_m_s2))
    mean_control_accel_ms2 = float(np.mean(a_u_m_s2))

    if verbose:
        print(f"min_radius: {min_radius:.6e}, r_e: {r_e:.6e}")

    info.update({
        "iterations": it,
        "feasible": min_radius >= r_e - 1e-6,
        "min_radius": min_radius,
        # Keep legacy keys but redefine them rigorously:
        # - cost: true nonnegative least-squares energy
        # - accel: max control acceleration magnitude (m/s^2)
        "cost": cost_true_energy,
        "cost_no_const": cost_no_const,
        "cost_true_energy": cost_true_energy,
        "accel": max_control_accel_ms2,
        "max_control_accel_ms2": max_control_accel_ms2,
        "mean_control_accel_ms2": mean_control_accel_ms2,
        "T_transfer_s": float(T),
        "sample_count": int(sample_count),
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
                                max_iter=120, tol=1e-8, verbose=True, debug=False, use_cache=True,
                                ignore_existing_cache=False,
                                v0=None, v1=None, a0=None, a1=None):
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
        ignore_existing_cache: If True, ignore existing cache and recompute fresh

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
    if use_cache and not ignore_existing_cache:
        print("Cache: ENABLED (read + write)")
    elif use_cache and ignore_existing_cache:
        print("Cache: REFRESH MODE (read disabled, write enabled)")
    else:
        print("Cache: DISABLED")
    print("=" * 60)

    for n_seg in segment_counts:
        if verbose:
            print(f"\nProcessing n_seg={n_seg}...")

        # Check if cached before running
        was_cached = False
        if use_cache and not ignore_existing_cache:
            # Match baseline behavior: cache key includes boundary values even if constraints
            # are currently not applied in the solver call.
            cache_key = get_cache_key(P_init, n_seg, r_e, max_iter, tol, 100, v0, v1, a0, a1)
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
            verbose=True,
            debug=debug,
            use_cache=use_cache,
            ignore_existing_cache=ignore_existing_cache,
            v0=v0, v1=v1, a0=a0, a1=a1
        )

        results.append((n_seg, P_opt, info))

        if verbose:
            cache_status = "[CACHED]" if was_cached else "[COMPUTED]"
            cost_print = info.get('cost_true_energy', info.get('cost', np.nan))
            max_u_print = info.get('max_control_accel_ms2', info.get('accel', np.nan))
            print(f"  ✓ n_seg={n_seg:>2d} {cache_status} | iter={info['iterations']:>3d} | "
                  f"cost={cost_print:.3e} | "
                  f"max_u={max_u_print:.3f} m/s² | feasible={info['feasible']}")

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

