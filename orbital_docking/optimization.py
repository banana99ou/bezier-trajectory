"""
Optimization functions for orbital docking trajectories.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import time
import warnings
from scipy.optimize import minimize, Bounds, LinearConstraint

try:
    import bezier_opt as _bezier_opt_rs
    _RUST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised in runtime environments without Rust build
    _bezier_opt_rs = None
    _RUST_IMPORT_ERROR = exc

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


def _require_rust_backend():
    """Return the Rust backend module or raise a clear import error."""
    if _bezier_opt_rs is None:
        raise ImportError(
            "Rust backend `bezier_opt` is not available. Build/install it from "
            "`rust_optimizer/pybind` before running the optimizer."
        ) from _RUST_IMPORT_ERROR
    return _bezier_opt_rs


def _linear_constraint_violation(c, x: np.ndarray) -> float:
    """
    Return max violation for a scipy.optimize.LinearConstraint at point x.

    For bounds:
      lb <= A x <= ub
    Violation is max( lb - Ax, Ax - ub, 0 ).
    """
    try:
        A = c.A
        lb = c.lb
        ub = c.ub
    except Exception:
        return float("nan")

    Ax = A @ x
    v = 0.0
    if lb is not None:
        v = max(v, float(np.max(lb - Ax)))
    if ub is not None:
        v = max(v, float(np.max(Ax - ub)))
    return float(max(v, 0.0))


def _koz_margin_stats(koz_constraint, x: np.ndarray) -> dict:
    """
    Compute KOZ half-space margin stats for the current linearization:
      margin = (A x) - lb  (since ub=+inf)

    Returns dict with keys:
      n_rows, min_margin, p01_margin, p05_margin, median_margin
    """
    try:
        A = koz_constraint.A
        lb = koz_constraint.lb
    except Exception:
        return {
            "n_rows": 0,
            "min_margin": float("nan"),
            "p01_margin": float("nan"),
            "p05_margin": float("nan"),
            "median_margin": float("nan"),
        }

    Ax = A @ x
    m = Ax - lb
    if m.size == 0:
        return {
            "n_rows": 0,
            "min_margin": float("nan"),
            "p01_margin": float("nan"),
            "p05_margin": float("nan"),
            "median_margin": float("nan"),
        }

    m_sorted = np.sort(m)
    def _pct(p: float) -> float:
        idx = int(np.clip(np.floor(p * (m_sorted.size - 1)), 0, m_sorted.size - 1))
        return float(m_sorted[idx])

    return {
        "n_rows": int(m.size),
        "min_margin": float(m_sorted[0]),
        "p01_margin": _pct(0.01),
        "p05_margin": _pct(0.05),
        "median_margin": float(np.median(m_sorted)),
    }


def _bernstein_basis(N: int, tau: float) -> np.ndarray:
    """
    Bernstein basis weights for degree N at tau.
    Returns shape (N+1,).
    """
    # Use the curve evaluator for numerical stability/consistency via comb in bezier.py
    from scipy.special import comb
    i = np.arange(N + 1)
    return comb(N, i) * (tau ** i) * ((1.0 - tau) ** (N - i))


def _bernstein_derivative_weights(N: int, tau: float) -> np.ndarray:
    """
    Weights for d/dtau of degree-N Bézier at tau, expressed in the original
    control-point basis:
        dp/dtau = sum_j w_der[j] * P_j

    Using the standard relation:
        dp/dtau = N * sum_{i=0}^{N-1} B_{N-1,i}(tau) * (P_{i+1} - P_i)
    """
    if N <= 0:
        return np.zeros(1)
    # Basis for degree N-1
    from scipy.special import comb

    i = np.arange(N)
    b_low = comb(N - 1, i) * (tau ** i) * ((1.0 - tau) ** (N - 1 - i))

    w_der = np.zeros(N + 1, dtype=float)
    # j = 0
    w_der[0] = -N * b_low[0]
    # j = 1..N-1
    for j in range(1, N):
        w_der[j] = N * (b_low[j - 1] - b_low[j])
    # j = N
    w_der[N] = N * b_low[N - 1]
    return w_der


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


def _build_ctrl_accel_quadratic(
    P_ref: np.ndarray,
    T: float,
    sample_count: int,
    objective: str = "energy",
    irls_eps: float = 1e-9,
    geom_reg: float = 0.0,
):
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

    objective = str(objective).lower().strip()
    if objective not in ("energy", "dv"):
        raise ValueError(f"Unknown objective={objective!r}; expected 'energy' or 'dv'.")

    # 1) Optional geometric regularization term via G_tilde (no sampling).
    # For 'energy' objective, this is part of the intended surrogate.
    # For 'dv' objective, keep it as optional stabilization (default 0.0).
    if curve_ref.G_tilde is not None:
        w_geom = 1.0 if objective == "energy" else float(geom_reg)
        if w_geom != 0.0:
            Q += w_geom * (1.0 / (T**4)) * np.kron(curve_ref.G_tilde, np.eye(dim))

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

        # Add ||A_i x - (B_i x + c_i)||^2 contributions.
        #
        # objective='energy':
        #   minimize Σ ||res_i||^2  (L2 energy)
        #
        # objective='dv':
        #   approximate minimize Σ ||res_i|| (delta-v proxy in tau-domain)
        #   via IRLS/majorization:
        #     ||r|| ≈ (1/alpha) ||r||^2 + const,  alpha = sqrt(||r_ref||^2 + eps)
        #   so weight_i = 1/alpha.
        # = x^T[(A-B)^T(A-B)]x - 2 c^T(A-B)x + c^T c
        M_i = A_i - B_i
        w_i = w_seg
        if objective == "dv":
            r_ref_res = (M_i @ x_ref) - c_i
            alpha = float(np.sqrt(float(r_ref_res @ r_ref_res) + float(irls_eps)))
            # Avoid division by 0; alpha >= sqrt(eps)
            w_i = w_seg * (1.0 / alpha)

        Q += w_i * (M_i.T @ M_i)
        q += w_i * (-M_i.T @ c_i)
        c_const += w_i * float(c_i @ c_i)

    H = 2.0 * Q
    f = 2.0 * q
    diag = {
        "sample_count": sample_count,
        "n_lin_seg": n_lin_seg,
        "T": float(T),
        "objective": objective,
        "irls_eps": float(irls_eps),
        "geom_reg": float(geom_reg),
    }
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
    objective_mode: str = "energy",
    dv_irls_eps: float = 1e-9,
    dv_geom_reg: float = 0.0,
    scp_prox_weight: float = 0.0,
    scp_trust_radius: float = 0.0,
    enforce_prograde: bool = False,
    prograde_n_samples: int = 16,
    verbose=True,
    debug=False,
    use_cache=True,
    ignore_existing_cache=False,
    store_history: bool = False,
    history_samples: int = 200,
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

    Args (extra diagnostics):
        store_history: If True, store per-outer-iteration telemetry in info['history'].
            The Rust backend currently does not emit per-outer-iteration history.
        history_samples: Number of tau samples used for per-iteration geometry metrics.

    Returns:
        (P_opt, info_dict) where info may include info['history'] when enabled.
    """
    from .constants import KOZ_RADIUS, TRANSFER_TIME_S
    if r_e is None:
        r_e = KOZ_RADIUS
    
    t0 = time.time()
    # Check cache first (optional bypass for forced fresh computation)
    if use_cache and not ignore_existing_cache:
        cache_key = get_cache_key(
            P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1,
            objective=objective_mode,
            scp_prox_weight=scp_prox_weight,
            scp_trust_radius=scp_trust_radius,
        )
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

    backend = _require_rust_backend()
    P_init = np.asarray(P_init, dtype=float)

    # ---- Rust backend solve ----
    P_opt, rust_info = backend.optimize_orbital_docking(
        p_init=P_init,
        n_seg=n_seg,
        r_e=r_e,
        max_iter=max_iter,
        tol=tol,
        v0=v0,
        v1=v1,
        a0=a0,
        a1=a1,
        sample_count=sample_count,
        objective_mode=objective_mode,
        dv_irls_eps=dv_irls_eps,
        dv_geom_reg=dv_geom_reg,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        enforce_prograde=enforce_prograde,
        prograde_n_samples=prograde_n_samples,
    )

    P = np.asarray(P_opt, dtype=float)
    Np1, dim = P.shape
    N = Np1 - 1
    T = float(TRANSFER_TIME_S)
    info = dict(rust_info)
    keep_history = bool(store_history or debug)
    it = int(info.get("iterations", max_iter))
    termination_reason = (
        "converged_delta_below_tol" if it < int(max_iter) else "stopped_max_iter"
    )
    last_delta = info.get("final_delta_norm")

    # Python-side validation of the Rust solution keeps downstream metrics comparable.
    curve = BezierCurve(P)
    ts_check = np.linspace(0, 1, 1000)
    pts = np.array([curve.point(t) for t in ts_check])
    radii = np.linalg.norm(pts, axis=1)
    min_radius = float(np.min(radii))

    x_final = P.reshape(-1)
    Hf, ff, cf, _ = _build_ctrl_accel_quadratic(
        P,
        T=T,
        sample_count=sample_count,
        objective=objective_mode,
        irls_eps=dv_irls_eps,
        geom_reg=dv_geom_reg,
    )
    cost_no_const = 0.5 * float(x_final @ (Hf @ x_final)) + float(ff @ x_final)
    cost_true_energy = cost_no_const + float(cf)

    ts_metrics = np.linspace(0.0, 1.0, 300)
    a_geom_km_s2 = np.array([curve.acceleration(t) for t in ts_metrics]) / (T**2)
    from .constants import EARTH_RADIUS_KM, EARTH_J2, EARTH_MU_SCALED

    a_grav_km_s2 = np.array(
        [_accel_total(curve.point(t), EARTH_MU_SCALED, EARTH_RADIUS_KM, EARTH_J2) for t in ts_metrics]
    )
    a_u_m_s2 = np.linalg.norm(a_geom_km_s2 - a_grav_km_s2, axis=1) * 1e3
    max_control_accel_ms2 = float(np.max(a_u_m_s2))
    mean_control_accel_ms2 = float(np.mean(a_u_m_s2))
    dv_proxy_m_s = float(T) * float(np.trapezoid(a_u_m_s2, ts_metrics))

    A_list = segment_matrices_equal_params(N, n_seg)
    koz_constraint = build_koz_constraints(A_list, P, r_e, dim)
    koz_linear_max_violation = _linear_constraint_violation(koz_constraint, x_final)

    bc_v0_rel_err = None
    bc_v1_rel_err = None
    if v0 is not None:
        v0_phys = N * (P[1] - P[0]) / T
        den = max(float(np.linalg.norm(v0)), 1e-12)
        bc_v0_rel_err = float(np.linalg.norm(v0_phys - v0) / den)
    if v1 is not None:
        v1_phys = N * (P[-1] - P[-2]) / T
        den = max(float(np.linalg.norm(v1)), 1e-12)
        bc_v1_rel_err = float(np.linalg.norm(v1_phys - v1) / den)

    if verbose:
        print(f"min_radius: {min_radius:.6e}, r_e: {r_e:.6e}, iterations: {it}")

    info.update(
        {
            "iterations": it,
            "max_iterations": int(max_iter),
            "termination_reason": termination_reason,
            "final_delta_norm": last_delta,
            "feasible": min_radius >= r_e - 1e-6,
            "min_radius": min_radius,
            "cost": cost_true_energy,
            "cost_no_const": cost_no_const,
            "cost_true_energy": cost_true_energy,
            "accel": max_control_accel_ms2,
            "max_control_accel_ms2": max_control_accel_ms2,
            "mean_control_accel_ms2": mean_control_accel_ms2,
            "dv_proxy_m_s": dv_proxy_m_s,
            "T_transfer_s": float(T),
            "sample_count": int(sample_count),
            "objective": str(objective_mode),
            "scp_prox_weight": float(scp_prox_weight),
            "scp_trust_radius": float(scp_trust_radius),
            "elapsed_time": time.time() - t0,
            "solver_backend": "rust",
            "koz_linear_max_violation": float(koz_linear_max_violation),
            "bc_v0_rel_err": bc_v0_rel_err,
            "bc_v1_rel_err": bc_v1_rel_err,
        }
    )
    if keep_history:
        info["history"] = []
        info["history_samples"] = int(max(20, int(history_samples)))

    if use_cache:
        cache_key = get_cache_key(
            P_init, n_seg, r_e, max_iter, tol, sample_count, v0, v1, a0, a1,
            objective=objective_mode,
            scp_prox_weight=scp_prox_weight,
            scp_trust_radius=scp_trust_radius,
        )
        cache_path = get_cache_path(cache_key, n_seg)
        save_to_cache(cache_path, P, info)
        if verbose:
            print(f"[Cache saved] Result saved for n_seg={n_seg}")

    return P, info


def _optimize_one_segment_count(payload: dict):
    """
    Worker-safe wrapper to run a single n_seg optimization.

    NOTE: This must be a top-level function (picklable) to support multiprocessing
    on platforms that use the 'spawn' start method (e.g., macOS).
    """
    # Avoid BLAS oversubscription when running multiple processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    n_seg = int(payload["n_seg"])
    P_init = payload["P_init"]
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=payload["r_e"],
        max_iter=payload["max_iter"],
        tol=payload["tol"],
        sample_count=payload["sample_count"],
        objective_mode=payload.get("objective", "energy"),
        dv_irls_eps=payload.get("dv_irls_eps", 1e-9),
        dv_geom_reg=payload.get("dv_geom_reg", 0.0),
        scp_prox_weight=payload.get("scp_prox_weight", 0.0),
        scp_trust_radius=payload.get("scp_trust_radius", 0.0),
        enforce_prograde=payload.get("enforce_prograde", False),
        verbose=payload["verbose"],
        debug=payload["debug"],
        use_cache=payload["use_cache"],
        ignore_existing_cache=payload["ignore_existing_cache"],
        v0=payload["v0"],
        v1=payload["v1"],
        a0=payload["a0"],
        a1=payload["a1"],
    )
    return n_seg, P_opt, info


def optimize_all_segment_counts(P_init, r_e=None, segment_counts=[2, 4, 8, 16, 32, 64], 
                                max_iter=120, tol=1e-8, verbose=True, debug=False, use_cache=True,
                                ignore_existing_cache=False,
                                v0=None, v1=None, a0=None, a1=None,
                                objective: str = "energy",
                                dv_irls_eps: float = 1e-9,
                                dv_geom_reg: float = 0.0,
                                scp_prox_weight: float = 0.0,
                                scp_trust_radius: float = 0.0,
                                enforce_prograde: bool = False,
                                n_jobs: int = 1):
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
        n_jobs: Number of worker processes to use for parallelizing across n_seg.
            - n_jobs=1: run serially (default, preserves legacy behavior)
            - n_jobs<=0: auto-select based on CPU count
            - n_jobs>1: run multiple n_seg optimizations concurrently

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

    n_jobs_eff = int(n_jobs) if n_jobs is not None else 1
    if n_jobs_eff <= 0:
        n_jobs_eff = int(os.cpu_count() or 1)

    # Pre-check cache existence for reporting (does not affect computation correctness).
    was_cached_map = {}
    if use_cache and not ignore_existing_cache:
        for n_seg in segment_counts:
            cache_key = get_cache_key(
                P_init, n_seg, r_e, max_iter, tol, 100, v0, v1, a0, a1,
                objective=objective,
                scp_prox_weight=scp_prox_weight,
                scp_trust_radius=scp_trust_radius,
            )
            cache_path = get_cache_path(cache_key, n_seg)
            if cache_path.exists():
                was_cached_map[int(n_seg)] = True
                cache_hits += 1
            else:
                was_cached_map[int(n_seg)] = False
                cache_misses += 1
    else:
        for n_seg in segment_counts:
            was_cached_map[int(n_seg)] = False

    if n_jobs_eff == 1 or len(segment_counts) <= 1:
        for n_seg in segment_counts:
            if verbose:
                print(f"\nProcessing n_seg={n_seg}...")

            P_opt, info = optimize_orbital_docking(
                P_init,
                n_seg=n_seg,
                r_e=r_e,
                max_iter=max_iter,
                tol=tol,
                sample_count=100,
                objective_mode=objective,
                dv_irls_eps=dv_irls_eps,
                dv_geom_reg=dv_geom_reg,
                scp_prox_weight=scp_prox_weight,
                scp_trust_radius=scp_trust_radius,
                enforce_prograde=enforce_prograde,
                verbose=True,
                debug=debug,
                use_cache=use_cache,
                ignore_existing_cache=ignore_existing_cache,
                v0=v0,
                v1=v1,
                a0=a0,
                a1=a1,
            )

            results.append((int(n_seg), P_opt, info))

            if verbose:
                cache_status = "[CACHED]" if was_cached_map.get(int(n_seg), False) else "[COMPUTED]"
                cost_print = info.get("cost_true_energy", info.get("cost", np.nan))
                max_u_print = info.get("max_control_accel_ms2", info.get("accel", np.nan))
                print(
                    f"  ✓ n_seg={int(n_seg):>2d} {cache_status} | iter={info['iterations']:>3d} | "
                    f"cost={cost_print:.3e} | "
                    f"max_u={max_u_print:.3f} m/s² | feasible={info['feasible']}"
                )
    else:
        if verbose:
            print(f"\nParallelizing over n_seg with n_jobs={n_jobs_eff} worker(s)")

        payloads = []
        for n_seg in segment_counts:
            if verbose:
                print(f"\nQueueing n_seg={n_seg}...")
            payloads.append(
                {
                    "n_seg": int(n_seg),
                    "P_init": P_init,
                    "r_e": r_e,
                    "max_iter": max_iter,
                    "tol": tol,
                    "sample_count": 100,
                    "objective": objective,
                    "dv_irls_eps": dv_irls_eps,
                    "dv_geom_reg": dv_geom_reg,
                    "scp_prox_weight": scp_prox_weight,
                    "scp_trust_radius": scp_trust_radius,
                    "enforce_prograde": enforce_prograde,
                    # Keep worker output quiet to avoid interleaved logs.
                    "verbose": False,
                    "debug": debug,
                    "use_cache": use_cache,
                    "ignore_existing_cache": ignore_existing_cache,
                    "v0": v0,
                    "v1": v1,
                    "a0": a0,
                    "a1": a1,
                }
            )

        with ProcessPoolExecutor(max_workers=n_jobs_eff) as ex:
            futs = [ex.submit(_optimize_one_segment_count, p) for p in payloads]
            for fut in as_completed(futs):
                n_seg_done, P_opt, info = fut.result()
                results.append((int(n_seg_done), P_opt, info))

                if verbose:
                    cache_status = "[CACHED]" if was_cached_map.get(int(n_seg_done), False) else "[COMPUTED]"
                    cost_print = info.get("cost_true_energy", info.get("cost", np.nan))
                    max_u_print = info.get("max_control_accel_ms2", info.get("accel", np.nan))
                    print(
                        f"  ✓ n_seg={int(n_seg_done):>2d} {cache_status} | iter={info['iterations']:>3d} | "
                        f"cost={cost_print:.3e} | "
                        f"max_u={max_u_print:.3f} m/s² | feasible={info['feasible']}"
                    )

    # Ensure deterministic order for downstream plotting/analysis.
    results.sort(key=lambda x: int(x[0]))

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

