"""
Rust-backed optimization functions for orbital docking trajectories.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import bezier_opt as _bezier_opt_rs
import numpy as np

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
    Optimize Bézier curve for orbital docking via the Rust backend.

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

    Compatibility args:
        store_history: Retained for API compatibility; ignored by the Rust backend.
        history_samples: Retained for API compatibility; ignored by the Rust backend.

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
    
    # ---- Rust backend ----
    P_opt, rust_info = _bezier_opt_rs.optimize_orbital_docking(
        p_init=P_init,
        n_seg=n_seg,
        r_e=r_e,
        max_iter=max_iter,
        tol=tol,
        v0=v0, v1=v1, a0=a0, a1=a1,
        sample_count=sample_count,
        objective_mode=objective_mode,
        dv_irls_eps=dv_irls_eps,
        dv_geom_reg=dv_geom_reg,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        enforce_prograde=enforce_prograde,
        prograde_n_samples=prograde_n_samples,
    )
    P = np.asarray(P_opt)
    info = dict(rust_info)

    # Fill in keys not provided by Rust
    info["elapsed_time"] = time.time() - t0
    info["max_iterations"] = int(max_iter)
    info["accel"] = info.get("max_control_accel_ms2", 0.0)
    info["T_transfer_s"] = float(TRANSFER_TIME_S)
    info["sample_count"] = int(sample_count)
    info["objective"] = str(objective_mode)
    info["scp_prox_weight"] = float(scp_prox_weight)
    info["scp_trust_radius"] = float(scp_trust_radius)
    if info.get("iterations", max_iter) < max_iter:
        info["termination_reason"] = "converged_delta_below_tol"
    else:
        info["termination_reason"] = "stopped_max_iter"

    it = info["iterations"]
    # ---- verbose + cache (shared tail) ----

    if verbose:
        print(f"min_radius: {info.get('min_radius', 'N/A')}, r_e: {r_e:.6e}, iterations: {it}")

    # Save to cache
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

