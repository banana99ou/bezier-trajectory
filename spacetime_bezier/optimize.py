"""
Optimization entrypoints for space-time Bezier trajectories.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from orbital_docking.bezier import BezierCurve
from orbital_docking.de_casteljau import segment_matrices_equal_params

from .constraints import (
    build_boundary_constraints,
    build_box_bounds,
    build_spacetime_koz_constraints,
    build_time_monotonicity,
)
from .geometry import bezier_curve, obstacle_array_bundle
from .objective import build_energy_objective, build_initial_guess

try:
    import bezier_opt as _bezier_opt_rs
except ImportError:  # pragma: no cover - exercised when the native extension is unavailable.
    _bezier_opt_rs = None


def _clip_trust_region(x_new: np.ndarray, x_ref: np.ndarray, trust_radius: float) -> np.ndarray:
    if trust_radius <= 0.0:
        return x_new
    step = x_new - x_ref
    step_norm = np.linalg.norm(step)
    if step_norm <= trust_radius or step_norm <= 1e-15:
        return x_new
    return x_ref + (trust_radius / step_norm) * step


def compute_min_clearance(P, obstacles: list[dict], dim: int, n_eval: int = 1500) -> float:
    """Evaluate the minimum clearance of a Bezier curve to all obstacles."""
    if not obstacles:
        return float("inf")

    pts = bezier_curve(np.asarray(P, dtype=float), num_pts=n_eval)
    spatial_dim = dim - 1
    worst = np.inf

    for obs in obstacles:
        pos0 = np.asarray(obs["pos0"], dtype=float)
        vel = np.asarray(obs["vel"], dtype=float)
        radius = float(obs["r"])
        t0 = float(obs.get("t_start", -np.inf))
        t1 = float(obs.get("t_end", np.inf))

        t_vals = pts[:, -1]
        active = (t_vals >= t0) & (t_vals <= t1)
        if not active.any():
            continue

        o_positions = pos0[None, :] + vel[None, :] * t_vals[active, None]
        dists = np.linalg.norm(pts[active, :spatial_dim] - o_positions, axis=1) - radius
        worst = min(worst, float(dists.min()))

    return float(worst)


def _optimize_spacetime_python(
    P_init: np.ndarray,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = 0.0,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    verbose: bool = True,
) -> np.ndarray:
    """Successive convexification solver using SciPy trust-constr."""
    P = np.asarray(P_init, dtype=float).copy()
    n_cp, dim = P.shape
    N = n_cp - 1

    A_list = segment_matrices_equal_params(N, n_seg)
    H_energy = build_energy_objective(N, dim)
    bc_con = build_boundary_constraints(n_cp, dim, P[0], P[-1])
    mono_con = build_time_monotonicity(n_cp, dim, min_dt=min_dt)
    bounds = build_box_bounds(
        P,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub_scale=time_ub_scale,
    )

    if verbose:
        print(
            f"SCP: N={N}, dim={dim}, n_seg={n_seg}, n_cp={n_cp}, n_obs={len(obstacles)}, backend=python"
        )

    best_P = P.copy()
    best_clearance = compute_min_clearance(best_P, obstacles, dim)
    for iteration in range(max_iter):
        x_ref = P.reshape(-1)
        koz_con = build_spacetime_koz_constraints(A_list, P, obstacles, dim)

        H = H_energy + float(scp_prox_weight) * np.eye(n_cp * dim)
        f = -float(scp_prox_weight) * x_ref

        constraints = [bc_con, mono_con]
        if koz_con is not None:
            constraints.append(koz_con)

        result = minimize(
            lambda x: 0.5 * x @ H @ x + f @ x,
            x_ref,
            jac=lambda x: H @ x + f,
            hess=lambda _x: H,
            method="trust-constr",
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 80, "gtol": 1e-9, "verbose": 0},
        )

        x_new = _clip_trust_region(np.asarray(result.x, dtype=float), x_ref, float(scp_trust_radius))
        P_new = x_new.reshape(n_cp, dim)
        delta = float(np.linalg.norm(P_new - P))
        clearance = compute_min_clearance(P_new, obstacles, dim)

        if verbose:
            print(
                f"  iter {iteration + 1}: delta={delta:.6f}, cost={float(result.fun):.4f}, clearance={clearance:.4f}"
            )

        P = P_new
        if clearance > 0.0 and clearance > best_clearance:
            best_P = P.copy()
            best_clearance = clearance
        if delta < tol:
            if verbose:
                print(f"  Converged at iter {iteration + 1}")
            break

    final_clearance = compute_min_clearance(P, obstacles, dim)
    if final_clearance < 0.0 and best_clearance > 0.0:
        if verbose:
            print("  Using best feasible iterate")
        return best_P
    return P


def _optimize_spacetime_rust(
    P_init: np.ndarray,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = 0.0,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Call the native Rust backend for the space-time optimizer."""
    if _bezier_opt_rs is None or not hasattr(_bezier_opt_rs, "optimize_spacetime_bezier"):
        raise RuntimeError("Rust space-time optimizer is not available in bezier_opt.")

    P_init = np.asarray(P_init, dtype=float)
    n_cp, dim = P_init.shape
    spatial_dim = dim - 1
    pos0, vel, radius, t_start, t_end = obstacle_array_bundle(obstacles, spatial_dim)
    time_upper = float(P_init[-1, -1]) * float(time_ub_scale)

    P_opt, info = _bezier_opt_rs.optimize_spacetime_bezier(
        p_init=P_init,
        obstacle_pos0=pos0,
        obstacle_vel=vel,
        obstacle_r=radius,
        obstacle_t_start=t_start,
        obstacle_t_end=t_end,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub=time_upper,
    )
    P_opt = np.asarray(P_opt, dtype=float)
    info = dict(info)
    info["backend"] = "rust"

    if verbose:
        iterations = int(info.get("iterations", -1))
        clearance = float(info.get("min_clearance", math.nan))
        print(
            f"SCP: N={n_cp - 1}, dim={dim}, n_seg={n_seg}, n_cp={n_cp}, n_obs={len(obstacles)}, backend=rust"
        )
        print(f"  rust result: iterations={iterations}, clearance={clearance:.4f}")

    return P_opt, info


def optimize_spacetime_from_control_points(
    P_init,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = 0.0,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    verbose: bool = True,
    backend: str = "auto",
) -> np.ndarray:
    """Optimize a space-time Bezier curve from an initial control polygon."""
    backend = str(backend).lower().strip()
    if backend not in {"auto", "python", "rust"}:
        raise ValueError(f"Unsupported backend={backend!r}")

    P_init = np.asarray(P_init, dtype=float)

    if backend in {"auto", "rust"}:
        try:
            P_rust, _ = _optimize_spacetime_rust(
                P_init,
                obstacles,
                n_seg=n_seg,
                max_iter=max_iter,
                tol=tol,
                scp_prox_weight=scp_prox_weight,
                scp_trust_radius=scp_trust_radius,
                min_dt=min_dt,
                coord_lb=coord_lb,
                coord_ub=coord_ub,
                time_lb=time_lb,
                time_ub_scale=time_ub_scale,
                verbose=verbose,
            )
            return P_rust
        except RuntimeError:
            if backend == "rust":
                raise
            if verbose:
                print("Rust space-time optimizer unavailable, falling back to Python.")

    return _optimize_spacetime_python(
        P_init,
        obstacles,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub_scale=time_ub_scale,
        verbose=verbose,
    )


def optimize_spacetime(
    N: int,
    dim: int,
    p_start,
    p_end,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = 0.0,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    verbose: bool = True,
    init_curve: dict | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Public optimizer entrypoint using either the Rust or Python backend."""
    n_cp = int(N) + 1
    P_init = build_initial_guess(p_start, p_end, n_cp, init_curve=init_curve)
    if dim != P_init.shape[1]:
        raise ValueError(f"Expected dim={dim}, got initial guess with dim={P_init.shape[1]}")
    return optimize_spacetime_from_control_points(
        P_init,
        obstacles,
        n_seg=n_seg,
        max_iter=max_iter,
        tol=tol,
        scp_prox_weight=scp_prox_weight,
        scp_trust_radius=scp_trust_radius,
        min_dt=min_dt,
        coord_lb=coord_lb,
        coord_ub=coord_ub,
        time_lb=time_lb,
        time_ub_scale=time_ub_scale,
        verbose=verbose,
        backend=backend,
    )


def optimize_scenario(
    scenario: dict,
    configs: list[tuple[int, int]],
    backend: str = "auto",
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float = 0.0,
    min_dt: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Run optimization for all requested degree/segment-count pairs."""
    obstacles = scenario["obstacles"]
    p_start = scenario["start"]
    p_end = scenario["end"]
    init_curve = scenario.get("init_curve")

    results = {}
    for N, n_seg in configs:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"[{scenario['name']}] degree={N}, segments={n_seg}")
            print(f"{'=' * 60}")

        P_opt = optimize_spacetime(
            N=N,
            dim=len(p_start),
            p_start=p_start,
            p_end=p_end,
            obstacles=obstacles,
            n_seg=n_seg,
            max_iter=max_iter,
            tol=tol,
            scp_prox_weight=scp_prox_weight,
            scp_trust_radius=scp_trust_radius,
            min_dt=min_dt,
            verbose=verbose,
            init_curve=init_curve,
            backend=backend,
        )

        clearance = compute_min_clearance(P_opt, obstacles, dim=len(p_start), n_eval=3000)
        if verbose:
            print(f"  Final clearance: {clearance:.4f}")

        key = f"N{N}_seg{n_seg}"
        results[key] = {
            "N": int(N),
            "n_seg": int(n_seg),
            "control_points": np.asarray(P_opt, dtype=float).tolist(),
            "min_clearance": float(clearance),
            "feasible": bool(clearance > 0.0),
        }

    feasible = {key: value for key, value in results.items() if value["feasible"]}
    if feasible:
        best_key = max(feasible, key=lambda key: feasible[key]["min_clearance"])
    else:
        best_key = max(results, key=lambda key: results[key]["min_clearance"])

    if verbose:
        print(
            f"[{scenario['name']}] Best: {best_key} (clearance={results[best_key]['min_clearance']:.4f})"
        )

    return {
        "name": scenario["name"],
        "title": scenario["title"],
        "best": best_key,
        "obstacles": obstacles,
        "start": p_start,
        "end": p_end,
        "T": scenario["T"],
        "init_curve": init_curve,
        "results": results,
    }


def optimize_scenarios(
    scenario_names: list[str],
    scenario_map: dict,
    existing_outputs: dict | None = None,
    backend: str = "auto",
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float = 0.0,
    min_dt: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Optimize the selected scenarios and merge them with any existing outputs."""
    all_outputs = dict(existing_outputs or {})
    for name in scenario_names:
        scenario_fn, configs = scenario_map[name]
        all_outputs[name] = optimize_scenario(
            scenario_fn(),
            configs,
            backend=backend,
            max_iter=max_iter,
            tol=tol,
            scp_prox_weight=scp_prox_weight,
            scp_trust_radius=scp_trust_radius,
            min_dt=min_dt,
            verbose=verbose,
        )
    return all_outputs
