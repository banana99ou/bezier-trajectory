"""
Optimization entrypoints for space-time Bezier trajectories.
"""

from __future__ import annotations

import math

import numpy as np

from .debug_stepper import (
    clip_trust_region,
    create_debug_stepper,
    create_debug_stepper_from_control_points,
)
from .geometry import bezier_curve, obstacle_array_bundle
from .objective import build_initial_guess

try:
    import bezier_opt as _bezier_opt_rs
except ImportError:  # pragma: no cover - exercised when the native extension is unavailable.
    _bezier_opt_rs = None


def _clip_trust_region(x_new: np.ndarray, x_ref: np.ndarray, trust_radius: float) -> np.ndarray:
    return clip_trust_region(np.asarray(x_new, dtype=float), np.asarray(x_ref, dtype=float), trust_radius)


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
    stepper = create_debug_stepper_from_control_points(
        np.asarray(P_init, dtype=float),
        obstacles,
        clearance_fn=compute_min_clearance,
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
    )

    if verbose:
        print(
            f"SCP: N={stepper.N}, dim={stepper.dim}, n_seg={n_seg}, n_cp={stepper.n_cp}, "
            f"n_obs={len(obstacles)}, backend=python"
        )
    final_control_points, _info = stepper.run_to_completion(verbose=verbose)
    return final_control_points


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


def create_spacetime_debug_stepper_from_control_points(
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
    backend: str = "python",
):
    backend = str(backend).lower().strip()
    if backend != "python":
        raise ValueError("Live optimizer stepping currently supports only the Python backend.")
    return create_debug_stepper_from_control_points(
        np.asarray(P_init, dtype=float),
        obstacles,
        clearance_fn=compute_min_clearance,
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
    )


def create_spacetime_debug_stepper(
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
    init_curve: dict | None = None,
    backend: str = "python",
):
    backend = str(backend).lower().strip()
    if backend != "python":
        raise ValueError("Live optimizer stepping currently supports only the Python backend.")
    return create_debug_stepper(
        N=N,
        dim=dim,
        p_start=p_start,
        p_end=p_end,
        obstacles=obstacles,
        clearance_fn=compute_min_clearance,
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
        init_curve=init_curve,
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
