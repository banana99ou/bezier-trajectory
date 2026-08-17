"""
Optimization entrypoints for space-time Bezier trajectories.

Public API only: request normalization, backend dispatch, batch orchestration.
Debugger factories live in ``rust_debug_stepper``; clearance computation and
obstacle geometry live in ``geometry``.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import compute_min_clearance, obstacle_array_bundle
from .objective import build_initial_guess

# Mirrors DEFAULT_TRUST_RADIUS in rust_optimizer/core/src/spacetime_optimizer.rs.
DEFAULT_TRUST_RADIUS = 0.5

try:
    import bezier_opt as _bezier_opt_rs
except ImportError:  # pragma: no cover - exercised when the native extension is unavailable.
    _bezier_opt_rs = None


# Mirrors `stop_reason` in rust_optimizer/core/src/spacetime_optimizer.rs.
# Only `merit_streak` and `stationary` are success claims; the rest mean the loop
# gave up, and the labels say so rather than leaving it to the reader.
_STOP_REASONS = {
    0: "iteration_cap (gave up)",
    1: "merit_streak",
    2: "trust_collapse (gave up unless certified)",
    3: "qp_failure (gave up)",
    4: "stationary",
    -1: "still running / not reported",
}


def _optimize_spacetime_rust(
    P_init: np.ndarray,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    cap_bulge_ratio: float = 2.0,
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
        cap_bulge_ratio=cap_bulge_ratio,
    )
    P_opt = np.asarray(P_opt, dtype=float)
    info = dict(info)
    info["backend"] = "rust"

    if verbose:
        iterations = int(info.get("iterations", -1))
        clearance = float(info.get("min_clearance", math.nan))
        feasible = bool(info.get("feasible", 0.0))
        delta = float(info.get("final_delta_norm", math.nan))
        total_slack = float(info.get("total_koz_slack", 0.0))
        print(
            f"SCP: N={n_cp - 1}, dim={dim}, n_seg={n_seg}, n_cp={n_cp}, n_obs={len(obstacles)}, backend=rust"
        )
        # Termination label comes from the solver, which is the only thing that knows
        # why it stopped. The previous version re-derived it here as
        # "delta < tol -> converged", which reports a small step as an optimality
        # claim -- and on the legacy path every step is accepted unconditionally, so a
        # small step can mean the iterate stopped moving for any reason at all.
        reason = _STOP_REASONS.get(int(info.get("stop_reason", -1)), "unknown")
        converged = bool(info.get("converged", 0.0))
        print(f"  rust result: iterations={iterations}/{max_iter}, clearance={clearance:.4f}")
        print(
            f"  termination: {reason}, converged={converged}, feasible={feasible}, "
            f"delta={delta:.2e}, koz_slack={total_slack:.2e}"
        )
        if info.get("returned_best_iterate", 0.0):
            print(
                "  NOTE: the final iterate penetrated an obstacle; returning the best "
                "feasible iterate seen instead. It is NOT the point the loop stopped on."
            )
        print(
            f"  steps: accept={int(info.get('accept_count', 0))} "
            f"reject={int(info.get('reject_count', 0))} "
            f"null={int(info.get('null_step_count', 0))} "
            f"bootstrap={int(info.get('bootstrap_count', 0))}"
        )
        print(
            f"  rho: mean={info.get('rho_mean', math.nan):.4f} "
            f"min={info.get('rho_min', math.nan):.4f} "
            f"max={info.get('rho_max', math.nan):.4f} "
            f"n={int(info.get('rho_samples', 0))}, "
            f"final_trust={info.get('final_trust', math.nan):.3e}"
        )
        print(
            f"  hull certificate violation at returned iterate: "
            f"{info.get('koz_violation_reference', math.nan):.3e}"
        )

    return P_opt, info


def optimize_spacetime_from_control_points(
    P_init,
    obstacles: list[dict],
    n_seg: int = 8,
    max_iter: int = 30,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.5,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    cap_bulge_ratio: float = 2.0,
    verbose: bool = True,
) -> tuple[np.ndarray, dict]:
    """Optimize a space-time Bezier curve from an initial control polygon.

    Returns (control_points, info) where info always contains 'backend'.
    """
    return _optimize_spacetime_rust(
        np.asarray(P_init, dtype=float),
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
        cap_bulge_ratio=cap_bulge_ratio,
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
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
    min_dt: float = 0.1,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
    cap_bulge_ratio: float = 2.0,
    verbose: bool = True,
    init_curve: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Public optimizer entrypoint.

    Returns (control_points, info) where info always contains 'backend'.
    """
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
        cap_bulge_ratio=cap_bulge_ratio,
        verbose=verbose,
    )


def optimize_scenario(
    scenario: dict,
    configs: list[tuple[int, int]],
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
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

        P_opt, opt_info = optimize_spacetime(
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
        )
        backend_used = opt_info["backend"]

        clearance = compute_min_clearance(P_opt, obstacles, dim=len(p_start), n_eval=3000)
        if verbose:
            print(f"  Final clearance: {clearance:.4f}")

        key = f"N{N}_seg{n_seg}"
        # `feasible` says the sampled curve misses the obstacles. `certified`
        # says the control-point hull satisfies the half-spaces it generates,
        # which is the property the paper claims. They are different, and a run
        # can pass the first while failing the second -- so both are recorded and
        # neither is allowed to stand in for the other.
        certificate = float(opt_info.get("koz_violation_reference", float("nan")))
        results[key] = {
            "N": int(N),
            "n_seg": int(n_seg),
            "control_points": np.asarray(P_opt, dtype=float).tolist(),
            "min_clearance": float(clearance),
            "feasible": bool(clearance > 0.0),
            "backend": backend_used,
            "converged": bool(opt_info.get("converged", 0.0)),
            "stop_reason": int(opt_info.get("stop_reason", -1)),
            "stop_label": _STOP_REASONS.get(int(opt_info.get("stop_reason", -1)), "unknown"),
            "iterations": int(opt_info.get("iterations", -1)),
            "certificate_violation": certificate,
            "certified": bool(certificate <= 1e-6),
            "accept_count": int(opt_info.get("accept_count", 0)),
            "reject_count": int(opt_info.get("reject_count", 0)),
            "returned_best_iterate": bool(opt_info.get("returned_best_iterate", 0.0)),
            "trust_radius": float(scp_trust_radius),
        }

    def _rank(item):
        _, v = item
        return (
            bool(v["feasible"]) and bool(v.get("certified", False)),
            bool(v["feasible"]),
            float(v["min_clearance"]),
        )

    best_key = max(results.items(), key=_rank)[0]

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
    max_iter: int = 200,
    tol: float = 1e-6,
    scp_prox_weight: float = 0.3,
    scp_trust_radius: float = DEFAULT_TRUST_RADIUS,
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
            max_iter=max_iter,
            tol=tol,
            scp_prox_weight=scp_prox_weight,
            scp_trust_radius=scp_trust_radius,
            min_dt=min_dt,
            verbose=verbose,
        )
    return all_outputs
