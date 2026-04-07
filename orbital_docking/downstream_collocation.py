from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import BFGS, LinearConstraint, NonlinearConstraint, minimize

from . import constants
from .bezier import BezierCurve
from .optimization import (
    _accel_total,
    _jacobian_numeric,
    generate_initial_control_points,
    optimize_orbital_docking,
)


DIM = 3


@dataclass(frozen=True)
class DirectCollocationProblem:
    r0: np.ndarray
    v0: np.ndarray
    rf: np.ndarray
    vf: np.ndarray
    transfer_time_s: float
    koz_radius_km: float
    n_intervals: int

    @property
    def n_nodes(self) -> int:
        return self.n_intervals + 1

    @property
    def dt(self) -> float:
        return self.transfer_time_s / self.n_intervals

    @property
    def tau_grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.n_nodes)


@dataclass(frozen=True)
class DirectCollocationConfig:
    maxiter: int = 200
    gtol: float = 1e-6
    barrier_tol: float = 1e-6
    verbose: bool = False


def _rotz(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotx(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _eci_from_circular_elements(
    radius_km: float,
    inc_deg: float,
    raan_deg: float,
    u_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    mu = constants.EARTH_MU_SCALED
    inc = np.deg2rad(inc_deg)
    raan = np.deg2rad(raan_deg)
    u = np.deg2rad(u_deg)

    r_pqw = np.array([radius_km * np.cos(u), radius_km * np.sin(u), 0.0])
    v_circ = np.sqrt(mu / radius_km)
    v_pqw = v_circ * np.array([-np.sin(u), np.cos(u), 0.0])

    q = _rotz(raan) @ _rotx(inc)
    return q @ r_pqw, q @ v_pqw


def make_demo_problem(n_intervals: int = 12) -> DirectCollocationProblem:
    progress_radius_km = constants.EARTH_RADIUS_KM + constants.PROGRESS_START_ALTITUDE_KM
    iss_radius_km = constants.EARTH_RADIUS_KM + constants.ISS_TARGET_ALTITUDE_KM
    inclination_deg = 51.64
    raan_deg = 0.0
    iss_u_deg = 45.0
    progress_u_deg = iss_u_deg - 30.0

    r0, v0 = _eci_from_circular_elements(
        progress_radius_km,
        inclination_deg,
        raan_deg,
        progress_u_deg,
    )
    rf, vf = _eci_from_circular_elements(
        iss_radius_km,
        inclination_deg,
        raan_deg,
        iss_u_deg,
    )
    return DirectCollocationProblem(
        r0=r0,
        v0=v0,
        rf=rf,
        vf=vf,
        transfer_time_s=float(constants.TRANSFER_TIME_S),
        koz_radius_km=float(constants.KOZ_RADIUS),
        n_intervals=int(n_intervals),
    )


def _pack_variables(r: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate([r.reshape(-1), v.reshape(-1), u.reshape(-1)])


def _unpack_variables(x: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_block = n_nodes * DIM
    r = x[:n_block].reshape(n_nodes, DIM)
    v = x[n_block : 2 * n_block].reshape(n_nodes, DIM)
    u = x[2 * n_block :].reshape(n_nodes, DIM)
    return r, v, u


def _gravity_accel(r_km: np.ndarray) -> np.ndarray:
    return _accel_total(
        r_km,
        constants.EARTH_MU_SCALED,
        constants.EARTH_RADIUS_KM,
        constants.EARTH_J2,
    )


def _gravity_jacobian(r_km: np.ndarray) -> np.ndarray:
    return _jacobian_numeric(
        lambda q: _accel_total(
            q,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        ),
        r_km,
    )


def _hermite_state_profile(problem: DirectCollocationProblem) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = problem.tau_grid[:, None]
    t_scale = problem.transfer_time_s

    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    r = (
        h00 * problem.r0
        + t_scale * h10 * problem.v0
        + h01 * problem.rf
        + t_scale * h11 * problem.vf
    )

    dh00 = 6.0 * s**2 - 6.0 * s
    dh10 = 3.0 * s**2 - 4.0 * s + 1.0
    dh01 = -6.0 * s**2 + 6.0 * s
    dh11 = 3.0 * s**2 - 2.0 * s
    v = (
        dh00 * problem.r0
        + t_scale * dh10 * problem.v0
        + dh01 * problem.rf
        + t_scale * dh11 * problem.vf
    ) / t_scale

    d2h00 = 12.0 * s - 6.0
    d2h10 = 6.0 * s - 4.0
    d2h01 = -12.0 * s + 6.0
    d2h11 = 6.0 * s - 2.0
    a = (
        d2h00 * problem.r0
        + t_scale * d2h10 * problem.v0
        + d2h01 * problem.rf
        + t_scale * d2h11 * problem.vf
    ) / (t_scale**2)
    return r, v, a


def build_naive_initial_guess(problem: DirectCollocationProblem) -> np.ndarray:
    r, v, a = _hermite_state_profile(problem)
    u = np.array([a_i - _gravity_accel(r_i) for r_i, a_i in zip(r, a)])
    return _pack_variables(r, v, u)


def build_bezier_warm_start(
    problem: DirectCollocationProblem,
    control_points: np.ndarray,
) -> np.ndarray:
    curve = BezierCurve(control_points)
    taus = problem.tau_grid
    r = np.array([curve.point(tau) for tau in taus])
    v = np.array([curve.velocity(tau) for tau in taus]) / problem.transfer_time_s
    a = np.array([curve.acceleration(tau) for tau in taus]) / (problem.transfer_time_s**2)

    # Force exact downstream boundary states so both initializations use the
    # same boundary information and variable blocks.
    r[0] = problem.r0
    r[-1] = problem.rf
    v[0] = problem.v0
    v[-1] = problem.vf

    u = np.array([a_i - _gravity_accel(r_i) for r_i, a_i in zip(r, a)])
    return _pack_variables(r, v, u)


def build_demo_bezier_warm_start(
    degree: int = 4,
    n_seg: int = 16,
    objective_mode: str = "energy",
    max_iter: int = 120,
    tol: float = 1e-8,
    use_cache: bool = True,
    ignore_existing_cache: bool = False,
) -> tuple[np.ndarray, dict]:
    problem = make_demo_problem()
    p_init = generate_initial_control_points(degree, problem.r0, problem.rf)
    return optimize_orbital_docking(
        p_init,
        n_seg=n_seg,
        r_e=problem.koz_radius_km,
        max_iter=max_iter,
        tol=tol,
        objective_mode=objective_mode,
        enforce_prograde=True,
        v0=None,
        v1=None,
        use_cache=use_cache,
        ignore_existing_cache=ignore_existing_cache,
        verbose=False,
        debug=False,
    )


def _build_boundary_constraint(problem: DirectCollocationProblem) -> LinearConstraint:
    n_nodes = problem.n_nodes
    n_vars = 3 * n_nodes * DIM
    n_block = n_nodes * DIM
    rows = []
    vals = []

    for block_start, target, node_idx in (
        (0, problem.r0, 0),
        (0, problem.rf, n_nodes - 1),
        (n_block, problem.v0, 0),
        (n_block, problem.vf, n_nodes - 1),
    ):
        for dim_idx in range(DIM):
            row = np.zeros(n_vars)
            row[block_start + node_idx * DIM + dim_idx] = 1.0
            rows.append(row)
            vals.append(target[dim_idx])

    a_mat = np.vstack(rows)
    b_vec = np.array(vals)
    return LinearConstraint(a_mat, b_vec, b_vec)


def _dynamics_defects(x: np.ndarray, problem: DirectCollocationProblem) -> np.ndarray:
    r, v, u = _unpack_variables(x, problem.n_nodes)
    dt = problem.dt
    defects = []
    for k in range(problem.n_intervals):
        f_k = u[k] + _gravity_accel(r[k])
        f_k1 = u[k + 1] + _gravity_accel(r[k + 1])
        defects.append(r[k + 1] - r[k] - 0.5 * dt * (v[k] + v[k + 1]))
        defects.append(v[k + 1] - v[k] - 0.5 * dt * (f_k + f_k1))
    return np.concatenate(defects)


def _dynamics_defects_jacobian(x: np.ndarray, problem: DirectCollocationProblem) -> np.ndarray:
    r, _v, _u = _unpack_variables(x, problem.n_nodes)
    n_nodes = problem.n_nodes
    n_vars = 3 * n_nodes * DIM
    dt = problem.dt
    jac = np.zeros((problem.n_intervals * 2 * DIM, n_vars))
    n_block = n_nodes * DIM
    u_block = 2 * n_block

    for k in range(problem.n_intervals):
        row_r = (2 * k) * DIM
        row_v = row_r + DIM
        r0_idx = k * DIM
        r1_idx = (k + 1) * DIM
        v0_idx = n_block + k * DIM
        v1_idx = n_block + (k + 1) * DIM
        u0_idx = u_block + k * DIM
        u1_idx = u_block + (k + 1) * DIM

        jac[row_r : row_r + DIM, r0_idx : r0_idx + DIM] = -np.eye(DIM)
        jac[row_r : row_r + DIM, r1_idx : r1_idx + DIM] = np.eye(DIM)
        jac[row_r : row_r + DIM, v0_idx : v0_idx + DIM] = -0.5 * dt * np.eye(DIM)
        jac[row_r : row_r + DIM, v1_idx : v1_idx + DIM] = -0.5 * dt * np.eye(DIM)

        g_jac_0 = _gravity_jacobian(r[k])
        g_jac_1 = _gravity_jacobian(r[k + 1])
        jac[row_v : row_v + DIM, r0_idx : r0_idx + DIM] = -0.5 * dt * g_jac_0
        jac[row_v : row_v + DIM, r1_idx : r1_idx + DIM] = -0.5 * dt * g_jac_1
        jac[row_v : row_v + DIM, v0_idx : v0_idx + DIM] = -np.eye(DIM)
        jac[row_v : row_v + DIM, v1_idx : v1_idx + DIM] = np.eye(DIM)
        jac[row_v : row_v + DIM, u0_idx : u0_idx + DIM] = -0.5 * dt * np.eye(DIM)
        jac[row_v : row_v + DIM, u1_idx : u1_idx + DIM] = -0.5 * dt * np.eye(DIM)

    return jac


def _koz_margins(x: np.ndarray, problem: DirectCollocationProblem) -> np.ndarray:
    r, _, _ = _unpack_variables(x, problem.n_nodes)
    radii = np.linalg.norm(r, axis=1)
    return radii - problem.koz_radius_km


def _koz_margins_jacobian(x: np.ndarray, problem: DirectCollocationProblem) -> np.ndarray:
    r, _, _ = _unpack_variables(x, problem.n_nodes)
    n_nodes = problem.n_nodes
    n_vars = 3 * n_nodes * DIM
    jac = np.zeros((n_nodes, n_vars))
    for k, r_k in enumerate(r):
        norm_r = np.linalg.norm(r_k)
        if norm_r > 1e-12:
            jac[k, k * DIM : (k + 1) * DIM] = r_k / norm_r
    return jac


def _objective_weights(problem: DirectCollocationProblem) -> np.ndarray:
    weights = np.ones(problem.n_nodes)
    weights[0] = 0.5
    weights[-1] = 0.5
    return weights


def _control_effort_objective(x: np.ndarray, problem: DirectCollocationProblem) -> float:
    _, _, u = _unpack_variables(x, problem.n_nodes)
    weights = _objective_weights(problem)[:, None]
    return 0.5 * problem.dt * float(np.sum(weights * (u**2)))


def _control_effort_gradient(x: np.ndarray, problem: DirectCollocationProblem) -> np.ndarray:
    _, _, u = _unpack_variables(x, problem.n_nodes)
    grad = np.zeros_like(x)
    n_block = problem.n_nodes * DIM
    weights = _objective_weights(problem)[:, None]
    grad[2 * n_block :] = (problem.dt * weights * u).reshape(-1)
    return grad


def _control_effort_hessian(_x: np.ndarray, problem: DirectCollocationProblem) -> np.ndarray:
    n_vars = 3 * problem.n_nodes * DIM
    hess = np.zeros((n_vars, n_vars))
    n_block = problem.n_nodes * DIM
    weights = _objective_weights(problem)
    for k, weight in enumerate(weights):
        idx = 2 * n_block + k * DIM
        hess[idx : idx + DIM, idx : idx + DIM] = problem.dt * weight * np.eye(DIM)
    return hess


def _segment_min_radius(r0: np.ndarray, r1: np.ndarray) -> float:
    seg = r1 - r0
    denom = float(seg @ seg)
    if denom <= 1e-12:
        return float(np.linalg.norm(r0))
    alpha = float(np.clip(-(r0 @ seg) / denom, 0.0, 1.0))
    p = r0 + alpha * seg
    return float(np.linalg.norm(p))


def _collect_metrics(
    x: np.ndarray,
    problem: DirectCollocationProblem,
    res,
    solve_time_s: float,
) -> dict:
    r, v, u = _unpack_variables(x, problem.n_nodes)
    boundary_constraint = _build_boundary_constraint(problem)
    boundary_residual = boundary_constraint.A @ x - boundary_constraint.lb
    dyn_defects = _dynamics_defects(x, problem)
    node_margins = _koz_margins(x, problem)
    segment_min_radius = min(
        _segment_min_radius(r[k], r[k + 1]) for k in range(problem.n_intervals)
    )
    segment_margin = segment_min_radius - problem.koz_radius_km
    objective_value = _control_effort_objective(x, problem)
    max_boundary_violation = float(np.max(np.abs(boundary_residual)))
    max_dyn_defect = float(np.max(np.abs(dyn_defects)))
    min_node_margin = float(np.min(node_margins))

    return {
        "success": bool(getattr(res, "success", False))
        and max_boundary_violation <= 1e-4
        and max_dyn_defect <= 1e-4
        and min_node_margin >= -1e-6,
        "solve_time_s": float(solve_time_s),
        "iteration_count": int(getattr(res, "nit", getattr(res, "niter", -1))),
        "final_objective": float(objective_value),
        "constraint_satisfaction": {
            "max_boundary_violation": max_boundary_violation,
            "max_dynamics_defect": max_dyn_defect,
            "max_eq_violation": float(max(max_boundary_violation, max_dyn_defect)),
            "min_node_margin_km": min_node_margin,
            "min_segment_margin_km": float(segment_margin),
        },
        "solver_status": int(getattr(res, "status", -1)),
        "solver_message": str(getattr(res, "message", "")),
        "node_count": int(problem.n_nodes),
        "r": r.tolist(),
        "v": v.tolist(),
        "u": u.tolist(),
    }


def solve_direct_collocation(
    problem: DirectCollocationProblem,
    x0: np.ndarray,
    config: DirectCollocationConfig | None = None,
):
    if config is None:
        config = DirectCollocationConfig()

    boundary_constraint = _build_boundary_constraint(problem)
    dynamics_constraint = NonlinearConstraint(
        lambda x: _dynamics_defects(x, problem),
        0.0,
        0.0,
        jac=lambda x: _dynamics_defects_jacobian(x, problem),
        hess=BFGS(),
    )
    koz_constraint = NonlinearConstraint(
        lambda x: _koz_margins(x, problem),
        0.0,
        np.inf,
        jac=lambda x: _koz_margins_jacobian(x, problem),
        hess=BFGS(),
    )

    t0 = time.time()
    res = minimize(
        lambda x: _control_effort_objective(x, problem),
        x0,
        method="trust-constr",
        jac=lambda x: _control_effort_gradient(x, problem),
        hess=lambda x: _control_effort_hessian(x, problem),
        constraints=[boundary_constraint, dynamics_constraint, koz_constraint],
        options={
            "maxiter": int(config.maxiter),
            "gtol": float(config.gtol),
            "barrier_tol": float(config.barrier_tol),
            "verbose": 1 if config.verbose else 0,
        },
    )
    solve_time_s = time.time() - t0
    metrics = _collect_metrics(res.x, problem, res, solve_time_s)
    return res, metrics


def run_downstream_comparison(
    control_points: np.ndarray,
    problem: DirectCollocationProblem | None = None,
    config: DirectCollocationConfig | None = None,
) -> dict:
    if problem is None:
        problem = make_demo_problem()
    if config is None:
        config = DirectCollocationConfig()

    naive_guess = build_naive_initial_guess(problem)
    warm_guess = build_bezier_warm_start(problem, control_points)

    _, naive_metrics = solve_direct_collocation(problem, naive_guess, config=config)
    _, warm_metrics = solve_direct_collocation(problem, warm_guess, config=config)
    return {
        "problem": {
            "transfer_time_s": float(problem.transfer_time_s),
            "koz_radius_km": float(problem.koz_radius_km),
            "n_intervals": int(problem.n_intervals),
            "n_nodes": int(problem.n_nodes),
        },
        "comparison": {
            "naive": naive_metrics,
            "bezier_warm_start": warm_metrics,
        },
    }
