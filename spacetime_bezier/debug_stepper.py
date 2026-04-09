"""
Live, user-steppable Python SCP optimizer.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from orbital_docking.de_casteljau import segment_matrices_equal_params

from .constraints import (
    build_boundary_constraints,
    build_box_bounds,
    build_spacetime_koz_constraints,
    build_time_monotonicity,
)
from .debug_trace import DebugFrame, to_serializable
from .geometry import bezier_curve
from .objective import build_energy_objective, build_initial_guess


def clip_trust_region(x_new: np.ndarray, x_ref: np.ndarray, trust_radius: float) -> np.ndarray:
    if trust_radius <= 0.0:
        return x_new
    step = x_new - x_ref
    step_norm = np.linalg.norm(step)
    if step_norm <= trust_radius or step_norm <= 1e-15:
        return x_new
    return x_ref + (trust_radius / step_norm) * step


def compute_min_clearance_fn(P, obstacles: list[dict], dim: int, clearance_fn, n_eval: int = 1500) -> float:
    return float(clearance_fn(P, obstacles, dim=dim, n_eval=n_eval))


def _constraint_payload(name: str, constraint) -> dict:
    if constraint is None:
        return {"name": name, "present": False}
    matrix = np.asarray(constraint.A, dtype=float)
    return {
        "name": name,
        "present": True,
        "matrix": matrix,
        "lower": np.asarray(constraint.lb, dtype=float),
        "upper": np.asarray(constraint.ub, dtype=float),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
    }


def _bounds_payload(bounds) -> dict:
    return {
        "lower": np.asarray(bounds.lb, dtype=float),
        "upper": np.asarray(bounds.ub, dtype=float),
        "shape": [int(len(bounds.lb))],
    }


def _segment_snapshots(A_list, P: np.ndarray, num_pts: int = 40) -> list[dict]:
    segments = []
    for segment_idx, A_seg in enumerate(A_list):
        Q = A_seg @ P
        centroid = Q.mean(axis=0)
        segments.append(
            {
                "segment_index": int(segment_idx),
                "control_points": np.asarray(Q, dtype=float),
                "centroid": np.asarray(centroid, dtype=float),
                "centroid_time": float(centroid[-1]),
                "sampled_curve": bezier_curve(np.asarray(Q, dtype=float), num_pts=num_pts),
            }
        )
    return segments


class SpacetimeOptimizerStepper:
    """State machine that exposes one optimizer stage at a time."""

    def __init__(
        self,
        P_init: np.ndarray,
        obstacles: list[dict],
        clearance_fn,
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
    ) -> None:
        self.obstacles = list(obstacles)
        self.P = np.asarray(P_init, dtype=float).copy()
        self.clearance_fn = clearance_fn
        self.n_cp, self.dim = self.P.shape
        self.N = self.n_cp - 1
        self.n_seg = int(n_seg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scp_prox_weight = float(scp_prox_weight)
        self.scp_trust_radius = float(scp_trust_radius)
        self.min_dt = float(min_dt)
        self.coord_lb = float(coord_lb)
        self.coord_ub = float(coord_ub)
        self.time_lb = float(time_lb)
        self.time_ub_scale = float(time_ub_scale)

        self.A_list = segment_matrices_equal_params(self.N, self.n_seg)
        self.H_energy = build_energy_objective(self.N, self.dim)
        self.boundary_constraint = build_boundary_constraints(self.n_cp, self.dim, self.P[0], self.P[-1])
        self.time_constraint = build_time_monotonicity(self.n_cp, self.dim, min_dt=self.min_dt)
        self.bounds = build_box_bounds(
            self.P,
            coord_lb=self.coord_lb,
            coord_ub=self.coord_ub,
            time_lb=self.time_lb,
            time_ub_scale=self.time_ub_scale,
        )

        self.initial_clearance = compute_min_clearance_fn(self.P, self.obstacles, self.dim, self.clearance_fn)
        self.best_P = self.P.copy()
        self.best_clearance = float(self.initial_clearance)
        self.final_control_points = self.P.copy()
        self.result_info: dict = {}

        self.iteration = 0
        self._phase = "init-guess"
        self._frame_id = 0
        self._done = False
        self._finalized = False
        self._converged = False
        self._last_delta = math.nan

        self._x_ref = None
        self._segment_data = []
        self._koz_constraint = None
        self._koz_debug = {"segments": [], "row_count": 0}
        self._constraints = []
        self._H = None
        self._f = None
        self._result = None
        self._raw_candidate = None
        self._accepted_candidate = None
        self._previous_control_points = None

    @property
    def done(self) -> bool:
        return self._done

    def _next_frame_id(self) -> int:
        frame_id = self._frame_id
        self._frame_id += 1
        return frame_id

    def _frame(self, stage: str, label: str, payload: dict, iteration: int | None = None) -> DebugFrame:
        return DebugFrame(
            frame_id=self._next_frame_id(),
            stage=stage,
            label=label,
            iteration=iteration,
            payload=to_serializable(payload),
        )

    def _step_init_guess(self) -> DebugFrame:
        self._phase = "boundary-constraints"
        return self._frame(
            "init-guess",
            "Initial control polygon",
            {
                "backend": "python",
                "control_points": self.P,
                "problem": {
                    "N": self.N,
                    "dim": self.dim,
                    "n_cp": self.n_cp,
                    "n_seg": self.n_seg,
                    "n_obs": len(self.obstacles),
                },
                "parameters": {
                    "max_iter": self.max_iter,
                    "tol": self.tol,
                    "scp_prox_weight": self.scp_prox_weight,
                    "scp_trust_radius": self.scp_trust_radius,
                    "min_dt": self.min_dt,
                },
                "initial_clearance": self.initial_clearance,
                "energy_matrix": self.H_energy,
            },
        )

    def _step_boundary_constraints(self) -> DebugFrame:
        self._phase = "time-monotonicity"
        return self._frame(
            "boundary-constraints",
            "Boundary constraints",
            {
                "control_points": self.P,
                "boundary_constraint": _constraint_payload("boundary", self.boundary_constraint),
            },
        )

    def _step_time_monotonicity(self) -> DebugFrame:
        self._phase = "box-bounds"
        return self._frame(
            "time-monotonicity",
            "Time monotonicity constraints",
            {
                "control_points": self.P,
                "time_constraint": _constraint_payload("time_monotonicity", self.time_constraint),
            },
        )

    def _step_box_bounds(self) -> DebugFrame:
        self._phase = "iteration-start" if self.max_iter > 0 else "finalize"
        return self._frame(
            "box-bounds",
            "Box bounds",
            {
                "control_points": self.P,
                "bounds": _bounds_payload(self.bounds),
            },
        )

    def _step_iteration_start(self) -> DebugFrame:
        if self.iteration >= self.max_iter:
            self._phase = "finalize"
            return self._step_finalize()
        self._x_ref = self.P.reshape(-1)
        self._phase = "segment-subdivision"
        return self._frame(
            "iteration-start",
            f"Iteration {self.iteration + 1} start",
            {
                "control_points": self.P,
                "best_clearance_so_far": self.best_clearance,
                "iteration_parameters": {
                    "scp_prox_weight": self.scp_prox_weight,
                    "scp_trust_radius": self.scp_trust_radius,
                    "tol": self.tol,
                },
            },
            iteration=self.iteration,
        )

    def _step_segment_subdivision(self) -> DebugFrame:
        self._segment_data = _segment_snapshots(self.A_list, self.P)
        self._phase = "koz-linearization"
        return self._frame(
            "segment-subdivision",
            f"Iteration {self.iteration + 1} subdivision",
            {
                "control_points": self.P,
                "segments": self._segment_data,
            },
            iteration=self.iteration,
        )

    def _step_koz_linearization(self) -> DebugFrame:
        self._koz_constraint, self._koz_debug = build_spacetime_koz_constraints(
            self.A_list,
            self.P,
            self.obstacles,
            self.dim,
            return_debug=True,
        )
        self._phase = "qp-assembly"
        return self._frame(
            "koz-linearization",
            f"Iteration {self.iteration + 1} supporting half-spaces",
            {
                "control_points": self.P,
                "segments": self._segment_data,
                "koz": {
                    "constraint": _constraint_payload("koz", self._koz_constraint),
                    "row_count": self._koz_debug["row_count"],
                    "segments": self._koz_debug["segments"],
                },
            },
            iteration=self.iteration,
        )

    def _step_qp_assembly(self) -> DebugFrame:
        self._H = self.H_energy + self.scp_prox_weight * np.eye(self.n_cp * self.dim)
        self._f = -self.scp_prox_weight * self._x_ref
        self._constraints = [self.boundary_constraint, self.time_constraint]
        if self._koz_constraint is not None:
            self._constraints.append(self._koz_constraint)
        self._phase = "solver-candidate"
        return self._frame(
            "qp-assembly",
            f"Iteration {self.iteration + 1} QP assembly",
            {
                "control_points": self.P,
                "qp": {
                    "matrix": self._H,
                    "gradient": self._f,
                    "constraint_count": len(self._constraints),
                    "constraint_shapes": [
                        _constraint_payload(f"constraint_{idx}", constraint)["shape"]
                        for idx, constraint in enumerate(self._constraints)
                    ],
                    "bounds": _bounds_payload(self.bounds),
                },
            },
            iteration=self.iteration,
        )

    def _step_solver_candidate(self) -> DebugFrame:
        self._result = minimize(
            lambda x: 0.5 * x @ self._H @ x + self._f @ x,
            self._x_ref,
            jac=lambda x: self._H @ x + self._f,
            hess=lambda _x: self._H,
            method="trust-constr",
            constraints=self._constraints,
            bounds=self.bounds,
            options={"maxiter": 80, "gtol": 1e-9, "verbose": 0},
        )
        self._raw_candidate = np.asarray(self._result.x, dtype=float).reshape(self.n_cp, self.dim)
        self._phase = "trust-region"
        return self._frame(
            "solver-candidate",
            f"Iteration {self.iteration + 1} solver candidate",
            {
                "control_points": self.P,
                "candidate_control_points": self._raw_candidate,
                "solver": {
                    "success": bool(getattr(self._result, "success", False)),
                    "status": int(getattr(self._result, "status", -1)),
                    "message": str(getattr(self._result, "message", "")),
                    "fun": float(getattr(self._result, "fun", math.nan)),
                    "nit": int(getattr(self._result, "nit", -1)),
                },
            },
            iteration=self.iteration,
        )

    def _step_trust_region(self) -> DebugFrame:
        raw_flat = np.asarray(self._result.x, dtype=float)
        accepted_flat = clip_trust_region(raw_flat, self._x_ref, self.scp_trust_radius)
        self._accepted_candidate = accepted_flat.reshape(self.n_cp, self.dim)
        raw_step_norm = float(np.linalg.norm(raw_flat - self._x_ref))
        used_step_norm = float(np.linalg.norm(accepted_flat - self._x_ref))
        clip_ratio = 1.0 if raw_step_norm <= 1e-15 else used_step_norm / raw_step_norm
        self._phase = "post-eval"
        return self._frame(
            "trust-region",
            f"Iteration {self.iteration + 1} trust-region",
            {
                "control_points": self.P,
                "candidate_control_points": self._raw_candidate,
                "accepted_control_points": self._accepted_candidate,
                "trust_region": {
                    "radius": self.scp_trust_radius,
                    "raw_step_norm": raw_step_norm,
                    "used_step_norm": used_step_norm,
                    "clip_ratio": clip_ratio,
                    "clipped": bool(not np.allclose(raw_flat, accepted_flat)),
                },
            },
            iteration=self.iteration,
        )

    def _step_post_eval(self) -> DebugFrame:
        self._previous_control_points = self.P.copy()
        self.P = self._accepted_candidate.copy()
        self._last_delta = float(np.linalg.norm(self.P - self._previous_control_points))
        clearance = compute_min_clearance_fn(self.P, self.obstacles, self.dim, self.clearance_fn)
        updated_best = False
        if clearance > 0.0 and clearance > self.best_clearance:
            self.best_P = self.P.copy()
            self.best_clearance = clearance
            updated_best = True
        self._converged = bool(self._last_delta < self.tol)
        frame = self._frame(
            "post-eval",
            f"Iteration {self.iteration + 1} evaluation",
            {
                "control_points_before": self._previous_control_points,
                "control_points": self.P,
                "metrics": {
                    "delta": self._last_delta,
                    "clearance": clearance,
                    "cost": float(getattr(self._result, "fun", math.nan)),
                    "updated_best_feasible": updated_best,
                    "converged": self._converged,
                },
            },
            iteration=self.iteration,
        )
        self.iteration += 1
        self._phase = "finalize" if self._converged or self.iteration >= self.max_iter else "iteration-start"
        return frame

    def _step_finalize(self) -> DebugFrame:
        if self._finalized:
            self._done = True
            return None
        last_iterate = self.P.copy()
        final_clearance = compute_min_clearance_fn(last_iterate, self.obstacles, self.dim, self.clearance_fn)
        used_best_feasible = False
        final_control_points = last_iterate
        if final_clearance < 0.0 and self.best_clearance > 0.0:
            final_control_points = self.best_P.copy()
            final_clearance = self.best_clearance
            used_best_feasible = True

        self.final_control_points = final_control_points.copy()
        self.result_info = {
            "backend": "python",
            "iterations": int(self.iteration),
            "min_clearance": float(final_clearance),
            "best_clearance": float(self.best_clearance),
            "used_best_feasible": bool(used_best_feasible),
            "final_delta_norm": float(self._last_delta),
            "converged": bool(self._converged),
        }
        self._finalized = True
        self._done = True
        return self._frame(
            "finalize",
            "Final result",
            {
                "control_points": self.final_control_points,
                "last_iterate_control_points": last_iterate,
                "best_control_points": self.best_P,
                "used_best_feasible": used_best_feasible,
                "final_clearance": final_clearance,
                "iterations_completed": self.iteration,
                "converged": self._converged,
            },
            iteration=None if self.iteration == 0 else self.iteration - 1,
        )

    def next_frame(self) -> DebugFrame | None:
        if self._done and self._finalized:
            return None
        if self._phase == "init-guess":
            return self._step_init_guess()
        if self._phase == "boundary-constraints":
            return self._step_boundary_constraints()
        if self._phase == "time-monotonicity":
            return self._step_time_monotonicity()
        if self._phase == "box-bounds":
            return self._step_box_bounds()
        if self._phase == "iteration-start":
            return self._step_iteration_start()
        if self._phase == "segment-subdivision":
            return self._step_segment_subdivision()
        if self._phase == "koz-linearization":
            return self._step_koz_linearization()
        if self._phase == "qp-assembly":
            return self._step_qp_assembly()
        if self._phase == "solver-candidate":
            return self._step_solver_candidate()
        if self._phase == "trust-region":
            return self._step_trust_region()
        if self._phase == "post-eval":
            return self._step_post_eval()
        if self._phase == "finalize":
            return self._step_finalize()
        raise RuntimeError(f"Unknown optimizer debug phase: {self._phase}")

    def run_to_completion(self, verbose: bool = False) -> tuple[np.ndarray, dict]:
        while True:
            frame = self.next_frame()
            if frame is None:
                break
            if verbose and frame.stage == "post-eval":
                metrics = frame.payload["metrics"]
                print(
                    f"  iter {frame.iteration + 1}: delta={metrics['delta']:.6f}, "
                    f"cost={metrics['cost']:.4f}, clearance={metrics['clearance']:.4f}"
                )
                if metrics["converged"]:
                    print(f"  Converged at iter {frame.iteration + 1}")
            if verbose and frame.stage == "finalize" and frame.payload["used_best_feasible"]:
                print("  Using best feasible iterate")
        return self.final_control_points.copy(), dict(self.result_info)


def create_debug_stepper_from_control_points(
    P_init,
    obstacles: list[dict],
    clearance_fn,
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
) -> SpacetimeOptimizerStepper:
    return SpacetimeOptimizerStepper(
        P_init=np.asarray(P_init, dtype=float),
        obstacles=obstacles,
        clearance_fn=clearance_fn,
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


def create_debug_stepper(
    N: int,
    dim: int,
    p_start,
    p_end,
    obstacles: list[dict],
    clearance_fn,
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
) -> SpacetimeOptimizerStepper:
    n_cp = int(N) + 1
    P_init = build_initial_guess(p_start, p_end, n_cp, init_curve=init_curve)
    if dim != P_init.shape[1]:
        raise ValueError(f"Expected dim={dim}, got initial guess with dim={P_init.shape[1]}")
    return create_debug_stepper_from_control_points(
        P_init,
        obstacles,
        clearance_fn=clearance_fn,
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
