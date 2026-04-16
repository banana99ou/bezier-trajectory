"""
Truthful Rust-backed debugger stepper built from Rust optimizer logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbital_docking.de_casteljau import segment_matrices_equal_params

from .debug_trace import DebugFrame
from .geometry import bezier_curve

RUST_DEBUG_LOG_PATH = Path("/Volumes/Sandisk/code/bezier-trajectory-merge/.cursor/debug-9abff6.log")


def _segment_snapshots(a_list, control_points: np.ndarray, num_pts: int = 40) -> list[dict]:
    segments = []
    for segment_idx, a_seg in enumerate(a_list):
        q = np.asarray(a_seg, dtype=float) @ control_points
        centroid = q.mean(axis=0)
        segments.append(
            {
                "segment_index": int(segment_idx),
                "control_points": np.asarray(q, dtype=float),
                "centroid": np.asarray(centroid, dtype=float),
                "centroid_time": float(centroid[-1]),
                "sampled_curve": bezier_curve(np.asarray(q, dtype=float), num_pts=num_pts),
            }
        )
    return segments


class RustOptimizerStepper:
    """Step through actual Rust optimizer execution without faking Python stages."""

    def __init__(
        self,
        p_init: np.ndarray,
        obstacles: list[dict],
        clearance_fn,
        rust_solver,
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
        self.P_init = np.asarray(p_init, dtype=float).copy()
        self.clearance_fn = clearance_fn
        self.rust_solver = rust_solver
        self.n_cp, self.dim = self.P_init.shape
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
        self.initial_clearance = float(
            self.clearance_fn(self.P_init, self.obstacles, dim=self.dim, n_eval=1500)
        )
        self.final_control_points = self.P_init.copy()
        self.result_info: dict = {}

        self._frames: list[DebugFrame] = []
        self._frame_cursor = 0
        self._frame_id = 0
        self._prepared = False

    @property
    def done(self) -> bool:
        return self._prepared and self._frame_cursor >= len(self._frames)

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
            payload=payload,
            profile={},
        )

    def _common_payload(self, control_points: np.ndarray | None) -> dict:
        payload = {
            "backend": "rust",
            "obstacles": self.obstacles,
            "koz": {"segments": [], "row_count": 0},
            "metrics": {},
            "solver": {},
            "trust_region": {},
            "diagnostics": {},
        }
        if control_points is not None:
            cp = np.asarray(control_points, dtype=float)
            payload["control_points"] = cp
            payload["segments"] = _segment_snapshots(self.A_list, cp)
        else:
            payload["segments"] = []
        return payload

    def _clear_log(self) -> None:
        try:
            RUST_DEBUG_LOG_PATH.unlink()
        except FileNotFoundError:
            pass

    def _read_log_entries(self) -> list[dict]:
        if not RUST_DEBUG_LOG_PATH.exists():
            return []
        entries = []
        for line in RUST_DEBUG_LOG_PATH.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
        return entries

    def _koz_payload_from_summary(self, summary: dict, control_points: np.ndarray | None) -> dict:
        koz = {
            "row_count": int(summary.get("n_rows", 0)),
            "segments": [],
            "summary": summary,
        }
        worst_segment = summary.get("worst_segment")
        worst_obstacle = summary.get("worst_obstacle")
        if worst_segment is None or worst_obstacle is None or control_points is None:
            return koz
        segment_snapshots = _segment_snapshots(self.A_list, np.asarray(control_points, dtype=float))
        selected_segment = next(
            (segment for segment in segment_snapshots if int(segment["segment_index"]) == int(worst_segment)),
            None,
        )
        if selected_segment is None:
            return koz
        selected_segment = dict(selected_segment)
        normal = list(summary.get("worst_normal_with_time_coeff", []))
        time_coefficient = float(normal[-1]) if normal else None
        spatial_normal = normal[:-1] if normal else []
        selected_segment["active_obstacles"] = [
            {
                "obstacle_index": int(worst_obstacle),
                "obstacle_name": self.obstacles[int(worst_obstacle)].get("name", f"obs{worst_obstacle}"),
                "geometry_type": "capsule",
                "center": summary.get("worst_obstacle_center", []),
                "support_point": summary.get("worst_support_point", []),
                "normal": spatial_normal,
                "time_coefficient": time_coefficient,
                "lower_bound": summary.get("worst_lb"),
                "lhs_current": summary.get("worst_lhs"),
                "margin_current": None
                if summary.get("worst_lhs") is None or summary.get("worst_lb") is None
                else float(summary["worst_lhs"]) - float(summary["worst_lb"]),
                "row_indices": [],
            }
        ]
        koz["segments"] = [selected_segment]
        return koz

    def _prepare_frames(self) -> None:
        if self._prepared:
            return

        init_payload = self._common_payload(self.P_init)
        init_payload["sampled_curve"] = bezier_curve(self.P_init, num_pts=200)
        init_payload["metrics"] = {"clearance": self.initial_clearance}
        init_payload["diagnostics"] = {
            "summary": "Actual Rust backend. Segment snapshots are derived from the Rust input control points.",
            "derived_visualization": True,
        }
        self._frames.append(self._frame("init-guess", "Initial control polygon", init_payload, iteration=None))

        subdivision_payload = self._common_payload(self.P_init)
        subdivision_payload["diagnostics"] = {
            "summary": "Equal-parameter subdivision derived from the same control polygon passed into Rust.",
            "derived_visualization": True,
        }
        self._frames.append(
            self._frame("segment-subdivision", "Segment subdivision", subdivision_payload, iteration=0)
        )

        obstacle_payload = self._common_payload(self.P_init)
        obstacle_payload["diagnostics"] = {
            "summary": "Showing raw Rust obstacle inputs. Backend-specific intermediate obstacle geometry is not exposed yet.",
            "not_available_for_backend": True,
        }
        self._frames.append(
            self._frame("obstacle-geometry", "Obstacle geometry", obstacle_payload, iteration=0)
        )

        objective_payload = self._common_payload(self.P_init)
        objective_payload["diagnostics"] = {
            "summary": "Objective metadata from the Rust optimizer configuration.",
            "derived_visualization": False,
        }
        objective_payload["objective"] = {
            "type": "spatial_acceleration_energy",
            "scp_prox_weight": self.scp_prox_weight,
            "trust_region_policy": "post_solve_clip",
            "scp_trust_radius": self.scp_trust_radius,
            "penalized_coordinates": list(range(self.dim - 1)),
            "penalizes_time_acceleration": False,
        }
        self._frames.append(
            self._frame("objective-assembly", "Objective assembly", objective_payload, iteration=0)
        )

        self._clear_log()
        p_opt, info = self.rust_solver(
            self.P_init,
            self.obstacles,
            n_seg=self.n_seg,
            max_iter=self.max_iter,
            tol=self.tol,
            scp_prox_weight=self.scp_prox_weight,
            scp_trust_radius=self.scp_trust_radius,
            min_dt=self.min_dt,
            coord_lb=self.coord_lb,
            coord_ub=self.coord_ub,
            time_lb=self.time_lb,
            time_ub_scale=self.time_ub_scale,
            verbose=False,
        )
        self.final_control_points = np.asarray(p_opt, dtype=float)
        self.result_info = dict(info)
        log_entries = self._read_log_entries()

        current_cp = self.P_init.copy()
        guessed_iteration = -1
        last_solver_payload = {}
        for entry in log_entries:
            data = dict(entry.get("data", {}))
            hypothesis_id = str(entry.get("hypothesisId", ""))
            if "control_points" in data:
                current_cp = np.asarray(data["control_points"], dtype=float)
            iteration = data.get("iteration")
            if iteration is None:
                if hypothesis_id == "H7_H8_H9_H10":
                    guessed_iteration += 1
                    iter_index = guessed_iteration
                else:
                    iter_index = max(0, guessed_iteration)
            else:
                iter_index = max(0, int(iteration) - 1)
                guessed_iteration = max(guessed_iteration, iter_index)

            if hypothesis_id == "H7_H8_H9_H10":
                payload = self._common_payload(current_cp)
                payload["koz"] = self._koz_payload_from_summary(data, current_cp)
                payload["diagnostics"] = {
                    "summary": "Actual Rust supporting-surface summary. Full per-row geometry is not exposed yet.",
                    "not_available_for_backend": False,
                }
                self._frames.append(
                    self._frame(
                        "supporting-surface-generation",
                        f"Iteration {iter_index + 1} supporting surfaces",
                        payload,
                        iteration=iter_index,
                    )
                )
            elif hypothesis_id == "H1_H3":
                payload = self._common_payload(current_cp)
                payload["koz"]["row_count"] = int(data.get("koz_rows", 0))
                payload["diagnostics"] = {
                    "summary": "Actual Rust constraint assembly counts.",
                    "total_rows": int(data.get("total_rows", 0)),
                    "best_clearance_before": data.get("best_clearance_before"),
                }
                self._frames.append(
                    self._frame(
                        "constraint-assembly",
                        f"Iteration {iter_index + 1} constraint assembly",
                        payload,
                        iteration=iter_index,
                    )
                )
            elif hypothesis_id == "H5_H6":
                payload = self._common_payload(current_cp)
                payload["solver"] = {
                    "name": "Clarabel",
                    "raw_status": data.get("status"),
                    "interpreted_status": data.get("status"),
                    "candidate_available": bool(
                        data.get("status") in {"Solved", "AlmostSolved"}
                    ),
                    "accepted": False,
                    "reason_for_rejection": None
                    if data.get("status") in {"Solved", "AlmostSolved"}
                    else "solve_qp rejected non-solved status",
                    "status": data.get("status"),
                }
                last_solver_payload = payload["solver"]
                self._frames.append(
                    self._frame("solver-call", f"Iteration {iter_index + 1} solver call", payload, iteration=iter_index)
                )
            elif hypothesis_id == "H1":
                payload = self._common_payload(current_cp)
                payload["solver"] = last_solver_payload or {
                    "name": "Clarabel",
                    "raw_status": "unknown",
                    "interpreted_status": "failed",
                    "candidate_available": False,
                    "accepted": False,
                    "reason_for_rejection": "qp solve returned none",
                }
                payload["trust_region"] = {
                    "radius": self.scp_trust_radius,
                    "candidate_available": False,
                }
                payload["diagnostics"] = {
                    "summary": "Rust candidate filter had no candidate because the solver returned no acceptable solution.",
                }
                self._frames.append(
                    self._frame(
                        "candidate-filter",
                        f"Iteration {iter_index + 1} candidate filter",
                        payload,
                        iteration=iter_index,
                    )
                )
            elif hypothesis_id == "H2":
                accepted_cp = np.asarray(data.get("accepted_control_points", current_cp), dtype=float)
                payload = self._common_payload(accepted_cp)
                payload["accepted_control_points"] = accepted_cp
                payload["trust_region"] = {
                    "radius": self.scp_trust_radius,
                    "raw_step_norm": data.get("step_norm"),
                    "used_step_norm": data.get("delta"),
                    "clipped": None
                    if self.scp_trust_radius <= 0.0 or data.get("step_norm") is None or data.get("delta") is None
                    else bool(float(data["delta"]) + 1e-12 < float(data["step_norm"])),
                }
                payload["diagnostics"] = {
                    "summary": "Accepted Rust candidate after optional post-solve clipping.",
                }
                self._frames.append(
                    self._frame(
                        "candidate-filter",
                        f"Iteration {iter_index + 1} candidate filter",
                        payload,
                        iteration=iter_index,
                    )
                )

                post_payload = self._common_payload(accepted_cp)
                post_payload["metrics"] = {
                    "delta": data.get("delta"),
                    "clearance": data.get("clearance_after"),
                }
                post_payload["diagnostics"] = {
                    "summary": "Actual Rust post-evaluation metrics.",
                    "tol_break": data.get("tol_break"),
                }
                self._frames.append(
                    self._frame("post-eval", f"Iteration {iter_index + 1} evaluation", post_payload, iteration=iter_index)
                )
                current_cp = accepted_cp
            elif hypothesis_id == "H4":
                payload = self._common_payload(self.final_control_points)
                payload["final_clearance"] = data.get("final_clearance")
                payload["last_iterate_control_points"] = self.final_control_points
                payload["best_control_points"] = self.final_control_points
                payload["diagnostics"] = {
                    "summary": "Final Rust return value.",
                    "iterations_reported": data.get("iterations_reported"),
                    "feasible": data.get("feasible"),
                }
                self._frames.append(self._frame("finalize", "Final result", payload, iteration=None))

        if not any(frame.stage == "finalize" for frame in self._frames):
            payload = self._common_payload(self.final_control_points)
            payload["final_clearance"] = float(
                self.clearance_fn(self.final_control_points, self.obstacles, dim=self.dim, n_eval=1500)
            )
            payload["last_iterate_control_points"] = self.final_control_points
            payload["best_control_points"] = self.final_control_points
            payload["diagnostics"] = {
                "summary": "Final Rust result with no matching finalize log entry.",
            }
            self._frames.append(self._frame("finalize", "Final result", payload, iteration=None))

        self._prepared = True

    def next_frame(self) -> DebugFrame | None:
        if not self._prepared:
            self._prepare_frames()
        if self._frame_cursor >= len(self._frames):
            return None
        frame = self._frames[self._frame_cursor]
        self._frame_cursor += 1
        return frame

    def run_to_completion(self, verbose: bool = False) -> tuple[np.ndarray, dict]:
        if not self._prepared:
            self._prepare_frames()
        while self.next_frame() is not None:
            pass
        return self.final_control_points.copy(), dict(self.result_info)
