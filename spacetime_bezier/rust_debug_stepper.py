"""
Rust-backed debug stepper driven by SpacetimeScpContext.step().

Each SCP iteration calls the Rust backend directly and reads the full
per-row KOZ metadata out of the return value. No log parsing.
"""

from __future__ import annotations

import numpy as np

from orbital_docking.de_casteljau import segment_matrices_equal_params

from .debug_trace import DebugFrame
from .geometry import bezier_curve, obstacle_array_bundle

try:
    import bezier_opt as _bezier_opt_rs
except ImportError:  # pragma: no cover
    _bezier_opt_rs = None


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
    """Step through Rust SCP execution one iteration at a time.

    Emits stage frames for each SCP iteration using per-row KOZ data
    returned by SpacetimeScpContext.step().
    """

    def __init__(
        self,
        p_init: np.ndarray,
        obstacles: list[dict],
        clearance_fn,
        rust_solver=None,  # retained for API compatibility, unused
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
        if _bezier_opt_rs is None or not hasattr(_bezier_opt_rs, "SpacetimeScpContext"):
            raise RuntimeError("Rust extension bezier_opt is not available or missing SpacetimeScpContext.")

        self.obstacles = list(obstacles)
        self.P_init = np.asarray(p_init, dtype=float).copy()
        self.clearance_fn = clearance_fn
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

    def _build_koz_payload(
        self,
        control_points: np.ndarray,
        seg_idx: np.ndarray,
        cp_idx: np.ndarray,
        obs_idx: np.ndarray,
        normals: np.ndarray,
        supports: np.ndarray,
        centers: np.ndarray,
        lbs: np.ndarray,
        margins: np.ndarray,
        slack: np.ndarray,
    ) -> dict:
        """Build a per-segment KOZ payload from Rust's per-row data.

        Groups rows by segment and builds active_obstacles entries with
        real normal/support/margin values for each obstacle-segment-CP triple.
        """
        n_rows = len(seg_idx)
        segment_snapshots = _segment_snapshots(self.A_list, np.asarray(control_points, dtype=float))
        # Index by segment
        by_segment: dict[int, list[int]] = {}
        for r in range(n_rows):
            by_segment.setdefault(int(seg_idx[r]), []).append(r)

        segments_out = []
        worst_margin = float("inf")
        worst_segment_idx = None
        for snap in segment_snapshots:
            s_idx = int(snap["segment_index"])
            if s_idx not in by_segment:
                segments_out.append(snap)
                continue
            seg = dict(snap)
            rows_here = by_segment[s_idx]

            # Aggregate per-obstacle (one entry per obstacle that touches this segment)
            obs_to_rows: dict[int, list[int]] = {}
            for r in rows_here:
                obs_to_rows.setdefault(int(obs_idx[r]), []).append(r)

            active_obs = []
            for o_idx, r_list in obs_to_rows.items():
                # Pick the tightest (lowest margin) row for this obstacle-segment pair
                tightest = min(r_list, key=lambda r: float(margins[r]))
                margin_val = float(margins[tightest])
                if margin_val < worst_margin:
                    worst_margin = margin_val
                    worst_segment_idx = s_idx

                normal = normals[tightest].tolist()
                active_obs.append({
                    "obstacle_index": o_idx,
                    "obstacle_name": self.obstacles[o_idx].get("name", f"obs{o_idx}"),
                    "geometry_type": "capsule",
                    "center": centers[tightest].tolist(),
                    "support_point": supports[tightest].tolist(),
                    "normal": normal[:-1],
                    "time_coefficient": float(normal[-1]),
                    "lower_bound": float(lbs[tightest]),
                    "lhs_current": float(lbs[tightest] + margins[tightest]),
                    "margin_current": margin_val,
                    "row_indices": r_list,
                    "slack": float(slack[tightest]) if len(slack) > tightest else 0.0,
                })
            seg["active_obstacles"] = active_obs
            segments_out.append(seg)

        return {
            "row_count": int(n_rows),
            "segments": segments_out,
            "worst_margin": worst_margin if worst_margin != float("inf") else None,
            "worst_segment": worst_segment_idx,
        }

    def _prepare_frames(self) -> None:
        if self._prepared:
            return

        # Static/initial frames
        init_payload = self._common_payload(self.P_init)
        init_payload["sampled_curve"] = bezier_curve(self.P_init, num_pts=200)
        init_payload["metrics"] = {"clearance": self.initial_clearance}
        init_payload["diagnostics"] = {"summary": "Initial control polygon passed into Rust."}
        self._frames.append(self._frame("init-guess", "Initial control polygon", init_payload, iteration=None))

        subdivision_payload = self._common_payload(self.P_init)
        subdivision_payload["diagnostics"] = {"summary": "Equal-parameter De Casteljau subdivision."}
        self._frames.append(
            self._frame("segment-subdivision", "Segment subdivision", subdivision_payload, iteration=0)
        )

        obstacle_payload = self._common_payload(self.P_init)
        obstacle_payload["diagnostics"] = {"summary": "Raw Rust obstacle inputs (capsule world-tubes in space-time)."}
        self._frames.append(
            self._frame("obstacle-geometry", "Obstacle geometry", obstacle_payload, iteration=0)
        )

        objective_payload = self._common_payload(self.P_init)
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

        # Build Rust context
        spatial_dim = self.dim - 1
        pos0, vel, radii, t_start, t_end = obstacle_array_bundle(self.obstacles, spatial_dim)
        time_upper = float(self.P_init[-1, -1]) * self.time_ub_scale

        ctx = _bezier_opt_rs.SpacetimeScpContext(
            p_init=self.P_init,
            obstacle_pos0=pos0,
            obstacle_vel=vel,
            obstacle_r=radii,
            obstacle_t_start=t_start,
            obstacle_t_end=t_end,
            n_seg=self.n_seg,
            min_dt=self.min_dt,
            coord_lb=self.coord_lb,
            coord_ub=self.coord_ub,
            time_lb=self.time_lb,
            time_ub=time_upper,
            scp_prox_weight=self.scp_prox_weight,
            scp_trust_radius=self.scp_trust_radius,
            elastic_weight=100.0,
            tol=self.tol,
        )

        # Track best feasible iterate (match Rust optimize_spacetime semantics)
        current_cp = self.P_init.copy()
        best_cp = current_cp.copy()
        best_clearance = self.initial_clearance

        for it in range(1, self.max_iter + 1):
            result = ctx.step(current_cp)
            p_new, info, seg_idx, cp_idx, obs_idx, normals, supports, centers, lbs, margins, slack = result
            p_new = np.asarray(p_new, dtype=float)
            seg_idx = np.asarray(seg_idx)
            cp_idx = np.asarray(cp_idx)
            obs_idx = np.asarray(obs_idx)
            normals = np.asarray(normals)
            supports = np.asarray(supports)
            centers = np.asarray(centers)
            lbs = np.asarray(lbs)
            margins = np.asarray(margins)
            slack = np.asarray(slack)
            status = str(info["solver_status"])

            # supporting-surface-generation frame — real per-row data
            ss_payload = self._common_payload(current_cp)
            ss_payload["koz"] = self._build_koz_payload(
                current_cp, seg_idx, cp_idx, obs_idx, normals, supports, centers, lbs, margins, slack
            )
            ss_payload["diagnostics"] = {
                "summary": f"Iteration {it} KOZ supporting surfaces ({info['koz_row_count']} rows).",
                "per_row_data": True,
            }
            self._frames.append(
                self._frame("supporting-surface-generation", f"Iteration {it} supporting surfaces",
                            ss_payload, iteration=it - 1)
            )

            # constraint-assembly frame
            ca_payload = self._common_payload(current_cp)
            ca_payload["koz"]["row_count"] = int(info["koz_row_count"])
            ca_payload["diagnostics"] = {
                "summary": f"Iteration {it} constraint assembly.",
                "koz_rows": int(info["koz_row_count"]),
            }
            self._frames.append(
                self._frame("constraint-assembly", f"Iteration {it} constraint assembly",
                            ca_payload, iteration=it - 1)
            )

            # solver-call frame
            sc_payload = self._common_payload(current_cp)
            sc_payload["solver"] = {
                "name": "Clarabel",
                "raw_status": status,
                "interpreted_status": status,
                "candidate_available": status in {"Solved", "Elastic"},
                "accepted": status in {"Solved", "Elastic"},
                "reason_for_rejection": None if status in {"Solved", "Elastic"} else "QP infeasible (elastic also failed)",
                "status": status,
                "elastic_used": status == "Elastic",
                "total_slack": float(info["total_slack"]),
                "max_slack": float(info["max_slack"]),
            }
            self._frames.append(
                self._frame("solver-call", f"Iteration {it} solver call",
                            sc_payload, iteration=it - 1)
            )

            if status == "Failed":
                # Stop here — no candidate produced
                fail_payload = self._common_payload(current_cp)
                fail_payload["solver"] = sc_payload["solver"]
                fail_payload["diagnostics"] = {"summary": "QP and elastic fallback both failed; optimizer halted."}
                self._frames.append(
                    self._frame("candidate-filter", f"Iteration {it} candidate filter",
                                fail_payload, iteration=it - 1)
                )
                break

            # candidate-filter frame (trust region clip)
            cf_payload = self._common_payload(p_new)
            cf_payload["accepted_control_points"] = p_new
            cf_payload["trust_region"] = {
                "radius": self.scp_trust_radius,
                "raw_step_norm": float(info["raw_step_norm"]),
                "used_step_norm": float(info["delta"]),
                "clipped": bool(
                    self.scp_trust_radius > 0.0
                    and float(info["delta"]) + 1e-12 < float(info["raw_step_norm"])
                ),
            }
            cf_payload["diagnostics"] = {"summary": "Accepted candidate after trust-region clipping."}
            self._frames.append(
                self._frame("candidate-filter", f"Iteration {it} candidate filter",
                            cf_payload, iteration=it - 1)
            )

            # post-eval frame
            pe_payload = self._common_payload(p_new)
            pe_payload["metrics"] = {
                "delta": float(info["delta"]),
                "clearance": float(info["clearance"]),
                "cost": float(info["cost"]),
                "total_koz_slack": float(info["total_slack"]),
            }
            pe_payload["diagnostics"] = {
                "summary": f"Iteration {it} post-evaluation.",
                "converged": bool(info["converged"]),
            }
            self._frames.append(
                self._frame("post-eval", f"Iteration {it} evaluation",
                            pe_payload, iteration=it - 1)
            )

            # Advance state
            current_cp = p_new
            if float(info["clearance"]) > 0.0 and float(info["clearance"]) > best_clearance:
                best_clearance = float(info["clearance"])
                best_cp = current_cp.copy()

            if bool(info["converged"]):
                break

        # Final iterate: use best feasible if current is infeasible
        final_clearance = float(
            self.clearance_fn(current_cp, self.obstacles, dim=self.dim, n_eval=1500)
        )
        used_best = False
        if final_clearance < 0.0 and best_clearance > 0.0:
            final_cp = best_cp
            final_clearance = best_clearance
            used_best = True
        else:
            final_cp = current_cp

        self.final_control_points = np.asarray(final_cp, dtype=float)
        self.result_info = {
            "backend": "rust",
            "iterations": int(it) if "it" in dir() else 0,
            "min_clearance": final_clearance,
            "best_clearance": best_clearance,
            "used_best_feasible": used_best,
        }

        finalize_payload = self._common_payload(self.final_control_points)
        finalize_payload["final_clearance"] = final_clearance
        finalize_payload["last_iterate_control_points"] = current_cp
        finalize_payload["best_control_points"] = best_cp
        finalize_payload["used_best_feasible"] = used_best
        finalize_payload["diagnostics"] = {
            "summary": f"Final result (clearance={final_clearance:.4f}, {'best-feasible' if used_best else 'last-iterate'}).",
        }
        self._frames.append(self._frame("finalize", "Final result", finalize_payload, iteration=None))

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
