"""
Rust-backed debug stepper driven by SpacetimeScpContext.step().

Each SCP iteration calls the Rust backend directly and reads the full
per-row KOZ metadata out of the return value. No log parsing.
"""

from __future__ import annotations

import numpy as np

from orbital_docking.de_casteljau import segment_matrices_equal_params

from .debug_trace import DebugFrame
from .geometry import bezier_curve, compute_min_clearance, obstacle_array_bundle
from .optimize import DEFAULT_ELASTIC_WEIGHT
from .objective import build_initial_guess

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
        # PAPER_1 statement (8): clamp the clip radius from below by
    # E + Delta*sqrt(d+1) so statement (7) holds unconditionally and the
    # construction is sound by construction, at the cost of conservatism where
    # the row binds. Off by default -- PAPER_1 calls the choice between the two
    # an OPEN EXPERIMENTAL QUESTION, so both are reachable and measurable.
    sound_clip: bool = True,
        elastic_weight: float = DEFAULT_ELASTIC_WEIGHT,
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
        self.sound_clip = bool(sound_clip)
        self.elastic_weight = float(elastic_weight)

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
        iter_idx: np.ndarray,
        normals: np.ndarray,
        supports: np.ndarray,
        centers: np.ndarray,
        lbs: np.ndarray,
        margins: np.ndarray,
        slack: np.ndarray,
    ) -> dict:
        """Build a per-segment KOZ payload from Rust's per-row data.

        Emits every row (not just the tightest per obstacle-segment pair).
        Summary fields on each active_obstacle entry reflect the tightest row
        so existing UI half-space rendering stays intact; full row list is
        attached as ``rows``.
        """
        n_rows = len(seg_idx)
        segment_snapshots = _segment_snapshots(self.A_list, np.asarray(control_points, dtype=float))
        n_cp_seg = int(self.A_list[0].shape[0]) if len(self.A_list) else 0

        # Index by segment
        by_segment: dict[int, list[int]] = {}
        for r in range(n_rows):
            by_segment.setdefault(int(seg_idx[r]), []).append(r)

        def _slack_of(r: int) -> float:
            return float(slack[r]) if r < len(slack) else 0.0

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

            # Group rows by obstacle within this segment
            obs_to_rows: dict[int, list[int]] = {}
            for r in rows_here:
                obs_to_rows.setdefault(int(obs_idx[r]), []).append(r)

            active_obs = []
            for o_idx, r_list in obs_to_rows.items():
                # Sort rows by cp_idx for deterministic UI ordering
                r_list_sorted = sorted(r_list, key=lambda r: int(cp_idx[r]))

                # Tightest row for summary/backward-compat display
                tightest = min(r_list_sorted, key=lambda r: float(margins[r]))
                tightest_margin = float(margins[tightest])
                if tightest_margin < worst_margin:
                    worst_margin = tightest_margin
                    worst_segment_idx = s_idx

                tight_normal = normals[tightest].tolist()

                rows_detail = [
                    {
                        "row_index": int(r),
                        "cp_idx": int(cp_idx[r]),
                        "iteration": int(iter_idx[r]) if r < len(iter_idx) else 0,
                        "normal": normals[r][:-1].tolist(),
                        "time_coefficient": float(normals[r][-1]),
                        "support_point": supports[r].tolist(),
                        "center": centers[r].tolist(),
                        "lower_bound": float(lbs[r]),
                        "lhs_current": float(lbs[r] + margins[r]),
                        "margin_current": float(margins[r]),
                        "slack": _slack_of(r),
                    }
                    for r in r_list_sorted
                ]

                active_obs.append({
                    "obstacle_index": o_idx,
                    "obstacle_name": self.obstacles[o_idx].get("name", f"obs{o_idx}"),
                    "geometry_type": "capsule",
                    # Summary (tightest row) — preserves existing UI rendering
                    "center": centers[tightest].tolist(),
                    "support_point": supports[tightest].tolist(),
                    "normal": tight_normal[:-1],
                    "time_coefficient": float(tight_normal[-1]),
                    "lower_bound": float(lbs[tightest]),
                    "lhs_current": float(lbs[tightest] + margins[tightest]),
                    "margin_current": tightest_margin,
                    "row_indices": r_list_sorted,
                    "slack": _slack_of(tightest),
                    # Per-row detail
                    "tightest_row_index": int(tightest),
                    "tightest_cp_idx": int(cp_idx[tightest]),
                    "row_count": len(r_list_sorted),
                    "rows": rows_detail,
                })
            seg["active_obstacles"] = active_obs
            segments_out.append(seg)

        # Flat list of every row for the margin table in the UI
        all_rows = [
            {
                "row_index": int(r),
                "segment_index": int(seg_idx[r]),
                "cp_idx": int(cp_idx[r]),
                "obstacle_index": int(obs_idx[r]),
                "obstacle_name": self.obstacles[int(obs_idx[r])].get("name", f"obs{int(obs_idx[r])}"),
                "iteration": int(iter_idx[r]) if r < len(iter_idx) else 0,
                "margin_current": float(margins[r]),
                "lower_bound": float(lbs[r]),
                "slack": _slack_of(r),
            }
            for r in range(n_rows)
        ]

        return {
            "row_count": int(n_rows),
            "segments": segments_out,
            "worst_margin": worst_margin if worst_margin != float("inf") else None,
            "worst_segment": worst_segment_idx,
            "all_rows": all_rows,
            "n_cp_seg": n_cp_seg,
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
            # NOT acceleration energy: the curve parameter is not time, so this
            # term is blind to timing (item B8, formulation decisions 1-3).
            "type": "parameter_domain_smoothness_regularizer",
            "scp_prox_weight": self.scp_prox_weight,
            "trust_region_policy": "post_solve_clip",
            "scp_trust_radius": self.scp_trust_radius,
            "penalized_coordinates": list(range(self.dim - 1)),
            "penalizes_time_coordinate": False,
        }
        self._frames.append(
            self._frame("objective-assembly", "Objective assembly", objective_payload, iteration=0)
        )

        # Build Rust context
        spatial_dim = self.dim - 1
        obstacle_ctrl, obstacle_radii = obstacle_array_bundle(self.obstacles, spatial_dim)
        time_upper = float(self.P_init[-1, -1]) * self.time_ub_scale

        ctx = _bezier_opt_rs.SpacetimeScpContext(
            p_init=self.P_init,
            obstacle_ctrl=obstacle_ctrl,
            obstacle_r=obstacle_radii,
            n_seg=self.n_seg,
            min_dt=self.min_dt,
            coord_lb=self.coord_lb,
            coord_ub=self.coord_ub,
            time_lb=self.time_lb,
            time_ub=time_upper,
            scp_prox_weight=self.scp_prox_weight,
            scp_trust_radius=self.scp_trust_radius,
            elastic_weight=self.elastic_weight,
            tol=self.tol,
            sound_clip=self.sound_clip,
        )

        # The iterate, the best-feasible iterate and the stopping decision all live
        # in the Rust state now. This loop reads them; it does not re-derive them.
        # Re-deriving is what made this a second solver.
        current_cp = self.P_init.copy()
        best_cp = current_cp.copy()
        # Overwritten from the solver state each iteration; never seeded from the
        # initial guess, which is not an iterate the solver produced.
        best_clearance = float("-inf")
        stop_reason = -1.0
        state_converged = False
        last_delta = float("nan")
        last_total_slack = 0.0
        last_max_slack = 0.0
        last_cost = float("nan")

        for it in range(1, self.max_iter + 1):
            result = ctx.step()
            (
                p_new, info,
                seg_idx, cp_idx, obs_idx, iter_idx,
                normals, supports, centers,
                lbs, margins, slack,
            ) = result
            p_new = np.asarray(p_new, dtype=float)
            seg_idx = np.asarray(seg_idx)
            cp_idx = np.asarray(cp_idx)
            obs_idx = np.asarray(obs_idx)
            iter_idx = np.asarray(iter_idx)
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
                current_cp, seg_idx, cp_idx, obs_idx, iter_idx,
                normals, supports, centers, lbs, margins, slack,
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

            # `p_new` is the ACCEPTED iterate: unchanged when the step was rejected.
            # The raw candidate is info["p_candidate"], and the two are kept distinct
            # because a rejected candidate must never be displayed as an iterate.
            current_cp = p_new
            last_delta = float(info["delta"])
            last_total_slack = float(info["total_slack"])
            last_max_slack = float(info["max_slack"])
            last_cost = float(info["cost"])
            best_clearance = float(info["best_clearance"])
            best_cp = np.asarray(ctx.best_control_points(), dtype=float)
            state_converged = bool(info["state_converged"])
            stop_reason = float(info["stop_reason"])

            # The solver decides when it is done, not this loop.
            if not bool(info["running"]):
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
        feasible = final_clearance > 0.0 or len(self.obstacles) == 0
        self.result_info = {
            "backend": "rust",
            "iterations": int(it) if "it" in dir() else 0,
            "feasible": bool(feasible),
            "min_clearance": float(final_clearance),
            "best_clearance": float(best_clearance),
            "used_best_feasible": bool(used_best),
            "cost": float(last_cost),
            "cost_true_energy": float(last_cost),
            "cost_no_const": float(last_cost),
            "final_delta_norm": float(last_delta),
            "total_koz_slack": float(last_total_slack),
            "max_koz_slack": float(last_max_slack),
            "max_control_accel_ms2": 0.0,
            "mean_control_accel_ms2": 0.0,
            "converged": bool(state_converged),
            "stop_reason": float(stop_reason),
            "returned_best_iterate": bool(used_best),
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

    def frames_as_dicts(self) -> list[dict]:
        """Return all collected frames as JSON-safe dicts. Requires run_to_completion first."""
        if not self._prepared:
            self._prepare_frames()
        return [f.to_dict() for f in self._frames]


def create_spacetime_debug_stepper_from_control_points(
    p_init,
    obstacles: list[dict],
    clearance_fn=compute_min_clearance,
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
    # PAPER_1 statement (8): clamp the clip radius from below by
    # E + Delta*sqrt(d+1) so statement (7) holds unconditionally and the
    # construction is sound by construction, at the cost of conservatism where
    # the row binds. Off by default -- PAPER_1 calls the choice between the two
    # an OPEN EXPERIMENTAL QUESTION, so both are reachable and measurable.
    sound_clip: bool = True,
    elastic_weight: float = DEFAULT_ELASTIC_WEIGHT,
) -> RustOptimizerStepper:
    """Create a Rust-backed debug stepper from an existing control polygon."""
    return RustOptimizerStepper(
        p_init=np.asarray(p_init, dtype=float),
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
        sound_clip=sound_clip,
        elastic_weight=elastic_weight,
    )


def create_spacetime_debug_stepper(
    N: int,
    dim: int,
    p_start,
    p_end,
    obstacles: list[dict],
    clearance_fn=compute_min_clearance,
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
    # PAPER_1 statement (8): clamp the clip radius from below by
    # E + Delta*sqrt(d+1) so statement (7) holds unconditionally and the
    # construction is sound by construction, at the cost of conservatism where
    # the row binds. Off by default -- PAPER_1 calls the choice between the two
    # an OPEN EXPERIMENTAL QUESTION, so both are reachable and measurable.
    sound_clip: bool = True,
    elastic_weight: float = DEFAULT_ELASTIC_WEIGHT,
    init_curve: dict | None = None,
) -> RustOptimizerStepper:
    """Create a Rust-backed debug stepper, building the initial guess from endpoints."""
    n_cp = int(N) + 1
    p_init = build_initial_guess(p_start, p_end, n_cp, init_curve=init_curve)
    if dim != p_init.shape[1]:
        raise ValueError(f"Expected dim={dim}, got initial guess with dim={p_init.shape[1]}")
    return create_spacetime_debug_stepper_from_control_points(
        p_init,
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
        sound_clip=sound_clip,
        elastic_weight=elastic_weight,
    )
