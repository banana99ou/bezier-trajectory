"""
Constraint builders for the space-time Bezier optimizer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from .geometry import obstacle_array_bundle

SPACETIME_CAPSULE_TIME_SCALE = 0.5


def _scaled_spacetime_capsule_data(
    centroid: np.ndarray,
    obs_pos0: np.ndarray,
    obs_vel: np.ndarray,
    obs_r: np.ndarray,
    obs_t0: np.ndarray,
    obs_t1: np.ndarray,
    time_start: float,
    time_end: float,
    time_scale: float,
) -> tuple[np.ndarray, ...]:
    """Closest-point geometry for obstacle capsules in (x, y, alpha*t)."""
    eff_t0 = np.maximum(obs_t0, float(time_start))
    eff_t1 = np.minimum(obs_t1, float(time_end))
    active = eff_t1 >= eff_t0
    if not active.any():
        empty_vec = np.zeros((0, centroid.size), dtype=float)
        empty_scalar = np.zeros(0, dtype=float)
        empty_idx = np.zeros(0, dtype=int)
        return empty_idx, empty_vec, empty_vec.copy(), empty_vec.copy(), empty_vec.copy(), empty_scalar

    spatial_dim = centroid.size - 1
    c_scaled = centroid.copy()
    c_scaled[-1] *= float(time_scale)

    start_pos = obs_pos0 + obs_vel * eff_t0[:, None]
    end_pos = obs_pos0 + obs_vel * eff_t1[:, None]
    a_scaled = np.column_stack([start_pos, time_scale * eff_t0])
    b_scaled = np.column_stack([end_pos, time_scale * eff_t1])
    axis = b_scaled - a_scaled
    denom = np.einsum("ij,ij->i", axis, axis)

    rel = c_scaled[None, :] - a_scaled
    tau = np.zeros(len(obs_r), dtype=float)
    nondegenerate = denom > 1e-12
    tau[nondegenerate] = np.einsum("ij,ij->i", rel[nondegenerate], axis[nondegenerate]) / denom[nondegenerate]
    tau = np.clip(tau, 0.0, 1.0)

    closest_scaled = a_scaled + tau[:, None] * axis
    diffs_scaled = c_scaled[None, :] - closest_scaled
    dists = np.linalg.norm(diffs_scaled, axis=1)
    valid = active & (dists > 1e-10)
    idx = np.where(valid)[0]
    if not len(idx):
        empty_vec = np.zeros((0, centroid.size), dtype=float)
        empty_scalar = np.zeros(0, dtype=float)
        empty_idx = np.zeros(0, dtype=int)
        return empty_idx, empty_vec, empty_vec.copy(), empty_vec.copy(), empty_vec.copy(), empty_scalar

    n_scaled = diffs_scaled[idx] / dists[idx, None]
    n_orig = n_scaled.copy()
    n_orig[:, -1] *= float(time_scale)

    support_scaled = closest_scaled[idx] + obs_r[idx, None] * n_scaled
    support_orig = support_scaled.copy()
    support_orig[:, -1] /= float(time_scale)

    closest_orig = closest_scaled[idx].copy()
    closest_orig[:, -1] /= float(time_scale)

    lb = np.einsum("ij,ij->i", n_orig, support_orig)
    return idx, n_orig, support_orig, closest_orig, n_scaled, lb


def build_spacetime_koz_constraints(
    A_list,
    P: np.ndarray,
    obstacles: list[dict],
    dim: int = 3,
    return_debug: bool = False,
):
    """
    Build supporting half-space constraints for lifted obstacle world-tubes.

    For a moving disk obstacle, the forbidden set in lifted space-time is
    `||x - (pos0 + vel * t)|| <= r`. Linearizing the side wall at segment
    centroid time `t_seg` yields a plane with full space-time normal
    `[n_hat, -dot(n_hat, vel)]`, not a purely spatial slice constraint.
    """
    if not obstacles:
        if return_debug:
            return None, {"segments": [], "row_count": 0}
        return None

    P = np.asarray(P, dtype=float)
    n_cp = P.shape[0]
    n_cp_seg = A_list[0].shape[0]
    spatial_dim = dim - 1
    obs_pos0, obs_vel, obs_r, obs_t0, obs_t1 = obstacle_array_bundle(obstacles, spatial_dim)
    plan_t0 = float(P[0, -1])
    plan_t1 = float(P[-1, -1])

    blocks_A = []
    blocks_lb = []
    segment_debug = []
    total_rows = 0

    for segment_idx, A_seg in enumerate(A_list):
        Q = A_seg @ P
        centroid = Q.mean(axis=0)
        segment_info = {
            "segment_index": int(segment_idx),
            "control_points": np.asarray(Q, dtype=float).tolist(),
            "centroid": np.asarray(centroid, dtype=float).tolist(),
            "centroid_time": float(centroid[-1]),
            "active_obstacles": [],
            "rows_added": 0,
        }

        idx, n_orig, support_orig, closest_orig, n_scaled, lb_per_obs = _scaled_spacetime_capsule_data(
            centroid=np.asarray(centroid, dtype=float),
            obs_pos0=obs_pos0,
            obs_vel=obs_vel,
            obs_r=obs_r,
            obs_t0=obs_t0,
            obs_t1=obs_t1,
            time_start=plan_t0,
            time_end=plan_t1,
            time_scale=SPACETIME_CAPSULE_TIME_SCALE,
        )
        if not len(idx):
            segment_debug.append(segment_info)
            continue

        vel_valid = obs_vel[idx]
        r_valid = obs_r[idx]
        n_valid = len(idx)

        n_rows = n_valid * n_cp_seg
        A_block = np.zeros((n_rows, n_cp * dim), dtype=float)
        for coord_idx in range(dim):
            coeffs = n_orig[:, coord_idx : coord_idx + 1, None] * A_seg[None, :, :]
            A_block[:, coord_idx::dim] = coeffs.reshape(n_rows, n_cp)

        lb_block = np.repeat(lb_per_obs, n_cp_seg)

        blocks_A.append(A_block)
        blocks_lb.append(lb_block)
        total_rows += n_rows

        for local_idx, obstacle_idx in enumerate(idx):
            support_point = support_orig[local_idx]
            obstacle_center = closest_orig[local_idx]
            plane_normal = n_orig[local_idx]
            row_start = local_idx * n_cp_seg
            row_end = row_start + n_cp_seg
            segment_info["active_obstacles"].append(
                {
                    "obstacle_index": int(obstacle_idx),
                    "obstacle_name": obstacles[obstacle_idx].get("name", f"obs{obstacle_idx}"),
                    "center": np.asarray(obstacle_center, dtype=float).tolist(),
                    "velocity": np.asarray(vel_valid[local_idx], dtype=float).tolist(),
                    "radius": float(r_valid[local_idx]),
                    "normal": np.asarray(plane_normal, dtype=float).tolist(),
                    "support_point": np.asarray(support_point, dtype=float).tolist(),
                    "normal_scaled": np.asarray(n_scaled[local_idx], dtype=float).tolist(),
                    "lower_bound": float(lb_per_obs[local_idx]),
                    "row_count": int(n_cp_seg),
                    "row_indices": list(range(row_start, row_end)),
                    "t_start": float(max(obs_t0[obstacle_idx], plan_t0)),
                    "t_end": float(min(obs_t1[obstacle_idx], plan_t1)),
                    "time_scale": float(SPACETIME_CAPSULE_TIME_SCALE),
                }
            )
        segment_info["rows_added"] = int(n_rows)
        segment_debug.append(segment_info)

    if not blocks_A:
        if return_debug:
            return None, {"segments": segment_debug, "row_count": 0}
        return None

    constraint = LinearConstraint(
        np.vstack(blocks_A),
        np.concatenate(blocks_lb),
        np.full(sum(len(block) for block in blocks_lb), np.inf),
    )
    if return_debug:
        return constraint, {"segments": segment_debug, "row_count": int(total_rows)}
    return constraint


def build_boundary_constraints(n_cp: int, dim: int, p_start, p_end) -> LinearConstraint:
    """Fix first and last control points via equality constraints."""
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    rows = []
    vals = []
    for coord_idx in range(dim):
        row0 = np.zeros(n_cp * dim, dtype=float)
        row0[coord_idx] = 1.0
        rows.append(row0)
        vals.append(p_start[coord_idx])

        rowN = np.zeros(n_cp * dim, dtype=float)
        rowN[(n_cp - 1) * dim + coord_idx] = 1.0
        rows.append(rowN)
        vals.append(p_end[coord_idx])

    A = np.array(rows, dtype=float)
    values = np.array(vals, dtype=float)
    return LinearConstraint(A, values, values)


def build_time_monotonicity(n_cp: int, dim: int, min_dt: float = 0.1) -> LinearConstraint:
    """Enforce P[i+1, t] - P[i, t] >= min_dt."""
    t_idx = dim - 1
    rows = []
    for cp_idx in range(n_cp - 1):
        row = np.zeros(n_cp * dim, dtype=float)
        row[(cp_idx + 1) * dim + t_idx] = 1.0
        row[cp_idx * dim + t_idx] = -1.0
        rows.append(row)

    A = np.array(rows, dtype=float)
    lb = np.full(len(rows), float(min_dt), dtype=float)
    ub = np.full(len(rows), np.inf, dtype=float)
    return LinearConstraint(A, lb, ub)


def build_box_bounds(
    p_init: np.ndarray,
    coord_lb: float = -20.0,
    coord_ub: float = 20.0,
    time_lb: float = 0.0,
    time_ub_scale: float = 1.5,
) -> Bounds:
    """Create box bounds while keeping the endpoints fixed."""
    p_init = np.asarray(p_init, dtype=float)
    n_cp, dim = p_init.shape
    lb = np.full(n_cp * dim, float(coord_lb), dtype=float)
    ub = np.full(n_cp * dim, float(coord_ub), dtype=float)

    for coord_idx in range(dim):
        lb[coord_idx] = ub[coord_idx] = p_init[0, coord_idx]
        last = (n_cp - 1) * dim + coord_idx
        lb[last] = ub[last] = p_init[-1, coord_idx]

    time_upper = float(p_init[-1, -1]) * float(time_ub_scale)
    for cp_idx in range(n_cp):
        t_col = cp_idx * dim + (dim - 1)
        lb[t_col] = float(time_lb)
        ub[t_col] = time_upper

    return Bounds(lb, ub)
