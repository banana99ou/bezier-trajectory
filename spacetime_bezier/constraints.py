"""
Constraint builders for the space-time Bezier optimizer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint

from .geometry import obstacle_array_bundle


def build_spacetime_koz_constraints(A_list, P: np.ndarray, obstacles: list[dict], dim: int = 3):
    """
    Build supporting half-space constraints for moving obstacles.

    The support plane is linearized at each segment centroid time.
    """
    if not obstacles:
        return None

    P = np.asarray(P, dtype=float)
    n_cp = P.shape[0]
    spatial_dim = dim - 1
    n_cp_seg = A_list[0].shape[0]
    obs_pos0, obs_vel, obs_r, obs_t0, obs_t1 = obstacle_array_bundle(obstacles, spatial_dim)

    blocks_A = []
    blocks_lb = []

    for A_seg in A_list:
        Q = A_seg @ P
        centroid = Q.mean(axis=0)
        t_seg = float(centroid[-1])
        c_spatial = centroid[:spatial_dim]

        active = (t_seg >= obs_t0) & (t_seg <= obs_t1)
        if not active.any():
            continue

        o_positions = obs_pos0 + obs_vel * t_seg
        diffs = c_spatial[None, :] - o_positions
        dists = np.linalg.norm(diffs, axis=1)
        valid = active & (dists > 1e-10)
        if not valid.any():
            continue

        idx = np.where(valid)[0]
        n_hats = diffs[idx] / dists[idx, None]
        o_pos_valid = o_positions[idx]
        r_valid = obs_r[idx]
        n_valid = len(idx)

        n_rows = n_valid * n_cp_seg
        A_block = np.zeros((n_rows, n_cp * dim), dtype=float)
        for spatial_idx in range(spatial_dim):
            coeffs = n_hats[:, spatial_idx : spatial_idx + 1, None] * A_seg[None, :, :]
            A_block[:, spatial_idx::dim] = coeffs.reshape(n_rows, n_cp)

        lb_per_obs = np.einsum("ok,ok->o", n_hats, o_pos_valid) + r_valid
        lb_block = np.repeat(lb_per_obs, n_cp_seg)

        blocks_A.append(A_block)
        blocks_lb.append(lb_block)

    if not blocks_A:
        return None

    return LinearConstraint(
        np.vstack(blocks_A),
        np.concatenate(blocks_lb),
        np.full(sum(len(block) for block in blocks_lb), np.inf),
    )


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
