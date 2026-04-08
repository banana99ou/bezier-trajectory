"""
Objective construction helpers for the space-time Bezier optimizer.
"""

from __future__ import annotations

import numpy as np

from orbital_docking.bezier import BezierCurve


def build_energy_objective(N: int, dim: int) -> np.ndarray:
    """
    Build the quadratic form for spatial acceleration energy only.

    The time coordinate is intentionally left unpenalized.
    """
    bc = BezierCurve(np.zeros((N + 1, dim), dtype=float))
    if bc.G_tilde is None:
        raise ValueError("Spatial acceleration energy requires Bezier degree >= 2.")

    spatial_dim = dim - 1
    n_cp = N + 1
    H = np.zeros((n_cp * dim, n_cp * dim), dtype=float)

    for coord_idx in range(spatial_dim):
        for i in range(n_cp):
            for j in range(n_cp):
                H[i * dim + coord_idx, j * dim + coord_idx] += bc.G_tilde[i, j]

    return 0.5 * (H + H.T)


def build_initial_guess(p_start, p_end, n_cp: int, init_curve: dict | None = None) -> np.ndarray:
    """
    Build the initial control polygon for SCP.

    The default is a straight line in space-time. For 2D spatial demos we can
    optionally add a quadratic-looking lateral bow so the initial curve swings
    toward a corner instead of passing through the middle of the workspace.
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    s_vals = np.linspace(0.0, 1.0, int(n_cp))
    P = np.array([(1.0 - s) * p_start + s * p_end for s in s_vals], dtype=float)

    if not init_curve or init_curve.get("mode", "straight") == "straight":
        return P

    if init_curve.get("mode") != "quadratic_bow":
        raise ValueError(f"Unknown init_curve mode: {init_curve.get('mode')}")

    spatial_dim = p_start.size - 1
    if spatial_dim < 2:
        return P

    bow = float(init_curve.get("bow", 0.0))
    if bow <= 0.0:
        return P

    chord = p_end[:2] - p_start[:2]
    chord_norm = np.linalg.norm(chord)
    if chord_norm < 1e-12:
        return P

    normal = np.array([-chord[1], chord[0]], dtype=float) / chord_norm
    workspace_center = init_curve.get("workspace_center")
    if workspace_center is not None:
        workspace_center = np.asarray(workspace_center, dtype=float)
        midpoint = 0.5 * (p_start[:2] + p_end[:2])
        if np.dot(workspace_center - midpoint, normal) > 0.0:
            normal *= -1.0
    normal *= float(init_curve.get("side", 1.0))

    bump = bow * (4.0 * s_vals * (1.0 - s_vals))
    P[:, :2] += bump[:, None] * normal
    return P
