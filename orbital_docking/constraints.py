"""
Constraint building functions for KOZ and boundary conditions.
"""

import numpy as np
from scipy.optimize import LinearConstraint
import warnings


def build_koz_constraints(A_list, P, r_e, dim=3, c_KOZ=None):
    """
    Build KOZ (Keep Out Zone) linear constraints for all segments.

    For each segment j:
    1. Compute CG (centroid) of control points: Qi = Ai @ P
    2. Generate unit vector nj from KOZ center (c_KOZ) to CG
    3. Create half-space constraint: nj^T @ Qi >= r_e

    Args:
        A_list: List of segment transformation matrices
        P: Control points (N+1, dim)
        r_e: KOZ radius
        dim: Spatial dimension
        c_KOZ: KOZ center, shape (dim,) or None (if None, uses origin and warns)

    Returns:
        LinearConstraint object
    """
    Np1 = A_list[0].shape[1]
    rows, lbs = [], []

    if c_KOZ is None:
        # warnings.warn("c_KOZ (KOZ center) not specified; defaulting to origin.")
        c_KOZ = np.zeros(dim)

    for Ai in A_list:
        Qi = Ai @ P  # Control points of segment i
        ci = Qi.mean(axis=0)  # Centroid (CG)

        # Unit vector from KOZ center to CG
        Nj = ci - c_KOZ
        Nj_norm = np.linalg.norm(Nj)
        if Nj_norm < 1e-12:
            # If CG coincides with KOZ center, skip this segment (degenerate case)
            continue

        nj = Nj / Nj_norm  # Unit normal vector

        # Create constraint for each control point in segment
        for k in range(Np1):
            row = np.zeros(Np1 * dim)
            for j in range(Np1):
                coeff = Ai[k, j]
                start = j * dim
                row[start:start+dim] += coeff * nj
            rows.append(row)
            # Supporting half-space for sphere centered at c_KOZ:
            #   n^T q >= n^T c_KOZ + r_e
            lbs.append(float(np.dot(nj, c_KOZ) + r_e))

    if len(rows) == 0:
        # No constraints generated
        A_const = np.zeros((1, Np1 * dim))
        lb_const = np.array([-np.inf])
        ub_const = np.array([np.inf])
    else:
        A_const = np.vstack(rows)
        lb_const = np.array(lbs)
        ub_const = np.full_like(lb_const, np.inf)

    return LinearConstraint(A_const, lb_const, ub_const)


def build_boundary_constraints(P_init, v0=None, v1=None, a0=None, a1=None, dim=3, T=1.0):
    """
    Build boundary condition equality constraints.

    Args:
        P_init: Initial control points (N+1, dim)
        v0: Initial velocity (optional, shape (dim,))
        v1: Final velocity (optional, shape (dim,))
        a0: Initial acceleration (optional, shape (dim,))
        a1: Final acceleration (optional, shape (dim,))
        dim: Spatial dimension
        T: Transfer time in seconds for converting tau-derivatives to physical derivatives

    Returns:
        list of LinearConstraint objects
    """
    Np1 = P_init.shape[0]
    N = Np1 - 1

    constraints = []

    # Position constraints: p(0) = P0, p(1) = PN
    # These are handled via bounds, not equality constraints
    # (we'll set bounds separately)

    # Velocity constraints (physical):
    # dr/dt = (1/T) dr/dtau
    vel_scale = 1.0 / float(T)
    if v0 is not None:
        # v(0) = (N/T) (P1 - P0)
        A = np.zeros((dim, Np1 * dim))
        for d in range(dim):
            A[d, 0 * dim + d] = -N * vel_scale
            A[d, 1 * dim + d] = N * vel_scale
        constraints.append(LinearConstraint(A, v0, v0))

    if v1 is not None:
        # v(1) = (N/T) (PN - PN-1)
        A = np.zeros((dim, Np1 * dim))
        for d in range(dim):
            A[d, (Np1 - 2) * dim + d] = -N * vel_scale
            A[d, (Np1 - 1) * dim + d] = N * vel_scale
        constraints.append(LinearConstraint(A, v1, v1))

    # Acceleration constraints (physical):
    # d2r/dt2 = (1/T^2) d2r/dtau2
    accel_scale = 1.0 / float(T) ** 2
    if a0 is not None and N >= 2:
        # a(0) = (N(N-1)/T^2) (P2 - 2P1 + P0)
        A = np.zeros((dim, Np1 * dim))
        c = N * (N - 1) * accel_scale
        for d in range(dim):
            A[d, 0 * dim + d] = c
            A[d, 1 * dim + d] = -2.0 * c
            A[d, 2 * dim + d] = c
        constraints.append(LinearConstraint(A, a0, a0))

    if a1 is not None and N >= 2:
        # a(1) = (N(N-1)/T^2) (PN - 2PN-1 + PN-2)
        A = np.zeros((dim, Np1 * dim))
        c = N * (N - 1) * accel_scale
        for d in range(dim):
            A[d, (Np1 - 3) * dim + d] = c
            A[d, (Np1 - 2) * dim + d] = -2.0 * c
            A[d, (Np1 - 1) * dim + d] = c
        constraints.append(LinearConstraint(A, a1, a1))

    return constraints

