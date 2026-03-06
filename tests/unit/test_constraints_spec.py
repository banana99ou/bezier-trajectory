"""
Unit tests for constraint math per Project Spec.

(1) Boundary conditions: build_boundary_constraints with v0,v1,a0,a1,T;
    for P satisfying the BC, assert LinearConstraint A@x equals lb (and ub) for x=vec(P).
(2) KOZ origin: build_koz_constraints with c_KOZ=None, one segment; RHS=r_e, normal unit.
(3) KOZ shifted center: c_KOZ non-zero; RHS = n^T c_KOZ + r_e per row.
"""

import numpy as np
import pytest

from orbital_docking.constraints import build_boundary_constraints, build_koz_constraints
from orbital_docking.de_casteljau import segment_matrices_equal_params


def vec_P(P):
    """Decision variable x = vec(P), row-major (P.ravel())."""
    return P.ravel()


# ---- (1) Boundary conditions ----


@pytest.fixture
def N():
    """Bézier degree for BC tests (N=3 per spec)."""
    return 3


@pytest.fixture
def dim():
    """Spatial dimension."""
    return 3


def test_build_boundary_constraints_Ax_equals_lb_ub(N, T, dim):
    """
    For P that satisfies v(0)=(N/T)(P1-P0) etc., assert each LinearConstraint
    A@x equals lb (and ub) for x=vec(P). Uses N=3 with fixture transfer time T.
    """
    # Build a P that satisfies the BC by definition: derive v0,v1,a0,a1 from P
    # so that the same P is a fixed point of the BC formulas.
    P0 = np.array([100.0, 200.0, 50.0])
    P1 = np.array([110.0, 210.0, 55.0])
    P2 = np.array([130.0, 230.0, 65.0])
    P3 = np.array([150.0, 250.0, 75.0])
    P = np.vstack([P0, P1, P2, P3])  # (N+1, dim)

    vel_scale = N / float(T)
    accel_scale = N * (N - 1) / float(T) ** 2

    v0 = vel_scale * (P1 - P0)
    v1 = vel_scale * (P3 - P2)
    a0 = accel_scale * (P2 - 2 * P1 + P0)
    a1 = accel_scale * (P3 - 2 * P2 + P1)

    constraints = build_boundary_constraints(
        P, v0=v0, v1=v1, a0=a0, a1=a1, dim=dim, T=T
    )
    x = vec_P(P)

    for lc in constraints:
        A = lc.A
        lb = lc.lb
        ub = lc.ub
        Ax = A @ x
        np.testing.assert_array_almost_equal(Ax, lb, err_msg="A@x should equal lb")
        np.testing.assert_array_almost_equal(Ax, ub, err_msg="A@x should equal ub")
        np.testing.assert_array_almost_equal(lb, ub, err_msg="Equality: lb should equal ub")


# ---- (2) KOZ origin ----


def test_koz_origin_rhs_r_e_and_unit_normal(T, r_e, default_N):
    """
    build_koz_constraints with c_KOZ=None (origin), one segment, known P.
    Assert each row's RHS (lb) equals r_e and that the normal is unit.
    """
    N = default_N
    dim = 3
    # P such that centroid is not at origin (so we get a non-degenerate normal)
    P = np.zeros((N + 1, dim))
    P[:, 0] = 1.0
    P[:, 1] = 0.5
    P[:, 2] = 0.0
    # One segment: A_list = [I]
    A_list = segment_matrices_equal_params(N, n_seg=1)

    with pytest.warns(UserWarning, match="c_KOZ.*defaulting to origin"):
        lc = build_koz_constraints(A_list, P, r_e, dim=dim, c_KOZ=None)

    # All rows should have lb = r_e (origin => n^T * 0 + r_e = r_e)
    np.testing.assert_array_almost_equal(
        lc.lb, np.full(len(lc.lb), r_e), err_msg="Each row RHS should equal r_e"
    )

    # Extract normal from first row (row encodes n in the first control point block for segment 0)
    Np1 = N + 1
    for row_idx in range(lc.A.shape[0]):
        row = lc.A[row_idx]
        # Row is [0..0, n, 0..0] for one control point; find non-zero block
        block = row.reshape(Np1, dim)
        # Which control point: row_idx corresponds to point (row_idx % Np1) in segment 0
        k = row_idx % Np1
        n = block[k]
        n_norm = np.linalg.norm(n)
        assert n_norm > 1e-10, "Normal should be non-zero"
        assert n_norm == pytest.approx(1.0), "Normal should be unit"


# ---- (3) KOZ shifted center ----


def test_koz_shifted_center_rhs_nTc_plus_r_e(r_e):
    """
    With c_KOZ not zero, assert RHS is n^T c_KOZ + r_e for each row.
    Use small synthetic A_list and P to force known n and numeric RHS.
    """
    N = 2
    dim = 3
    # Centroid along x: P all at [2,0,0] => c = [2,0,0]
    P = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    c_KOZ = np.array([1.0, 0.0, 0.0])
    # n = (c - c_KOZ) / ||c - c_KOZ|| = [1,0,0]; n^T c_KOZ + r_e = 1 + r_e
    expected_rhs = float(np.dot(np.array([1.0, 0.0, 0.0]), c_KOZ) + r_e)

    A_list = segment_matrices_equal_params(N, n_seg=1)
    lc = build_koz_constraints(A_list, P, r_e, dim=dim, c_KOZ=c_KOZ)

    np.testing.assert_array_almost_equal(
        lc.lb, np.full(len(lc.lb), expected_rhs),
        err_msg="Each row RHS should equal n^T c_KOZ + r_e"
    )
    # Sanity: normal should be [1,0,0]
    first_row = lc.A[0]
    n_from_row = first_row[:dim]
    np.testing.assert_array_almost_equal(
        n_from_row, np.array([1.0, 0.0, 0.0]),
        err_msg="Normal should be [1,0,0]"
    )
