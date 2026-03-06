"""
Unit tests asserting Project Spec for De Casteljau: segment_matrices_equal_params
returns n_seg matrices; for known P, each segment Q_i = A_i @ P has correct length.
"""

import numpy as np
import pytest

from orbital_docking.de_casteljau import segment_matrices_equal_params


def test_segment_matrices_returns_n_seg_matrices():
    """segment_matrices_equal_params(N, n_seg) returns a list of n_seg matrices."""
    for N in [2, 3, 4]:
        for n_seg in [1, 2, 4, 8]:
            A_list = segment_matrices_equal_params(N, n_seg)
            assert len(A_list) == n_seg


def test_segment_matrices_each_is_N_plus_1_square():
    """Each returned matrix has shape (N+1, N+1)."""
    for N in [2, 3, 4]:
        for n_seg in [1, 2, 4]:
            A_list = segment_matrices_equal_params(N, n_seg)
            for A in A_list:
                assert A.shape == (N + 1, N + 1)


def test_Q_i_equals_A_i_at_P_has_correct_length(P_init):
    """
    For known P, each segment Q_i = A_i @ P has correct length (N+1 control points per segment).
    Uses conftest P_init (degree 3, shape (4,3)).
    """
    N = 3
    n_seg = 4
    P = P_init  # (4, 3)
    A_list = segment_matrices_equal_params(N, n_seg)
    assert len(A_list) == n_seg
    for A in A_list:
        Q = A @ P  # (N+1, 3)
        assert Q.shape == (N + 1, 3), "Each segment must have N+1 control points in 3D"


def test_Q_i_equals_A_i_at_P_known_N_n_seg(rng):
    """For random P, Q_i = A_i @ P has shape (N+1, dim) for various N and n_seg."""
    for N in [2, 3]:
        for n_seg in [1, 2, 3]:
            P = rng.uniform(-1.0, 1.0, (N + 1, 3))
            A_list = segment_matrices_equal_params(N, n_seg)
            for A in A_list:
                Q = A @ P
                assert Q.shape == (N + 1, 3)


def test_n_seg_1_returns_identity():
    """n_seg=1 returns single identity matrix (no subdivision)."""
    for N in [1, 2, 3]:
        A_list = segment_matrices_equal_params(N, 1)
        assert len(A_list) == 1
        np.testing.assert_array_almost_equal(A_list[0], np.eye(N + 1))
