"""
De Casteljau subdivision functions for Bézier curve segmentation.
"""

import numpy as np


def de_casteljau_split_1d(N, tau, basis_index):
    """
    Compute De Casteljau subdivision coefficients for a single basis vector.
    """
    w = np.zeros(N+1)
    w[basis_index] = 1.0
    left = [w[0]]
    right = [w[-1]]
    W = w.copy()

    for _ in range(1, N+1):
        W = (1 - tau) * W[:-1] + tau * W[1:]
        left.append(W[0])
        right.append(W[-1])

    L = np.array(left)
    R = np.array(right[::-1])
    return L, R


def de_casteljau_split_matrices(N, tau):
    """Compute subdivision matrices S_left and S_right."""
    S_left = np.zeros((N+1, N+1))
    S_right = np.zeros((N+1, N+1))

    for j in range(N+1):
        L, R = de_casteljau_split_1d(N, tau, j)
        S_left[:, j] = L
        S_right[:, j] = R
    return S_left, S_right


def segment_matrices_equal_params(N, n_seg):
    """
    Generate segment matrices for equal-parameter splitting.
    Returns list of (N+1, N+1) matrices, one per segment.
    """
    if n_seg < 1:
        raise ValueError("n_seg must be >= 1")
    if n_seg == 1:
        return [np.eye(N+1)]

    mats = []
    remainder = np.eye(N+1)

    for k in range(n_seg, 1, -1):
        tau = 1.0 / k
        S_L, S_R = de_casteljau_split_matrices(N, tau)
        mats.append(S_L @ remainder)
        remainder = S_R @ remainder
    mats.append(remainder)
    return mats

