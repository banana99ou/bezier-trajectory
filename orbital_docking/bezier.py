"""
Bézier curve implementation with D/E matrices for derivatives.
"""

import numpy as np
from scipy.special import comb


def get_D_matrix(N):
    """
    Compute derivative matrix D for Bézier curve of degree N.

    From Project_Spec.md:
    [D]_i,j = N × { -1 if j=i, 1 if j=i+1, 0 otherwise }

    Args:
        N: Degree of Bézier curve

    Returns:
        D: (N, N+1) matrix
    """
    D = np.zeros((N, N+1))
    for i in range(N):
        D[i, i] = -N
        D[i, i+1] = N
    return D


def get_E_matrix(N):
    """
    Compute elevation matrix E for Bézier curve of degree N.
    Elevates degree from N to N+1.

    From Project_Spec.md equation:
    E_{N→N+1} with specific structure

    Args:
        N: Original degree

    Returns:
        E: (N+2, N+1) matrix
    """
    E = np.zeros((N+2, N+1))

    # First row: i=1, j=1 → 1
    E[0, 0] = 1.0

    # Last row: i=N+2, j=N+1 → 1
    E[N+1, N] = 1.0

    # Middle rows: 2 ≤ i ≤ N+1 (1-indexed) → rows 1 to N (0-indexed)
    for row_idx in range(1, N+1):  # 0-indexed row: 1 to N
        i_1idx = row_idx + 1  # Convert to 1-indexed for formula
        # j = i-1 (1-indexed) → col = row_idx - 1 (0-indexed)
        E[row_idx, row_idx - 1] = (N + 2 - i_1idx) / (N + 1)
        # j = i (1-indexed) → col = row_idx (0-indexed)
        E[row_idx, row_idx] = (i_1idx - 1) / (N + 1)

    return E


class BezierCurve:
    """
    Bézier curve implementation using D/E matrices for derivatives.
    """

    def __init__(self, control_points):
        P = np.array(control_points, dtype=float)
        if P.ndim != 2:  # cus np.array makes control_points a 2D matrix
            raise ValueError("control_points must be (N+1, dim)")
        self.control_points = P
        self.degree = P.shape[0] - 1  # = N
        self.dimension = P.shape[1]  # P.shape = (N+1, dim)

        # Precompute D and E matrices
        self.D = get_D_matrix(self.degree)
        if self.degree > 0:
            self.E = get_E_matrix(self.degree - 1)  # E elevates from N-1 to N

    def point(self, tau):
        """Evaluate curve at parameter tau using Bernstein basis."""
        N, d = self.degree, self.dimension
        out = np.zeros(d)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * self.control_points[i]
        return out

    def velocity_control_points(self):
        """
        Compute velocity control points using V = EDP.
        Returns control points for velocity curve (degree N, N+1 control points).
        """
        if self.degree == 0:
            return np.zeros((1, self.dimension))

        # V = E @ D @ P
        # P is (N+1, dim), D is (N, N+1), E is (N+1, N)
        # Result: (N+1, dim)
        V_ctrl = self.E @ self.D @ self.control_points
        return V_ctrl

    def acceleration_control_points(self):
        """
        Compute acceleration control points using A = EDEDP.
        Returns control points for acceleration curve.
        """
        # Acceleration control points using EDEDP as instructed.
        if self.degree < 2:
            return np.zeros(((self.degree+1), self.dimension))
        # Use EDEDP directly per the analytic/formal instructions.
        A_ctrl = self.E @ self.D @ self.E @ self.D @ self.control_points
        # (Resulting shape: (N+1, d);
        return A_ctrl

    def velocity(self, tau):
        """Evaluate velocity at parameter tau."""
        if self.degree == 0:
            return np.zeros(self.dimension)
        V_ctrl = self.velocity_control_points()
        # Velocity curve has degree N (same as original)
        N = self.degree
        out = np.zeros(self.dimension)
        for i in range(N + 1):
            b = comb(N, i) * (tau ** i) * ((1 - tau) ** (N - i))
            out += b * V_ctrl[i]
        return out

    def acceleration(self, tau):
        """Evaluate acceleration at parameter tau."""
        if self.degree < 2:
            return np.zeros(((self.degree+1), self.dimension))
        A_ctrl = self.acceleration_control_points()
        # Acceleration curve has degree N due to the E matrix
        N_accel = self.degree
        out = np.zeros(self.dimension)
        for i in range(N_accel + 1):
            b = comb(N_accel, i) * (tau ** i) * ((1 - tau) ** (N_accel - i))
            out += b * A_ctrl[i]
        return out

