"""
Integration tests: run optimizer and verify KOZ linear constraints at solution.
"""

import numpy as np
import pytest

from orbital_docking import optimize_orbital_docking
from orbital_docking.constraints import build_koz_constraints
from orbital_docking.de_casteljau import segment_matrices_equal_params

# Tolerance for constraint satisfaction A@x >= lb
KOZ_CONSTRAINT_TOL = 1e-6


def test_optimizer_koz_constraints_satisfied_at_solution(P_init, r_e, default_n_seg):
    """
    Run optimizer (default c_KOZ=0); at solution, rebuild KOZ constraints
    and verify all linear constraints A@x >= lb are satisfied.
    """
    n_seg = default_n_seg
    max_iter = 10
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=r_e,
        max_iter=max_iter,
        tol=1e-6,
        use_cache=False,
        verbose=False,
    )
    Np1, dim = P_opt.shape
    N = Np1 - 1
    A_list = segment_matrices_equal_params(N, n_seg)
    c_KOZ = np.zeros(dim)
    koz_constraint = build_koz_constraints(A_list, P_opt, r_e, dim=dim, c_KOZ=c_KOZ)
    x = P_opt.reshape(-1)
    lhs = koz_constraint.A @ x
    # Constraint is lb <= A@x <= ub; for KOZ we have one-sided A@x >= lb
    violations = lhs < (koz_constraint.lb - KOZ_CONSTRAINT_TOL)
    assert not np.any(violations), (
        f"KOZ constraint violations: {np.sum(violations)} rows; "
        f"min margin = {np.min(lhs - koz_constraint.lb):.6e}"
    )
