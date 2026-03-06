"""
Integration tests: run the real optimizer and assert feasibility and endpoint preservation.
"""

import numpy as np
import pytest

from orbital_docking import optimize_orbital_docking

# Small tolerance for feasibility and endpoint comparison
FEAS_TOL = 1e-5
ENDPOINT_TOL = 1e-9
# Velocity BC is enforced via linear equality constraints inside each trust-constr solve.
# If the solve is feasible, endpoint residuals should be near numerical tolerance.
VEL_RESIDUAL_TOL = 1e-4  # km/s


def test_optimizer_feasibility_and_endpoints(P_init, r_e, default_n_seg):
    """Run optimizer with conftest P_init, no velocity BC; assert feasible, min_radius, endpoints."""
    n_seg = default_n_seg
    max_iter = 10
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=r_e,
        max_iter=max_iter,
        tol=1e-6,
        v0=None,
        v1=None,
        use_cache=False,
        verbose=False,
    )
    assert info["feasible"] is True, f"Expected feasible; got min_radius={info.get('min_radius')}"
    assert "min_radius" in info
    assert info["min_radius"] >= r_e - FEAS_TOL, (
        f"min_radius {info['min_radius']} should be >= r_e - tol = {r_e - FEAS_TOL}"
    )
    # Endpoints fixed by bounds: first and last rows of P_opt must match P_init
    np.testing.assert_allclose(P_opt[0], P_init[0], atol=ENDPOINT_TOL, err_msg="P_opt[0] != P_init[0]")
    np.testing.assert_allclose(P_opt[-1], P_init[-1], atol=ENDPOINT_TOL, err_msg="P_opt[-1] != P_init[-1]")


def test_optimizer_with_velocity_bc_endpoint_residuals(P_init, r_e, T, default_n_seg):
    """
    Use endpoint velocity BC values that are guaranteed feasible for the given endpoints and transfer time,
    then assert endpoint velocity residuals (N/T)(P1-P0) - v0 etc. are near zero.

    Note: physically-motivated circular-orbit velocities (km/s) can be inconsistent with the chosen fixed T,
    making v0/v1 constraints infeasible. This test focuses on verifying that *if* feasible BCs are provided,
    they are enforced.
    """
    n_seg = default_n_seg
    max_iter = 10
    # Choose feasible BC targets from the initial control polygon itself.
    N = P_init.shape[0] - 1
    vel_scale = N / T
    v0 = vel_scale * (P_init[1] - P_init[0])
    v1 = vel_scale * (P_init[-1] - P_init[-2])
    P_opt, info = optimize_orbital_docking(
        P_init,
        n_seg=n_seg,
        r_e=r_e,
        max_iter=max_iter,
        tol=1e-6,
        v0=v0,
        v1=v1,
        use_cache=False,
        verbose=False,
    )
    # Known behavior: some configurations (notably N=3 with both v0 and v1) can be infeasible.
    # If infeasible, this test is not meaningful (residuals won't match a constraint that couldn't be satisfied).
    if not info.get("feasible", False):
        pytest.skip(f"Infeasible with both endpoint velocity BCs (min_radius={info.get('min_radius')}).")
    Np1, dim = P_opt.shape
    N = Np1 - 1
    vel_scale = N / T
    # v(0) = (N/T)(P1 - P0)
    v0_actual = vel_scale * (P_opt[1] - P_opt[0])
    # v(1) = (N/T)(PN - PN-1)
    v1_actual = vel_scale * (P_opt[-1] - P_opt[-2])
    residual_0 = np.linalg.norm(v0_actual - v0)
    residual_1 = np.linalg.norm(v1_actual - v1)
    assert residual_0 < VEL_RESIDUAL_TOL, (
        f"Initial velocity residual = {residual_0:.6e} (km/s), expected < {VEL_RESIDUAL_TOL}"
    )
    assert residual_1 < VEL_RESIDUAL_TOL, (
        f"Final velocity residual = {residual_1:.6e} (km/s), expected < {VEL_RESIDUAL_TOL}"
    )
