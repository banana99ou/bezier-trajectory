"""
Unit tests for gravity and objective math in the optimizer.

Covers: two-body acceleration (direction and magnitude), J2 perturbation,
total gravity (two-body + J2), and the control-accel quadratic builder
(H symmetric, PSD, constant term c >= 0).
"""

import numpy as np
import pytest

from orbital_docking import constants
from orbital_docking.optimization import (
    _accel_two_body,
    _accel_j2,
    _accel_total,
    _build_ctrl_accel_quadratic,
)


# -----------------------------------------------------------------------------
# Two-body: accel_two_body(r) points toward origin, magnitude mu / norm(r)^2
# -----------------------------------------------------------------------------


def test_accel_two_body_direction_points_toward_origin(rng):
    """accel_two_body(r) is antiparallel to r (points toward origin)."""
    mu = constants.EARTH_MU_SCALED
    for _ in range(5):
        r = rng.uniform(1000.0, 8000.0, size=3)
        rn = np.linalg.norm(r)
        if rn < 1e-6:
            continue
        a = _accel_two_body(r, mu)
        # a = -(mu/r^3) r  =>  a is antiparallel to r
        expected_dir = -r / rn
        actual_dir = a / (np.linalg.norm(a) + 1e-20)
        np.testing.assert_allclose(actual_dir, expected_dir, rtol=1e-9, atol=1e-12)


def test_accel_two_body_magnitude_mu_over_r_squared(rng):
    """|accel_two_body(r)| = mu / norm(r)^2."""
    mu = constants.EARTH_MU_SCALED
    for _ in range(5):
        r = rng.uniform(1000.0, 8000.0, size=3)
        rn = np.linalg.norm(r)
        if rn < 1e-6:
            continue
        a = _accel_two_body(r, mu)
        expected_mag = mu / (rn**2)
        actual_mag = np.linalg.norm(a)
        np.testing.assert_allclose(actual_mag, expected_mag, rtol=1e-9, atol=1e-12)


# -----------------------------------------------------------------------------
# J2: _accel_j2 at known r (z-axis or symmetry at equator)
# -----------------------------------------------------------------------------


def test_accel_j2_on_z_axis_matches_hand_computed():
    """_accel_j2 at r = [0, 0, z] has only z-component with known formula."""
    mu = constants.EARTH_MU_SCALED
    r_e = constants.EARTH_RADIUS_KM
    j2 = constants.EARTH_J2
    z_km = 7000.0  # e.g. 7000 km from center
    r = np.array([0.0, 0.0, z_km])
    r2 = z_km**2
    rn = abs(z_km)
    r5 = rn**5
    factor = 1.5 * j2 * mu * (r_e**2) / r5
    k = 5.0 * (z_km**2) / r2  # = 5 on z-axis
    # a_x = factor * x * (k - 1) = 0, a_y = 0, a_z = factor * z * (k - 3) = 2 * factor * z
    expected_az = 2.0 * factor * z_km
    a = _accel_j2(r, mu, r_e, j2)
    np.testing.assert_allclose(a[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(a[1], 0.0, atol=1e-15)
    np.testing.assert_allclose(a[2], expected_az, rtol=1e-10, atol=1e-15)


def test_accel_j2_at_equator_z_component_zero():
    """On equator (z=0), J2 acceleration has zero z-component (symmetry)."""
    mu = constants.EARTH_MU_SCALED
    r_e = constants.EARTH_RADIUS_KM
    j2 = constants.EARTH_J2
    R = 6700.0  # km, equatorial distance
    r = np.array([R, 0.0, 0.0])
    a = _accel_j2(r, mu, r_e, j2)
    np.testing.assert_allclose(a[2], 0.0, atol=1e-15)


# -----------------------------------------------------------------------------
# _accel_total = two-body + J2 (sanity check)
# -----------------------------------------------------------------------------


def test_accel_total_equals_two_body_plus_j2(rng):
    """_accel_total(r) == _accel_two_body(r) + _accel_j2(r)."""
    mu = constants.EARTH_MU_SCALED
    r_e = constants.EARTH_RADIUS_KM
    j2 = constants.EARTH_J2
    r = rng.uniform(6400.0, 7500.0, size=3)
    a_total = _accel_total(r, mu, r_e, j2)
    a_two = _accel_two_body(r, mu)
    a_j2 = _accel_j2(r, mu, r_e, j2)
    np.testing.assert_allclose(a_total, a_two + a_j2, rtol=1e-12, atol=1e-15)


# -----------------------------------------------------------------------------
# _build_ctrl_accel_quadratic: H symmetric, PSD, c >= 0
# -----------------------------------------------------------------------------


def test_build_ctrl_accel_quadratic_H_symmetric(T, P_init):
    """H from _build_ctrl_accel_quadratic is symmetric."""
    P_ref = np.asarray(P_init, dtype=float)
    sample_count = 4
    H, f, c_const, _ = _build_ctrl_accel_quadratic(P_ref, T=T, sample_count=sample_count)
    np.testing.assert_allclose(H, H.T, rtol=0, atol=1e-12)


def test_build_ctrl_accel_quadratic_H_positive_semidefinite(T, P_init):
    """H from _build_ctrl_accel_quadratic is positive semidefinite."""
    P_ref = np.asarray(P_init, dtype=float)
    sample_count = 4
    H, f, c_const, _ = _build_ctrl_accel_quadratic(P_ref, T=T, sample_count=sample_count)
    evals = np.linalg.eigvalsh(H)
    assert np.all(evals >= -1e-10), f"min eigenvalue {evals.min()}"


def test_build_ctrl_accel_quadratic_constant_term_non_negative(T, P_init):
    """Objective 0.5 x'Hx + f'x + c has c >= 0."""
    P_ref = np.asarray(P_init, dtype=float)
    sample_count = 4
    H, f, c_const, _ = _build_ctrl_accel_quadratic(P_ref, T=T, sample_count=sample_count)
    assert c_const >= -1e-12, f"constant term c = {c_const}"


def test_build_ctrl_accel_quadratic_small_P_ref(T):
    """With a small reference trajectory (degree 2), H is symmetric PSD and c >= 0."""
    # Minimal (3, 3) = degree 2 Bézier so G_tilde exists
    P_ref = np.array([
        [6700.0, 0.0, 0.0],
        [6800.0, 200.0, 50.0],
        [6900.0, 0.0, 0.0],
    ], dtype=float)
    sample_count = 2
    H, f, c_const, _ = _build_ctrl_accel_quadratic(P_ref, T=T, sample_count=sample_count)
    np.testing.assert_allclose(H, H.T, rtol=0, atol=1e-12)
    evals = np.linalg.eigvalsh(H)
    assert np.all(evals >= -1e-10), f"min eigenvalue {evals.min()}"
    assert c_const >= -1e-12, f"constant term c = {c_const}"
