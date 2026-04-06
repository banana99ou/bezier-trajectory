"""
Deeper validation tests for the J2 logic and its local linearization behavior.
"""

import numpy as np

from orbital_docking import constants
from orbital_docking.j2_validation import (
    accel_total_from_c20_numeric_gradient,
    default_validation_cases,
    error_summary,
    j2_from_c20,
    spherical_to_cartesian_km,
)
from orbital_docking.optimization import _accel_j2, _accel_total, _jacobian_numeric
from orbital_docking.visualization import accel_gravity_total_km_s2


def test_c20_to_j2_matches_repo_constant():
    """The normalized EGM C20 relation should recover the repo's J2 constant."""
    c20 = -0.484165143790815e-03
    np.testing.assert_allclose(j2_from_c20(c20), constants.EARTH_J2, rtol=1e-6, atol=0.0)


def test_numeric_gradient_reference_matches_closed_form_j2():
    """
    Independent C20-potential baseline should agree with the production J2 formula.
    """
    c20 = -constants.EARTH_J2 / np.sqrt(5.0)
    for case in default_validation_cases():
        radius_km = constants.EARTH_RADIUS_KM + case.altitude_km
        r_km = spherical_to_cartesian_km(radius_km, case.latitude_deg, case.longitude_deg)
        a_ref = accel_total_from_c20_numeric_gradient(
            r_km,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            c20,
        )
        a_model = _accel_total(
            r_km,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        )
        err = error_summary(a_model, a_ref)
        assert err["abs_norm"] < 2e-11, f"{case.sample_id}: abs error {err['abs_norm']}"
        assert err["rel_norm"] < 5e-9, f"{case.sample_id}: rel error {err['rel_norm']}"


def test_visualization_total_gravity_matches_optimizer_total_gravity(rng):
    """The visualization helper must stay numerically identical to the optimizer model."""
    for _ in range(8):
        altitude_km = float(rng.uniform(245.0, 20000.0))
        latitude_deg = float(rng.uniform(-85.0, 85.0))
        longitude_deg = float(rng.uniform(-180.0, 180.0))
        radius_km = constants.EARTH_RADIUS_KM + altitude_km
        r_km = spherical_to_cartesian_km(radius_km, latitude_deg, longitude_deg)
        a_opt = _accel_total(
            r_km,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        )
        a_viz = accel_gravity_total_km_s2(r_km)
        np.testing.assert_allclose(a_viz, a_opt, rtol=0.0, atol=1e-15)


def test_j2_direction_changes_between_equator_and_pole():
    """
    J2 correction points inward at the equator and outward along the pole.
    """
    r_equator = np.array([constants.EARTH_RADIUS_KM + 400.0, 0.0, 0.0], dtype=float)
    r_pole = np.array([0.0, 0.0, constants.EARTH_RADIUS_KM + 400.0], dtype=float)
    a_equator = _accel_j2(
        r_equator,
        constants.EARTH_MU_SCALED,
        constants.EARTH_RADIUS_KM,
        constants.EARTH_J2,
    )
    a_pole = _accel_j2(
        r_pole,
        constants.EARTH_MU_SCALED,
        constants.EARTH_RADIUS_KM,
        constants.EARTH_J2,
    )
    assert float(np.dot(a_equator, r_equator)) < 0.0
    assert float(np.dot(a_pole, r_pole)) > 0.0


def test_total_gravity_linearization_is_first_order_accurate():
    """
    The affine gravity model should show quadratic residual shrinkage as the step halves.
    """
    r_ref = spherical_to_cartesian_km(constants.EARTH_RADIUS_KM + 400.0, 42.0, 70.0)

    def f(r_km: np.ndarray) -> np.ndarray:
        return _accel_total(
            r_km,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        )

    J = _jacobian_numeric(f, r_ref)
    g_ref = f(r_ref)
    direction = np.array([0.4, -0.7, 0.59], dtype=float)
    direction /= np.linalg.norm(direction)
    dr_large = 0.2 * direction
    dr_small = 0.1 * direction

    err_large = np.linalg.norm(f(r_ref + dr_large) - (g_ref + J @ dr_large))
    err_small = np.linalg.norm(f(r_ref + dr_small) - (g_ref + J @ dr_small))

    assert err_large > 0.0
    assert err_small < 0.35 * err_large, (
        f"Expected near-quadratic shrinkage, got err_large={err_large}, err_small={err_small}"
    )
