"""
Shared pytest fixtures for bezier-trajectory tests.

Fixtures provide a fixed-seed RNG, default scenario parameters (N, n_seg, T, r_e),
3D endpoints P_start/P_end from constants (Progress-to-ISS inspired radii),
and P_init from generate_initial_control_points.
"""

import numpy as np
import pytest

from orbital_docking import constants, generate_initial_control_points


@pytest.fixture
def rng():
    """NumPy random Generator with fixed seed for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def default_N():
    """Default Bézier curve degree (cubic)."""
    return 3


@pytest.fixture
def default_n_seg():
    """Default number of segments for optimization."""
    return 4


@pytest.fixture
def T():
    """Default transfer time in seconds."""
    return constants.TRANSFER_TIME_S


@pytest.fixture
def r_e():
    """KOZ radius from Earth center (km), from constants."""
    return constants.KOZ_RADIUS


@pytest.fixture
def P_start():
    """
    Start endpoint (Progress-like chaser position) in 3D.

    Prefer angle parameters from constants if present; otherwise fall back to the
    simple coplanar demo geometry.
    """
    if hasattr(constants, "THETA_END_DEG") and hasattr(constants, "PHASE_LAG_DEG"):
        theta_end = np.deg2rad(constants.THETA_END_DEG)
        theta_start = theta_end - np.deg2rad(constants.PHASE_LAG_DEG)
    else:
        theta_start = -np.pi / 4.0
    return np.array([
        constants.CHASER_RADIUS * np.cos(theta_start),
        constants.CHASER_RADIUS * np.sin(theta_start),
        0.0,
    ])


@pytest.fixture
def P_end():
    """
    End endpoint (ISS-like target position) in 3D.

    Prefer angle parameters from constants if present; otherwise fall back to the
    simple coplanar demo geometry.
    """
    if hasattr(constants, "THETA_END_DEG"):
        theta_end = np.deg2rad(constants.THETA_END_DEG)
    else:
        theta_end = np.pi / 4.0
    return np.array([
        constants.ISS_RADIUS * np.cos(theta_end),
        constants.ISS_RADIUS * np.sin(theta_end),
        0.0,
    ])


@pytest.fixture
def P_init(default_N, P_start, P_end):
    """Initial control points from straight-line interpolation (degree default_N)."""
    return generate_initial_control_points(default_N, P_start, P_end)


def _ccw_equatorial_tangent(pos_xyz: np.ndarray) -> np.ndarray:
    """Unit tangent in xy-plane, counter-clockwise (for boundary velocity fixtures)."""
    x, y, _ = pos_xyz
    r_xy = np.hypot(x, y)
    if r_xy < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return np.array([-y / r_xy, x / r_xy, 0.0])


@pytest.fixture
def v0(P_start, r_e):
    """Initial velocity (km/s): tangential to KOZ, counter-clockwise in xy-plane."""
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / r_e)
    return v_mag * _ccw_equatorial_tangent(P_start)


@pytest.fixture
def v1(P_end, r_e):
    """Final velocity (km/s): tangential to KOZ, counter-clockwise in xy-plane."""
    v_mag = np.sqrt(constants.EARTH_MU_SCALED / r_e)
    return v_mag * _ccw_equatorial_tangent(P_end)
