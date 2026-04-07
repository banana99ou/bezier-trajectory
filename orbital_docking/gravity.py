"""
Gravity and J2 helpers shared by tests and diagnostics.
"""

from __future__ import annotations

import numpy as np


def _accel_two_body(r_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Two-body gravitational acceleration in km/s^2."""
    r = np.asarray(r_km, dtype=float)
    rn = np.linalg.norm(r)
    return (-mu_km3_s2 / (rn**3)) * r


def _accel_j2(r_km: np.ndarray, mu_km3_s2: float, r_e_km: float, j2: float) -> np.ndarray:
    """
    J2 perturbation acceleration in km/s^2 (Earth symmetry axis aligned with ECI Z).
    """
    r = np.asarray(r_km, dtype=float)
    x, y, z = r
    r2 = float(x * x + y * y + z * z)
    rn = np.sqrt(r2)
    if rn < 1e-12:
        return np.zeros(3)

    z2 = z * z
    r5 = rn**5
    factor = 1.5 * j2 * mu_km3_s2 * (r_e_km**2) / r5
    k = 5.0 * z2 / r2
    ax = factor * x * (k - 1.0)
    ay = factor * y * (k - 1.0)
    az = factor * z * (k - 3.0)
    return np.array([ax, ay, az], dtype=float)


def _accel_total(r_km: np.ndarray, mu_km3_s2: float, r_e_km: float, j2: float) -> np.ndarray:
    """Two-body + J2 total gravitational acceleration in km/s^2."""
    return _accel_two_body(r_km, mu_km3_s2) + _accel_j2(r_km, mu_km3_s2, r_e_km, j2)


def _jacobian_numeric(f, r0: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """
    Central-difference Jacobian of f: R^3 -> R^3 at r0.
    h is in km (default 1e-3 km = 1 m).
    """
    r0 = np.asarray(r0, dtype=float)
    J = np.zeros((3, 3), dtype=float)
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = h
        fp = f(r0 + dr)
        fm = f(r0 - dr)
        J[:, i] = (fp - fm) / (2.0 * h)
    return J
