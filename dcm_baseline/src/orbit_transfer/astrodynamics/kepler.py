"""케플러 전파 (Universal Variable Formulation).

Curtis, "Orbital Mechanics for Engineering Students", Algorithm 3.4 기반.
"""

import numpy as np


def _stumpff_c2(psi):
    """Stumpff function c2(psi)."""
    if psi > 1e-6:
        return (1.0 - np.cos(np.sqrt(psi))) / psi
    elif psi < -1e-6:
        return (np.cosh(np.sqrt(-psi)) - 1.0) / (-psi)
    else:
        return 1.0 / 2.0


def _stumpff_c3(psi):
    """Stumpff function c3(psi)."""
    if psi > 1e-6:
        sq = np.sqrt(psi)
        return (sq - np.sin(sq)) / (psi * sq)
    elif psi < -1e-6:
        sq = np.sqrt(-psi)
        return (np.sinh(sq) - sq) / ((-psi) * sq)
    else:
        return 1.0 / 6.0


def kepler_propagate(r0, v0, dt, mu):
    """케플러 전파 (Universal Variable Formulation).

    Parameters
    ----------
    r0 : ndarray (3,)
        초기 위치 벡터 [km]
    v0 : ndarray (3,)
        초기 속도 벡터 [km/s]
    dt : float
        전파 시간 [s]
    mu : float
        Gravitational parameter [km^3/s^2]

    Returns
    -------
    r : ndarray (3,)
        최종 위치 벡터 [km]
    v : ndarray (3,)
        최종 속도 벡터 [km/s]
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    sqrt_mu = np.sqrt(mu)

    # dot(r0, v0) / sqrt(mu)  -- used in the universal variable formulas
    rdotv_over_sqrtmu = np.dot(r0, v0) / sqrt_mu

    # Reciprocal of semi-major axis: alpha = 1/a
    alpha = 2.0 / r0_mag - v0_mag**2 / mu

    # Initial guess for chi (universal variable)
    if alpha > 1e-10:
        # Elliptic orbit
        chi = sqrt_mu * dt * alpha
    elif alpha < -1e-10:
        # Hyperbolic orbit
        a = 1.0 / alpha
        chi = (np.sign(dt) * np.sqrt(-a)
               * np.log((-2.0 * mu * alpha * dt)
                        / (np.dot(r0, v0)
                           + np.sign(dt) * np.sqrt(-mu * a)
                           * (1.0 - r0_mag * alpha))))
    else:
        # Parabolic
        h = np.cross(r0, v0)
        h_mag = np.linalg.norm(h)
        p = h_mag**2 / mu
        s = 0.5 * np.arctan(1.0 / (3.0 * np.sqrt(mu / p**3) * dt))
        w = np.arctan(np.cbrt(np.tan(s)))
        chi = np.sqrt(p) * 2.0 / np.tan(2.0 * w)

    # Newton-Raphson iteration to solve Kepler's equation in universal variables
    max_iter = 100
    tol = 1e-12

    for _ in range(max_iter):
        psi = chi**2 * alpha
        c2 = _stumpff_c2(psi)
        c3 = _stumpff_c3(psi)

        chi2 = chi * chi
        chi3 = chi2 * chi

        # Universal Kepler equation: F(chi) = 0
        # t(chi) = chi^3 * c3 + (r0.v0/sqrt(mu)) * chi^2 * c2 + r0_mag * chi * (1 - psi*c3)
        # dt/dchi = chi^2 * c2 + (r0.v0/sqrt(mu)) * chi * (1 - psi*c3) + r0_mag * (1 - psi*c2)
        # Note: dt/dchi = r(chi) / sqrt(mu) ... but it's cleaner to use the explicit formula

        fn = (chi3 * c3
              + rdotv_over_sqrtmu * chi2 * c2
              + r0_mag * chi * (1.0 - psi * c3)) - sqrt_mu * dt

        # r(chi) = chi^2 * c2 + (r0.v0/sqrt(mu)) * chi * (1 - psi*c3) + r0_mag * (1 - psi*c2)
        r_chi = (chi2 * c2
                 + rdotv_over_sqrtmu * chi * (1.0 - psi * c3)
                 + r0_mag * (1.0 - psi * c2))

        # Newton update: delta = -F / F'
        # F' = dt/dchi = r_chi  (since F = t(chi) - sqrt(mu)*dt, F' = dt_dchi)
        # Actually F' = d/dchi of the first terms = r_chi
        delta_chi = -fn / r_chi
        chi = chi + delta_chi

        if abs(delta_chi) < tol:
            break

    # Recompute final Stumpff values
    psi = chi**2 * alpha
    c2 = _stumpff_c2(psi)
    c3 = _stumpff_c3(psi)

    chi2 = chi * chi
    chi3 = chi2 * chi

    # Lagrange coefficients f, g
    f = 1.0 - chi2 / r0_mag * c2
    g = dt - chi3 / sqrt_mu * c3

    # New position
    r = f * r0 + g * v0
    r_mag = np.linalg.norm(r)

    # Lagrange coefficients fdot, gdot
    fdot = sqrt_mu / (r_mag * r0_mag) * chi * (psi * c3 - 1.0)
    gdot = 1.0 - chi2 / r_mag * c2

    # New velocity
    v = fdot * r0 + gdot * v0

    return r, v
