"""Classical orbital elements ↔ ECI 상태벡터 변환."""

import numpy as np
import casadi as ca


def oe_to_rv(oe, mu):
    """Classical orbital elements → ECI position/velocity vectors.

    Parameters
    ----------
    oe : tuple of float
        (a, e, i, Omega, omega, nu)
        a     : semi-major axis [km]
        e     : eccentricity [-]
        i     : inclination [rad]
        Omega : RAAN [rad]
        omega : argument of periapsis [rad]
        nu    : true anomaly [rad]
    mu : float
        Gravitational parameter [km^3/s^2]

    Returns
    -------
    r : ndarray (3,)
        Position vector in ECI [km]
    v : ndarray (3,)
        Velocity vector in ECI [km/s]
    """
    a, e, i, Omega, omega, nu = oe

    # Semi-latus rectum
    p = a * (1.0 - e**2)

    # Distance
    r_mag = p / (1.0 + e * np.cos(nu))

    # Position and velocity in perifocal frame
    r_pqw = np.array([r_mag * np.cos(nu),
                       r_mag * np.sin(nu),
                       0.0])

    v_pqw = np.sqrt(mu / p) * np.array([-np.sin(nu),
                                          e + np.cos(nu),
                                          0.0])

    # Rotation matrix: perifocal → ECI
    cos_O = np.cos(Omega)
    sin_O = np.sin(Omega)
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    R = np.array([
        [cos_O * cos_w - sin_O * sin_w * cos_i,
         -cos_O * sin_w - sin_O * cos_w * cos_i,
         sin_O * sin_i],
        [sin_O * cos_w + cos_O * sin_w * cos_i,
         -sin_O * sin_w + cos_O * cos_w * cos_i,
         -cos_O * sin_i],
        [sin_w * sin_i,
         cos_w * sin_i,
         cos_i],
    ])

    r = R @ r_pqw
    v = R @ v_pqw

    return r, v


def rv_to_oe(r, v, mu):
    """ECI position/velocity → classical orbital elements.

    Parameters
    ----------
    r : ndarray (3,)
        Position vector in ECI [km]
    v : ndarray (3,)
        Velocity vector in ECI [km/s]
    mu : float
        Gravitational parameter [km^3/s^2]

    Returns
    -------
    oe : tuple of float
        (a, e, i, Omega, omega, nu)

    Notes
    -----
    - e < 1e-10 인 경우 omega=0, nu=argument of latitude 로 처리
    - i < 1e-10 인 경우 Omega=0 으로 처리
    """
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # Angular momentum
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # Node vector
    K = np.array([0.0, 0.0, 1.0])
    n = np.cross(K, h)
    n_mag = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)

    # Semi-major axis (vis-viva)
    energy = 0.5 * v_mag**2 - mu / r_mag
    a = -mu / (2.0 * energy)

    # Inclination
    i = np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0))

    # RAAN
    if i < 1e-10:
        # Equatorial orbit: Omega undefined, set to 0
        Omega = 0.0
    else:
        if n_mag < 1e-14:
            Omega = 0.0
        else:
            Omega = np.arccos(np.clip(n[0] / n_mag, -1.0, 1.0))
            if n[1] < 0.0:
                Omega = 2.0 * np.pi - Omega

    # Argument of periapsis and true anomaly
    if e < 1e-10:
        # Circular orbit: omega undefined, set to 0
        omega = 0.0
        # True anomaly = argument of latitude
        if i < 1e-10:
            # Circular equatorial: use true longitude
            nu = np.arctan2(r[1], r[0])
            if nu < 0.0:
                nu += 2.0 * np.pi
        else:
            # Circular inclined: use argument of latitude
            if n_mag < 1e-14:
                nu = 0.0
            else:
                n_hat = n / n_mag
                nu = np.arccos(np.clip(np.dot(n_hat, r / r_mag), -1.0, 1.0))
                # Check sign via cross product
                if np.dot(np.cross(n, r), h) < 0.0:
                    nu = 2.0 * np.pi - nu
    else:
        if i < 1e-10:
            # Elliptic equatorial
            omega = np.arctan2(e_vec[1], e_vec[0])
            if omega < 0.0:
                omega += 2.0 * np.pi
        else:
            if n_mag < 1e-14:
                omega = 0.0
            else:
                omega = np.arccos(np.clip(np.dot(n, e_vec) / (n_mag * e), -1.0, 1.0))
                if e_vec[2] < 0.0:
                    omega = 2.0 * np.pi - omega

        # True anomaly
        cos_nu = np.clip(np.dot(e_vec, r) / (e * r_mag), -1.0, 1.0)
        nu = np.arccos(cos_nu)
        if np.dot(r, v) < 0.0:
            nu = 2.0 * np.pi - nu

    return (a, e, i, Omega, omega, nu)


def oe_to_rv_casadi(oe, mu):
    """Classical orbital elements → ECI (CasADi symbolic version).

    Parameters
    ----------
    oe : casadi.MX or list/tuple of 6 MX symbols
        (a, e, i, Omega, omega, nu)
    mu : float or casadi.MX
        Gravitational parameter [km^3/s^2]

    Returns
    -------
    r : casadi.MX (3, 1)
        Position vector in ECI [km]
    v : casadi.MX (3, 1)
        Velocity vector in ECI [km/s]
    """
    if isinstance(oe, (list, tuple)):
        a, e, i, Omega, omega, nu = oe
    else:
        a = oe[0]
        e = oe[1]
        i = oe[2]
        Omega = oe[3]
        omega = oe[4]
        nu = oe[5]

    # Semi-latus rectum
    p = a * (1.0 - e**2)

    # Distance
    r_mag = p / (1.0 + e * ca.cos(nu))

    # Position and velocity in perifocal frame
    r_pqw = ca.vertcat(r_mag * ca.cos(nu),
                        r_mag * ca.sin(nu),
                        0.0)

    v_pqw = ca.sqrt(mu / p) * ca.vertcat(-ca.sin(nu),
                                           e + ca.cos(nu),
                                           0.0)

    # Rotation matrix: perifocal → ECI
    cos_O = ca.cos(Omega)
    sin_O = ca.sin(Omega)
    cos_w = ca.cos(omega)
    sin_w = ca.sin(omega)
    cos_i = ca.cos(i)
    sin_i = ca.sin(i)

    R = ca.vertcat(
        ca.horzcat(cos_O * cos_w - sin_O * sin_w * cos_i,
                    -cos_O * sin_w - sin_O * cos_w * cos_i,
                    sin_O * sin_i),
        ca.horzcat(sin_O * cos_w + cos_O * sin_w * cos_i,
                    -sin_O * sin_w + cos_O * cos_w * cos_i,
                    -cos_O * sin_i),
        ca.horzcat(sin_w * sin_i,
                    cos_w * sin_i,
                    cos_i),
    )

    r = R @ r_pqw
    v = R @ v_pqw

    return r, v
