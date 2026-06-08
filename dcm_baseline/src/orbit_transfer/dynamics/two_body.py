"""이체 중력 가속도."""

import numpy as np

from orbit_transfer.constants import MU_EARTH


def gravity_acceleration(r, mu=MU_EARTH):
    """중력 가속도 계산.

    Args:
        r: 위치 벡터 [km], shape (3,) - NumPy 또는 CasADi
        mu: 중력 상수 [km^3/s^2]

    Returns:
        a_grav: 중력 가속도 [km/s^2], shape (3,)
    """
    try:
        import casadi as ca

        if isinstance(r, (ca.MX, ca.SX, ca.DM)):
            r_norm = ca.norm_2(r)
            return -mu / r_norm**3 * r
    except ImportError:
        pass

    r_norm = np.linalg.norm(r)
    return -mu / r_norm**3 * r
