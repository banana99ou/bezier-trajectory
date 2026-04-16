"""J2 편평도 섭동 가속도."""

import numpy as np

from orbit_transfer.constants import MU_EARTH, J2 as J2_DEFAULT, R_E as R_E_DEFAULT


def j2_acceleration(r, mu=MU_EARTH, J2=J2_DEFAULT, R_E=R_E_DEFAULT):
    """J2 섭동 가속도 (ECI 좌표계).

    Args:
        r: 위치 벡터 [km], shape (3,) - NumPy 또는 CasADi
        mu: 중력 상수 [km^3/s^2]
        J2: J2 계수
        R_E: 지구 적도 반지름 [km]

    Returns:
        a_J2: J2 섭동 가속도 [km/s^2], shape (3,)
    """
    try:
        import casadi as ca

        if isinstance(r, (ca.MX, ca.SX, ca.DM)):
            x = r[0]
            y = r[1]
            z = r[2]
            r_norm = ca.norm_2(r)
            r2 = r_norm**2
            r5 = r_norm**5
            coeff = -1.5 * J2 * mu * R_E**2 / r5
            z2_over_r2 = z**2 / r2

            ax = coeff * x * (1.0 - 5.0 * z2_over_r2)
            ay = coeff * y * (1.0 - 5.0 * z2_over_r2)
            az = coeff * z * (3.0 - 5.0 * z2_over_r2)
            return ca.vertcat(ax, ay, az)
    except ImportError:
        pass

    x, y, z = r[0], r[1], r[2]
    r_norm = np.linalg.norm(r)
    r2 = r_norm**2
    r5 = r_norm**5
    coeff = -1.5 * J2 * mu * R_E**2 / r5
    z2_over_r2 = z**2 / r2

    ax = coeff * x * (1.0 - 5.0 * z2_over_r2)
    ay = coeff * y * (1.0 - 5.0 * z2_over_r2)
    az = coeff * z * (3.0 - 5.0 * z2_over_r2)
    return np.array([ax, ay, az])
