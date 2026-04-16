"""우주선 운동방정식 (NumPy / CasADi)."""

import casadi as ca
import numpy as np

from orbit_transfer.constants import MU_EARTH
from orbit_transfer.dynamics.two_body import gravity_acceleration
from orbit_transfer.dynamics.j2_perturbation import j2_acceleration
from orbit_transfer.dynamics.drag import exponential_drag


def spacecraft_eom_numpy(x, u, mu=MU_EARTH, include_j2=True, include_drag=False):
    """NumPy 버전 운동방정식.

    Args:
        x: 상태 벡터 [r(3), v(3)], shape (6,)
        u: 제어 입력 (추력 가속도) [km/s^2], shape (3,)
        mu: 중력 상수 [km^3/s^2]
        include_j2: J2 섭동 포함 여부
        include_drag: 항력 포함 여부

    Returns:
        xdot: 상태 미분 [v(3), a(3)], shape (6,)
    """
    r = x[:3]
    v = x[3:]

    a = gravity_acceleration(r, mu)

    if include_j2:
        a = a + j2_acceleration(r, mu=mu)

    if include_drag:
        a = a + exponential_drag(r, v)

    a = a + u

    return np.concatenate([v, a])


def create_dynamics_function(mu=MU_EARTH, include_j2=True):
    """CasADi 심볼릭 운동방정식 Function 객체 생성.

    CasADi에서는 drag를 포함하지 않음 (고도 구간 분기 불가).

    Args:
        mu: 중력 상수 [km^3/s^2]
        include_j2: J2 섭동 포함 여부

    Returns:
        ca.Function('eom', [x, u], [xdot])
    """
    x = ca.MX.sym("x", 6)
    u = ca.MX.sym("u", 3)

    r = x[:3]
    v = x[3:]

    a = gravity_acceleration(r, mu)

    if include_j2:
        a = a + j2_acceleration(r, mu=mu)

    a = a + u

    xdot = ca.vertcat(v, a)

    return ca.Function("eom", [x, u], [xdot])
