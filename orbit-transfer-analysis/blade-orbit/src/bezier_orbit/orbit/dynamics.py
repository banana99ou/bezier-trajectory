"""ECI 궤도 운동방정식 및 상태 Jacobian (정규화 변수).

이론: docs/reports/002_orbital_dynamics/

정규화 단위에서 mu* = 1. 모든 함수의 입출력은 정규화된 변수.
상태벡터 x = [r; v] ∈ R^6, r = [x, y, z], v = [vx, vy, vz].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..normalize import R_EARTH, J2_EARTH


# ── 2체 + J2 동역학 (Level 0) ───────────────────────────────────
def eom_twobody_j2(
    x: NDArray,
    u: NDArray | None = None,
    *,
    Re_star: float = 1.0,
    J2: float = J2_EARTH,
    include_j2: bool = True,
) -> NDArray:
    """정규화된 2체 + J2 운동방정식.

    ẋ = f(x) + B·u

    Parameters
    ----------
    x : ndarray, shape (6,)
        상태벡터 [r; v] (정규화).
    u : ndarray, shape (3,), optional
        추진 가속도 (정규화). None이면 0.
    Re_star : float
        정규화된 지구 반경 R_earth / DU.
    J2 : float
        J2 계수.
    include_j2 : bool
        J2 섭동 포함 여부 (기본 True).

    Returns
    -------
    xdot : ndarray, shape (6,)
        상태 미분 [v; a].
    """
    r = x[:3]
    v = x[3:6]
    r_mag = np.linalg.norm(r)
    r3 = r_mag**3

    # 2체 중심 중력 (mu* = 1)
    a_grav = -r / r3

    # J2 섭동
    if include_j2:
        a_grav = a_grav + _accel_j2(r, r_mag, Re_star, J2)

    # 추진 가속도
    a_total = a_grav
    if u is not None:
        a_total = a_total + np.asarray(u)

    return np.concatenate([v, a_total])


def _accel_j2(r: NDArray, r_mag: float, Re_star: float, J2: float) -> NDArray:
    """J2 섭동 가속도 (정규화).

    a_J2 = (3 J2 Re*^2) / (2 r^5) · [x(5z²/r² - 1), y(5z²/r² - 1), z(5z²/r² - 3)]
    """
    x, y, z = r
    r2 = r_mag**2
    r5 = r_mag**5
    z2_r2 = z**2 / r2

    coeff = 1.5 * J2 * Re_star**2 / r5
    return coeff * np.array([
        x * (5.0 * z2_r2 - 1.0),
        y * (5.0 * z2_r2 - 1.0),
        z * (5.0 * z2_r2 - 3.0),
    ])


# ── 상태 Jacobian A(x) ─────────────────────────────────────────
def jacobian_twobody_j2(
    x: NDArray,
    *,
    Re_star: float = 1.0,
    J2: float = J2_EARTH,
    include_j2: bool = True,
) -> NDArray:
    """상태 Jacobian A = ∂f/∂x ∈ R^{6×6}.

    A = [[0, I], [A_rr, 0]]

    2체 중력 기여: ∂(-r/r³)/∂r = -I/r³ + 3·r·r^T/r⁵
    J2 기여: 해석적 Jacobian.

    Parameters
    ----------
    x : ndarray, shape (6,)
        상태벡터 [r; v] (정규화).
    Re_star : float
        정규화된 지구 반경.
    J2 : float
        J2 계수.
    include_j2 : bool
        J2 Jacobian 포함 여부.

    Returns
    -------
    A : ndarray, shape (6, 6)
        상태 Jacobian.
    """
    r = x[:3]
    r_mag = np.linalg.norm(r)
    r2 = r_mag**2
    r3 = r_mag**3
    r5 = r_mag**5

    # 2체 중력 Jacobian: ∂(-r/r³)/∂r = -I/r³ + 3·r·rT/r⁵
    I3 = np.eye(3)
    A_rr = -I3 / r3 + 3.0 * np.outer(r, r) / r5

    # J2 Jacobian
    if include_j2:
        A_rr = A_rr + _jacobian_j2(r, r_mag, Re_star, J2)

    A = np.zeros((6, 6))
    A[:3, 3:6] = I3      # ∂ṙ/∂v = I
    A[3:6, :3] = A_rr    # ∂v̇/∂r
    # A[3:6, 3:6] = 0 (대기항력 없으면 속도 Jacobian = 0)
    return A


def _jacobian_j2(r: NDArray, r_mag: float, Re_star: float, J2: float) -> NDArray:
    """J2 섭동의 위치 Jacobian ∂a_J2/∂r ∈ R^{3×3}.

    해석적 유도. 보고서 002 식 (9) 및 006 식 (8) 참고.
    """
    x, y, z = r
    r2 = r_mag**2
    r7 = r_mag**7

    coeff = 1.5 * J2 * Re_star**2

    z2 = z**2
    z2_r2 = z2 / r2

    # 공통 계수들
    c1 = coeff / r_mag**5

    # ∂a_J2/∂r 의 각 원소를 직접 계산
    # a_J2_x = coeff * x * (5z²/r² - 1) / r⁵  (r⁵는 이미 coeff에 포함)
    # 편미분을 chain rule로 전개

    # 보조 변수
    r4 = r2 * r2

    # 방법: 텐서 대수 대신 직접 전개
    # f_i = coeff/r⁵ · r_i · (5z²/r² - α_i) where α = [1,1,3]
    # ∂f_i/∂r_j 계산

    Jac = np.zeros((3, 3))

    alpha = np.array([1.0, 1.0, 3.0])
    ri = np.array([x, y, z])

    for i in range(3):
        for j in range(3):
            # ∂/∂r_j [ (coeff / r⁵) · r_i · (5z²/r² - α_i) ]
            # = coeff · ∂/∂r_j [ r_i · (5z²/r² - α_i) / r⁵ ]

            # Let g = r_i · (5z²/r² - α_i) / r⁵
            # g = r_i · (5z² - α_i · r²) / r⁷

            # Numerator: h = r_i · (5z² - α_i · r²)
            # ∂h/∂r_j = δ_{ij} · (5z² - α_i·r²) + r_i · (10z·δ_{j2} - 2·α_i·r_j)
            # where δ_{j2} means j==2 (z component)

            h = ri[i] * (5.0 * z2 - alpha[i] * r2)

            dh_drj = 0.0
            if i == j:
                dh_drj += 5.0 * z2 - alpha[i] * r2
            dh_drj += ri[i] * (-2.0 * alpha[i] * ri[j])
            if j == 2:  # z component
                dh_drj += ri[i] * 10.0 * z

            # ∂(1/r⁷)/∂r_j = -7·r_j/r⁹
            # ∂g/∂r_j = dh_drj / r⁷ + h · (-7·r_j / r⁹)
            Jac[i, j] = coeff * (dh_drj / r7 - 7.0 * h * ri[j] / (r7 * r2))

    return Jac


# ── 수치 적분 유틸리티 ──────────────────────────────────────────
def propagate_rk4(
    x0: NDArray,
    tau_span: tuple[float, float],
    n_steps: int,
    *,
    u_func=None,
    Re_star: float = 1.0,
    J2: float = J2_EARTH,
    include_j2: bool = True,
    t_f: float = 1.0,
    extra_accel_func=None,
) -> tuple[NDArray, NDArray]:
    """RK4 수치 적분.

    정규화 시간 τ ∈ [τ_start, τ_end] 구간을 적분.
    물리 시간 dt = t_f · dτ.

    Parameters
    ----------
    x0 : ndarray, shape (6,)
        초기 상태.
    tau_span : (float, float)
        정규화 시간 구간.
    n_steps : int
        적분 스텝 수.
    u_func : callable, optional
        추진 가속도 함수 u(τ) → (3,).
    Re_star : float
        정규화된 지구 반경.
    J2 : float
        J2 계수.
    include_j2 : bool
        J2 포함 여부.
    t_f : float
        비행시간 (정규화).
    extra_accel_func : callable, optional
        추가 섭동 가속도 함수 (τ, x) → (3,).
        Level 1/2 섭동을 주입하는 데 사용.

    Returns
    -------
    tau_arr : ndarray, shape (n_steps+1,)
        시간 배열.
    x_arr : ndarray, shape (n_steps+1, 6)
        상태 궤적.
    """
    tau_arr = np.linspace(tau_span[0], tau_span[1], n_steps + 1)
    dtau = tau_arr[1] - tau_arr[0]

    x_arr = np.zeros((n_steps + 1, 6))
    x_arr[0] = x0

    def rhs(tau: float, x: NDArray) -> NDArray:
        u = None if u_func is None else u_func(tau)
        # ẋ/dt* = f(x) + B·u  →  dx/dτ = t_f · (f(x) + B·u)
        xdot = t_f * eom_twobody_j2(
            x, u, Re_star=Re_star, J2=J2, include_j2=include_j2,
        )
        # 고차 섭동 가속도 추가
        if extra_accel_func is not None:
            a_extra = extra_accel_func(tau, x)
            xdot[3:6] += t_f * a_extra
        return xdot

    for k in range(n_steps):
        tau_k = tau_arr[k]
        xk = x_arr[k]

        k1 = rhs(tau_k, xk)
        k2 = rhs(tau_k + 0.5 * dtau, xk + 0.5 * dtau * k1)
        k3 = rhs(tau_k + 0.5 * dtau, xk + 0.5 * dtau * k2)
        k4 = rhs(tau_k + dtau, xk + dtau * k3)

        x_arr[k + 1] = xk + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return tau_arr, x_arr
