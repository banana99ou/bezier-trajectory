"""초기값 생성 모듈."""

import numpy as np

from ..constants import MU_EARTH, R_E
from ..astrodynamics.orbital_elements import oe_to_rv
from ..astrodynamics.kepler import kepler_propagate
from ..types import TransferConfig


def linear_interpolation_guess(config: TransferConfig, N_points: int):
    """선형 보간 초기값 생성.

    출발 궤도와 도착 궤도 각각을 케플러 전파한 후,
    두 궤적을 시간 비율로 부드럽게 혼합(blending)하여 초기값을 생성한다.
    이렇게 하면 모든 중간점이 물리적으로 의미 있는 궤도 위에 놓이게 된다.

    Args:
        config: 전이 구성
        N_points: collocation 점 수 (61 for M=30)

    Returns:
        t: 시간 배열 [s], shape (N_points,)
        x_guess: 상태 초기값 (6, N_points)
        u_guess: 제어 초기값 (3, N_points)  [zeros]
        nu0_guess: 출발 true anomaly [rad]
        nuf_guess: 도착 true anomaly [rad]

    Algorithm:
        1. nu0=0, nuf=pi (coplanar에서는 apse line 방향)
        2. 출발 궤도에서 케플러 전파 (forward)
        3. 도착 궤도에서 케플러 역전파 (backward)
        4. 두 궤적을 cosine blending으로 혼합
    """
    T = config.T_max
    t = np.linspace(0, T, N_points)

    # True anomaly 초기값
    nu0_guess = 0.0
    nuf_guess = np.pi

    # 출발 궤도 요소: (a0, e0, i0, Omega=0, omega=0, nu0)
    oe0 = (config.a0, config.e0, config.i0, 0.0, 0.0, nu0_guess)
    r0, v0 = oe_to_rv(oe0, MU_EARTH)

    # 도착 궤도 요소: (af, ef, if_, Omega=0, omega=0, nuf)
    oef = (config.af, config.ef, config.if_, 0.0, 0.0, nuf_guess)
    rf, vf = oe_to_rv(oef, MU_EARTH)

    # 출발 궤도 전파 (forward) 및 도착 궤도 역전파 (backward)
    x_depart = np.zeros((6, N_points))
    x_arrive = np.zeros((6, N_points))

    x_depart[:3, 0] = r0
    x_depart[3:, 0] = v0
    x_arrive[:3, -1] = rf
    x_arrive[3:, -1] = vf

    for k in range(1, N_points):
        dt_fwd = t[k]
        rk, vk = kepler_propagate(r0, v0, dt_fwd, MU_EARTH)
        x_depart[:3, k] = rk
        x_depart[3:, k] = vk

    for k in range(N_points - 2, -1, -1):
        dt_bwd = t[k] - T  # 음수
        rk, vk = kepler_propagate(rf, vf, dt_bwd, MU_EARTH)
        x_arrive[:3, k] = rk
        x_arrive[3:, k] = vk

    # Cosine blending: alpha(t) = 0.5 * (1 - cos(pi * t/T))
    x_guess = np.zeros((6, N_points))
    for k in range(N_points):
        alpha = 0.5 * (1.0 - np.cos(np.pi * t[k] / T)) if T > 0 else 0.0
        x_guess[:, k] = (1.0 - alpha) * x_depart[:, k] + alpha * x_arrive[:, k]

    u_guess = np.zeros((3, N_points))

    return t, x_guess, u_guess, nu0_guess, nuf_guess


def keplerian_guess(config: TransferConfig, N_points: int):
    """케플러 전파 기반 초기값 생성.

    출발 궤도에서 케플러 전파한 궤적을 초기값으로 사용.
    coplanar: nu0=0 (apse line)
    plane change: nu0=pi/2 (line of nodes 방향)

    Args:
        config: 전이 구성
        N_points: collocation 점 수

    Returns:
        t: 시간 배열 [s], shape (N_points,)
        x_guess: 상태 초기값 (6, N_points)
        u_guess: 제어 초기값 (3, N_points)  [zeros]
        nu0_guess: 출발 true anomaly [rad]
        nuf_guess: 도착 true anomaly [rad]
    """
    T = config.T_max
    t = np.linspace(0, T, N_points)

    # Plane change 여부에 따라 nu0 결정
    is_plane_change = abs(config.delta_i) > 1e-10
    nu0_guess = np.pi / 2 if is_plane_change else 0.0
    nuf_guess = np.pi

    # 출발 궤도 상태벡터
    oe0 = (config.a0, config.e0, config.i0, 0.0, 0.0, nu0_guess)
    r0, v0 = oe_to_rv(oe0, MU_EARTH)

    # 케플러 전파
    x_guess = np.zeros((6, N_points))
    x_guess[:3, 0] = r0
    x_guess[3:, 0] = v0

    for k in range(1, N_points):
        dt = t[k] - t[0]
        rk, vk = kepler_propagate(r0, v0, dt, MU_EARTH)
        x_guess[:3, k] = rk
        x_guess[3:, k] = vk

    u_guess = np.zeros((3, N_points))

    return t, x_guess, u_guess, nu0_guess, nuf_guess
