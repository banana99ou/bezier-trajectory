"""Hermite-Simpson collocation (Pass 1)."""

import casadi as ca
import numpy as np

from ..config import HS_NUM_SEGMENTS, IPOPT_OPTIONS_PASS1
from ..constants import MU_EARTH, R_E
from ..dynamics.eom import create_dynamics_function
from ..astrodynamics.orbital_elements import oe_to_rv_casadi
from ..types import TransferConfig, TrajectoryResult


class HermiteSimpsonCollocation:
    """Hermite-Simpson collocation (Pass 1).

    M=30 균일 구간, 61 collocation points (31 주 노드 + 30 중간점)
    결정변수: x (6x61), u (3x61), nu0, nuf
    총 변수: 6*61 + 3*61 = 549 + 2 = 551

    비용: Simpson 적분 J = sum (h/6)[||u_k||^2 + 4||u_{k+0.5}||^2 + ||u_{k+1}||^2]

    제약:
    - Simpson continuity: x_{k+1} = x_k + (h/6)(f_k + 4*f_{k+0.5} + f_{k+1})
    - Hermite midpoint: x_{k+0.5} = 0.5*(x_k + x_{k+1}) + (h/8)(f_k - f_{k+1})
    - 경계조건: x(0) = oe_to_rv(alpha0, nu0), x(T) = oe_to_rv(alphaf, nuf)
    - 추력 상한: ||u_k|| <= u_max for all k
    - 최소 고도: ||r_k|| >= R_E + h_min for all k
    """

    def __init__(self, config: TransferConfig, M: int = HS_NUM_SEGMENTS):
        self.config = config
        self.M = M
        self.N_points = 2 * M + 1  # 61
        self.eom_func = create_dynamics_function(include_j2=True)

    def solve(self, x_guess=None, u_guess=None, nu0_guess=None, nuf_guess=None,
              ipopt_options=None):
        """NLP를 구성하고 풀기.

        Args:
            x_guess: 상태 초기값 (6, N_points)
            u_guess: 제어 초기값 (3, N_points)
            nu0_guess: 출발 true anomaly 초기값 [rad]
            nuf_guess: 도착 true anomaly 초기값 [rad]
            ipopt_options: IPOPT 옵션 (기본값: IPOPT_OPTIONS_PASS1)

        Returns:
            TrajectoryResult
        """
        opti = ca.Opti()

        # 결정변수
        X = opti.variable(6, self.N_points)  # 상태
        U = opti.variable(3, self.N_points)  # 제어
        nu0 = opti.variable()                # 출발 true anomaly
        nuf = opti.variable()                # 도착 true anomaly

        # T_f 결정변수 (전이시간 자유변수)
        T_f = opti.variable()
        opti.subject_to(T_f >= self.config.T_min)
        opti.subject_to(T_f <= self.config.T_max)
        h = T_f / self.M  # 심볼릭 구간 폭

        # 비용함수 (Simpson 적분)
        J = 0
        for k in range(self.M):
            idx_k = 2 * k
            idx_m = 2 * k + 1
            idx_k1 = 2 * k + 2
            u_k = U[:, idx_k]
            u_m = U[:, idx_m]
            u_k1 = U[:, idx_k1]
            J += (h / 6.0) * (
                ca.dot(u_k, u_k) + 4 * ca.dot(u_m, u_m) + ca.dot(u_k1, u_k1)
            )
        opti.minimize(J)

        # Collocation 제약
        for k in range(self.M):
            idx_k = 2 * k
            idx_m = 2 * k + 1
            idx_k1 = 2 * k + 2

            x_k = X[:, idx_k]
            x_m = X[:, idx_m]
            x_k1 = X[:, idx_k1]
            u_k = U[:, idx_k]
            u_m = U[:, idx_m]
            u_k1 = U[:, idx_k1]

            f_k = self.eom_func(x_k, u_k)
            f_m = self.eom_func(x_m, u_m)
            f_k1 = self.eom_func(x_k1, u_k1)

            # Simpson continuity
            opti.subject_to(
                x_k1 == x_k + (h / 6.0) * (f_k + 4 * f_m + f_k1)
            )

            # Hermite midpoint
            opti.subject_to(
                x_m == 0.5 * (x_k + x_k1) + (h / 8.0) * (f_k - f_k1)
            )

        # 경계조건 (ECI 상태벡터 직접 매칭)
        a0 = self.config.a0
        i0 = self.config.i0
        oe0 = ca.vertcat(a0, self.config.e0, i0, 0.0, 0.0, nu0)
        r0, v0 = oe_to_rv_casadi(oe0, MU_EARTH)
        rv0 = ca.vertcat(r0, v0)
        opti.subject_to(X[:, 0] == rv0)

        af = self.config.af
        if_ = self.config.if_
        oef = ca.vertcat(af, self.config.ef, if_, 0.0, 0.0, nuf)
        rf, vf = oe_to_rv_casadi(oef, MU_EARTH)
        rvf = ca.vertcat(rf, vf)
        opti.subject_to(X[:, -1] == rvf)

        # 추력 상한 (각 점에서)
        u_max_sq = self.config.u_max ** 2
        for k in range(self.N_points):
            opti.subject_to(ca.dot(U[:, k], U[:, k]) <= u_max_sq)

        # 최소 고도 (각 점에서)
        r_min_sq = (R_E + self.config.h_min) ** 2
        for k in range(self.N_points):
            opti.subject_to(ca.dot(X[:3, k], X[:3, k]) >= r_min_sq)

        # 초기값 설정
        opti.set_initial(T_f, self.config.T_max)
        if x_guess is not None:
            opti.set_initial(X, x_guess)
        if u_guess is not None:
            opti.set_initial(U, u_guess)
        if nu0_guess is not None:
            opti.set_initial(nu0, nu0_guess)
        else:
            opti.set_initial(nu0, 0.0)
        if nuf_guess is not None:
            opti.set_initial(nuf, nuf_guess)
        else:
            opti.set_initial(nuf, np.pi)

        # IPOPT 솔버 설정
        opts = dict(IPOPT_OPTIONS_PASS1)
        if ipopt_options:
            opts.update(ipopt_options)
        opti.solver('ipopt', opts)

        try:
            sol = opti.solve()
            converged = True
            cost = float(sol.value(J))
            T_val = float(sol.value(T_f))
            t = np.linspace(0, T_val, self.N_points)
            x = np.array(sol.value(X))
            u = np.array(sol.value(U))
            nu0_val = float(sol.value(nu0))
            nuf_val = float(sol.value(nuf))
        except RuntimeError:
            converged = False
            cost = float('inf')
            T_val = float(opti.debug.value(T_f))
            # T_val이 음수 또는 0이면 T_max로 대체 (debug 값 신뢰 불가)
            if T_val <= 0:
                T_val = self.config.T_max
            t = np.linspace(0, T_val, self.N_points)
            x = np.array(opti.debug.value(X))
            u = np.array(opti.debug.value(U))
            nu0_val = float(opti.debug.value(nu0))
            nuf_val = float(opti.debug.value(nuf))

        # 피크 탐지 + 분류
        from ..classification.peak_detection import detect_peaks
        from ..classification.classifier import classify_profile

        u_mag = np.linalg.norm(u, axis=0)
        n_peaks, _, _ = detect_peaks(t, u_mag, T_val)
        profile_class = classify_profile(n_peaks)

        return TrajectoryResult(
            converged=converged,
            cost=cost,
            t=t,
            x=x,
            u=u,
            nu0=nu0_val,
            nuf=nuf_val,
            n_peaks=n_peaks,
            profile_class=profile_class,
            T_f=T_val,
        )
