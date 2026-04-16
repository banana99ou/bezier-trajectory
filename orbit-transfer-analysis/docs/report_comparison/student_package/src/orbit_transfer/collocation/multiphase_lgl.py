"""Multi-Phase LGL Pseudospectral Collocation (Pass 2)."""

import casadi as ca
import numpy as np

from ..config import IPOPT_OPTIONS_PASS2
from ..constants import MU_EARTH, R_E
from ..dynamics.eom import create_dynamics_function
from ..astrodynamics.orbital_elements import oe_to_rv_casadi
from ..types import TransferConfig, TrajectoryResult
from .lgl_nodes import (
    compute_lgl_nodes,
    compute_lgl_weights,
    compute_differentiation_matrix,
)


class MultiPhaseLGLCollocation:
    """Multi-Phase LGL Pseudospectral Collocation (Pass 2).

    각 Phase별 LGL collocation + 경계 linkage 제약.
    피크 구간: 15 노드, coasting 구간: 8 노드.

    NLP 구조:
    - Phase별 상태/제어 변수 X^(k), U^(k)
    - Phase 경계 시각은 고정 (Pass 1 피크탐지 결과 기반)
    - 비용: Phase별 LGL 적분 합산
    - 제약:
      - Phase별 collocation: (2/dt_k) * D @ X^(k) = f(X^(k), U^(k))
      - Linkage: X^(k)[-1] = X^(k+1)[0]
      - 경계조건: X^(1)[0] = rv0(nu0), X^(P)[-1] = rvf(nuf)
      - 추력 상한, 최소 고도
    """

    def __init__(self, config: TransferConfig, phases: list, T_fixed: float = None):
        """
        Args:
            config: 전이 구성
            phases: Phase 구조 리스트
                각 dict: {'t_start', 't_end', 'n_nodes', 'type'}
            T_fixed: Pass 1에서 결정된 전이시간 [s] (고정)
        """
        self.config = config
        self.phases = phases
        self.T_fixed = T_fixed
        self.eom_func = create_dynamics_function(include_j2=True)

    def solve(self, x_phases=None, u_phases=None, nu0_guess=0.0,
              nuf_guess=np.pi, ipopt_options=None):
        """Multi-Phase NLP 구성 및 풀기.

        Args:
            x_phases: Phase별 상태 초기값 리스트
            u_phases: Phase별 제어 초기값 리스트
            nu0_guess: 출발 true anomaly
            nuf_guess: 도착 true anomaly
            ipopt_options: IPOPT 옵션

        Returns:
            TrajectoryResult
        """
        opti = ca.Opti()

        n_phases = len(self.phases)

        # Phase별 변수 생성
        X_vars = []  # list of (6, N_k) MX
        U_vars = []  # list of (3, N_k) MX
        for phase in self.phases:
            N_k = phase['n_nodes']
            X_vars.append(opti.variable(6, N_k))
            U_vars.append(opti.variable(3, N_k))

        nu0 = opti.variable()
        nuf = opti.variable()

        # Phase별 LGL 데이터 사전 계산
        lgl_data = []
        for phase in self.phases:
            N_k = phase['n_nodes']
            N_deg = N_k - 1  # LGL 다항식 차수
            tau = compute_lgl_nodes(N_deg)
            w = compute_lgl_weights(N_deg, tau)
            D = compute_differentiation_matrix(N_deg, tau)
            dt_k = phase['t_end'] - phase['t_start']
            lgl_data.append({
                'tau': tau, 'w': w, 'D': D, 'dt_k': dt_k,
                'N_k': N_k, 'N_deg': N_deg,
            })

        # 비용함수: Phase별 LGL 적분 합산
        J = 0
        for p in range(n_phases):
            N_k = lgl_data[p]['N_k']
            w = lgl_data[p]['w']
            dt_k = lgl_data[p]['dt_k']

            for j in range(N_k):
                u_j = U_vars[p][:, j]
                J += (dt_k / 2.0) * w[j] * ca.dot(u_j, u_j)
        opti.minimize(J)

        # Phase별 collocation 제약
        for p in range(n_phases):
            N_k = lgl_data[p]['N_k']
            D = lgl_data[p]['D']
            dt_k = lgl_data[p]['dt_k']

            for i in range(N_k):
                # dx/dt at node i: (2/dt_k) * sum_j D[i,j] * X^(p)[:,j]
                dx_dt = ca.MX.zeros(6, 1)
                for j in range(N_k):
                    dx_dt += D[i, j] * X_vars[p][:, j]
                dx_dt = (2.0 / dt_k) * dx_dt

                # EoM at node i
                f_i = self.eom_func(X_vars[p][:, i], U_vars[p][:, i])

                opti.subject_to(dx_dt == f_i)

        # Linkage 제약 (Phase 경계 연속성)
        for p in range(n_phases - 1):
            opti.subject_to(X_vars[p][:, -1] == X_vars[p + 1][:, 0])

        # 경계조건
        a0 = self.config.a0
        i0 = self.config.i0
        oe0 = ca.vertcat(a0, self.config.e0, i0, 0.0, 0.0, nu0)
        r0, v0 = oe_to_rv_casadi(oe0, MU_EARTH)
        rv0 = ca.vertcat(r0, v0)
        opti.subject_to(X_vars[0][:, 0] == rv0)

        af = self.config.af
        if_ = self.config.if_
        oef = ca.vertcat(af, self.config.ef, if_, 0.0, 0.0, nuf)
        rf, vf = oe_to_rv_casadi(oef, MU_EARTH)
        rvf = ca.vertcat(rf, vf)
        opti.subject_to(X_vars[-1][:, -1] == rvf)

        # 추력 상한 + 최소 고도 (모든 Phase의 모든 노드)
        u_max_sq = self.config.u_max ** 2
        r_min_sq = (R_E + self.config.h_min) ** 2
        for p in range(n_phases):
            N_k = self.phases[p]['n_nodes']
            for k in range(N_k):
                opti.subject_to(
                    ca.dot(U_vars[p][:, k], U_vars[p][:, k]) <= u_max_sq
                )
                opti.subject_to(
                    ca.dot(X_vars[p][:3, k], X_vars[p][:3, k]) >= r_min_sq
                )

        # 초기값 설정
        if x_phases is not None:
            for p in range(n_phases):
                opti.set_initial(X_vars[p], x_phases[p])
        if u_phases is not None:
            for p in range(n_phases):
                opti.set_initial(U_vars[p], u_phases[p])
        opti.set_initial(nu0, nu0_guess)
        opti.set_initial(nuf, nuf_guess)

        # IPOPT 설정
        opts = dict(IPOPT_OPTIONS_PASS2)
        if ipopt_options:
            opts.update(ipopt_options)
        opti.solver('ipopt', opts)

        try:
            sol = opti.solve()
            converged = True
            cost = float(sol.value(J))
            # 결과 조립
            t_all, x_all, u_all = self._assemble_result(
                sol, X_vars, U_vars, lgl_data
            )
            nu0_val = float(sol.value(nu0))
            nuf_val = float(sol.value(nuf))
        except RuntimeError:
            converged = False
            cost = float('inf')
            t_all, x_all, u_all = self._assemble_result_debug(
                opti, X_vars, U_vars, lgl_data
            )
            nu0_val = float(opti.debug.value(nu0))
            nuf_val = float(opti.debug.value(nuf))

        # 피크 탐지 + 분류
        from ..classification.peak_detection import detect_peaks
        from ..classification.classifier import classify_profile

        T = self.T_fixed if self.T_fixed is not None else self.config.T_max
        u_mag = np.linalg.norm(u_all, axis=0)
        phase_bounds = [(p['t_start'], p['t_end']) for p in self.phases]
        n_peaks, _, _ = detect_peaks(t_all, u_mag, T,
                                     phase_boundaries=phase_bounds)
        profile_class = classify_profile(n_peaks)

        return TrajectoryResult(
            converged=converged, cost=cost,
            t=t_all, x=x_all, u=u_all,
            nu0=nu0_val, nuf=nuf_val,
            n_peaks=n_peaks, profile_class=profile_class,
            T_f=T,
            solver_stats={
                'phase_boundaries': [
                    (p['t_start'], p['t_end']) for p in self.phases
                ],
            },
        )

    def _assemble_result(self, sol, X_vars, U_vars, lgl_data):
        """솔루션에서 연속 시간/상태/제어 배열 조립."""
        t_list, x_list, u_list = [], [], []
        for p, phase in enumerate(self.phases):
            tau = lgl_data[p]['tau']
            t_start = phase['t_start']
            t_end = phase['t_end']
            t_phys = t_start + (t_end - t_start) / 2.0 * (tau + 1.0)

            x_p = np.array(sol.value(X_vars[p]))
            u_p = np.array(sol.value(U_vars[p]))

            if p > 0:
                # linkage 점 중복 제거
                t_phys = t_phys[1:]
                x_p = x_p[:, 1:]
                u_p = u_p[:, 1:]

            t_list.append(t_phys)
            x_list.append(x_p)
            u_list.append(u_p)

        t_all = np.concatenate(t_list)
        x_all = np.hstack(x_list)
        u_all = np.hstack(u_list)

        return _enforce_monotonicity(t_all, x_all, u_all)

    def _assemble_result_debug(self, opti, X_vars, U_vars, lgl_data):
        """debug 값으로 조립."""
        t_list, x_list, u_list = [], [], []
        for p, phase in enumerate(self.phases):
            tau = lgl_data[p]['tau']
            t_start = phase['t_start']
            t_end = phase['t_end']
            t_phys = t_start + (t_end - t_start) / 2.0 * (tau + 1.0)

            x_p = np.array(opti.debug.value(X_vars[p]))
            u_p = np.array(opti.debug.value(U_vars[p]))

            if p > 0:
                t_phys = t_phys[1:]
                x_p = x_p[:, 1:]
                u_p = u_p[:, 1:]

            t_list.append(t_phys)
            x_list.append(x_p)
            u_list.append(u_p)

        t_all = np.concatenate(t_list)
        x_all = np.hstack(x_list)
        u_all = np.hstack(u_list)

        return _enforce_monotonicity(t_all, x_all, u_all)


def _enforce_monotonicity(t, x, u):
    """시간 배열의 엄격한 단조증가를 강제한다.

    1. 시간순 정렬
    2. 중복 시간 제거 (나중 phase 값 유지)
    3. 비단조 구간이 남아 있으면 미세 보정

    Args:
        t: 시간 배열 (N,)
        x: 상태 배열 (6, N)
        u: 제어 배열 (3, N)

    Returns:
        t, x, u: 엄격히 단조증가하는 시간 배열과 대응 상태/제어
    """
    # 1. 시간순 정렬
    sort_idx = np.argsort(t, kind='stable')
    t = t[sort_idx].copy()
    x = x[:, sort_idx].copy()
    u = u[:, sort_idx].copy()

    # 2. 중복 시간 제거 (동일 시각의 마지막 값 유지)
    if len(t) > 1:
        # 역순으로 unique를 찾으면 나중 값(이후 phase)을 유지
        _, unique_idx = np.unique(t[::-1], return_index=True)
        unique_idx = len(t) - 1 - unique_idx  # 원래 인덱스로 변환
        unique_idx = np.sort(unique_idx)
        t = t[unique_idx]
        x = x[:, unique_idx]
        u = u[:, unique_idx]

    # 3. 엄격한 단조증가 강제
    if len(t) > 1:
        eps = max(np.finfo(float).eps * np.abs(t[-1] - t[0]), 1e-14)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = t[i - 1] + eps

    return t, x, u
