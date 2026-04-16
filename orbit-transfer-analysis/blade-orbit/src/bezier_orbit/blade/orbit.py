"""BLADE 궤도전이 SCP.

Phase 2: 비선형 궤도 동역학 + J2 세차 경계조건 + 시간 최적화.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from bezier_orbit.bezier.basis import (
    bernstein,
    bernstein_eval,
    definite_integral,
    gram_matrix,
    int_matrix,
    double_int_matrix,
)
from bezier_orbit.bezier.algebra import (
    degree_elevate,
    degree_reduce,
    degree_elevate_multi,
    degree_reduce_multi,
    gravity_composition_pipeline,
)
from bezier_orbit.normalize import (
    CanonicalUnits,
    MU_EARTH,
    R_EARTH,
    J2_EARTH,
    from_orbit,
)
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.orbit.dynamics import (
    propagate_rk4,
    eom_twobody_j2,
    jacobian_twobody_j2,
)

from .basis import boundary_matrices, continuity_matrices


# ── J2 세차 속도 ─────────────────────────────────────────────────

def j2_precession_rates(
    a: float, e: float, inc: float,
    mu: float = MU_EARTH, J2: float = J2_EARTH, Re: float = R_EARTH,
) -> tuple[float, float]:
    """J2 세차에 의한 RAAN, AOP 변화율 [rad/s].

    Ω̇ = -1.5 n J2 (Re/p)² cos(i)
    ω̇ = 1.5 n J2 (Re/p)² (2 - 2.5 sin²(i))

    Returns
    -------
    raan_dot, aop_dot : [rad/s]
    """
    n = math.sqrt(mu / a**3)
    p = a * (1.0 - e**2)
    if p < 1e-6 * a:
        p = 1e-6 * a  # 고이심률 안전장치
    factor = 1.5 * n * J2 * (Re / p) ** 2
    raan_dot = -factor * math.cos(inc)
    aop_dot = factor * (2.0 - 2.5 * math.sin(inc) ** 2)
    return raan_dot, aop_dot


# ── 궤도 경계조건 ────────────────────────────────────────────────

@dataclasses.dataclass
class OrbitBC:
    """궤도 경계조건 (Keplerian 요소, 기준 시점)."""

    a: float        # 반장축 [km]
    e: float        # 이심률
    inc: float      # 경사각 [rad]
    raan: float     # RAAN [rad] (기준 시점)
    aop: float      # 근점인수 [rad] (기준 시점)
    ta: float = 0.0  # 진근점이각 [rad] (기준 시점)

    def at_time(
        self, t: float, mu: float = MU_EARTH,
    ) -> tuple[NDArray, NDArray]:
        """J2 세차 반영 + 케플러 전파 후 Cartesian 상태 [물리 단위].

        Parameters
        ----------
        t : 경과 시간 [s] (물리 단위).
        mu : 중력 상수.

        Returns
        -------
        r, v : (3,) 위치/속도 [km, km/s].
        """
        raan_dot, aop_dot = j2_precession_rates(self.a, self.e, self.inc, mu)
        n = math.sqrt(mu / self.a**3)

        raan_t = self.raan + raan_dot * t
        aop_t = self.aop + aop_dot * t
        # 평균운동으로 진근점이각 전파 (원형 근사)
        ta_t = self.ta + n * t

        return keplerian_to_cartesian(
            self.a, self.e, self.inc, raan_t, aop_t, ta_t, mu,
        )


# ── BLADE 궤도전이 문제 ──────────────────────────────────────────

@dataclasses.dataclass
class BLADEValidation:
    """BLADE 사후검증 리포트."""

    bc_violation_rk4: float      # 독립 RK4 재전파 BC violation
    bc_violation_r: float        # 위치 성분
    bc_violation_v: float        # 속도 성분
    max_thrust_norm: float       # 궤적 전체 peak ||u||
    thrust_violation: float      # max(0, peak - u_max), 제약 없으면 0
    energy_error: float          # |E_final - E_target| / |E_target|
    passed: bool                 # 전체 통과 여부
    details: dict                # 항목별 pass/fail


@dataclasses.dataclass
class BLADESCPResult:
    """BLADE 궤도전이 SCP 결과."""

    p_segments: list[NDArray]   # 각 세그먼트 최적 제어점 (n+1, 3)
    cost: float
    converged: bool
    n_iter: int
    bc_violation: float
    bc_history: list[float]
    cost_history: list[float]
    status: str
    ta_opt: float | None = None  # 최적 도착 ta [rad] (ta_free=True일 때)
    bc_violation_r: float | None = None     # 위치 BC violation
    bc_violation_v: float | None = None     # 속도 BC violation
    thrust_violation: float | None = None   # max(0, peak_thrust - u_max)
    x_final: NDArray | None = None          # 마지막 전파 종점 상태 [6]
    validation: BLADEValidation | None = None  # 사후검증 리포트


@dataclasses.dataclass
class BLADEOrbitProblem:
    """BLADE 궤도전이 문제 정의."""

    dep: OrbitBC                # 출발 궤도
    arr: OrbitBC                # 도착 궤도
    t_f: float                  # 비행시간 [TU] (정규화)
    K: int = 10                 # 세그먼트 수
    n: int = 2                  # 세그먼트 차수
    u_max: float | None = None  # 추력 제약 (정규화)
    canonical_units: CanonicalUnits | None = None
    max_iter: int = 30
    tol_bc: float = 1e-3
    relax_alpha: float = 0.4
    trust_region: float = 10.0
    l1_lambda: float = 0.0
    n_steps_per_seg: int = 20   # 세그먼트당 RK4 스텝
    coupling_order: int = 0     # 0: 기존(직접 적분), 1: STM 커플링
    coupling_from_start: bool = False  # True이면 iteration 0부터 커플링 활성화
    warm_start: list | None = None  # 초기 제어점 리스트 (K개, 각 (n+1, 3))
    velocity_only_bc: bool = False  # True이면 속도 BC만 사용 (위치 BC 제거)
    ta_free: bool = False  # True이면 도착 ta를 QP 결정변수로 편입 (phase search 불필요)
    algebraic_drift: bool = False  # True이면 Bernstein 대수로 드리프트 계산 (RK4 제거)
    gc_K: int = 8       # r^{-3/2} Chebyshev 근사 차수
    gc_R: int = 12      # gravity pipeline 차수 축소 목표
    validate: bool = False  # True이면 solve 후 자동 RK4 재전파 검증


# ── 참조 궤적 전파 ───────────────────────────────────────────────

def _propagate_blade_reference(
    p_segments: list[NDArray],
    n: int,
    K: int,
    deltas: NDArray,
    t_f: float,
    x0: NDArray,
    Re_star: float,
    n_steps_per_seg: int = 20,
) -> list[NDArray]:
    """세그먼트별 RK4 전파.

    Returns
    -------
    seg_trajs : list of (n_steps_per_seg+1, 6) — 각 세그먼트의 궤적.
    """
    seg_trajs = []
    x_current = x0.copy()
    tau_start = 0.0

    for k in range(K):
        tau_end = tau_start + deltas[k]
        p_k = p_segments[k]  # (n+1, 3)

        def u_func(tau, _pk=p_k, _ts=tau_start, _dk=deltas[k]):
            # 국소 파라미터 τ_k ∈ [0,1]
            tau_local = (tau - _ts) / _dk
            tau_local = np.clip(tau_local, 0.0, 1.0)
            B = bernstein(n, tau_local)  # (n+1,)
            return B @ _pk

        _, x_arr = propagate_rk4(
            x_current,
            (tau_start, tau_end),
            n_steps_per_seg,
            u_func=u_func,
            Re_star=Re_star,
            include_j2=True,
            t_f=t_f,
        )
        seg_trajs.append(x_arr)
        x_current = x_arr[-1].copy()
        tau_start = tau_end

    return seg_trajs


# ── STM 기반 세그먼트별 전파 ──────────────────────────────────────

def _propagate_segment_with_stm(
    x0: NDArray,
    p_k: NDArray,
    n: int,
    tau_start: float,
    delta_k: float,
    t_f: float,
    Re_star: float,
    n_steps: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """세그먼트 1개를 STM과 함께 RK4 적분.

    상태(6) + STM(6×6=36) = 42차원 증강 ODE:
        dx/dτ = t_f · f(x, u)
        dΦ/dτ = t_f · A(x) · Φ

    Returns
    -------
    x_arr : (n_steps+1, 6)   상태 궤적
    phi_arr : (n_steps+1, 6, 6)  각 시점의 STM Φ(τ, τ_start)
    phi_seg : (6, 6)   세그먼트 전체 STM Φ(τ_end, τ_start)
    """
    tau_end = tau_start + delta_k
    tau_arr = np.linspace(tau_start, tau_end, n_steps + 1)
    dtau = tau_arr[1] - tau_arr[0]

    x_arr = np.zeros((n_steps + 1, 6))
    phi_arr = np.zeros((n_steps + 1, 6, 6))

    x_arr[0] = x0
    phi_arr[0] = np.eye(6)

    # 증강 상태: y = [x(6), Φ_flat(36)] = 42차원
    y0 = np.zeros(42)
    y0[:6] = x0
    y0[6:] = np.eye(6).ravel()

    def u_func(tau):
        tau_local = (tau - tau_start) / delta_k
        tau_local = np.clip(tau_local, 0.0, 1.0)
        B = bernstein(n, tau_local)
        return B @ p_k

    def rhs(tau, y):
        x = y[:6]
        Phi = y[6:].reshape(6, 6)

        # 상태 동역학
        u = u_func(tau)
        xdot = t_f * eom_twobody_j2(x, u, Re_star=Re_star)

        # STM 동역학: dΦ/dτ = t_f · A(x) · Φ
        A = jacobian_twobody_j2(x, Re_star=Re_star)
        Phi_dot = t_f * (A @ Phi)

        dy = np.zeros(42)
        dy[:6] = xdot
        dy[6:] = Phi_dot.ravel()
        return dy

    y = y0.copy()
    for k in range(n_steps):
        tau_k = tau_arr[k]

        k1 = rhs(tau_k, y)
        k2 = rhs(tau_k + 0.5 * dtau, y + 0.5 * dtau * k1)
        k3 = rhs(tau_k + 0.5 * dtau, y + 0.5 * dtau * k2)
        k4 = rhs(tau_k + dtau, y + dtau * k3)

        y = y + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        x_arr[k + 1] = y[:6]
        phi_arr[k + 1] = y[6:].reshape(6, 6)

    return x_arr, phi_arr, phi_arr[-1]


def _propagate_all_segments_with_stm(
    p_segments: list[NDArray],
    n: int,
    K: int,
    deltas: NDArray,
    t_f: float,
    x0: NDArray,
    Re_star: float,
    n_steps_per_seg: int,
) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
    """모든 세그먼트를 STM과 함께 전파.

    Returns
    -------
    seg_trajs : list of (n_steps+1, 6) — 세그먼트별 궤적
    seg_phi_arrs : list of (n_steps+1, 6, 6) — 세그먼트별 STM 이력
    seg_stms : list of (6, 6) — 세그먼트별 전체 STM
    """
    seg_trajs = []
    seg_phi_arrs = []
    seg_stms = []
    x_current = x0.copy()
    tau_start = 0.0

    for k in range(K):
        x_arr, phi_arr, phi_seg = _propagate_segment_with_stm(
            x_current, p_segments[k], n,
            tau_start, deltas[k], t_f, Re_star, n_steps_per_seg,
        )
        seg_trajs.append(x_arr)
        seg_phi_arrs.append(phi_arr)
        seg_stms.append(phi_seg)
        x_current = x_arr[-1].copy()
        tau_start += deltas[k]

    return seg_trajs, seg_phi_arrs, seg_stms


# ── 커플링 행렬 계산 ─────────────────────────────────────────────

def _compute_coupling_matrices(
    seg_phi_arrs: list[NDArray],
    seg_stms: list[NDArray],
    n: int,
    K: int,
    deltas: NDArray,
    t_f: float,
) -> list[NDArray]:
    """STM 기반 커플링 행렬 계산.

    각 세그먼트 k에 대해, 제어점 p_k가 최종 상태에 미치는 감도:
        δx_final = M_total_k · vec(p_k)

    여기서 vec(p_k) = [p_k[:,0]; p_k[:,1]; p_k[:,2]], 길이 3(n+1).

    내부 계산:
    1. M_local_k: 세그먼트 내 Bernstein 가중 STM 적분
    2. Φ_forward: 세그먼트 k 이후 전체 STM 곱
    3. M_total_k = Φ_forward · M_local_k

    Returns
    -------
    M_total_list : list of (6, 3*(n+1)) — 세그먼트별 커플링 행렬
    """
    # 1. 역방향 STM 체인 (segment k → final)
    #    Φ_forward[k] = Φ_K · ... · Φ_{k+1}   (k 이후 ~ 끝까지)
    #    Φ_forward[K-1] = I  (마지막 세그먼트)
    phi_forward = [None] * K
    phi_forward[K - 1] = np.eye(6)
    for k in range(K - 2, -1, -1):
        # Φ(final, end_of_k) = Φ(final, end_of_{k+1}) · Φ_{k+1}
        phi_forward[k] = phi_forward[k + 1] @ seg_stms[k + 1]

    M_total_list = []

    for k in range(K):
        phi_arr_k = seg_phi_arrs[k]  # (n_steps+1, 6, 6)
        n_steps = phi_arr_k.shape[0] - 1
        h_k = t_f * deltas[k]  # 세그먼트 물리 시간 (정규화)

        # Φ_k(end) = phi_arr_k[-1] = seg_stms[k]
        phi_end = seg_stms[k]

        # M_local: (6, 3*(n+1))
        # M_local[:, ax*(n+1) + i] = h_k · Σ_j w_j · Φ_k(end, τ_j) · [0; e_ax] · b_i(τ_j)
        #
        # Φ_k(end, τ_j) = Φ_k(end) · Φ_k(τ_j)^{-1}
        # 추력은 속도 성분에만 작용하므로: Φ_k(end, τ_j) · [0₃; I₃] = Φ_k(end, τ_j)[:, 3:6]

        M_local = np.zeros((6, 3 * (n + 1)))

        # Simpson 적분 가중치
        weights = np.zeros(n_steps + 1)
        dtau_local = 1.0 / n_steps
        for j in range(n_steps + 1):
            if j == 0 or j == n_steps:
                weights[j] = dtau_local / 3.0
            elif j % 2 == 1:
                weights[j] = 4.0 * dtau_local / 3.0
            else:
                weights[j] = 2.0 * dtau_local / 3.0

        for j in range(n_steps + 1):
            tau_local = j / n_steps
            B_j = bernstein(n, tau_local)  # (n+1,)

            # Φ_k(end, τ_j) = Φ_k(end) · Φ_k(τ_j)^{-1}
            phi_j = phi_arr_k[j]  # Φ_k(τ_j, τ_start)
            try:
                phi_end_from_j = phi_end @ np.linalg.inv(phi_j)
            except np.linalg.LinAlgError:
                phi_end_from_j = phi_end  # fallback

            # 추력은 속도에 작용: [0₃; I₃]
            # Ψ_j = phi_end_from_j[:, 3:6], shape (6, 3)
            Psi_j = phi_end_from_j[:, 3:6]

            w_j = weights[j]
            for ax in range(3):
                for i in range(n + 1):
                    col = ax * (n + 1) + i
                    # M_local[:, col] += h_k · w_j · Ψ_j[:, ax] · B_j[i]
                    M_local[:, col] += h_k * w_j * Psi_j[:, ax] * B_j[i]

        # M_total = Φ_forward · M_local
        M_total = phi_forward[k] @ M_local
        M_total_list.append(M_total)

    return M_total_list


def _build_blade_orbit_bc_coupled(
    K: int,
    n: int,
    deltas: NDArray,
    x_ref_final: NDArray,
    rf: NDArray,
    vf: NDArray,
    M_total_list: list[NDArray],
    p_segments: list[NDArray],
) -> tuple[NDArray, NDArray]:
    """STM 커플링 행렬 기반 경계조건 구축.

    선형화된 BC:
        Σ_k M_total_k · p_k^new = (target - x_ref_final) + Σ_k M_total_k · p_k^ref

    최적화 변수 z는 축별 배치:
        z = [q_x_all; q_y_all; q_z_all], q_k_ax = Δ_k · p_k[:, ax]

    Returns
    -------
    A_eq : (6, 3*K*(n+1))
    b_eq : (6,)
    """
    seg_dim = K * (n + 1)
    total_dim = 3 * seg_dim

    target = np.concatenate([rf, vf])  # [r_f; v_f]

    # 잔차: target - x_ref_final (현재 참조 궤적 종점과의 차이)
    residual = target - x_ref_final

    # b_eq = residual + Σ M_total_k · vec(p_ref_k)
    b_eq = residual.copy()
    for k in range(K):
        vec_pk = np.concatenate([p_segments[k][:, ax] for ax in range(3)])
        b_eq += M_total_list[k] @ vec_pk

    # A_eq: M_total_k의 열을 z 변수 배치에 맞게 재배치
    # z[ax*seg_dim + k*(n+1) : ax*seg_dim + (k+1)*(n+1)] = q_k_ax = Δ_k · p_k_ax
    # p_k_ax = q_k_ax / Δ_k
    # M_total_k[:, ax*(n+1)+i] · p_k[i, ax] = M_total_k[:, ax*(n+1)+i] / Δ_k · q_k[i, ax]
    A_eq = np.zeros((6, total_dim))
    for k in range(K):
        M_k = M_total_list[k]  # (6, 3*(n+1))
        for ax in range(3):
            for i in range(n + 1):
                col_M = ax * (n + 1) + i       # M_total 내 열 인덱스
                col_z = ax * seg_dim + k * (n + 1) + i  # z 변수 인덱스
                A_eq[:, col_z] = M_k[:, col_M] / deltas[k]

    return A_eq, b_eq


# ── 세그먼트별 드리프트 적분 ─────────────────────────────────────

def _compute_blade_drifts(
    seg_trajs: list[NDArray],
    deltas: NDArray,
    Re_star: float,
) -> tuple[list[NDArray], list[NDArray]]:
    """세그먼트별 드리프트 적분 (c_v_k, c_r_k).

    각 세그먼트 내에서 중력 가속도를 적분하여
    드리프트 기여(속도/위치)를 계산한다.

    Returns
    -------
    c_v_list : list of (3,) — 세그먼트별 속도 드리프트.
    c_r_list : list of (3,) — 세그먼트별 위치 드리프트.
    """
    c_v_list = []
    c_r_list = []

    for k, x_arr in enumerate(seg_trajs):
        n_pts = x_arr.shape[0]
        tau_local = np.linspace(0.0, 1.0, n_pts)  # 국소 τ ∈ [0,1]
        dt = 1.0 / (n_pts - 1) if n_pts > 1 else 1.0

        # 중력 가속도 (2체 + J2)
        r_all = x_arr[:, :3]
        r_mag = np.linalg.norm(r_all, axis=1, keepdims=True)
        f_grav = -r_all / r_mag**3

        # J2
        z_comp = r_all[:, 2:3]
        r_mag_sq = r_mag**2
        coeff = 1.5 * J2_EARTH * Re_star**2 / r_mag**5
        fac = 5.0 * z_comp**2 / r_mag_sq
        j2_x = coeff * r_all[:, 0:1] * (fac - 1.0)
        j2_y = coeff * r_all[:, 1:2] * (fac - 1.0)
        j2_z = coeff * r_all[:, 2:3] * (fac - 3.0)
        f_grav = f_grav + np.hstack([j2_x, j2_y, j2_z])

        # 드리프트 적분 (구간 [0,1] 위)
        c_v_k = np.trapezoid(f_grav, tau_local, axis=0)  # (3,)

        # 위치 드리프트: ∫₀¹ ∫₀^s f(σ) dσ ds
        cumint = np.cumsum(
            0.5 * (f_grav[:-1] + f_grav[1:]) * dt, axis=0,
        )
        cumint = np.vstack([np.zeros((1, 3)), cumint])
        c_r_k = np.trapezoid(cumint, tau_local, axis=0)  # (3,)

        c_v_list.append(c_v_k)
        c_r_list.append(c_r_k)

    return c_v_list, c_r_list


# ── 대수적 드리프트 (Bernstein 대수 파이프라인) ────────────────────

def _compute_blade_drifts_algebraic(
    p_segments: list[NDArray],
    n: int,
    K: int,
    deltas: NDArray,
    t_f: float,
    r0: NDArray,
    v0: NDArray,
    Re_star: float,
    gc_K: int = 8,
    gc_R: int = 12,
    n_grav_iter: int = 2,
) -> tuple[list[NDArray], list[NDArray], NDArray, list[NDArray], list[NDArray]]:
    """Bernstein 대수 파이프라인으로 세그먼트별 드리프트 적분 계산.

    RK4 전파 없이 Bernstein 대수 연산만으로 중력 드리프트를 계산한다.
    자기일관 보정(self-consistent gravity correction)으로 위치 정확도를 확보.

    Parameters
    ----------
    p_segments : list of (n+1, 3)
        세그먼트별 추력 제어점.
    n, K : int
        세그먼트 차수, 수.
    deltas : (K,)
        세그먼트 길이 (정규화, 합=1).
    t_f : float
        비행시간 (정규화).
    r0, v0 : (3,)
        초기 위치/속도 (정규화).
    Re_star : float
        정규화 지구 반경.
    gc_K : int
        r^{-3/2} Chebyshev 근사 차수.
    gc_R : int
        gravity pipeline 차수 축소 목표.
    n_grav_iter : int
        자기일관 보정 반복 횟수.

    Returns
    -------
    c_v_list : list of (3,)
        세그먼트별 속도 드리프트.
    c_r_list : list of (3,)
        세그먼트별 위치 드리프트.
    x_ref_final : (6,)
        최종 세그먼트 끝점 참조 상태 [r; v] (정규화).
    seg_boundaries : list of (3,)
        세그먼트 경계 위치 (K+1개). seg_boundaries[k] = r_k.
    seg_pos_cps : list of (D_pos+1, 3)
        세그먼트별 위치 Bernstein 제어점.
    """
    Ibar_n = double_int_matrix(n)  # (n+3, n+1)
    I_n = int_matrix(n)            # (n+2, n+1)
    e_I = I_n[-1, :]               # 속도 적분 끝점 가중치
    P0 = n + 2                     # 추력-only 위치 차수
    D_pos = max(P0, 8)             # 작업 위치 차수

    # J2 적분용 상수 (루프 밖에서 1회만 계산)
    _j2_tau_pts = np.linspace(0, 1, 30)
    _j2_dt = _j2_tau_pts[1] - _j2_tau_pts[0]

    c_v_list = []
    c_r_list = []
    seg_boundaries = []
    seg_pos_cps = []  # 세그먼트별 위치 제어점 (D_pos 차수)
    r_k = r0.copy()
    v_k = v0.copy()
    seg_boundaries.append(r_k.copy())

    for k in range(K):
        dk = deltas[k]
        pk = p_segments[k]  # (n+1, 3)

        # ── 1) 추력-only 위치 제어점 (차수 P0 = n+2) ──
        ell = np.arange(P0 + 1) / P0
        q_r0 = (r_k[np.newaxis, :]
                + t_f * dk * v_k[np.newaxis, :] * ell[:, np.newaxis]
                + t_f**2 * dk**2 * (Ibar_n @ pk))  # (P0+1, 3) 배치

        # D_pos로 승급 — 다축 배치
        q_r = degree_elevate_multi(q_r0, D_pos)  # (D_pos+1, 3)

        # ── 2) 자기일관 중력 보정 ──
        for _ in range(n_grav_iter):
            q_ag = gravity_composition_pipeline(q_r, K=gc_K, R=gc_R)
            Rd = q_ag.shape[0] - 1
            q_F = int_matrix(Rd) @ q_ag
            q_G = int_matrix(Rd + 1) @ q_F  # 이중적분, 차수 Rd+2

            tgt_deg = q_G.shape[0] - 1
            q_r0_elev = degree_elevate_multi(q_r0, tgt_deg)  # (tgt+1, 3) 배치
            q_r_full = q_r0_elev + t_f**2 * dk**2 * q_G
            q_r = degree_reduce_multi(q_r_full, D_pos)  # (D_pos+1, 3) 배치

        # ── 3) 최종 드리프트 적분 (2체 중력) ──
        q_ag_f = gravity_composition_pipeline(q_r, K=gc_K, R=gc_R)
        Rd_f = q_ag_f.shape[0] - 1
        c_v_k = definite_integral(q_ag_f)
        c_r_k = definite_integral(int_matrix(Rd_f) @ q_ag_f)

        # ── 4) J2 보정 (수치 적분, 30점) ──
        r_eval = bernstein_eval(D_pos, q_r, _j2_tau_pts)
        r_mag = np.linalg.norm(r_eval, axis=1)
        z_comp = r_eval[:, 2]
        coeff = 1.5 * J2_EARTH * Re_star**2 / r_mag**5
        fac5 = 5.0 * z_comp**2 / r_mag**2
        fac_xy = coeff * (fac5 - 1.0)
        j2_a = np.empty((30, 3))
        j2_a[:, 0] = fac_xy * r_eval[:, 0]
        j2_a[:, 1] = fac_xy * r_eval[:, 1]
        j2_a[:, 2] = coeff * z_comp * (fac5 - 3.0)
        c_v_k = c_v_k + np.trapezoid(j2_a, _j2_tau_pts, axis=0)
        cumint = np.empty((30, 3))
        cumint[0] = 0.0
        cumint[1:] = np.cumsum(0.5 * (j2_a[:-1] + j2_a[1:]) * _j2_dt, axis=0)
        c_r_k = c_r_k + np.trapezoid(cumint, _j2_tau_pts, axis=0)

        c_v_list.append(c_v_k)
        c_r_list.append(c_r_k)
        seg_pos_cps.append(q_r.copy())

        # ── 5) 다음 세그먼트 초기 조건 갱신 ──
        r_k = q_r[-1, :].copy()  # Bernstein 끝점 = 정확한 끝점
        dv_thrust = t_f * dk * (e_I @ pk)  # (3,) 배치
        v_k = v_k + dv_thrust + t_f * dk * c_v_k
        seg_boundaries.append(r_k.copy())

    x_ref_final = np.concatenate([r_k, v_k])
    return c_v_list, c_r_list, x_ref_final, seg_boundaries, seg_pos_cps


def _gravity_gradient(r: NDArray, Re_star: float) -> NDArray:
    """중력 기울기 텐서 ∂a_grav/∂r (2체 + J2, 정규화 단위).

    Parameters
    ----------
    r : (3,)
        정규화 위치.
    Re_star : float
        정규화 지구 반경 (R_E / DU).

    Returns
    -------
    G : (3, 3)
        ∂a_grav/∂r 텐서.
    """
    rmag = np.linalg.norm(r)
    rmag2 = rmag * rmag
    rmag3 = rmag * rmag2
    rmag5 = rmag2 * rmag3
    rhat = r / rmag

    # 2체 기울기: -1/r³ (I - 3 r̂r̂^T)
    G = (-1.0 / rmag3) * (np.eye(3) - 3.0 * np.outer(rhat, rhat))

    # J2 기울기 (해석적)
    x, y, z = r
    Re2 = Re_star * Re_star
    rmag7 = rmag5 * rmag2
    J2c = 1.5 * J2_EARTH * Re2

    # ∂f_J2/∂r 성분별 계산 (Montenbruck & Gill, Table 3.2)
    z2 = z * z
    fac = 5.0 * z2 / rmag2
    fac7 = 7.0 * z2 / rmag2

    # 대각 성분
    G[0, 0] += J2c * (1.0 / rmag5 * (fac - 1.0)
                      - 5.0 * x * x / rmag7 * (fac7 - 3.0))
    G[1, 1] += J2c * (1.0 / rmag5 * (fac - 1.0)
                      - 5.0 * y * y / rmag7 * (fac7 - 3.0))
    G[2, 2] += J2c * (1.0 / rmag5 * (fac - 3.0)
                      - 5.0 * z2 / rmag7 * (fac7 - 5.0))

    # 비대각 성분 (대칭)
    G[0, 1] += J2c * (-5.0 * x * y / rmag7 * (fac7 - 3.0))
    G[1, 0] = G[0, 1]
    G[0, 2] += J2c * (-5.0 * x * z / rmag7 * (fac7 - 5.0)
                      + 2.0 * x * z / rmag7)
    G[2, 0] = G[0, 2]
    G[1, 2] += J2c * (-5.0 * y * z / rmag7 * (fac7 - 5.0)
                      + 2.0 * y * z / rmag7)
    G[2, 1] = G[1, 2]

    return G


def _compute_algebraic_coupling(
    p_segments: list[NDArray],
    n: int,
    K: int,
    deltas: NDArray,
    t_f: float,
    seg_boundaries: list[NDArray],
    x_ref_final: NDArray,
    Re_star: float,
    seg_pos_cps: list[NDArray] | None = None,
    n_sub: int = 4,
) -> list[NDArray]:
    """대수적 커플링 행렬 계산 (RK4-free STM 근사).

    세그먼트 내부를 n_sub 서브스텝으로 분할하여 중력 기울기 텐서를 계산하고,
    서브스텝 STM을 체인하여 정밀한 세그먼트 간 커플링을 구현한다.

    Parameters
    ----------
    seg_boundaries : list of (3,)
        K+1개 세그먼트 경계 위치 (정규화).
    x_ref_final : (6,)
        최종 참조 상태 [r; v].
    seg_pos_cps : list of (D+1, 3), optional
        세그먼트별 위치 Bernstein 제어점. 없으면 경계 선형보간 사용.
    n_sub : int
        세그먼트 내 서브스텝 수 (기본 4).

    Returns
    -------
    M_total_list : list of (6, 3*(n+1))
        세그먼트별 커플링 행렬.
    """
    I_n = int_matrix(n)
    Ibar_n = double_int_matrix(n)
    e_I = I_n[-1, :]        # (n+1,)
    e_Ibar = Ibar_n[-1, :]  # (n+1,)

    # 1. 세그먼트별 STM (서브스텝 체인)
    seg_stms = []
    # 세그먼트 내부 STM (from τ_j to end) — M_local 계산용
    seg_internal_stms = []  # list of list of (6,6)

    for k in range(K):
        h_k = t_f * deltas[k]
        h_sub = h_k / n_sub

        # 서브스텝 위치 평가
        tau_sub = np.linspace(0, 1, n_sub + 1)
        if seg_pos_cps is not None:
            D_pos = seg_pos_cps[k].shape[0] - 1
            r_sub = bernstein_eval(D_pos, seg_pos_cps[k], tau_sub)
        else:
            r_sub = np.array([
                (1 - t) * seg_boundaries[k] + t * seg_boundaries[k + 1]
                for t in tau_sub
            ])

        # 서브스텝 STM 계산 + 체인
        sub_stms = []
        for s in range(n_sub):
            r_mid_s = 0.5 * (r_sub[s] + r_sub[s + 1])
            G_s = _gravity_gradient(r_mid_s, Re_star)
            h2G = 0.5 * h_sub * h_sub * G_s
            Phi_s = np.eye(6)
            Phi_s[:3, :3] += h2G
            Phi_s[:3, 3:6] = h_sub * np.eye(3)
            Phi_s[3:6, :3] = h_sub * G_s
            Phi_s[3:6, 3:6] += h2G
            sub_stms.append(Phi_s)

        # 전체 세그먼트 STM = Φ_{n_sub-1} · ... · Φ_1 · Φ_0
        Phi_seg = np.eye(6)
        for s in range(n_sub):
            Phi_seg = sub_stms[s] @ Phi_seg  # 주의: 순서
        # 수정: Phi_seg = sub_stms[-1] @ ... @ sub_stms[0]
        Phi_seg = np.eye(6)
        for s in range(n_sub):
            Phi_seg = sub_stms[s] @ Phi_seg
        seg_stms.append(Phi_seg)

        # 내부 STM: Φ(end, τ_j) for each sub-step boundary
        # Φ(end, τ_j) = Φ_{n-1} · ... · Φ_j
        internal = [None] * (n_sub + 1)
        internal[n_sub] = np.eye(6)
        for s in range(n_sub - 1, -1, -1):
            internal[s] = internal[s + 1] @ sub_stms[s]
        seg_internal_stms.append(internal)

    # 2. 역방향 STM 체인 (segment k → final)
    phi_forward = [None] * K
    phi_forward[K - 1] = np.eye(6)
    for k in range(K - 2, -1, -1):
        phi_forward[k] = phi_forward[k + 1] @ seg_stms[k + 1]

    # 3. 세그먼트별 커플링 행렬 (Simpson 적분)
    M_total_list = []
    for k in range(K):
        dk = deltas[k]
        h_k = t_f * dk

        # M_local = h_k · ∫₀¹ Φ(end, τ)[:, 3:6] · B_n(τ) dτ
        # Simpson 적분 (n_sub 서브스텝 경계에서 평가)
        M_local = np.zeros((6, 3 * (n + 1)))
        tau_eval = np.linspace(0, 1, n_sub + 1)

        for j, tau_j in enumerate(tau_eval):
            Psi_j = seg_internal_stms[k][j][:, 3:6]  # (6, 3)
            B_j = bernstein(n, tau_j)  # (n+1,)

            # Simpson 가중치
            if j == 0 or j == n_sub:
                w_j = 1.0 / (3.0 * n_sub)
            elif j % 2 == 1:
                w_j = 4.0 / (3.0 * n_sub)
            else:
                w_j = 2.0 / (3.0 * n_sub)

            for ax in range(3):
                for i in range(n + 1):
                    col = ax * (n + 1) + i
                    M_local[:, col] += h_k * w_j * Psi_j[:, ax] * B_j[i]

        # Φ_forward @ M_local → 최종 상태에 대한 감도
        M_total = phi_forward[k] @ M_local
        M_total_list.append(M_total)

    return M_total_list


# ── 도착 상태 ta Jacobian ─────────────────────────────────────────

def _target_jacobian_ta(
    arr: OrbitBC,
    t_f_phys: float,
    cu: CanonicalUnits,
    delta: float = 1e-7,
) -> tuple[NDArray, NDArray]:
    """도착 상태의 ta에 대한 Jacobian (중심차분, 정규화 단위).

    Parameters
    ----------
    arr : OrbitBC
        도착 궤도 (현재 ta_ref 반영).
    t_f_phys : float
        물리 비행시간 [s].
    cu : CanonicalUnits
        정규화 단위계.
    delta : float
        유한차분 스텝 [rad].

    Returns
    -------
    drf_dta : (3,) — ∂rf/∂ta [정규화 위치/rad].
    dvf_dta : (3,) — ∂vf/∂ta [정규화 속도/rad].
    """
    arr_p = dataclasses.replace(arr, ta=arr.ta + delta)
    arr_m = dataclasses.replace(arr, ta=arr.ta - delta)

    rf_p, vf_p = arr_p.at_time(t_f_phys)
    rf_m, vf_m = arr_m.at_time(t_f_phys)

    drf_dta = (cu.nondim_pos(rf_p) - cu.nondim_pos(rf_m)) / (2.0 * delta)
    dvf_dta = (cu.nondim_vel(vf_p) - cu.nondim_vel(vf_m)) / (2.0 * delta)

    return drf_dta, dvf_dta


# ── 3D 경계조건 구성 ─────────────────────────────────────────────

def _build_blade_orbit_bc(
    K: int, n: int, deltas: NDArray, t_f: float,
    r0: NDArray, v0: NDArray, rf: NDArray, vf: NDArray,
    c_v_list: list[NDArray], c_r_list: list[NDArray],
) -> tuple[NDArray, NDArray]:
    """3D BLADE 경계조건 (q_k = Δ_k p_k 변수 기준).

    기존 SCP (report 006/007)의 경계조건과 동일한 구조를 따르되,
    단일 Z 대신 세그먼트별 q_k를 사용.

    정규화 시간 τ ∈ [0,1] 에서 물리 시간 t = t_f τ.
    속도: v(1) = v0 + Σ_k [t_f Δ_k e_I^T p_k + t_f Δ_k c_v_k]
         = v0 + Σ_k [t_f e_I^T q_k + t_f Δ_k c_v_k]
    위치: r(1) = r0 + t_f v0 + Σ_k [v_thrust(t_{k-1}) Δ_k t_f
         + t_f^2 Δ_k^2 e_Ī^T p_k + t_f^2 Δ_k^2 c_r_k]

    여기서 v_thrust(t_{k-1}) = Σ_{j<k} t_f Δ_j e_I^T p_j (추력에 의한 속도 기여만)

    Returns
    -------
    A_eq : (6, 3*K*(n+1))
    b_eq : (6,)
    """
    seg_dim = K * (n + 1)
    total_dim = 3 * seg_dim

    I_n = int_matrix(n)
    Ibar_n = double_int_matrix(n)
    e_I = I_n[-1, :]        # (n+1,)
    e_Ibar = Ibar_n[-1, :]  # (n+1,)

    A_eq = np.zeros((6, total_dim))
    b_eq = np.zeros(6)

    # 총 드리프트 (모든 세그먼트 합산)
    c_v_total = sum(deltas[k] * c_v_list[k] for k in range(K))

    # 총 드리프트 위치 기여 (순차 누적)
    c_r_total = np.zeros(3)
    v_grav_cum = np.zeros(3)  # 중력에 의한 누적 속도
    for k in range(K):
        c_r_total += v_grav_cum * deltas[k]  # 이전 중력 속도의 위치 기여
        c_r_total += deltas[k]**2 * c_r_list[k]  # 세그먼트 내 중력 위치 기여
        v_grav_cum += deltas[k] * c_v_list[k]

    # basis.py의 검증된 boundary_matrices를 재사용 (p_k 기준)
    A_v_p, A_r_p = boundary_matrices(K, n, deltas, t_f)
    # A_v_p, A_r_p는 p_k에 대한 행렬. q_k = Δ_k p_k이므로
    # A @ p = b → A @ (q/Δ) = b → (A/Δ) @ q = b
    # 세그먼트 j의 블록: A[j_block] / Δ_j

    for ax in range(3):
        offset = ax * seg_dim

        # 속도 BC: A_v_p @ p = b_v → (A_v_p/Δ) @ q = b_v
        for j in range(K):
            s_src = j * (n + 1)
            s_dst = offset + j * (n + 1)
            A_eq[ax, s_dst:s_dst + n + 1] = A_v_p[0, s_src:s_src + n + 1] / deltas[j]

        b_eq[ax] = (vf[ax] - v0[ax]) - t_f * c_v_total[ax]

        # 위치 BC: A_r_p @ p = b_r → (A_r_p/Δ) @ q = b_r
        for j in range(K):
            s_src = j * (n + 1)
            s_dst = offset + j * (n + 1)
            A_eq[3 + ax, s_dst:s_dst + n + 1] = A_r_p[0, s_src:s_src + n + 1] / deltas[j]

        b_eq[3 + ax] = (rf[ax] - r0[ax]) - t_f * v0[ax] - t_f**2 * c_r_total[ax]

    return A_eq, b_eq


# ── 캐시된 SOCP (CVXPY Parameter 기반) ────────────────────────────

class _CachedSOCP:
    """CVXPY Problem을 Parameter로 캐시하여 반복 재구성 오버헤드를 제거.

    문제 구조(변수, 제약 원뿔)는 1회 구성하고,
    매 반복 변하는 데이터(A_eq, b_eq, J_ta, z_ref, bc_penalty, t_f, u_max)를
    Parameter.value로 업데이트한다.

    모듈 레벨 ``_get_cached_socp()``로 (K, n, ta_free, trust_region) 키 기반
    캐싱하여, 동일 구조의 반복 호출에서 첫 컴파일 비용(~77ms)을 제거한다.
    """

    def __init__(
        self,
        K: int,
        n: int,
        has_u_max: bool,
        ta_free: bool,
        use_slack: bool,
        trust_region: float,
    ):
        deltas = np.full(K, 1.0 / K)
        seg_dim = K * (n + 1)
        total_dim = 3 * seg_dim
        self.total_dim = total_dim

        # G_lifted는 (K, n)으로 결정 — 캐시 가능
        G_n = gram_matrix(n)
        G_lifted = np.zeros((seg_dim, seg_dim))
        for k in range(K):
            s = k * (n + 1)
            e = s + (n + 1)
            G_lifted[s:e, s:e] = G_n / deltas[k]

        # ── 변수 ──
        z = cp.Variable(total_dim)
        self.z = z

        dta = cp.Variable() if ta_free else None
        self.dta = dta

        bc_slack = cp.Variable(6) if use_slack else None
        self.bc_slack = bc_slack

        # ── Parameters (매 반복 업데이트) ──
        self.A_eq_p = cp.Parameter((6, total_dim))
        self.b_eq_p = cp.Parameter(6)
        self.J_ta_p = cp.Parameter(6) if ta_free else None
        self.z_ref_p = cp.Parameter(total_dim)
        self.tr_ub_p = cp.Parameter(total_dim)  # z_ref + δ
        self.tr_lb_p = cp.Parameter(total_dim)  # z_ref - δ
        self.bc_pen_p = cp.Parameter(nonneg=True) if use_slack else None
        # t_f, u_max도 Parameter로 — 다른 문제에서 재사용 가능
        self.inv_tf_p = cp.Parameter(nonneg=True)  # 1/t_f
        self.u_max_p = cp.Parameter(nonneg=True) if has_u_max else None

        # 초기값 (iteration 0 용)
        self._no_tr_bound = 20.0 * trust_region
        self.z_ref_p.value = np.zeros(total_dim)
        self.tr_ub_p.value = np.full(total_dim, self._no_tr_bound)
        self.tr_lb_p.value = np.full(total_dim, -self._no_tr_bound)
        self.A_eq_p.value = np.zeros((6, total_dim))
        self.b_eq_p.value = np.zeros(6)
        self.inv_tf_p.value = 1.0
        if self.J_ta_p is not None:
            self.J_ta_p.value = np.zeros(6)
        if self.bc_pen_p is not None:
            self.bc_pen_p.value = 1e4
        if self.u_max_p is not None:
            self.u_max_p.value = 1.0

        # ── 목적함수 ──
        # cost = (1/t_f) * Σ quad_form(zax, G_lifted)
        raw_cost = 0.0
        for ax in range(3):
            offset = ax * seg_dim
            zax = z[offset:offset + seg_dim]
            raw_cost = raw_cost + cp.quad_form(zax, G_lifted)
        cost_expr = self.inv_tf_p * raw_cost

        if use_slack:
            cost_expr = cost_expr + self.bc_pen_p * cp.norm1(bc_slack)

        # ── 제약 ──
        constraints = []

        # 1. BC 등식
        if ta_free and use_slack:
            constraints.append(
                self.A_eq_p @ z - self.J_ta_p * dta == self.b_eq_p + bc_slack
            )
        elif ta_free:
            constraints.append(
                self.A_eq_p @ z - self.J_ta_p * dta == self.b_eq_p
            )
        elif use_slack:
            constraints.append(self.A_eq_p @ z == self.b_eq_p + bc_slack)
        else:
            constraints.append(self.A_eq_p @ z == self.b_eq_p)

        # 2. Trust region (box constraint — DPP 호환)
        constraints.append(z <= self.tr_ub_p)
        constraints.append(z >= self.tr_lb_p)

        # 3. ta trust region
        if ta_free:
            constraints.append(cp.abs(dta) <= np.radians(30))

        # 4. SOCP 추력 제약 — u_max를 Parameter로
        if has_u_max:
            M_grid = max(2 * n, 4)
            tau_pts = np.linspace(0, 1, M_grid + 1)
            for k in range(K):
                for tau_j in tau_pts:
                    bj = bernstein(n, tau_j)
                    u_comps = []
                    for ax in range(3):
                        s = ax * seg_dim + k * (n + 1)
                        u_comps.append((bj @ z[s:s + n + 1]) / deltas[k])
                    constraints.append(
                        cp.norm(cp.hstack(u_comps)) <= self.u_max_p
                    )

        self.problem = cp.Problem(cp.Minimize(cost_expr), constraints)
        self.trust_region = trust_region

        self._compiled = False

    def solve(
        self,
        A_eq: NDArray,
        b_eq: NDArray,
        J_ta: NDArray | None,
        z_ref: NDArray | None,
        iteration: int,
        bc_penalty: float = 1e4,
        t_f: float = 1.0,
        u_max: float | None = None,
    ) -> tuple[str, float | None]:
        """Parameter 업데이트 후 솔브."""
        self.A_eq_p.value = A_eq
        self.b_eq_p.value = b_eq
        self.inv_tf_p.value = 1.0 / t_f

        if self.J_ta_p is not None and J_ta is not None:
            self.J_ta_p.value = J_ta

        if self.u_max_p is not None and u_max is not None:
            self.u_max_p.value = u_max

        if iteration == 0 or z_ref is None:
            self.tr_ub_p.value = np.full(self.total_dim, self._no_tr_bound)
            self.tr_lb_p.value = np.full(self.total_dim, -self._no_tr_bound)
        else:
            self.tr_ub_p.value = z_ref + self.trust_region
            self.tr_lb_p.value = z_ref - self.trust_region

        if self.bc_pen_p is not None:
            self.bc_pen_p.value = bc_penalty

        try:
            self.problem.solve(
                solver=cp.CLARABEL, verbose=False, warm_start=self._compiled,
            )
            self._compiled = True
        except Exception:
            try:
                self.problem.solve(
                    solver=cp.SCS, verbose=False, max_iters=10000,
                )
            except Exception:
                return "solver_error", None

        return self.problem.status, self.problem.value


# ── 모듈 레벨 SOCP 캐시 ──────────────────────────────────────

_socp_cache: dict[tuple, _CachedSOCP] = {}


def _get_cached_socp(
    K: int, n: int, u_max: float | None,
    ta_free: bool, trust_region: float,
) -> _CachedSOCP:
    """구조적 키로 _CachedSOCP 인스턴스를 캐시/반환.

    동일 (K, n, has_u_max, ta_free, trust_region)이면 CVXPY 컴파일을
    건너뛰어 ~77ms 절약.
    """
    key = (K, n, u_max is not None, ta_free, trust_region)
    if key not in _socp_cache:
        _socp_cache[key] = _CachedSOCP(
            K=K, n=n,
            has_u_max=(u_max is not None),
            ta_free=ta_free,
            use_slack=True,
            trust_region=trust_region,
        )
    return _socp_cache[key]


# ── 사후검증 ─────────────────────────────────────────────────────

def validate_blade_solution(
    prob: BLADEOrbitProblem,
    result: BLADESCPResult,
    *,
    n_steps_per_seg: int = 40,
    tol_bc: float | None = None,
    tol_thrust: float = 1e-3,
    tol_energy: float = 1e-2,
) -> BLADEValidation:
    """독립 RK4 재전파로 BLADE 솔루션을 검증한다.

    Parameters
    ----------
    prob : BLADEOrbitProblem
    result : BLADESCPResult
    n_steps_per_seg : 세그먼트당 RK4 스텝 (기본 40, 솔버의 2배)
    tol_bc : BC violation 허용치 (None이면 prob.tol_bc 사용)
    tol_thrust : 추력 제약 위반 허용치
    tol_energy : 궤도 에너지 상대 오차 허용치
    """
    if tol_bc is None:
        tol_bc = prob.tol_bc

    cu = prob.canonical_units or from_orbit(prob.dep.a)
    t_f = prob.t_f
    K = prob.K
    n = prob.n
    deltas = np.ones(K) / K

    # 초기/목표 상태
    r0_phys, v0_phys = prob.dep.at_time(0.0)
    x0 = np.concatenate([cu.nondim_pos(r0_phys), cu.nondim_vel(v0_phys)])

    t_f_phys = cu.dim_time(t_f)
    arr_bc = prob.arr if result.ta_opt is None else dataclasses.replace(prob.arr, ta=result.ta_opt)
    rf_phys, vf_phys = arr_bc.at_time(t_f_phys)
    rf = cu.nondim_pos(rf_phys)
    vf = cu.nondim_vel(vf_phys)

    Re_star = R_EARTH / cu.DU

    details: dict = {}

    # 1. 독립 RK4 재전파
    seg_trajs = _propagate_blade_reference(
        result.p_segments, n, K, deltas, t_f, x0, Re_star,
        n_steps_per_seg=n_steps_per_seg,
    )
    x_final_rk4 = seg_trajs[-1][-1]
    bc_viol_r = float(np.linalg.norm(x_final_rk4[:3] - rf))
    bc_viol_v = float(np.linalg.norm(x_final_rk4[3:6] - vf))
    bc_viol_rk4 = max(bc_viol_r, bc_viol_v)
    bc_passed = bc_viol_rk4 < tol_bc
    details["bc_rk4"] = bc_passed
    details["false_convergence"] = result.converged and not bc_passed

    # 2. 추력 크기 검사 (dense grid)
    max_thrust_norm = 0.0
    for k in range(K):
        p_k = result.p_segments[k]
        tau_local = np.linspace(0.0, 1.0, 10)
        for tau_l in tau_local:
            B = bernstein(n, tau_l)
            u_val = B @ p_k  # (3,)
            u_norm = float(np.linalg.norm(u_val))
            if u_norm > max_thrust_norm:
                max_thrust_norm = u_norm

    thrust_viol = max(0.0, max_thrust_norm - prob.u_max) if prob.u_max is not None else 0.0
    thrust_passed = thrust_viol < tol_thrust if prob.u_max is not None else True
    details["thrust"] = thrust_passed

    # 3. 궤도 에너지 오차
    r_final = x_final_rk4[:3]
    v_final = x_final_rk4[3:6]
    r_norm = float(np.linalg.norm(r_final))
    v_norm2 = float(np.dot(v_final, v_final))
    E_final = v_norm2 / 2.0 - 1.0 / r_norm  # 정규화 (mu*=1)
    a_arr_star = arr_bc.a / cu.DU
    E_target = -1.0 / (2.0 * a_arr_star)
    energy_error = abs(E_final - E_target) / max(abs(E_target), 1e-12)
    energy_passed = energy_error < tol_energy
    details["energy"] = energy_passed

    # 4. 비용 정상성
    cost_ok = np.isfinite(result.cost) and result.cost > 0
    details["cost_finite"] = cost_ok

    passed = bc_passed and thrust_passed and energy_passed and cost_ok

    return BLADEValidation(
        bc_violation_rk4=bc_viol_rk4,
        bc_violation_r=bc_viol_r,
        bc_violation_v=bc_viol_v,
        max_thrust_norm=max_thrust_norm,
        thrust_violation=thrust_viol,
        energy_error=energy_error,
        passed=passed,
        details=details,
    )


# ── BLADE SCP 솔버 ───────────────────────────────────────────────

def solve_blade_scp(prob: BLADEOrbitProblem) -> BLADESCPResult:
    """BLADE 궤도전이 SCP."""
    # 정규화
    cu = prob.canonical_units or from_orbit(prob.dep.a)
    t_f = prob.t_f
    K, n = prob.K, prob.n
    deltas = np.full(K, 1.0 / K)
    seg_dim = K * (n + 1)
    total_dim = 3 * seg_dim

    # 경계조건 (J2 세차 반영)
    t_f_phys = cu.dim_time(t_f)
    r0_phys, v0_phys = prob.dep.at_time(0.0)

    r0 = cu.nondim_pos(r0_phys)
    v0 = cu.nondim_vel(v0_phys)
    Re_star = cu.R_earth_star

    # ta_free: 도착 ta를 SCP 변수로 편입 (루프 내에서 갱신)
    ta_ref = prob.arr.ta
    if not prob.ta_free:
        rf_phys, vf_phys = prob.arr.at_time(t_f_phys)
        rf = cu.nondim_pos(rf_phys)
        vf = cu.nondim_vel(vf_phys)

    # 초기 제어점 (warm-start 또는 영벡터)
    if prob.warm_start is not None:
        p_segments = [pk.copy() for pk in prob.warm_start]
    else:
        p_segments = [np.zeros((n + 1, 3)) for _ in range(K)]

    x0 = np.concatenate([r0, v0])
    G_n = gram_matrix(n)

    # Lifted Gram: 최적화 변수 q_k = Δ_k·p_k 에 대한 올바른 Gram 행렬
    # cost = (1/t_f) Σ_k Δ_k p_k^T G_n p_k = (1/t_f) Σ_k q_k^T (G_n/Δ_k) q_k
    G_lifted = np.zeros((seg_dim, seg_dim))
    for k in range(K):
        s = k * (n + 1)
        e = s + (n + 1)
        G_lifted[s:e, s:e] = G_n / deltas[k]

    bc_history = []
    cost_history = []

    want_coupling = prob.coupling_order >= 1
    # 하이브리드: 초기에는 0차(강건), bc_viol이 충분히 작아지면 커플링(정밀)
    coupling_switch_tol = 10.0 * prob.tol_bc  # 목표의 10배 이내에서 전환
    coupling_active = prob.coupling_from_start and want_coupling

    # 캐시된 SOCP (algebraic_drift 전용 — 문제 구조 1회 구성)
    cached_socp = None
    if prob.algebraic_drift and prob.l1_lambda == 0:
        cached_socp = _get_cached_socp(
            K=K, n=n, u_max=prob.u_max,
            ta_free=prob.ta_free,
            trust_region=prob.trust_region,
        )

    for iteration in range(prob.max_iter):
        # 0. ta_free: 매 반복 시작 시 도착 상태 및 Jacobian 갱신
        if prob.ta_free:
            arr_iter = dataclasses.replace(prob.arr, ta=ta_ref)
            rf_phys, vf_phys = arr_iter.at_time(t_f_phys)
            rf = cu.nondim_pos(rf_phys)
            vf = cu.nondim_vel(vf_phys)
            drf_dta, dvf_dta = _target_jacobian_ta(arr_iter, t_f_phys, cu)

        # 1. 참조 궤적 전파 + 경계조건 구성
        if prob.algebraic_drift:
            # Bernstein 대수 파이프라인 (RK4 없음)
            c_v_list, c_r_list, x_ref_final, seg_boundaries, seg_pos_cps = (
                _compute_blade_drifts_algebraic(
                    p_segments, n, K, deltas, t_f, r0, v0, Re_star,
                    gc_K=prob.gc_K, gc_R=prob.gc_R,
                )
            )
            # 대수적 커플링 (서브스텝 STM)
            M_total_list = _compute_algebraic_coupling(
                p_segments, n, K, deltas, t_f,
                seg_boundaries, x_ref_final, Re_star,
                seg_pos_cps=seg_pos_cps,
            )
            A_eq, b_eq = _build_blade_orbit_bc_coupled(
                K, n, deltas, x_ref_final, rf, vf,
                M_total_list, p_segments,
            )
            coupling_active = True  # 커플링 활성 표시 (ta_free J_ta 순서용)
        else:
            n_steps = max(prob.n_steps_per_seg, 5 * n)

            # 커플링 전환 판정
            if want_coupling and not coupling_active and iteration > 3:
                if bc_history and bc_history[-1] < coupling_switch_tol:
                    coupling_active = True

            if coupling_active:
                seg_trajs, seg_phi_arrs, seg_stms = _propagate_all_segments_with_stm(
                    p_segments, n, K, deltas, t_f, x0, Re_star, n_steps,
                )
                M_total_list = _compute_coupling_matrices(
                    seg_phi_arrs, seg_stms, n, K, deltas, t_f,
                )
                x_ref_final = seg_trajs[-1][-1]
                A_eq, b_eq = _build_blade_orbit_bc_coupled(
                    K, n, deltas, x_ref_final, rf, vf,
                    M_total_list, p_segments,
                )
            else:
                seg_trajs = _propagate_blade_reference(
                    p_segments, n, K, deltas, t_f, x0, Re_star, n_steps,
                )
                c_v_list, c_r_list = _compute_blade_drifts(seg_trajs, deltas, Re_star)
                A_eq, b_eq = _build_blade_orbit_bc(
                    K, n, deltas, t_f, r0, v0, rf, vf, c_v_list, c_r_list,
                )

        # velocity_only_bc: 위치 BC 행 제거 (속도 BC만 유지)
        if prob.velocity_only_bc and not coupling_active:
            A_eq = A_eq[:3, :]
            b_eq = b_eq[:3]

        # 4. QP/SOCP 솔브
        if cached_socp is not None:
            # ── 캐시된 SOCP 경로 (algebraic_drift) ──
            if prob.ta_free:
                J_ta = np.concatenate([drf_dta, dvf_dta])  # coupled: [r; v]

            z_ref_vec = None
            if iteration > 0:
                z_ref_vec = np.concatenate([
                    np.concatenate([p_segments[k][:, ax] for k in range(K)])
                    for ax in range(3)
                ]) * np.tile(np.repeat(deltas, n + 1), 3)

            bc_penalty = (
                1e6 if bc_history and bc_history[-1] < 5.0 * prob.tol_bc
                else 1e4
            )

            status, cost_val = cached_socp.solve(
                A_eq=A_eq, b_eq=b_eq,
                J_ta=J_ta if prob.ta_free else None,
                z_ref=z_ref_vec,
                iteration=iteration,
                bc_penalty=bc_penalty,
                t_f=t_f,
                u_max=prob.u_max,
            )

            if status == "solver_error":
                return BLADESCPResult(
                    p_segments=p_segments, cost=float("inf"),
                    converged=False, n_iter=iteration,
                    bc_violation=float("inf"),
                    bc_history=bc_history, cost_history=cost_history,
                    status="solver_error",
                )
            if status not in ("optimal", "optimal_inaccurate"):
                return BLADESCPResult(
                    p_segments=p_segments, cost=float("inf"),
                    converged=False, n_iter=iteration,
                    bc_violation=float("inf"),
                    bc_history=bc_history, cost_history=cost_history,
                    status=status,
                )

            z_opt = cached_socp.z.value
            dta_ref = cached_socp.dta

        else:
            # ── 기존 CVXPY 경로 (매 반복 재구성) ──
            z = cp.Variable(total_dim)

            if prob.ta_free:
                dta = cp.Variable()

            cost_expr = 0.0
            for ax in range(3):
                offset = ax * seg_dim
                zax = z[offset:offset + seg_dim]
                cost_expr = cost_expr + cp.quad_form(zax, G_lifted)
            cost_expr = cost_expr / t_f

            if prob.l1_lambda > 0:
                l1_term = 0.0
                for k in range(K):
                    for ax in range(3):
                        s = ax * seg_dim + k * (n + 1)
                        l1_term = l1_term + cp.norm(z[s:s + n + 1], 1)
                cost_expr = cost_expr + prob.l1_lambda * l1_term

            use_bc_slack = prob.algebraic_drift
            if use_bc_slack:
                bc_slack = cp.Variable(b_eq.shape[0])
                bc_penalty = (
                    1e6 if bc_history and bc_history[-1] < 5.0 * prob.tol_bc
                    else 1e4
                )

            if prob.ta_free:
                if coupling_active:
                    J_ta = np.concatenate([drf_dta, dvf_dta])
                else:
                    J_ta = np.concatenate([dvf_dta, drf_dta])
                if prob.velocity_only_bc and not coupling_active:
                    J_ta = J_ta[:3]
                if use_bc_slack:
                    constraints = [A_eq @ z - J_ta * dta == b_eq + bc_slack]
                else:
                    constraints = [A_eq @ z - J_ta * dta == b_eq]
            else:
                if use_bc_slack:
                    constraints = [A_eq @ z == b_eq + bc_slack]
                else:
                    constraints = [A_eq @ z == b_eq]

            if iteration > 0:
                z_ref = np.concatenate([
                    np.concatenate([p_segments[k][:, ax] for k in range(K)])
                    for ax in range(3)
                ]) * np.tile(np.repeat(deltas, n + 1), 3)
                constraints.append(cp.norm_inf(z - z_ref) <= prob.trust_region)

            if prob.ta_free:
                constraints.append(cp.abs(dta) <= np.radians(30))

            if prob.u_max is not None:
                M_grid = max(2 * n, 4)
                tau_pts = np.linspace(0, 1, M_grid + 1)
                for k in range(K):
                    for tau_j in tau_pts:
                        bj = bernstein(n, tau_j)
                        u_comps = []
                        for ax in range(3):
                            s = ax * seg_dim + k * (n + 1)
                            u_comps.append((bj @ z[s:s + n + 1]) / deltas[k])
                        constraints.append(
                            cp.norm(cp.hstack(u_comps)) <= prob.u_max
                        )

            if use_bc_slack:
                cost_expr = cost_expr + bc_penalty * cp.norm1(bc_slack)

            problem = cp.Problem(cp.Minimize(cost_expr), constraints)
            try:
                problem.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                try:
                    problem.solve(solver=cp.SCS, verbose=False, max_iters=10000)
                except Exception:
                    return BLADESCPResult(
                        p_segments=p_segments, cost=float("inf"),
                        converged=False, n_iter=iteration,
                        bc_violation=float("inf"),
                        bc_history=bc_history, cost_history=cost_history,
                        status="solver_error",
                    )

            if problem.status not in ("optimal", "optimal_inaccurate"):
                return BLADESCPResult(
                    p_segments=p_segments, cost=float("inf"),
                    converged=False, n_iter=iteration,
                    bc_violation=float("inf"),
                    bc_history=bc_history, cost_history=cost_history,
                    status=problem.status,
                )

            z_opt = z.value
            dta_ref = dta if prob.ta_free else None
            cost_val = problem.value

        # 5. 제어점 추출 + step damping
        p_new = []
        for k in range(K):
            pk = np.zeros((n + 1, 3))
            for ax in range(3):
                s = ax * seg_dim + k * (n + 1)
                pk[:, ax] = z_opt[s:s + n + 1] / deltas[k]  # p_k = q_k / Δ_k
            p_new.append(pk)

        # 적응적 relaxation: 수렴 진행 시 alpha 증가
        if iteration == 0:
            alpha = 1.0
        elif prob.algebraic_drift and iteration > 5 and len(bc_history) >= 2:
            # BC가 감소 추세이면 alpha 점진 증가
            if bc_history[-1] < bc_history[-2]:
                alpha = min(1.0, prob.relax_alpha * 1.5)
            else:
                alpha = prob.relax_alpha * 0.8
        else:
            alpha = prob.relax_alpha
        p_segments = [
            (1.0 - alpha) * p_old + alpha * p_n
            for p_old, p_n in zip(p_segments, p_new)
        ]

        # 5b. 추력 제약 위반 검사 (제어점 norm → convex hull 상한)
        _thrust_viol = None
        if prob.u_max is not None:
            _max_thrust = max(
                np.linalg.norm(p_segments[k], axis=1).max()
                for k in range(K)
            )
            _thrust_viol = max(0.0, _max_thrust - prob.u_max)

        # ta_free: δta 적용하여 ta_ref 갱신
        _dta_var = dta_ref if cached_socp is not None else (dta if prob.ta_free else None)
        if prob.ta_free and _dta_var is not None and _dta_var.value is not None:
            ta_ref = ta_ref + alpha * float(_dta_var.value)
            # 갱신된 ta로 목표 상태 재계산 (수렴 판정용)
            arr_updated = dataclasses.replace(prob.arr, ta=ta_ref)
            rf_phys_new, vf_phys_new = arr_updated.at_time(t_f_phys)
            rf = cu.nondim_pos(rf_phys_new)
            vf = cu.nondim_vel(vf_phys_new)

        # 6. 수렴 판정
        # BC violation: 참조 궤적 종점 vs 목표
        if prob.algebraic_drift:
            x_final = x_ref_final
        elif coupling_active:
            x_final = seg_trajs[-1][-1]
        else:
            x_final = seg_trajs[-1][-1]
        bc_viol_v = np.linalg.norm(x_final[3:6] - vf)
        bc_viol_r = np.linalg.norm(x_final[:3] - rf)
        bc_viol = max(bc_viol_v, bc_viol_r)
        bc_history.append(bc_viol)
        cost_history.append(cost_val)

        _ta_opt = ta_ref if prob.ta_free else None

        if bc_viol < prob.tol_bc and iteration > 2:
            result = BLADESCPResult(
                p_segments=p_segments, cost=cost_val,
                converged=True, n_iter=iteration + 1,
                bc_violation=bc_viol,
                bc_history=bc_history, cost_history=cost_history,
                status="converged",
                ta_opt=_ta_opt,
                bc_violation_r=bc_viol_r,
                bc_violation_v=bc_viol_v,
                thrust_violation=_thrust_viol,
                x_final=x_final.copy(),
            )
            if prob.validate:
                result.validation = validate_blade_solution(prob, result)
            return result

        # 7. 정체(stagnation) 감지 — 조기 종료
        _stag_window = 8
        if iteration >= _stag_window and iteration > 5:
            _recent = bc_history[-_stag_window:]
            _mid = _stag_window // 2
            _best_old = min(_recent[:_mid])
            _best_new = min(_recent[_mid:])
            _mean_r = np.mean(_recent)
            _std_r = np.std(_recent)
            _oscillating = _std_r / max(_mean_r, 1e-12) > 0.3
            _no_progress = _best_new >= 0.95 * _best_old
            if _oscillating and _no_progress:
                result = BLADESCPResult(
                    p_segments=p_segments, cost=cost_history[-1],
                    converged=False, n_iter=iteration + 1,
                    bc_violation=bc_viol,
                    bc_history=bc_history, cost_history=cost_history,
                    status="stagnated",
                    ta_opt=_ta_opt,
                    bc_violation_r=bc_viol_r,
                    bc_violation_v=bc_viol_v,
                    thrust_violation=_thrust_viol,
                    x_final=x_final.copy(),
                )
                if prob.validate:
                    result.validation = validate_blade_solution(prob, result)
                return result

    result = BLADESCPResult(
        p_segments=p_segments, cost=cost_history[-1] if cost_history else float("inf"),
        converged=False, n_iter=prob.max_iter,
        bc_violation=bc_history[-1] if bc_history else float("inf"),
        bc_history=bc_history, cost_history=cost_history,
        status="max_iter",
        ta_opt=ta_ref if prob.ta_free else None,
        bc_violation_r=bc_viol_r if bc_history else None,
        bc_violation_v=bc_viol_v if bc_history else None,
        thrust_violation=_thrust_viol if bc_history else None,
        x_final=x_final.copy() if bc_history else None,
    )
    if prob.validate:
        result.validation = validate_blade_solution(prob, result)
    return result


# ── Outer Loop (시간 최적화) ─────────────────────────────────────

@dataclasses.dataclass
class BLADEOuterResult:
    """BLADE outer loop 결과."""

    t_f_opt: float
    inner_result: BLADESCPResult
    t_f_history: list[float]
    cost_history: list[float]


def blade_outer_loop(
    prob: BLADEOrbitProblem,
    t_f_bounds: tuple[float, float],
    n_grid: int = 5,
) -> BLADEOuterResult:
    """t_f 격자 탐색으로 최적 비행시간 결정."""
    t_f_grid = np.linspace(t_f_bounds[0], t_f_bounds[1], n_grid)

    best_cost = float("inf")
    best_result = None
    best_tf = t_f_grid[0]
    tf_hist = []
    cost_hist = []

    for t_f_val in t_f_grid:
        prob_i = dataclasses.replace(prob, t_f=float(t_f_val))
        # J2 세차로 경계조건 자동 갱신됨 (OrbitBC.at_time)
        result = solve_blade_scp(prob_i)
        tf_hist.append(float(t_f_val))
        cost_hist.append(result.cost)

        if result.converged and result.cost < best_cost:
            best_cost = result.cost
            best_result = result
            best_tf = float(t_f_val)

    if best_result is None:
        warnings.warn(
            f"blade_outer_loop: 전체 그리드 수렴 실패 "
            f"[{t_f_bounds[0]:.3f}, {t_f_bounds[1]:.3f}], "
            f"t_f={float(t_f_grid[n_grid // 2]):.3f} fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        best_result = solve_blade_scp(
            dataclasses.replace(prob, t_f=float(t_f_grid[n_grid // 2])),
        )
        best_tf = float(t_f_grid[n_grid // 2])

    return BLADEOuterResult(
        t_f_opt=best_tf,
        inner_result=best_result,
        t_f_history=tf_hist,
        cost_history=cost_hist,
    )


# ── Continuation (호모토피 연속법) ─────────────────────────────────

def blade_continuation(
    prob: BLADEOrbitProblem,
    n_steps: int = 8,
) -> BLADESCPResult:
    """호모토피 연속법으로 대궤도 전이 수렴 개선.

    출발 궤도에서 점진적으로 도착 궤도까지 목표를 이동하며,
    각 단계의 해를 다음 단계의 warm-start로 사용한다.

    Parameters
    ----------
    prob : BLADEOrbitProblem
        최종 목표 문제 정의.
    n_steps : int
        연속 단계 수. 클수록 안정적이지만 느림.

    Returns
    -------
    BLADESCPResult
        최종 단계의 SCP 결과.
    """
    cu = prob.canonical_units or from_orbit(prob.dep.a)

    # 출발/도착 궤도 파라미터 보간
    dep = prob.dep
    arr = prob.arr
    a_dep, a_arr = dep.a, arr.a
    e_dep, e_arr = dep.e, arr.e
    inc_dep, inc_arr = dep.inc, arr.inc

    p_warm = None
    result = None

    for step in range(1, n_steps + 1):
        frac = step / n_steps

        # 중간 목표 궤도
        a_mid = a_dep + frac * (a_arr - a_dep)
        e_mid = e_dep + frac * (e_arr - e_dep)
        inc_mid = inc_dep + frac * (inc_arr - inc_dep)

        arr_mid = OrbitBC(
            a=a_mid, e=e_mid, inc=inc_mid,
            raan=arr.raan, aop=arr.aop, ta=arr.ta,
        )

        prob_step = dataclasses.replace(
            prob,
            arr=arr_mid,
            warm_start=p_warm,
        )

        result = solve_blade_scp(prob_step)
        p_warm = [pk.copy() for pk in result.p_segments]

    return result


# ── 도착 위상 최적화 ───────────────────────────────────────────────

def blade_phase_search(
    prob: BLADEOrbitProblem,
    n_grid: int = 12,
    refine: bool = True,
) -> tuple[BLADESCPResult, float]:
    """도착 진근점이각(ta) 격자 탐색으로 최적 위상 결정.

    원형 궤도에서 ta는 도착 위상만 바꾸므로, 최적 위상에서
    추력 비용이 크게 감소할 수 있다.

    Parameters
    ----------
    prob : BLADEOrbitProblem
        기본 문제 정의 (arr.ta는 무시하고 재탐색).
    n_grid : int
        초기 격자 수 (기본 12 → 30° 간격).
    refine : bool
        True이면 최적 부근에서 2° 간격으로 세밀 탐색.

    Returns
    -------
    result : BLADESCPResult
        최적 ta에서의 SCP 결과.
    ta_opt : float
        최적 도착 ta [rad].
    """
    arr = prob.arr
    ta_grid = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)

    best_cost = float("inf")
    best_result = None
    best_ta = 0.0

    for ta_val in ta_grid:
        arr_i = OrbitBC(
            a=arr.a, e=arr.e, inc=arr.inc,
            raan=arr.raan, aop=arr.aop, ta=ta_val,
        )
        prob_i = dataclasses.replace(prob, arr=arr_i)
        result_i = solve_blade_scp(prob_i)

        if result_i.converged and result_i.cost < best_cost:
            best_cost = result_i.cost
            best_result = result_i
            best_ta = ta_val

    # 세밀 탐색
    if refine and best_result is not None:
        ta_fine = np.arange(
            best_ta - np.radians(15),
            best_ta + np.radians(15),
            np.radians(2),
        )
        for ta_val in ta_fine:
            arr_i = OrbitBC(
                a=arr.a, e=arr.e, inc=arr.inc,
                raan=arr.raan, aop=arr.aop, ta=ta_val,
            )
            prob_i = dataclasses.replace(prob, arr=arr_i)
            result_i = solve_blade_scp(prob_i)

            if result_i.converged and result_i.cost < best_cost:
                best_cost = result_i.cost
                best_result = result_i
                best_ta = ta_val

    if best_result is None:
        warnings.warn(
            "blade_phase_search: 전체 그리드 수렴 실패, "
            "초기 ta로 fallback",
            RuntimeWarning,
            stacklevel=2,
        )
        best_result = solve_blade_scp(prob)

    return best_result, best_ta
