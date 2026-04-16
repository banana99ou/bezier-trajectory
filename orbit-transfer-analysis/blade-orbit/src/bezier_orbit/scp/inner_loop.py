"""SCP Inner Loop: QP/SOCP 반복 풀기 (CVXPY).

이론: docs/reports/006_scp_formulation/

t_f 고정 상태에서 Z = t_f · P_u 에 대한 SCP 반복:
1. 참조 궤도 선형화 → Jacobian, 상수항
2. 경계조건 선형 등식 제약 구성
3. QP/SOCP 풀기
4. 궤도 전파 → 새 참조 궤도
5. 수렴 판정
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from ..bezier.basis import gram_matrix, bernstein, int_matrix, double_int_matrix
from ..bezier.constraints import thrust_socp_matrices
from ..normalize import J2_EARTH
from ..orbit.dynamics import propagate_rk4, eom_twobody_j2, jacobian_twobody_j2
from ..orbit.perturbations import compute_perturbations
from .drift import compute_drift, compute_coupling_matrices
from .problem import SCPProblem, build_lifted_boundary_constraints


@dataclasses.dataclass
class SCPResult:
    """SCP inner loop 결과."""

    Z_opt: NDArray              # (N+1, 3) 최적 보조변수
    P_u_opt: NDArray            # (N+1, 3) 복원된 제어점
    cost: float                 # 최적 비용
    n_iter: int                 # SCP 반복 횟수
    converged: bool             # 수렴 여부
    bc_violation: float         # 종말 경계조건 위반량
    ctrl_change_history: list[float]  # 제어점 변화 이력
    cost_history: list[float]         # 비용 이력
    bc_violation_history: list[float] = dataclasses.field(default_factory=list)
    trust_radius_history: list[float] = dataclasses.field(default_factory=list)
    convergence_reason: str = ""      # "ctrl", "cost", "max_iter", "solver_fail"
    solve_time_s: float = 0.0        # 총 소요 시간 [초]
    solver_used: str = ""             # "SCS" / "CLARABEL"


def solve_inner_loop(
    prob: SCPProblem,
    Z_init: NDArray | None = None,
) -> SCPResult:
    """SCP inner loop: t_f 고정, Z에 대한 반복 QP/SOCP.

    Parameters
    ----------
    prob : SCPProblem
        문제 정의.
    Z_init : (N+1, 3), optional
        초기 Z 추정. None이면 0으로 시작.

    Returns
    -------
    result : SCPResult
    """
    N = prob.N
    t_f = prob.t_f
    Re_star = 1.0
    if prob.canonical_units is not None:
        Re_star = prob.canonical_units.R_earth_star

    # 초기화
    if Z_init is None:
        Z_init = np.zeros((N + 1, 3))

    t_start = time.perf_counter()

    Z_prev = Z_init.copy()
    ctrl_history = []
    cost_history = []
    bc_viol_history: list[float] = []
    trust_history: list[float] = []
    delta = prob.trust_region  # trust region 반경
    prev_bc_viol = float("inf")
    last_solver_used = ""

    # 적응적 RK4 스텝 수: 반복 중 적절 해상도, 최종 검증 고해상도
    n_steps_iter = max(200, 5 * prob.N)     # 반복 중 (정확한 선형화)
    n_steps_final = max(500, 10 * prob.N)   # 최종 검증 (정밀 BC)

    drift_method = prob.drift_config.method

    # 초기 참조 궤도
    P_u_ref = Z_prev / t_f
    ref_traj = None
    prev_phi = None  # 중력 보정 워밍스타트
    if drift_method == "rk4":
        ref_traj = _propagate_reference(prob, P_u_ref, Re_star, n_steps=n_steps_iter)

    use_coupling = prob.drift_config.use_coupling

    for iteration in range(prob.max_iter):
        # ── 1. 드리프트 적분 ───────────────────────────────
        if drift_method == "rk4":
            c_v, c_r = _compute_drift_integrals(prob, ref_traj, Re_star)
        else:
            c_v, c_r, prev_phi = compute_drift(
                prob, P_u_ref, Re_star, prev_phi=prev_phi,
            )

        # ── 2. 커플링 행렬 (2번째 반복부터) ────────────────
        M_v, M_r, Z_ref_coupling = None, None, None
        if use_coupling and iteration >= 1:
            if drift_method == "rk4":
                M_v, M_r = _compute_coupling_matrices(prob, ref_traj, Re_star)
            else:
                M_v, M_r = compute_coupling_matrices(prob, P_u_ref, Re_star)
            Z_ref_coupling = Z_prev

        # ── 3. 경계조건 제약 구성 ─────────────────────────
        A_eq, b_eq = build_lifted_boundary_constraints(
            prob, c_v=c_v, c_r=c_r,
            M_v=M_v, M_r=M_r, Z_ref=Z_ref_coupling,
        )

        # ── 3a. 경로 제약 구성 (궤도 반경 부등식, 3번째 반복부터) ──
        if prob.has_path_constraints and iteration >= 2:
            from ..bezier.constraints import path_constraint_matrices

            A_ub, b_ub = path_constraint_matrices(
                P_u_ref, prob.r0, prob.v0, t_f, N,
                r_min=prob.r_min, r_max=prob.r_max,
                K_subdiv=prob.path_K_subdiv,
                gravity_drift=prev_phi,
            )
            prob._path_A_ub = A_ub
            prob._path_b_ub = b_ub

        # ── 4. QP/SOCP 풀기 ─────────────────────────────
        # 첫 2회 반복은 trust region 없이 자유롭게 탐색
        use_tr = iteration >= 2
        Z_opt, cost, solver_name = _solve_convex_subproblem(
            prob, A_eq, b_eq, Re_star,
            Z_ref=Z_prev if use_tr else None,
            trust_radius=delta if use_tr else None,
        )
        last_solver_used = solver_name

        if Z_opt is None:
            # Solver 실패 — trust region 축소 후 재시도
            if use_tr:
                delta = max(delta * prob.trust_shrink, prob.trust_min)
            cost_history.append(float("inf"))
            ctrl_history.append(0.0)
            bc_viol_history.append(float("inf"))
            trust_history.append(delta)
            continue

        # ── 4. 완화 (step damping) ────────────────────────
        # 첫 반복은 full step, 이후 relaxation
        alpha = 1.0 if iteration == 0 else prob.relax_alpha
        Z_new = (1.0 - alpha) * Z_prev + alpha * Z_opt

        cost_history.append(cost)
        trust_history.append(delta)

        # ── 5. 수렴 판정 ──────────────────────────────────
        ctrl_change = np.linalg.norm(Z_new - Z_prev, "fro")
        ctrl_history.append(ctrl_change)

        # 참조 갱신 및 BC 위반 계산
        P_u_new = Z_new / t_f
        if drift_method == "rk4":
            ref_traj = _propagate_reference(
                prob, P_u_new, Re_star, n_steps=n_steps_iter,
            )
            bc_viol = _bc_violation_from_traj(prob, ref_traj)
        else:
            bc_viol = _bc_violation_algebraic(prob, P_u_new, c_v, c_r)

        # Trust region 적응 (3번째 반복부터)
        if use_tr:
            if bc_viol < prev_bc_viol:
                delta = min(delta * prob.trust_expand, 1e6)
            else:
                delta = max(delta * prob.trust_shrink, prob.trust_min)
        prev_bc_viol = bc_viol
        bc_viol_history.append(bc_viol)

        # 수렴 조건: 절대 ctrl_change OR 상대 비용 변화
        cost_converged = (
            len(cost_history) >= 2
            and cost_history[-2] < float("inf")
            and abs(cost - cost_history[-2]) < 1e-4 * max(abs(cost), 1e-12)
        )
        ctrl_converged = ctrl_change < prob.tol_ctrl
        if (ctrl_converged or cost_converged) and bc_viol < prob.tol_bc:
            # 수렴 시 고해상도 RK4로 정확한 BC 위반 검증
            ref_final = _propagate_reference(
                prob, P_u_new, Re_star, n_steps=n_steps_final,
            )
            bc_viol_exact = _bc_violation_from_traj(prob, ref_final)
            reason = "ctrl" if ctrl_converged else "cost"
            return SCPResult(
                Z_opt=Z_new, P_u_opt=P_u_new,
                cost=cost, n_iter=iteration + 1,
                converged=True, bc_violation=bc_viol_exact,
                ctrl_change_history=ctrl_history, cost_history=cost_history,
                bc_violation_history=bc_viol_history,
                trust_radius_history=trust_history,
                convergence_reason=reason,
                solve_time_s=time.perf_counter() - t_start,
                solver_used=last_solver_used,
            )

        Z_prev = Z_new.copy()
        P_u_ref = P_u_new

    # 최대 반복 도달 — 고해상도 전파로 정확한 BC 위반 산출
    P_u_opt = Z_prev / t_f
    ref_final = _propagate_reference(
        prob, P_u_opt, Re_star, n_steps=n_steps_final,
    )
    bc_viol_exact = _bc_violation_from_traj(prob, ref_final)
    return SCPResult(
        Z_opt=Z_prev, P_u_opt=P_u_opt,
        cost=cost_history[-1] if cost_history else float("inf"),
        n_iter=prob.max_iter, converged=False,
        bc_violation=bc_viol_exact,
        ctrl_change_history=ctrl_history, cost_history=cost_history,
        bc_violation_history=bc_viol_history,
        trust_radius_history=trust_history,
        convergence_reason="max_iter",
        solve_time_s=time.perf_counter() - t_start,
        solver_used=last_solver_used,
    )


def _solve_convex_subproblem(
    prob: SCPProblem,
    A_eq: NDArray,
    b_eq: NDArray,
    Re_star: float,
    Z_ref: NDArray | None = None,
    trust_radius: float | None = None,
) -> tuple[NDArray | None, float, str]:
    """CVXPY로 볼록 부분문제 풀기.

    min  (1/t_f) · Σ z_k^T G_N z_k
    s.t. A_eq @ vec(Z) = b_eq
         ‖Z^T B_N(τ_j)‖ ≤ t_f · u_max  (SOCP, optional)
         ‖z - z_ref‖_∞ ≤ δ              (trust region, optional)
    """
    N = prob.N
    t_f = prob.t_f
    dim = 3 * (N + 1)

    # 결정변수: z = vec(Z) ∈ R^{3(N+1)}
    z = cp.Variable(dim)

    # 목적함수: (1/t_f) · z^T H z
    G = gram_matrix(N)
    cost = 0.0
    for k in range(3):
        zk = z[k * (N + 1):(k + 1) * (N + 1)]
        cost = cost + cp.quad_form(zk, G)
    cost = cost / t_f

    # 제약조건
    constraints = [A_eq @ z == b_eq]

    # Trust region 제약
    if Z_ref is not None and trust_radius is not None:
        z_ref_vec = np.concatenate([Z_ref[:, k] for k in range(3)])
        constraints.append(cp.norm_inf(z - z_ref_vec) <= trust_radius)

    # SOCP 추력 제약
    if prob.u_max is not None:
        tau_grid, B_grid = thrust_socp_matrices(N, prob.thrust_grid_M)
        rhs = t_f * prob.u_max  # ‖Z^T B‖ ≤ t_f · u_max

        for j in range(len(tau_grid)):
            bj = B_grid[j, :]  # (N+1,)
            # u_j = [z_x^T bj, z_y^T bj, z_z^T bj]
            u_components = []
            for k in range(3):
                zk = z[k * (N + 1):(k + 1) * (N + 1)]
                u_components.append(bj @ zk)
            u_vec = cp.hstack(u_components)
            constraints.append(cp.norm(u_vec) <= rhs)

    # 경로 제약 (궤도 반경 부등식)
    if prob.has_path_constraints and hasattr(prob, '_path_A_ub') and prob._path_A_ub.shape[0] > 0:
        A_ub = prob._path_A_ub
        b_ub = prob._path_b_ub
        if A_ub.shape[0] > 0:
            constraints.append(A_ub @ z <= b_ub)

    # 풀기
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective, constraints)

    solver_used = "SCS"
    try:
        problem.solve(solver=cp.SCS, verbose=False, max_iters=10000)
    except cp.SolverError:
        solver_used = "CLARABEL"
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
        except (cp.SolverError, Exception):
            return None, float("inf"), solver_used

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None, float("inf"), solver_used

    z_val = z.value
    if z_val is None:
        return None, float("inf"), solver_used

    # vec(Z) → (N+1, 3)
    Z_opt = np.column_stack([
        z_val[k * (N + 1):(k + 1) * (N + 1)] for k in range(3)
    ])

    return Z_opt, float(problem.value), solver_used


def _propagate_reference(
    prob: SCPProblem,
    P_u: NDArray,
    Re_star: float,
    n_steps: int | None = None,
) -> NDArray:
    """참조 궤도 전파 (제어 입력 포함).

    Parameters
    ----------
    n_steps : int, optional
        RK4 스텝 수. None이면 기본값 max(200, 10*N).

    Returns (n_steps+1, 6) 상태 궤적.
    """
    x0 = np.concatenate([prob.r0, prob.v0])
    if n_steps is None:
        n_steps = max(200, 10 * prob.N)

    def u_func(tau):
        B = bernstein(prob.N, tau)
        return np.dot(B, P_u)

    # 고차 섭동 클로저 구성 (Level 1/2)
    extra_accel = _build_extra_accel_func(prob, Re_star)

    _, x_arr = propagate_rk4(
        x0, (0.0, 1.0), n_steps,
        u_func=u_func, Re_star=Re_star,
        include_j2=(prob.perturbation_level >= 0),
        t_f=prob.t_f,
        extra_accel_func=extra_accel,
    )
    return x_arr


def _build_extra_accel_func(
    prob: SCPProblem,
    Re_star: float,
):
    """perturbation_level에 따른 추가 섭동 가속도 클로저 생성.

    Level 0: None (J2는 dynamics.py에서 처리)
    Level 1: J3–J6, 대기항력
    Level 2: Level 1 + SRP, 3체 섭동
    """
    if prob.perturbation_level < 1:
        return None

    # 정규화 기준량 추출
    cu = prob.canonical_units
    DU = cu.DU if cu is not None else 1.0
    TU = cu.TU if cu is not None else 1.0

    # 클로저에 필요한 파라미터 캡처
    level = prob.perturbation_level
    max_jn = prob.max_jn_degree
    Cd_Am = prob.Cd_A_over_m
    Cr_Am = prob.Cr_A_over_m
    r_sun_func = prob.r_sun_func
    r_moon_func = prob.r_moon_func
    mu_sun = prob.mu_sun_star
    mu_moon = prob.mu_moon_star

    def extra_accel(tau: float, x) -> np.ndarray:
        r = x[:3]
        v = x[3:6]

        # 천체 위치 (Level 2)
        r_sun = r_sun_func(tau) if (level >= 2 and r_sun_func is not None) else None
        r_moon = r_moon_func(tau) if (level >= 2 and r_moon_func is not None) else None

        return compute_perturbations(
            r, v,
            level=level,
            Re_star=Re_star,
            DU=DU, TU=TU,
            max_jn_degree=max_jn,
            Cd_A_over_m=Cd_Am,
            Cr_A_over_m=Cr_Am,
            r_sun=r_sun, r_moon=r_moon,
            mu_sun_star=mu_sun, mu_moon_star=mu_moon,
        )

    return extra_accel


def _compute_drift_integrals(
    prob: SCPProblem,
    ref_traj: NDArray,
    Re_star: float,
) -> tuple[NDArray, NDArray]:
    """참조 궤도의 드리프트 적분 기여 c_v, c_r 계산 (벡터화).

    비선형 드리프트 f_0(τ)를 참조 궤도에서 수치 적분.

    c_v = ∫₀¹ f_0(τ) dτ  (속도 변화 중 중력+섭동 기여)
    c_r = ∫₀¹∫₀^s f_0(σ) dσ ds  (위치 변화 중 중력+섭동 기여)
    """
    n_pts = ref_traj.shape[0]
    tau_arr = np.linspace(0, 1, n_pts)
    dt = tau_arr[1] - tau_arr[0]

    # f_0(τ) = 중력+섭동 가속도 — 벡터화 계산
    r_all = ref_traj[:, :3]  # (n_pts, 3)
    r_mag = np.linalg.norm(r_all, axis=1)  # (n_pts,)
    r3 = r_mag**3

    # 2체 중심 중력 (mu* = 1)
    f0_arr = -r_all / r3[:, None]

    # J2 섭동 (벡터화)
    if prob.perturbation_level >= 0:
        r2 = r_mag**2
        r5 = r_mag**5
        z = r_all[:, 2]
        z2_r2 = z**2 / r2
        coeff = 1.5 * J2_EARTH * Re_star**2 / r5  # (n_pts,)

        j2_accel = np.empty_like(r_all)
        j2_accel[:, 0] = coeff * r_all[:, 0] * (5.0 * z2_r2 - 1.0)
        j2_accel[:, 1] = coeff * r_all[:, 1] * (5.0 * z2_r2 - 1.0)
        j2_accel[:, 2] = coeff * z * (5.0 * z2_r2 - 3.0)
        f0_arr = f0_arr + j2_accel

    # 고차 섭동 (Level 1/2) — 포인트별 루프
    if prob.perturbation_level >= 1:
        cu = prob.canonical_units
        DU = cu.DU if cu is not None else 1.0
        TU = cu.TU if cu is not None else 1.0
        v_all = ref_traj[:, 3:6]

        for m in range(n_pts):
            r_sun = None
            r_moon = None
            if prob.perturbation_level >= 2:
                tau_m = tau_arr[m]
                if prob.r_sun_func is not None:
                    r_sun = prob.r_sun_func(tau_m)
                if prob.r_moon_func is not None:
                    r_moon = prob.r_moon_func(tau_m)

            f0_arr[m] += compute_perturbations(
                r_all[m], v_all[m],
                level=prob.perturbation_level,
                Re_star=Re_star,
                DU=DU, TU=TU,
                max_jn_degree=prob.max_jn_degree,
                Cd_A_over_m=prob.Cd_A_over_m,
                Cr_A_over_m=prob.Cr_A_over_m,
                r_sun=r_sun, r_moon=r_moon,
                mu_sun_star=prob.mu_sun_star,
                mu_moon_star=prob.mu_moon_star,
            )

    # c_v = ∫₀¹ f_0(τ) dτ  (사다리꼴)
    c_v = np.trapezoid(f0_arr, tau_arr, axis=0)

    # c_r = ∫₀¹∫₀^s f_0(σ) dσ ds  (누적 적분 후 재적분)
    cumint_f0 = np.cumsum(
        0.5 * (f0_arr[:-1] + f0_arr[1:]) * dt, axis=0,
    )
    cumint_f0 = np.vstack([np.zeros((1, 3)), cumint_f0])

    c_r = np.trapezoid(cumint_f0, tau_arr, axis=0)

    return c_v, c_r


def _gravity_accel_vec(r_all: NDArray, Re_star: float, include_j2: bool) -> NDArray:
    """2체+J2 중력 가속도 벡터화. (M, 3) → (M, 3)."""
    r_mag = np.linalg.norm(r_all, axis=1)
    r3 = r_mag**3
    a = -r_all / r3[:, None]

    if include_j2:
        r2 = r_mag**2
        r5 = r_mag**5
        z = r_all[:, 2]
        z2_r2 = z**2 / r2
        coeff = 1.5 * J2_EARTH * Re_star**2 / r5
        a[:, 0] += coeff * r_all[:, 0] * (5.0 * z2_r2 - 1.0)
        a[:, 1] += coeff * r_all[:, 1] * (5.0 * z2_r2 - 1.0)
        a[:, 2] += coeff * z * (5.0 * z2_r2 - 3.0)

    return a


def _compute_drift_midpoint(
    prob: SCPProblem,
    ref_traj: NDArray,
    Re_star: float,
    K: int = 8,
) -> tuple[NDArray, NDArray]:
    """구간별 midpoint rule로 드리프트 적분 근사 (보고서 009).

    [0,1]을 K등분, 각 구간 중점에서 중력 가속도를 평가.
    RK4 전파 없이 참조 궤도 위의 이산 점만 사용.

    Returns c_v (3,), c_r (3,).
    """
    n_traj = ref_traj.shape[0]
    tau_traj = np.linspace(0, 1, n_traj)
    include_j2 = prob.perturbation_level >= 0

    tau_breaks = np.linspace(0, 1, K + 1)
    tau_mids = 0.5 * (tau_breaks[:-1] + tau_breaks[1:])
    dtau = 1.0 / K

    # 구간 중점 위치 (보간)
    r_mids = np.column_stack([
        np.interp(tau_mids, tau_traj, ref_traj[:, k]) for k in range(3)
    ])  # (K, 3)

    # 중력 가속도 일괄 평가
    a_mids = _gravity_accel_vec(r_mids, Re_star, include_j2)  # (K, 3)

    # c_v = Σ a(r_mid_k) · Δτ  (midpoint rule)
    c_v = np.sum(a_mids * dtau, axis=0)

    # c_r: 구간 경계에서도 평가하여 이중 적분 근사
    r_breaks = np.column_stack([
        np.interp(tau_breaks, tau_traj, ref_traj[:, k]) for k in range(3)
    ])  # (K+1, 3)
    a_breaks = _gravity_accel_vec(r_breaks, Re_star, include_j2)

    # F(s) = ∫₀^s a(r(σ)) dσ  (사다리꼴 누적)
    F = np.cumsum(0.5 * (a_breaks[:-1] + a_breaks[1:]) * dtau, axis=0)
    F = np.vstack([np.zeros((1, 3)), F])  # (K+1, 3)

    # c_r = ∫₀¹ F(s) ds  (사다리꼴)
    c_r = np.trapezoid(F, tau_breaks, axis=0)

    return c_v, c_r


def _compute_coupling_matrices(
    prob: SCPProblem,
    ref_traj: NDArray,
    Re_star: float,
) -> tuple[NDArray, NDArray]:
    """커플링 행렬 M_v, M_r 계산 (보고서 009 식 (11)).

    드리프트 적분의 제어점 감도:
      c_v ≈ c_v_ref + M_v · vec(ΔP_u)
      c_r ≈ c_r_ref + M_r · vec(ΔP_u)

    M_v = t_f · Σ w_m · kron(A(r*_m), b̄(τ_m)^T)
    여기서 b̄(τ) = Ī_N^T · B_{N+2}(τ) 는 이중적분 위치 기저.

    Parameters
    ----------
    Returns M_v (3, 3(N+1)), M_r (3, 3(N+1)).
    """
    N = prob.N
    t_f = prob.t_f
    n_pts = ref_traj.shape[0]
    include_j2 = prob.perturbation_level >= 0

    tau_arr = np.linspace(0, 1, n_pts)
    dt = tau_arr[1] - tau_arr[0]

    Ibar_N = double_int_matrix(N)  # (N+3, N+1)
    dim = 3 * (N + 1)

    # 각 구적점에서 kron(A, b̄^T) 계산하여 합산
    # K_arr[m] = A_rr(τ_m) ⊗ b̄(τ_m)^T → (3, 3(N+1))
    # B_{N+2}(τ)를 일괄 평가
    B_N2_all = bernstein(N + 2, tau_arr)  # (n_pts, N+3)
    bbar_all = np.dot(B_N2_all, Ibar_N)   # (n_pts, N+1)

    # 사다리꼴 가중치
    w = np.ones(n_pts) * dt
    w[0] *= 0.5
    w[-1] *= 0.5

    # M_v, K_arr 누적
    M_v = np.zeros((3, dim))
    K_arr = np.zeros((n_pts, 3, dim))  # M_r 계산용

    for m in range(n_pts):
        r = ref_traj[m, :3]
        x_state = np.zeros(6)
        x_state[:3] = r
        A_full = jacobian_twobody_j2(
            x_state, Re_star=Re_star, include_j2=include_j2,
        )
        A_rr = A_full[3:6, :3]  # (3,3) ∂a_grav/∂r

        bbar_m = bbar_all[m]  # (N+1,)
        K_m = np.kron(A_rr, bbar_m.reshape(1, -1))  # (3, 3(N+1))
        K_arr[m] = K_m
        M_v += w[m] * K_m

    M_v *= t_f

    # M_r: 이중 적분 — F(s) = ∫₀^s K(σ) dσ, M_r = t_f · ∫₀¹ F(s) ds
    # 사다리꼴 누적 적분
    cumK = np.cumsum(
        0.5 * (K_arr[:-1] + K_arr[1:]) * dt, axis=0,
    )  # (n_pts-1, 3, dim)
    cumK = np.concatenate(
        [np.zeros((1, 3, dim)), cumK], axis=0,
    )  # (n_pts, 3, dim)

    # 사다리꼴로 ∫₀¹ F(s) ds
    M_r = np.trapezoid(cumK, tau_arr, axis=0)  # (3, dim)
    M_r *= t_f

    return M_v, M_r


def _bc_violation_algebraic(
    prob: SCPProblem,
    P_u: NDArray,
    c_v: NDArray,
    c_r: NDArray,
) -> float:
    """경계조건 공식으로 대수적 BC 위반량 계산 (RK4 불필요).

    v_f_pred = v_0 + t_f·c_v + t_f·I_N[-1,:]·P_u
    r_f_pred = r_0 + t_f·v_0 + t_f²·c_r + t_f·Ī_N[-1,:]·P_u
    """
    N = prob.N
    t_f = prob.t_f

    eI = int_matrix(N)[-1, :]          # (N+1,)
    eIbar = double_int_matrix(N)[-1, :]  # (N+1,)

    v_pred = prob.v0 + t_f * c_v + t_f * (eI @ P_u)
    r_pred = prob.r0 + t_f * prob.v0 + t_f**2 * c_r + t_f**2 * (eIbar @ P_u)

    v_err = np.linalg.norm(v_pred - prob.vf)
    r_err = np.linalg.norm(r_pred - prob.rf)
    return float(r_err + v_err)


def _bc_violation_from_traj(
    prob: SCPProblem,
    ref_traj: NDArray,
) -> float:
    """이미 전파된 궤적에서 종말 경계조건 위반량 계산."""
    x_final = ref_traj[-1]
    r_err = np.linalg.norm(x_final[:3] - prob.rf)
    v_err = np.linalg.norm(x_final[3:6] - prob.vf)
    return float(r_err + v_err)


def _check_bc_violation(
    prob: SCPProblem,
    P_u: NDArray,
    Re_star: float,
) -> float:
    """종말 경계조건 위반량 계산."""
    ref_traj = _propagate_reference(prob, P_u, Re_star)
    return _bc_violation_from_traj(prob, ref_traj)
