"""드리프트 적분 계산 모듈 — 3가지 방법 지원.

SCP 반복에서 중력+섭동 드리프트 적분 c_v, c_r을 계산하는
세 가지 방법을 통합 인터페이스로 제공한다.

방법:
  - "rk4"       : RK4 수치 전파 + 사다리꼴 적분 (기존, 정확도 최고)
  - "affine"    : 구간별 midpoint rule (보고서 009)
  - "bernstein" : Bernstein 대수 파이프라인 (보고서 010, 디폴트)

affine과 bernstein 방법은 제어 전용 Bernstein 위치에 중력 자기일관
보정(self-consistent gravity correction)을 적용하여 실제 궤적에 가까운
참조 위치를 구성한다.

이론:
  - docs/reports/009_affine_drift/
  - docs/reports/010_bernstein_algebra/
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from ..bezier.algebra import degree_elevate, degree_reduce, gravity_composition_pipeline
from ..bezier.basis import (
    bernstein,
    bernstein_eval,
    definite_integral,
    double_int_matrix,
    int_matrix,
)
from ..normalize import J2_EARTH


@dataclasses.dataclass
class DriftConfig:
    """드리프트 적분 계산 설정.

    Parameters
    ----------
    method : str
        "rk4" | "affine" | "bernstein" (디폴트 "bernstein").
    affine_K : int
        아핀 방법의 구간 수 (디폴트 8).
    affine_order : int
        0: midpoint rule, 1: +Jacobian 보정 (디폴트 0).
    bernstein_K : int
        r^{-3/2} Chebyshev-Bernstein 근사 차수 (디폴트 8).
    bernstein_R : int
        차수 축소 목표 (디폴트 20).
    n_gravity_iter : int
        자기일관 중력 보정 최대 반복 횟수 (디폴트 8).
    gravity_correction : str
        "algebraic" (디폴트) | "numerical".
    gravity_correction_K : int | None
        중력 보정용 파이프라인 K (None이면 bernstein_K 사용, 기본 4).
    gravity_tol : float
        자기일관 보정 조기 종료 허용치 (디폴트 1e-6).
    """

    method: str = "bernstein"
    # 아핀 방법 (보고서 009)
    affine_K: int = 8
    affine_order: int = 0
    # Bernstein 방법 (보고서 010)
    bernstein_K: int = 8
    bernstein_R: int = 20
    # 자기일관 중력 보정
    n_gravity_iter: int = 8
    gravity_correction: str = "algebraic"  # "algebraic" | "numerical"
    gravity_correction_K: int | None = 4   # 보정용 K (낮은 값으로 속도 향상)
    gravity_tol: float = 1e-6              # 조기 종료 허용치
    use_coupling: bool = True              # 커플링 행렬 M_v, M_r 사용 여부


# ── 공용 유틸리티 ─────────────────────────────────────────────


def position_control_points(
    P_u: NDArray,
    r0: NDArray,
    v0: NDArray,
    t_f: float,
    N: int,
) -> NDArray:
    """추력 제어점으로부터 위치 궤적의 Bernstein 제어점 구성 (제어 기여만).

    적분 체인: u ∈ B_N → v ∈ B_{N+1} → r ∈ B_{N+2}

    q_r[:,k] = r0[k] + t_f · v0[k] · ℓ + t_f² · Ī_N · P_u[:,k]

    Parameters
    ----------
    P_u : (N+1, 3) — 추력 가속도 제어점
    r0, v0 : (3,) — 초기 위치/속도 (정규화)
    t_f : float — 비행시간 (정규화)
    N : int — 베지어 차수

    Returns
    -------
    q_r : (N+3, 3) — 위치 궤적 Bernstein 제어점
    """
    Ibar_N = double_int_matrix(N)  # (N+3, N+1)
    ell = np.arange(N + 3) / (N + 2)  # [0, 1/(N+2), ..., 1]

    q_r = np.empty((N + 3, 3))
    for k in range(3):
        q_r[:, k] = r0[k] + t_f * v0[k] * ell + t_f**2 * (Ibar_N @ P_u[:, k])
    return q_r


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


def _corrected_position(
    P_u: NDArray,
    r0: NDArray,
    v0: NDArray,
    t_f: float,
    N: int,
    Re_star: float,
    include_j2: bool,
    n_iter: int = 8,
    correction_mode: str = "algebraic",
    K: int = 8,
    R: int = 20,
    n_pts: int = 60,
    tol: float = 1e-6,
    prev_phi: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """자기일관 중력 보정이 포함된 위치 Bernstein 제어점.

    제어 전용 위치에서 시작하여, 중력의 위치 기여(이중 적분)를
    반복적으로 보정하여 실제 궤적에 가까운 참조 위치를 구성한다.
    워밍스타트(prev_phi)와 적응적 조기 종료(tol)를 지원한다.

    Returns
    -------
    q_r : (N+3, 3) — 보정된 위치 Bernstein 제어점
    phi : (N+3, 3) — 중력 드리프트 제어점 (다음 호출의 워밍스타트용)
    """
    if correction_mode == "algebraic":
        return _corrected_position_algebraic(
            P_u, r0, v0, t_f, N, Re_star, include_j2,
            n_iter=n_iter, K=K, R=R, tol=tol, prev_phi=prev_phi,
        )
    else:
        return _corrected_position_numerical(
            P_u, r0, v0, t_f, N, Re_star, include_j2,
            n_iter=n_iter, n_pts=n_pts, tol=tol, prev_phi=prev_phi,
        )


def _degree_elevate_2d(p: NDArray, target_deg: int) -> NDArray:
    """degree_elevate의 2D 확장. (Q+1, d) → (target_deg+1, d)."""
    if p.ndim == 1:
        return degree_elevate(p, target_deg)
    return np.column_stack([
        degree_elevate(p[:, k], target_deg) for k in range(p.shape[1])
    ])


def _corrected_position_algebraic(
    P_u: NDArray,
    r0: NDArray,
    v0: NDArray,
    t_f: float,
    N: int,
    Re_star: float,
    include_j2: bool,
    n_iter: int = 8,
    K: int = 8,
    R: int = 20,
    tol: float = 1e-6,
    prev_phi: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """대수적 Bernstein 파이프라인으로 중력 보정 (보고서 010 방식).

    중력 가속도 → 적분행렬 곱 → 이중 적분 → 차수 맞추기 → 위치 보정.
    워밍스타트 및 적응적 조기 종료를 지원한다.

    Returns
    -------
    q_r : (N+3, 3) — 보정된 위치 Bernstein 제어점
    q_Phi_matched : (N+3, 3) — 마지막 중력 드리프트 (워밍스타트용)
    """
    deg_r = N + 2
    corr_K = K  # 보정용 K (낮은 값으로 속도 향상)
    q_r_control = position_control_points(P_u, r0, v0, t_f, N)  # (N+3, 3)

    I_R = int_matrix(R)
    I_R1 = int_matrix(R + 1)
    deg_phi = R + 2

    # 워밍스타트: 이전 반복의 Φ로 초기화
    if prev_phi is not None:
        q_Phi_matched = prev_phi
        q_r = q_r_control + t_f**2 * q_Phi_matched
    else:
        q_Phi_matched = np.zeros((deg_r + 1, 3))
        q_r = q_r_control.copy()

    for _ in range(n_iter):
        q_agrav = gravity_composition_pipeline(q_r, K=corr_K, R=R)

        q_F = I_R @ q_agrav
        q_Phi = I_R1 @ q_F

        if deg_phi > deg_r:
            q_Phi_new = degree_reduce(q_Phi, deg_r)
        elif deg_phi < deg_r:
            q_Phi_new = _degree_elevate_2d(q_Phi, deg_r)
        else:
            q_Phi_new = q_Phi

        # 적응적 조기 종료
        delta = np.linalg.norm(q_Phi_new - q_Phi_matched)
        q_Phi_matched = q_Phi_new
        q_r = q_r_control + t_f**2 * q_Phi_matched

        if delta < tol:
            break

    return q_r, q_Phi_matched


def _corrected_position_numerical(
    P_u: NDArray,
    r0: NDArray,
    v0: NDArray,
    t_f: float,
    N: int,
    Re_star: float,
    include_j2: bool,
    n_iter: int = 8,
    n_pts: int = 60,
    tol: float = 1e-6,
    prev_phi: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """수치적분으로 중력 보정 (점별 평가 + 사다리꼴 적분).

    Returns
    -------
    q_r : (N+3, 3) — 보정된 위치
    phi_cp : (N+3, 3) — 마지막 중력 드리프트 (워밍스타트용)
    """
    deg = N + 2
    q_r_control = position_control_points(P_u, r0, v0, t_f, N)

    tau = np.linspace(0, 1, n_pts)
    dt = tau[1] - tau[0]
    B_mat = bernstein(deg, tau)

    if prev_phi is not None:
        phi_cp = prev_phi
        q_r = q_r_control + phi_cp
    else:
        phi_cp = np.zeros((deg + 1, 3))
        q_r = q_r_control.copy()

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        for _ in range(n_iter):
            r_pts = B_mat @ q_r
            a_pts = _gravity_accel_vec(r_pts, Re_star, include_j2)

            cum_a = np.cumsum(0.5 * (a_pts[:-1] + a_pts[1:]) * dt, axis=0)
            cum_a = np.vstack([np.zeros((1, 3)), cum_a])
            phi = np.cumsum(0.5 * (cum_a[:-1] + cum_a[1:]) * dt, axis=0)
            phi = np.vstack([np.zeros((1, 3)), phi])

            phi_cp_new, _, _, _ = np.linalg.lstsq(B_mat, t_f**2 * phi, rcond=None)

            delta_norm = np.linalg.norm(phi_cp_new - phi_cp)
            phi_cp = phi_cp_new
            q_r = q_r_control + phi_cp

            if delta_norm < tol:
                break

    return q_r, phi_cp


def _drift_from_position(
    q_r: NDArray,
    deg: int,
    Re_star: float,
    include_j2: bool,
    n_pts: int = 60,
) -> tuple[NDArray, NDArray]:
    """보정된 위치 제어점에서 c_v, c_r을 수치 적분으로 계산.

    Parameters
    ----------
    q_r : (deg+1, 3) — 위치 Bernstein 제어점
    deg : int — 위치 곡선 차수

    Returns
    -------
    c_v, c_r : (3,) each
    """
    tau = np.linspace(0, 1, n_pts)
    dt = tau[1] - tau[0]
    r_pts = bernstein_eval(deg, q_r, tau)  # (n_pts, 3)
    a_pts = _gravity_accel_vec(r_pts, Re_star, include_j2)

    # c_v = ∫₀¹ a_grav(τ) dτ
    c_v = np.trapezoid(a_pts, tau, axis=0)

    # c_r = ∫₀¹∫₀^s a_grav(σ) dσ ds
    cumint = np.cumsum(0.5 * (a_pts[:-1] + a_pts[1:]) * dt, axis=0)
    cumint = np.vstack([np.zeros((1, 3)), cumint])
    c_r = np.trapezoid(cumint, tau, axis=0)

    return c_v, c_r


# ── 디스패처 ──────────────────────────────────────────────────


def compute_drift(
    prob,  # SCPProblem (순환 import 방지를 위해 타입 힌트 생략)
    P_u: NDArray,
    Re_star: float,
    prev_phi: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """드리프트 적분 c_v, c_r 계산 (방법 자동 분기).

    Parameters
    ----------
    prob : SCPProblem
    P_u : (N+1, 3) — 추력 제어점
    Re_star : float — 정규화 지구 반경
    prev_phi : (N+3, 3) | None — 이전 중력 드리프트 (워밍스타트용)

    Returns
    -------
    c_v : (3,) — 속도 드리프트
    c_r : (3,) — 위치 드리프트
    phi : (N+3, 3) | None — 중력 드리프트 (다음 호출의 워밍스타트용, RK4는 None)
    """
    cfg = prob.drift_config
    method = cfg.method

    if method == "affine":
        return _compute_drift_affine(
            prob, P_u, Re_star, K=cfg.affine_K, order=cfg.affine_order,
            n_gravity_iter=cfg.n_gravity_iter, prev_phi=prev_phi,
        )
    elif method == "bernstein":
        return _compute_drift_bernstein(
            prob, P_u, Re_star, K=cfg.bernstein_K, R=cfg.bernstein_R,
            n_gravity_iter=cfg.n_gravity_iter, prev_phi=prev_phi,
        )
    elif method == "rk4":
        cv, cr = _compute_drift_rk4(prob, P_u, Re_star)
        return cv, cr, None
    else:
        raise ValueError(f"Unknown drift method: {method!r}")


# ── 방법 1: RK4 (기존) ───────────────────────────────────────


def _compute_drift_rk4(
    prob,
    P_u: NDArray,
    Re_star: float,
    n_steps: int | None = None,
) -> tuple[NDArray, NDArray]:
    """RK4 전파 + 사다리꼴 적분으로 c_v, c_r 계산."""
    from .inner_loop import _compute_drift_integrals, _propagate_reference

    if n_steps is None:
        n_steps = max(50, prob.N)
    ref_traj = _propagate_reference(prob, P_u, Re_star, n_steps=n_steps)
    return _compute_drift_integrals(prob, ref_traj, Re_star)


# ── 방법 2: 아핀 드리프트 (보고서 009) ───────────────────────


def _compute_drift_affine(
    prob,
    P_u: NDArray,
    Re_star: float,
    K: int = 8,
    order: int = 0,
    n_gravity_iter: int = 8,
    prev_phi: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """구간별 midpoint rule로 드리프트 적분 근사 (보고서 009).

    자기일관 중력 보정된 위치에서 구간별 중력을 평가.
    """
    N = prob.N
    include_j2 = prob.perturbation_level >= 0
    cfg = prob.drift_config
    corr_K = cfg.gravity_correction_K or cfg.bernstein_K

    q_r, phi = _corrected_position(
        P_u, prob.r0, prob.v0, prob.t_f, N,
        Re_star, include_j2, n_iter=n_gravity_iter,
        correction_mode=cfg.gravity_correction,
        K=corr_K, R=cfg.bernstein_R,
        tol=cfg.gravity_tol, prev_phi=prev_phi,
    )

    tau_breaks = np.linspace(0, 1, K + 1)
    tau_mids = 0.5 * (tau_breaks[:-1] + tau_breaks[1:])
    dtau = 1.0 / K

    # 구간 중점 위치
    r_mids = bernstein_eval(N + 2, q_r, tau_mids)  # (K, 3)
    a_mids = _gravity_accel_vec(r_mids, Re_star, include_j2)  # (K, 3)

    if order >= 1:
        from ..orbit.dynamics import jacobian_twobody_j2

        M_sub = 4
        a_corrected = np.zeros_like(a_mids)
        for seg in range(K):
            tau_sub = np.linspace(tau_breaks[seg], tau_breaks[seg + 1], M_sub + 2)[1:-1]
            r_sub = bernstein_eval(N + 2, q_r, tau_sub)
            r_mid = r_mids[seg]

            x_state = np.zeros(6)
            x_state[:3] = r_mid
            A_full = jacobian_twobody_j2(
                x_state, Re_star=Re_star, include_j2=include_j2,
            )
            A_rr = A_full[3:6, :3]

            dr = r_sub - r_mid
            correction = np.mean(dr @ A_rr.T, axis=0)
            a_corrected[seg] = a_mids[seg] + correction

        a_mids = a_corrected

    # c_v = Σ a(r_mid_k) · Δτ
    c_v = np.sum(a_mids * dtau, axis=0)

    # c_r: 구간 경계에서 평가하여 이중 적분
    r_breaks = bernstein_eval(N + 2, q_r, tau_breaks)
    a_breaks = _gravity_accel_vec(r_breaks, Re_star, include_j2)

    F = np.cumsum(0.5 * (a_breaks[:-1] + a_breaks[1:]) * dtau, axis=0)
    F = np.vstack([np.zeros((1, 3)), F])

    c_r = np.trapezoid(F, tau_breaks, axis=0)

    # 고차 섭동 보정 (Level 1/2) — J3~J6, 항력, SRP, 3체
    if prob.perturbation_level >= 1:
        delta_cv, delta_cr = _higher_perturbation_correction(
            prob, q_r, Re_star,
        )
        c_v += delta_cv
        c_r += delta_cr

    return c_v, c_r, phi


# ── 방법 3: Bernstein 대수 (보고서 010) ──────────────────────


def _compute_drift_bernstein(
    prob,
    P_u: NDArray,
    Re_star: float,
    K: int = 8,
    R: int = 20,
    n_gravity_iter: int = 8,
    prev_phi: NDArray | None = None,
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Bernstein 대수 파이프라인으로 드리프트 적분 계산 (보고서 010).

    자기일관 보정된 위치에서 Bernstein 대수 파이프라인으로
    2체 중력을 계산하고, J2는 수치 보정한다.
    """
    N = prob.N
    include_j2 = prob.perturbation_level >= 0
    cfg = prob.drift_config
    corr_K = cfg.gravity_correction_K or cfg.bernstein_K

    q_r, phi = _corrected_position(
        P_u, prob.r0, prob.v0, prob.t_f, N,
        Re_star, include_j2, n_iter=n_gravity_iter,
        correction_mode=cfg.gravity_correction,
        K=corr_K, R=R,
        tol=cfg.gravity_tol, prev_phi=prev_phi,
    )

    # 2체 중력 — Bernstein 대수 파이프라인
    q_agrav = gravity_composition_pipeline(q_r, K=K, R=R)  # (R+1, 3)

    # c_v = ∫₀¹ a_grav(τ) dτ = mean(제어점) (Bernstein 정적분 성질)
    c_v_2body = definite_integral(q_agrav)  # (3,)

    # c_r = ∫₀¹∫₀^s a_grav(σ) dσ ds = mean(I_R · q_agrav)
    I_R = int_matrix(R)  # (R+2, R+1)
    q_F = I_R @ q_agrav  # (R+2, 3)
    c_r_2body = definite_integral(q_F)  # (3,)

    # 섭동 보정 (J2 + 고차)
    delta_cv, delta_cr = _perturbation_correction(prob, q_r, Re_star)
    c_v = c_v_2body + delta_cv
    c_r = c_r_2body + delta_cr

    return c_v, c_r, phi


def _perturbation_correction(
    prob,
    q_r: NDArray,
    Re_star: float,
    n_eval: int = 50,
) -> tuple[NDArray, NDArray]:
    """섭동 보정항(J2~J6, 항력, SRP, 3체)을 보정된 위치에서 수치 적분으로 계산.

    2체 중력은 파이프라인/midpoint에서 이미 포함되었으므로,
    여기서는 **섭동 가속도만** 적분한다.

    perturbation_level에 따라:
      0: J2만
      1: J2 + J3~J6 + 대기항력
      2: Level 1 + SRP + 3체

    Parameters
    ----------
    prob : SCPProblem
    q_r : (N+3, 3) — 보정된 위치 Bernstein 제어점
    Re_star : float
    n_eval : int — 평가점 수

    Returns
    -------
    delta_c_v : (3,) — 섭동 속도 드리프트 보정
    delta_c_r : (3,) — 섭동 위치 드리프트 보정
    """
    from ..orbit.perturbations import compute_perturbations

    N = prob.N
    P_deg = N + 2
    tau_arr = np.linspace(0, 1, n_eval)
    dt = tau_arr[1] - tau_arr[0]

    r_all = bernstein_eval(P_deg, q_r, tau_arr)  # (n_eval, 3)

    # ── J2 섭동 (벡터화) ──
    pert_accel = np.zeros_like(r_all)

    if prob.perturbation_level >= 0:
        r_mag = np.linalg.norm(r_all, axis=1)
        r2 = r_mag**2
        r5 = r_mag**5
        z = r_all[:, 2]
        z2_r2 = z**2 / r2
        coeff = 1.5 * J2_EARTH * Re_star**2 / r5

        pert_accel[:, 0] = coeff * r_all[:, 0] * (5.0 * z2_r2 - 1.0)
        pert_accel[:, 1] = coeff * r_all[:, 1] * (5.0 * z2_r2 - 1.0)
        pert_accel[:, 2] = coeff * z * (5.0 * z2_r2 - 3.0)

    # ── Level 1/2 섭동 (포인트별) ──
    if prob.perturbation_level >= 1:
        from ..bezier.basis import diff_matrix as _diff_matrix

        cu = prob.canonical_units
        DU = cu.DU if cu is not None else 1.0
        TU = cu.TU if cu is not None else 1.0

        # 속도: Bernstein 미분 행렬로 정확하게 계산
        # dr/dτ의 제어점 = D_{N+2} · q_r, 차수 N+1
        # v = (dr/dτ) / t_f
        D_N2 = _diff_matrix(P_deg)  # (N+2, N+3)
        q_v = (D_N2 @ q_r) / prob.t_f  # (N+2, 3): 속도 제어점
        v_all = bernstein_eval(P_deg - 1, q_v, tau_arr)  # (n_eval, 3)

        for m in range(n_eval):
            r_sun = None
            r_moon = None
            if prob.perturbation_level >= 2:
                tau_m = tau_arr[m]
                if prob.r_sun_func is not None:
                    r_sun = prob.r_sun_func(tau_m)
                if prob.r_moon_func is not None:
                    r_moon = prob.r_moon_func(tau_m)

            pert_accel[m] += compute_perturbations(
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

    # ── 적분 ──
    delta_c_v = np.trapezoid(pert_accel, tau_arr, axis=0)

    cumint = np.cumsum(0.5 * (pert_accel[:-1] + pert_accel[1:]) * dt, axis=0)
    cumint = np.vstack([np.zeros((1, 3)), cumint])
    delta_c_r = np.trapezoid(cumint, tau_arr, axis=0)

    return delta_c_v, delta_c_r


def _higher_perturbation_correction(
    prob,
    q_r: NDArray,
    Re_star: float,
    n_eval: int = 50,
) -> tuple[NDArray, NDArray]:
    """Level 1/2 고차 섭동만 계산 (J2 제외).

    affine 방법에서는 _gravity_accel_vec이 J2를 이미 포함하므로,
    여기서는 J3~J6, 항력, SRP, 3체만 적분한다.
    """
    from ..orbit.perturbations import compute_perturbations

    N = prob.N
    P_deg = N + 2
    tau_arr = np.linspace(0, 1, n_eval)
    dt = tau_arr[1] - tau_arr[0]

    r_all = bernstein_eval(P_deg, q_r, tau_arr)

    cu = prob.canonical_units
    DU = cu.DU if cu is not None else 1.0
    TU = cu.TU if cu is not None else 1.0

    # 속도: Bernstein 미분 행렬로 정확하게 계산
    from ..bezier.basis import diff_matrix as _diff_matrix
    D_N2 = _diff_matrix(P_deg)
    q_v = (D_N2 @ q_r) / prob.t_f
    v_all = bernstein_eval(P_deg - 1, q_v, tau_arr)

    pert_accel = np.zeros_like(r_all)
    for m in range(n_eval):
        r_sun = None
        r_moon = None
        if prob.perturbation_level >= 2:
            tau_m = tau_arr[m]
            if prob.r_sun_func is not None:
                r_sun = prob.r_sun_func(tau_m)
            if prob.r_moon_func is not None:
                r_moon = prob.r_moon_func(tau_m)

        pert_accel[m] = compute_perturbations(
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

    delta_c_v = np.trapezoid(pert_accel, tau_arr, axis=0)

    cumint = np.cumsum(0.5 * (pert_accel[:-1] + pert_accel[1:]) * dt, axis=0)
    cumint = np.vstack([np.zeros((1, 3)), cumint])
    delta_c_r = np.trapezoid(cumint, tau_arr, axis=0)

    return delta_c_v, delta_c_r


# ── 커플링 행렬 ──────────────────────────────────────────────


def compute_coupling_matrices(
    prob,
    P_u: NDArray,
    Re_star: float,
    q_r: NDArray | None = None,
    n_pts: int = 40,
) -> tuple[NDArray, NDArray]:
    """커플링 행렬 M_v, M_r 계산 (보고서 009 식 (11)).

    보정된 위치 곡선에서 Jacobian을 평가하여 드리프트의 제어점 감도를 구한다.
    RK4 ref_traj 대신 Bernstein 위치 평가를 사용하므로 모든 방법에서 호출 가능.

      c_v ≈ c_v_ref + M_v · vec(ΔP_u)
      c_r ≈ c_r_ref + M_r · vec(ΔP_u)

    Parameters
    ----------
    q_r : (N+3, 3) | None — 보정된 위치 제어점. None이면 내부에서 구성.
    n_pts : int — 구적점 수

    Returns
    -------
    M_v : (3, 3(N+1))
    M_r : (3, 3(N+1))
    """
    from ..orbit.dynamics import jacobian_twobody_j2

    N = prob.N
    t_f = prob.t_f
    include_j2 = prob.perturbation_level >= 0
    dim = 3 * (N + 1)

    # 위치 제어점이 없으면 구성
    if q_r is None:
        q_r = position_control_points(P_u, prob.r0, prob.v0, t_f, N)

    Ibar_N = double_int_matrix(N)  # (N+3, N+1)

    tau_arr = np.linspace(0, 1, n_pts)
    dt = tau_arr[1] - tau_arr[0]

    # B_{N+2}(τ) 평가 → b̄(τ) = Ī_N^T · B_{N+2}(τ) 기저
    B_N2_all = bernstein(N + 2, tau_arr)  # (n_pts, N+3)
    bbar_all = np.dot(B_N2_all, Ibar_N)   # (n_pts, N+1)

    # 위치 평가 (Bernstein) — np.dot으로 BLAS spurious 경고 회피
    r_all = np.dot(B_N2_all, q_r)  # (n_pts, 3)

    # 사다리꼴 가중치
    w = np.ones(n_pts) * dt
    w[0] *= 0.5
    w[-1] *= 0.5

    # M_v 누적
    M_v = np.zeros((3, dim))
    K_arr = np.zeros((n_pts, 3, dim))

    for m in range(n_pts):
        x_state = np.zeros(6)
        x_state[:3] = r_all[m]
        A_full = jacobian_twobody_j2(
            x_state, Re_star=Re_star, include_j2=include_j2,
        )
        A_rr = A_full[3:6, :3]  # (3,3)

        bbar_m = bbar_all[m]  # (N+1,)
        K_m = np.kron(A_rr, bbar_m.reshape(1, -1))  # (3, 3(N+1))
        K_arr[m] = K_m
        M_v += w[m] * K_m

    M_v *= t_f

    # M_r: 이중 적분
    cumK = np.cumsum(
        0.5 * (K_arr[:-1] + K_arr[1:]) * dt, axis=0,
    )
    cumK = np.concatenate(
        [np.zeros((1, 3, dim)), cumK], axis=0,
    )
    M_r = np.trapezoid(cumK, tau_arr, axis=0)
    M_r *= t_f

    return M_v, M_r
