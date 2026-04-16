"""SCP 문제 정의 및 경계조건 선형 제약 구성.

이론: docs/reports/006_scp_formulation/

SCP 문제 데이터:
- 초기/종말 상태 (정규화)
- 비행시간 t_f
- 베지어 차수 N
- 섭동 레벨
- 추력 제약 옵션

경계조건 → 적분 행렬을 통해 제어점의 선형 등식 제약으로 변환.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..normalize import CanonicalUnits
from ..bezier.basis import int_matrix, double_int_matrix, gram_matrix, bernstein
from .drift import DriftConfig


@dataclasses.dataclass
class SCPProblem:
    """SCP 궤도전이 최적화 문제 정의.

    모든 값은 정규화된 단위.
    """

    # 경계조건
    r0: NDArray          # (3,) 초기 위치
    v0: NDArray          # (3,) 초기 속도
    rf: NDArray          # (3,) 종말 위치
    vf: NDArray          # (3,) 종말 속도

    # 비행시간 (outer loop에서 결정, inner loop에서 고정)
    t_f: float

    # 베지어 차수
    N: int = 12

    # 추력 제약
    u_max: float | None = None  # None이면 추력 무제약 (QP)

    # 섭동 레벨
    perturbation_level: int = 0  # 0: J2 only, 1: J3-J6+drag, 2: SRP+3body

    # 정규화 단위 (물리 단위 변환용)
    canonical_units: CanonicalUnits | None = None

    # SOCP 이산화 격자 수 (추력 제약용)
    thrust_grid_M: int | None = None  # None → 2N

    # ── Level 1 파라미터 (J3–J6, 대기항력) ────────────────────
    max_jn_degree: int = 6                   # 최고 Jn 차수
    Cd_A_over_m: float = 0.01               # 항력계수×면적/질량 [m²/kg]

    # ── Level 2 파라미터 (SRP, 3체 섭동) ──────────────────────
    Cr_A_over_m: float = 0.01               # SRP 계수×면적/질량 [m²/kg]
    r_sun_func: Callable[[float], NDArray] | None = None   # τ → r_sun* (정규화)
    r_moon_func: Callable[[float], NDArray] | None = None  # τ → r_moon* (정규화)
    mu_sun_star: float = 0.0                # 정규화된 태양 중력상수
    mu_moon_star: float = 0.0               # 정규화된 달 중력상수

    # SCP 파라미터
    max_iter: int = 20
    tol_ctrl: float = 1e-6      # 제어점 변화 수렴 허용치
    tol_bc: float = 1e-6        # 경계조건 위반 허용치

    # Trust region
    trust_region: float = 10.0  # 초기 trust region 반경
    trust_expand: float = 1.5   # BC 개선 시 확대 비율
    trust_shrink: float = 0.5   # BC 악화 시 축소 비율
    trust_min: float = 0.01     # 최소 trust region

    # 완화 계수 (step damping)
    relax_alpha: float = 0.4    # Z_new = (1-α)·Z_prev + α·Z_qp

    # 드리프트 적분 계산 방법 (디폴트: Bernstein 대수)
    drift_config: DriftConfig = dataclasses.field(default_factory=DriftConfig)

    # 경로 제약 (궤도 반경 부등식)
    r_min: float | None = None  # 궤도 반경 하한 (정규화), None이면 무제약
    r_max: float | None = None  # 궤도 반경 상한 (정규화)
    path_K_subdiv: int = 8      # 경로 제약 드 카스텔조 세분화 구간 수
    path_M_sub: int = 3         # 구간당 평가점 수

    @property
    def has_path_constraints(self) -> bool:
        """경로 제약이 있는지."""
        return self.r_min is not None or self.r_max is not None

    @property
    def is_socp(self) -> bool:
        """추력 제약 있으면 SOCP, 없으면 QP."""
        return self.u_max is not None


def build_boundary_constraints(
    prob: SCPProblem,
    *,
    c_v: NDArray | None = None,
    c_r: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """경계조건을 제어점의 선형 등식 제약으로 변환.

    보고서 004 식 (21)–(22):
      v_f = v_0 + t_f·c_v + t_f · e_{N+1}^T · I_N · P_u
      r_f = r_0 + t_f·v_0 + t_f·c_r + t_f² · e_{N+2}^T · Ī_N · P_u

    정리하면:
      A_eq @ vec(P_u) = b_eq   (6 equations, 3(N+1) unknowns)

    Parameters
    ----------
    prob : SCPProblem
    c_v : (3,) 참조 궤도의 속도 적분 기여. 기본값 0 (케플러 초기화).
    c_r : (3,) 참조 궤도의 위치 적분 기여. 기본값 0.

    Returns
    -------
    A_eq : (6, 3*(N+1))
    b_eq : (6,)
    """
    N = prob.N
    t_f = prob.t_f

    I_N = int_matrix(N)             # (N+2, N+1)
    Ibar_N = double_int_matrix(N)   # (N+3, N+1)

    # e_{N+1}^T @ I_N → (N+1,)  (적분 행렬의 마지막 행)
    eI = I_N[N + 1, :]     # (N+1,)
    # e_{N+2}^T @ Ī_N → (N+1,)
    eIbar = Ibar_N[N + 2, :]  # (N+1,)

    if c_v is None:
        c_v = np.zeros(3)
    if c_r is None:
        c_r = np.zeros(3)

    # 속도 경계조건: t_f · eI @ P_u[:,k] = (vf_k - v0_k) - t_f·c_v_k
    # 위치 경계조건: t_f² · eIbar @ P_u[:,k] = (rf_k - r0_k) - t_f·v0_k - t_f·c_r_k

    dim = 3 * (N + 1)
    A_eq = np.zeros((6, dim))
    b_eq = np.zeros(6)

    for k in range(3):
        col_start = k * (N + 1)
        col_end = col_start + (N + 1)

        # 속도 조건 (행 k)
        A_eq[k, col_start:col_end] = t_f * eI
        b_eq[k] = (prob.vf[k] - prob.v0[k]) - t_f * c_v[k]

        # 위치 조건 (행 3+k)
        A_eq[3 + k, col_start:col_end] = t_f**2 * eIbar
        b_eq[3 + k] = (prob.rf[k] - prob.r0[k]) - t_f * prob.v0[k] - t_f**2 * c_r[k]

    return A_eq, b_eq


def build_lifted_boundary_constraints(
    prob: SCPProblem,
    *,
    c_v: NDArray | None = None,
    c_r: NDArray | None = None,
    M_v: NDArray | None = None,
    M_r: NDArray | None = None,
    Z_ref: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """보조변수 승강 Z = t_f · P_u 를 사용한 경계조건.

    보고서 007 식 (3)–(4):
      v_f = v_0 + t_f·c_v + eI @ Z[:,k]
      r_f = r_0 + t_f·v_0 + t_f·c_r + t_f · eIbar @ Z[:,k]

    커플링 행렬 (보고서 009) 제공 시, 드리프트 감도 반영:
      c_v ≈ c_v_ref + M_v · vec(ΔP_u)
      → 속도: (eI + M_v[k,:]) · Z = ... + M_v · vec(Z_ref)

    A_eq @ vec(Z) = b_eq

    Parameters
    ----------
    M_v, M_r : (3, 3(N+1)), optional
        커플링 행렬. None이면 0차 근사 (기존 동작).
    Z_ref : (N+1, 3), optional
        참조 보조변수 (M_v/M_r 사용 시 필수).

    Returns
    -------
    A_eq : (6, 3*(N+1))
    b_eq : (6,)
    """
    N = prob.N
    t_f = prob.t_f

    I_N = int_matrix(N)
    Ibar_N = double_int_matrix(N)

    eI = I_N[N + 1, :]
    eIbar = Ibar_N[N + 2, :]

    if c_v is None:
        c_v = np.zeros(3)
    if c_r is None:
        c_r = np.zeros(3)

    dim = 3 * (N + 1)
    A_eq = np.zeros((6, dim))
    b_eq = np.zeros(6)

    for k in range(3):
        col_start = k * (N + 1)
        col_end = col_start + (N + 1)

        # 속도: eI @ Z_k = (vf_k - v0_k) - t_f·c_v_k
        A_eq[k, col_start:col_end] = eI
        b_eq[k] = (prob.vf[k] - prob.v0[k]) - t_f * c_v[k]

        # 위치: t_f · eIbar @ Z_k = (rf_k - r0_k) - t_f·v0_k - t_f²·c_r_k
        A_eq[3 + k, col_start:col_end] = t_f * eIbar
        b_eq[3 + k] = (prob.rf[k] - prob.r0[k]) - t_f * prob.v0[k] - t_f**2 * c_r[k]

    # 커플링 행렬 반영 (보고서 009 식 (19))
    # 속도: (eI + M_v[k,:]/t_f) · Z = b + M_v · vec(Z_ref)/t_f
    # 위치: (t_f·eIbar + M_r[k,:]) · Z = b + M_r · vec(Z_ref)
    if M_v is not None and Z_ref is not None:
        z_ref_vec = np.concatenate([Z_ref[:, k] for k in range(3)])
        for k in range(3):
            # M_v acts on vec(P_u), and P_u = Z/t_f
            # Δc_v = M_v · vec(ΔP_u) = M_v · vec(ΔZ)/t_f
            # t_f·Δc_v = M_v · vec(ΔZ)
            # 속도: eI · Z_k + M_v[k,:] · vec(Z) = ... + M_v[k,:] · vec(Z_ref)
            A_eq[k, :] += M_v[k, :]
            b_eq[k] += M_v[k, :] @ z_ref_vec

    if M_r is not None and Z_ref is not None:
        z_ref_vec = np.concatenate([Z_ref[:, k] for k in range(3)])
        for k in range(3):
            # Δc_r = M_r · vec(ΔP_u) = M_r · vec(ΔZ)/t_f
            # t_f²·Δc_r = t_f · M_r · vec(ΔZ)
            # 위치: t_f·eIbar · Z_k + t_f·M_r[k,:] · vec(Z) = ... + t_f·M_r[k,:] · vec(Z_ref)
            A_eq[3 + k, :] += t_f * M_r[k, :]
            b_eq[3 + k] += t_f * M_r[k, :] @ z_ref_vec

    return A_eq, b_eq
