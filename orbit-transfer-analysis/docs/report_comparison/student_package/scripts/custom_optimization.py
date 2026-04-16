#!/usr/bin/env python
"""사용자 정의 궤도전이 최적화 실험 스크립트.

이 스크립트는 직접 콜로케이션(Direct Collocation) 기반 최적화 문제의
핵심 구성 요소를 한 파일 안에서 쉽게 수정할 수 있도록 작성되었습니다.

수정 가능한 항목 (각 섹션에 표시됨):
    [A] 임무 파라미터 (궤도 조건, 시간 제약)
    [B] 비용함수 (L², L¹ 근사, 종단 비용 추가 등)
    [C] 물리 모델 (J2 섭동 포함 여부, 중력 상수)
    [D] 이산화 설정 (구간 수, 허용 오차)
    [E] 추가 제약조건 (선택 사항)

실행 예:
    python scripts/custom_optimization.py
    python scripts/custom_optimization.py --compare   # Hohmann 기준 비교 포함
    python scripts/custom_optimization.py --outdir results/my_experiment
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import casadi as ca
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_pkg_root = Path(__file__).parent.parent
sys.path.insert(0, str(_pkg_root / "src"))

from orbit_transfer.constants import MU_EARTH, R_E
from orbit_transfer.types import TransferConfig
from orbit_transfer.astrodynamics.orbital_elements import oe_to_rv_casadi
from orbit_transfer.optimizer.initial_guess import linear_interpolation_guess


# ============================================================
# [A] 임무 파라미터  ← 여기를 수정하세요
# ============================================================
#
# h0          : 초기 원궤도 고도 [km]
# delta_a     : 장반경 변화 [km]  (양수: 고도 증가, 0: 동고도)
# delta_i     : 경사각 변화 [deg] (0: 경사각 유지)
# T_max_normed: 최대 비행시간 / 초기 궤도 주기 T₀
#               T_max_normed = 0.5 → Hohmann 전이와 동일한 시간 상한
#               T_max_normed < 0.5 → Hohmann 불가, 연속 추력만 성공
# u_max       : 최대 추력 가속도 [km/s²]  (예: 0.01 ≈ 10 mN/kg 수준)
# h_min       : 비행 중 최소 허용 고도 [km] (재진입 방지)
# e0, ef      : 초기/목표 궤도 이심률 (0: 원궤도)

MISSION = TransferConfig(
    h0          = 400.0,   # 초기 고도 [km]
    delta_a     = 0.0,     # 장반경 변화 [km]
    delta_i     = 10.0,    # 경사각 변화 [deg]
    T_max_normed= 0.7,     # T_max / T₀
    u_max       = 0.01,    # 최대 추력 가속도 [km/s²]
    h_min       = 200.0,   # 최소 허용 고도 [km]
    e0          = 0.0,     # 초기 이심률
    ef          = 0.0,     # 목표 이심률
)


# ============================================================
# [B] 비용함수 선택  ← 여기를 수정하세요
# ============================================================
#
# "L2"          : Simpson 적분  J = ∫ ||u||²  dt   (에너지 최소화, 기본값)
# "L1_approx"   : J = ∫ sqrt(||u||² + eps²)  dt   (연료 최소화 근사)
#                 eps가 작을수록 실제 L¹에 가까워지지만 수렴이 어려워짐
# "L1_time"     : J = T_f + w * ∫ ||u||²  dt      (시간 + 에너지 혼합)
#                 w_time_energy 로 시간 vs. 에너지 가중치 조절
# "custom"      : 아래 build_objective() 함수를 직접 구현

COST_TYPE = "L2"

# L1_approx용 정규화 파라미터 (작을수록 실제 L¹에 가깝지만 수렴 어려움)
L1_EPS = 1e-4

# L1_time 혼합 시 에너지 가중치 (w): J = T_f  +  w * ∫||u||² dt
W_TIME_ENERGY = 0.5


# ============================================================
# [C] 물리 모델  ← 여기를 수정하세요
# ============================================================
#
# INCLUDE_J2: J2 섭동 포함 여부
#   True  → 지구 편평도 효과 포함 (현실적, 계산 비용 약간 증가)
#   False → 순수 2체 문제 (이상적, 계산 빠름)

INCLUDE_J2 = True


# ============================================================
# [D] 이산화 및 솔버 설정  ← 여기를 수정하세요
# ============================================================
#
# N_SEGMENTS: Hermite-Simpson 구간 수
#   클수록 정확하지만 NLP 규모가 커져 느려짐
#   권장: 20~50 (기본 30)
#
# IPOPT_TOL: IPOPT 허용 오차 (기본 1e-4)
#   작을수록 정밀하지만 수렴 실패율 증가
#
# IPOPT_MAX_ITER: IPOPT 최대 반복 횟수

N_SEGMENTS   = 30
IPOPT_TOL    = 1e-4
IPOPT_MAX_ITER = 500


# ============================================================
# [E] 추가 제약조건  ← 필요한 경우 아래 함수에 추가하세요
# ============================================================
#
# opti  : CasADi Opti 인스턴스
# X     : 상태 행렬  (6 × N_points)  — [rx, ry, rz, vx, vy, vz]
# U     : 제어 행렬  (3 × N_points)  — [ux, uy, uz]
# T_f   : 비행시간 결정변수 (스칼라)
# N     : N_points = 2*N_SEGMENTS + 1
#
# 예시:
#   # 최대 속도 제한 (지구 재진입 속도 방지)
#   V_MAX = 10.0  # km/s
#   for k in range(N):
#       opti.subject_to(ca.dot(X[3:, k], X[3:, k]) <= V_MAX**2)
#
#   # 비행시간 하한 추가
#   opti.subject_to(T_f >= 500.0)  # 최소 500초

def add_extra_constraints(opti, X, U, T_f, N):
    """추가 제약조건을 여기에 구현하세요. 기본은 빈 함수(추가 없음)."""
    pass   # ← 이 아래에 원하는 제약조건을 추가하세요


# ============================================================
# 핵심 구현부 — 아래는 수정하지 않아도 됩니다
# (단, 수식 이해 후 수정 가능)
# ============================================================

def build_objective(opti, U, h, M, T_f):
    """비용함수 구성.

    [B] COST_TYPE에 따라 비용함수를 반환한다.
    """
    J = 0
    for k in range(M):
        idx_k  = 2 * k
        idx_m  = 2 * k + 1
        idx_k1 = 2 * k + 2
        u_k  = U[:, idx_k]
        u_m  = U[:, idx_m]
        u_k1 = U[:, idx_k1]

        if COST_TYPE == "L2":
            # J = ∫ ||u||² dt  (Simpson 적분)
            J += (h / 6.0) * (
                ca.dot(u_k, u_k) + 4 * ca.dot(u_m, u_m) + ca.dot(u_k1, u_k1)
            )

        elif COST_TYPE == "L1_approx":
            # J = ∫ sqrt(||u||² + ε²) dt  (L¹ 근사)
            eps2 = L1_EPS ** 2
            J += (h / 6.0) * (
                ca.sqrt(ca.dot(u_k, u_k)  + eps2)
                + 4 * ca.sqrt(ca.dot(u_m, u_m)  + eps2)
                + ca.sqrt(ca.dot(u_k1, u_k1) + eps2)
            )

        elif COST_TYPE == "L1_time":
            # J = T_f + w * ∫ ||u||² dt
            J += (h / 6.0) * W_TIME_ENERGY * (
                ca.dot(u_k, u_k) + 4 * ca.dot(u_m, u_m) + ca.dot(u_k1, u_k1)
            )
        elif COST_TYPE == "custom":
            # ── [B-custom] 직접 구현 구역 ───────────────────────────────────
            # 여기에 원하는 비용함수를 구현하세요.
            # 예: 종단 비용 + 적분 비용 혼합
            J += (h / 6.0) * (
                ca.dot(u_k, u_k) + 4 * ca.dot(u_m, u_m) + ca.dot(u_k1, u_k1)
            )
            # ─────────────────────────────────────────────────────────────────
        else:
            raise ValueError(f"알 수 없는 COST_TYPE: {COST_TYPE!r}")

    # L1_time: 비행시간 자체도 비용에 포함
    if COST_TYPE == "L1_time":
        J = T_f + J

    return J


def solve_custom(config: TransferConfig) -> dict:
    """사용자 설정으로 Hermite-Simpson Collocation NLP를 구성하고 풀기."""
    from orbit_transfer.dynamics.eom import create_dynamics_function

    M = N_SEGMENTS
    N = 2 * M + 1  # collocation points

    # 동역학 함수 ([C] INCLUDE_J2)
    eom = create_dynamics_function(mu=MU_EARTH, include_j2=INCLUDE_J2)

    opti = ca.Opti()

    # 결정변수
    X   = opti.variable(6, N)   # 상태  [rx ry rz vx vy vz]
    U   = opti.variable(3, N)   # 제어  [ux uy uz]
    nu0 = opti.variable()        # 출발 true anomaly [rad]
    nuf = opti.variable()        # 도착 true anomaly [rad]
    T_f = opti.variable()        # 비행시간 [s]

    # 비행시간 범위 제약
    opti.subject_to(T_f >= config.T_min)
    opti.subject_to(T_f <= config.T_max)
    h_step = T_f / M             # 심볼릭 구간 폭

    # ─ 비용함수 ([B]) ────────────────────────────────────────────────────────
    J = build_objective(opti, U, h_step, M, T_f)
    opti.minimize(J)

    # ─ Hermite-Simpson 연속성 제약 ───────────────────────────────────────────
    for k in range(M):
        i_k  = 2 * k
        i_m  = 2 * k + 1
        i_k1 = 2 * k + 2

        x_k  = X[:, i_k];  u_k  = U[:, i_k]
        x_m  = X[:, i_m];  u_m  = U[:, i_m]
        x_k1 = X[:, i_k1]; u_k1 = U[:, i_k1]

        f_k  = eom(x_k,  u_k)
        f_m  = eom(x_m,  u_m)
        f_k1 = eom(x_k1, u_k1)

        # Simpson continuity: x_{k+1} = x_k + (h/6)(f_k + 4f_m + f_{k+1})
        opti.subject_to(
            x_k1 == x_k + (h_step / 6.0) * (f_k + 4 * f_m + f_k1)
        )
        # Hermite midpoint: x_m = 0.5*(x_k+x_{k+1}) + (h/8)*(f_k - f_{k+1})
        opti.subject_to(
            x_m == 0.5 * (x_k + x_k1) + (h_step / 8.0) * (f_k - f_k1)
        )

    # ─ 경계조건 (출발·도착 ECI 상태벡터) ────────────────────────────────────
    oe0 = ca.vertcat(config.a0, config.e0, config.i0, 0.0, 0.0, nu0)
    r0, v0 = oe_to_rv_casadi(oe0, MU_EARTH)
    opti.subject_to(X[:, 0] == ca.vertcat(r0, v0))

    oef = ca.vertcat(config.af, config.ef, config.if_, 0.0, 0.0, nuf)
    rf, vf = oe_to_rv_casadi(oef, MU_EARTH)
    opti.subject_to(X[:, -1] == ca.vertcat(rf, vf))

    # ─ 추력 상한: ||u_k||² ≤ u_max² ────────────────────────────────────────
    for k in range(N):
        opti.subject_to(ca.dot(U[:, k], U[:, k]) <= config.u_max ** 2)

    # ─ 최소 고도: ||r_k|| ≥ R_E + h_min ────────────────────────────────────
    r_min_sq = (R_E + config.h_min) ** 2
    for k in range(N):
        opti.subject_to(ca.dot(X[:3, k], X[:3, k]) >= r_min_sq)

    # ─ 추가 제약조건 ([E]) ───────────────────────────────────────────────────
    add_extra_constraints(opti, X, U, T_f, N)

    # ─ 초기값 설정 ───────────────────────────────────────────────────────────
    t_g, x_g, u_g, nu0_g, nuf_g = linear_interpolation_guess(config, N)
    opti.set_initial(X, x_g)
    opti.set_initial(U, u_g)
    opti.set_initial(T_f, config.T_max)
    opti.set_initial(nu0, nu0_g)
    opti.set_initial(nuf, nuf_g)

    # ─ IPOPT 솔버 ([D]) ──────────────────────────────────────────────────────
    opts = {
        "ipopt.tol":              IPOPT_TOL,
        "ipopt.constr_viol_tol":  IPOPT_TOL,
        "ipopt.max_iter":         IPOPT_MAX_ITER,
        "ipopt.linear_solver":    "mumps",
        "ipopt.mu_strategy":      "adaptive",
        "ipopt.print_level":      0,
    }
    opti.solver("ipopt", opts)

    # ─ 풀기 ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        converged = True
        cost   = float(sol.value(J))
        T_val  = float(sol.value(T_f))
        x_sol  = np.array(sol.value(X))
        u_sol  = np.array(sol.value(U))
        nu0_v  = float(sol.value(nu0))
        nuf_v  = float(sol.value(nuf))
    except RuntimeError as e:
        converged = False
        cost   = float("inf")
        T_val  = float(opti.debug.value(T_f)) or config.T_max
        if T_val <= 0:
            T_val = config.T_max
        x_sol  = np.array(opti.debug.value(X))
        u_sol  = np.array(opti.debug.value(U))
        nu0_v  = float(opti.debug.value(nu0))
        nuf_v  = float(opti.debug.value(nuf))
        print(f"  [경고] 수렴 실패: {e}")
    elapsed = time.perf_counter() - t0

    t_arr  = np.linspace(0, T_val, N)
    u_mag  = np.linalg.norm(u_sol, axis=0)

    # 비용 지표 계산
    # L1 비용 (Simpson 적분 of ||u||)
    dt = T_val / M
    l1 = 0.0
    for k in range(M):
        i_k = 2*k; i_m = 2*k+1; i_k1 = 2*k+2
        l1 += (dt / 6.0) * (u_mag[i_k] + 4*u_mag[i_m] + u_mag[i_k1])

    # L2 비용 (Simpson 적분 of ||u||²)
    l2 = 0.0
    for k in range(M):
        i_k = 2*k; i_m = 2*k+1; i_k1 = 2*k+2
        l2 += (dt / 6.0) * (u_mag[i_k]**2 + 4*u_mag[i_m]**2 + u_mag[i_k1]**2)

    T0 = config.T0
    return {
        "converged": converged,
        "cost_raw":  cost,
        "cost_l1":   l1,
        "cost_l2":   l2,
        "T_f":       T_val,
        "T_f_norm":  T_val / T0,
        "nu0_deg":   np.degrees(nu0_v),
        "nuf_deg":   np.degrees(nuf_v),
        "elapsed":   elapsed,
        "t":         t_arr,
        "x":         x_sol,
        "u":         u_sol,
        "u_mag":     u_mag,
        "N":         N,
    }


def run_hohmann_reference(config: TransferConfig) -> dict:
    """참조용 Hohmann 전이 결과."""
    from orbit_transfer.benchmark import TransferBenchmark
    bm = TransferBenchmark(config)
    bm.run_hohmann()
    res = bm.results.get("hohmann")
    if res is None or not res.converged:
        return {}
    m = res.metrics
    return {
        "dv_total":  m.get("dv_total", float("nan")),
        "tof":       m.get("tof", float("nan")),
        "tof_norm":  m.get("tof_norm", float("nan")),
    }


def plot_results(result: dict, config: TransferConfig,
                 hohmann: dict | None, outpath: str) -> None:
    """최적화 결과 시각화."""
    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    t  = result["t"]
    x  = result["x"]
    u  = result["u"]
    um = result["u_mag"]
    T_f = result["T_f"]
    T0  = config.T0
    t_n = t / T_f   # 정규화 시간

    # ── (a) 추력 크기 프로파일 ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_n, um * 1e3, color="#4CAF50", lw=1.6)
    ax1.fill_between(t_n, 0, um * 1e3, color="#4CAF50", alpha=0.18)
    ax1.set_xlabel(r"Normalized time $t/T_f$")
    ax1.set_ylabel(r"$\|\mathbf{u}\|$ [$\times 10^{-3}$ km/s²]")
    ax1.set_title("(a) Thrust magnitude profile")
    ax1.set_xlim(0, 1); ax1.set_ylim(bottom=0)

    # 비용 정보
    info = (f"$L^1$ = {result['cost_l1']:.4f} km/s\n"
            f"$L^2$ = {result['cost_l2']:.5f}\n"
            f"$T_f/T_0$ = {result['T_f_norm']:.3f}\n"
            f"Cost type: {COST_TYPE}")
    ax1.text(0.98, 0.97, info, transform=ax1.transAxes,
             ha="right", va="top", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

    # ── (b) 추력 3성분 ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, (lbl, col) in enumerate(zip(
        ["$u_x$", "$u_y$", "$u_z$"],
        ["#E53935", "#1565C0", "#558B2F"]
    )):
        ax2.plot(t_n, u[idx] * 1e3, color=col, lw=1.2, label=lbl)
    ax2.axhline(0, color="gray", lw=0.6, ls=":")
    ax2.set_xlabel(r"$t/T_f$")
    ax2.set_ylabel(r"$u_i$ [$\times 10^{-3}$ km/s²]")
    ax2.set_title("(b) Thrust components")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(0, 1)

    # ── (c) 고도 변화 ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    r_km = np.linalg.norm(x[:3], axis=0)
    h_km = r_km - R_E
    ax3.plot(t_n, h_km, color="#FF9800", lw=1.6)
    ax3.axhline(config.h_min, color="#E53935", ls="--", lw=1.0,
                label=f"$h_{{min}}$ = {config.h_min:.0f} km")
    ax3.set_xlabel(r"$t/T_f$")
    ax3.set_ylabel("Altitude [km]")
    ax3.set_title("(c) Altitude profile")
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 1)

    # ── (d) 3D 궤적 ───────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2], projection="3d")
    rx, ry, rz = x[0], x[1], x[2]
    ax4.plot(rx, ry, rz, color="#4CAF50", lw=1.6, label="Collocation (custom)", zorder=3)
    ax4.scatter(rx[0],  ry[0],  rz[0],  color="#4CAF50", s=50, marker="o")
    ax4.scatter(rx[-1], ry[-1], rz[-1], color="#4CAF50", s=70, marker="*")

    # 지구 구체
    u_ = np.linspace(0, 2*np.pi, 30)
    v_ = np.linspace(0, np.pi, 30)
    xs = R_E * np.outer(np.cos(u_), np.sin(v_))
    ys = R_E * np.outer(np.sin(u_), np.sin(v_))
    zs = R_E * np.outer(np.ones(30), np.cos(v_))
    ax4.plot_surface(xs, ys, zs, color="#1565C0", alpha=0.15, linewidth=0)

    ax4.set_xlabel("X [km]", labelpad=3)
    ax4.set_ylabel("Y [km]", labelpad=3)
    ax4.set_zlabel("Z [km]", labelpad=3)
    ax4.set_title("(d) Transfer trajectory (ECI frame)", pad=6)
    ax4.tick_params(labelsize=7)
    ax4.legend(fontsize=8)

    # ── (e) 비용 비교 막대 ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    labels_bar = ["Collocation\n(custom)"]
    vals_bar   = [result["cost_l1"]]
    colors_bar = ["#4CAF50"]

    if hohmann:
        labels_bar.append("Hohmann\n(reference)")
        vals_bar.append(hohmann.get("dv_total", float("nan")))
        colors_bar.append("#2196F3")

    bars = ax5.bar(range(len(labels_bar)), vals_bar, color=colors_bar,
                   alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals_bar):
        if np.isfinite(v):
            ax5.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax5.set_xticks(range(len(labels_bar)))
    ax5.set_xticklabels(labels_bar, fontsize=8.5)
    ax5.set_ylabel("$L^1$ cost / $\\Delta v$ [km/s]")
    ax5.set_title("(e) Cost comparison")

    # ── 공통 제목 ─────────────────────────────────────────────────────────────
    conv_str = "CONVERGED" if result["converged"] else "FAILED"
    status   = f"[{conv_str}]  J2={'on' if INCLUDE_J2 else 'off'}  M={N_SEGMENTS}"
    fig.suptitle(
        f"Custom Optimization  —  "
        f"h₀={config.h0:.0f} km, Δa={config.delta_a:.0f} km, "
        f"Δi={config.delta_i:.1f}°,  "
        f"$T_{{max}}/T_0$={config.T_max_normed:.2f}  |  {status}",
        fontsize=10, y=1.01,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[plot] 저장: {outpath}")


def print_summary(result: dict, hohmann: dict | None) -> None:
    """결과 요약 출력."""
    print("\n" + "=" * 55)
    print("  최적화 결과 요약")
    print("=" * 55)
    status = "수렴 (OK)" if result["converged"] else "수렴 실패 (FAILED)"
    print(f"  상태          : {status}")
    print(f"  비용함수 유형  : {COST_TYPE}")
    print(f"  원시 비용 (J) : {result['cost_raw']:.6f}")
    print(f"  L¹ 비용       : {result['cost_l1']:.6f} km/s")
    print(f"  L² 비용       : {result['cost_l2']:.6f} km²/s³")
    print(f"  비행시간 T_f  : {result['T_f']:.1f} s  ({result['T_f_norm']:.4f} T₀)")
    print(f"  출발 이각 ν₀  : {result['nu0_deg']:.2f}°")
    print(f"  도착 이각 ν_f : {result['nuf_deg']:.2f}°")
    print(f"  계산 시간     : {result['elapsed']:.2f} s")
    if hohmann:
        print("-" * 55)
        dv  = hohmann.get("dv_total", float("nan"))
        tof = hohmann.get("tof_norm", float("nan"))
        print(f"  [참조] Hohmann Δv        : {dv:.6f} km/s")
        print(f"  [참조] Hohmann T_f/T₀    : {tof:.4f}")
        if np.isfinite(dv) and dv > 0:
            ratio = result["cost_l1"] / dv
            print(f"  Collocation L¹ / Hoh Δv : {ratio:.4f}")
    print("=" * 55 + "\n")


def export_csv(result: dict, outdir: str) -> None:
    """결과를 CSV로 저장."""
    import csv

    # 궤적 CSV
    traj_path = os.path.join(outdir, "custom_trajectory.csv")
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t[s]", "x[km]", "y[km]", "z[km]",
                         "vx[km/s]", "vy[km/s]", "vz[km/s]",
                         "ux[km/s2]", "uy[km/s2]", "uz[km/s2]",
                         "u_mag[km/s2]"])
        x = result["x"]; u = result["u"]; um = result["u_mag"]
        for k in range(result["N"]):
            writer.writerow([
                f"{result['t'][k]:.4f}",
                f"{x[0,k]:.6f}", f"{x[1,k]:.6f}", f"{x[2,k]:.6f}",
                f"{x[3,k]:.6f}", f"{x[4,k]:.6f}", f"{x[5,k]:.6f}",
                f"{u[0,k]:.8e}", f"{u[1,k]:.8e}", f"{u[2,k]:.8e}",
                f"{um[k]:.8e}",
            ])

    # 지표 CSV
    metrics_path = os.path.join(outdir, "custom_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "converged", "cost_type", "cost_raw", "cost_l1", "cost_l2",
            "T_f_s", "T_f_norm", "nu0_deg", "nuf_deg",
            "N_segments", "include_j2", "elapsed_s",
        ])
        writer.writeheader()
        writer.writerow({
            "converged":  result["converged"],
            "cost_type":  COST_TYPE,
            "cost_raw":   f"{result['cost_raw']:.8e}",
            "cost_l1":    f"{result['cost_l1']:.8e}",
            "cost_l2":    f"{result['cost_l2']:.8e}",
            "T_f_s":      f"{result['T_f']:.4f}",
            "T_f_norm":   f"{result['T_f_norm']:.6f}",
            "nu0_deg":    f"{result['nu0_deg']:.4f}",
            "nuf_deg":    f"{result['nuf_deg']:.4f}",
            "N_segments": N_SEGMENTS,
            "include_j2": INCLUDE_J2,
            "elapsed_s":  f"{result['elapsed']:.3f}",
        })

    print(f"[export] 궤적 CSV: {traj_path}")
    print(f"[export] 지표 CSV: {metrics_path}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="사용자 정의 궤도전이 최적화 실험",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--compare", action="store_true",
                        help="Hohmann 기준 결과와 비교")
    parser.add_argument("--outdir", default="results/custom",
                        help="결과 저장 디렉토리 (기본: results/custom)")
    parser.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"],
                        help="그림 저장 형식")
    args = parser.parse_args()

    config = MISSION
    os.makedirs(args.outdir, exist_ok=True)

    print("\n" + "=" * 55)
    print("  사용자 정의 직접 콜로케이션 최적화")
    print("=" * 55)
    print(f"  h₀       = {config.h0:.0f} km")
    print(f"  Δa       = {config.delta_a:.0f} km")
    print(f"  Δi       = {config.delta_i:.1f}°")
    print(f"  T_max/T₀ = {config.T_max_normed:.2f}  ({config.T_max:.0f} s)")
    print(f"  u_max    = {config.u_max:.4f} km/s²")
    print(f"  비용함수  = {COST_TYPE}")
    print(f"  J2 섭동  = {INCLUDE_J2}")
    print(f"  N 구간   = {N_SEGMENTS}")
    print(f"  IPOPT tol= {IPOPT_TOL}")
    print()

    # 최적화 실행
    print("최적화 실행 중...")
    result = solve_custom(config)

    # Hohmann 기준 계산
    hohmann = None
    if args.compare:
        print("Hohmann 기준 계산 중...")
        hohmann = run_hohmann_reference(config)

    # 결과 출력
    print_summary(result, hohmann)

    # CSV 저장
    export_csv(result, args.outdir)

    # 그림 저장
    figpath = os.path.join(args.outdir, f"custom_result.{args.fmt}")
    plot_results(result, config, hohmann, figpath)

    print(f"\n완료. 결과 위치: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
