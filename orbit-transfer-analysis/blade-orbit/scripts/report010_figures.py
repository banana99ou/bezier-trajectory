#!/usr/bin/env python3
"""보고서 010 수치 예제 및 그림 생성.

Bernstein 대수를 통한 중력 합성 파이프라인 검증.
"""

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bezier_orbit.bezier.basis import (
    bernstein, int_matrix, double_int_matrix, gram_matrix, definite_integral,
)
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.orbit.dynamics import propagate_rk4
from bezier_orbit.normalize import J2_EARTH
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "reports" / "010_bernstein_algebra"

# ── Bernstein 대수 함수 ──────────────────────────────────────

def bern_product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """두 Bernstein 다항식의 곱의 제어점 계산.

    f(τ)∈B_N, g(τ)∈B_M → (f·g)(τ)∈B_{N+M}
    식: (f·g)_k = Σ C(N,j)C(M,k-j)/C(N+M,k) · f_j · g_{k-j}
    """
    N = len(p) - 1
    M = len(q) - 1
    result = np.zeros(N + M + 1)
    for k in range(N + M + 1):
        j_min = max(0, k - M)
        j_max = min(N, k)
        for j in range(j_min, j_max + 1):
            coeff = math.comb(N, j) * math.comb(M, k - j) / math.comb(N + M, k)
            result[k] += coeff * p[j] * q[k - j]
    return result


def bern_compose(h: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Bernstein 합성 h(g(τ)) 계산.

    h(s)∈B_K (s∈[0,1]), g(τ)∈B_M (g:[0,1]→[0,1])
    결과: B_{KM}차

    방법: h(g) = Σ h_i C(K,i) g^i (1-g)^{K-i}
    각 항을 순차 Bernstein 곱으로 계산.
    """
    K = len(h) - 1
    M = len(g) - 1

    # g의 거듭제곱 사전 계산: g^0, g^1, ..., g^K
    one_minus_g = -g.copy()
    one_minus_g[0] += 1.0  # (1-g)의 제어점: 1 - g_i only works for degree 0
    # 올바른 방법: (1-g)(τ) 의 Bernstein 표현
    # 1(τ) ∈ B_M: 모든 제어점이 1
    ones = np.ones(M + 1)
    one_minus_g = ones - g  # 제어점 빼기 (같은 차수이므로 유효)

    # g^i 계산 (순차 곱)
    g_powers = [np.ones(1)]  # g^0 = 1 ∈ B_0
    for i in range(1, K + 1):
        g_powers.append(bern_product(g_powers[-1], g))

    # (1-g)^i 계산
    omg_powers = [np.ones(1)]  # (1-g)^0 = 1 ∈ B_0
    for i in range(1, K + 1):
        omg_powers.append(bern_product(omg_powers[-1], one_minus_g))

    # 합산: h(g) = Σ h_i C(K,i) g^i (1-g)^{K-i}
    # 각 항은 B_{iM + (K-i)M} = B_{KM}차이지만 곱 과정에서 차수가 다를 수 있음
    # → 차수 올림(degree elevation)으로 통일
    target_deg = K * M
    result = np.zeros(target_deg + 1)

    for i in range(K + 1):
        term = bern_product(g_powers[i], omg_powers[K - i])
        # 차수 올림 필요 시
        term = degree_elevate(term, target_deg)
        result += h[i] * math.comb(K, i) * term

    return result


def degree_elevate(p: np.ndarray, target_deg: int) -> np.ndarray:
    """Bernstein 차수 올림: N차 → target_deg차."""
    N = len(p) - 1
    if N == target_deg:
        return p.copy()
    if N > target_deg:
        raise ValueError(f"Cannot elevate from degree {N} to {target_deg}")

    q = p.copy()
    for _ in range(target_deg - N):
        n = len(q) - 1
        new_q = np.zeros(n + 2)
        for i in range(n + 2):
            if i == 0:
                new_q[i] = q[0]
            elif i == n + 1:
                new_q[i] = q[n]
            else:
                alpha = i / (n + 1)
                new_q[i] = alpha * q[i - 1] + (1 - alpha) * q[i]
        q = new_q
    return q


def degree_reduce_l2(p: np.ndarray, R: int) -> np.ndarray:
    """L² 최적 차수 축소: Q차 → R차.

    min ∫₀¹ |p(τ) - q(τ)|² dτ
    해: q = G_R^{-1} C_{Q→R} p
    """
    Q = len(p) - 1
    if R >= Q:
        return degree_elevate(p, R) if R > Q else p.copy()

    # Gram 행렬 G_R
    G_R = gram_matrix(R)

    # 교차 Gram 행렬 C_{Q→R}: [C]_{i,j} = C(R,i)C(Q,j) / [C(R+Q,i+j)(R+Q+1)]
    C = np.zeros((R + 1, Q + 1))
    for i in range(R + 1):
        for j in range(Q + 1):
            C[i, j] = math.comb(R, i) * math.comb(Q, j) / (math.comb(R + Q, i + j) * (R + Q + 1))

    return np.linalg.solve(G_R, C @ p)


def chebyshev_to_bernstein_approx(func, K, a=0.0, b=1.0):
    """함수를 [a,b]에서 K차 Bernstein으로 근사.

    방법: Chebyshev 노드에서 샘플 → Bernstein 기저로 최소자승 피팅.
    (정확한 Chebyshev→Bernstein 변환 대신 실용적 접근)
    """
    # K+1개 Chebyshev 노드 ([0,1]으로 매핑)
    M_sample = max(3 * K, 30)  # 오버샘플링
    nodes_01 = 0.5 * (1 - np.cos(np.pi * np.arange(M_sample + 1) / M_sample))

    # 함수 값
    s_nodes = a + (b - a) * nodes_01
    f_vals = np.array([func(s) for s in s_nodes])

    # Bernstein 기저 행렬 (M_sample+1, K+1)
    B = bernstein(K, nodes_01)

    # 최소자승
    coeffs, _, _, _ = np.linalg.lstsq(B, f_vals, rcond=None)
    return coeffs


def position_control_points(P_u, r0, v0, t_f, N):
    """P_u로부터 위치 궤적의 Bernstein 제어점 계산.

    r(τ) = r₀ + t_f·v₀·τ + t_f²·B_{N+2}(τ)^T·Ī_N·P_u  (각 성분)

    반환: (N+3, 3) 제어점 행렬
    """
    Ibar = double_int_matrix(N)  # (N+3, N+1)

    # τ의 (N+2)차 Bernstein 제어점: ℓ_i = i/(N+2)
    ell = np.arange(N + 3) / (N + 2)

    q_r = np.zeros((N + 3, 3))
    for k in range(3):
        q_r[:, k] = r0[k] + t_f * v0[k] * ell + t_f**2 * Ibar @ P_u[:, k]

    return q_r


def eval_bernstein(p, tau_arr):
    """Bernstein 곡선 평가."""
    N = len(p) - 1
    B = bernstein(N, tau_arr)  # (len(tau), N+1)
    return B @ p


# ── 시나리오 설정 ─────────────────────────────────────────────

def get_scenario_b():
    """시나리오 B: LEO → 1.2 a0, t_f=3.62, N=12."""
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
    return SCPProblem(
        r0=r0, v0=v0, rf=rf, vf=vf,
        t_f=3.62, N=12,
        u_max=None, max_iter=50,
        tol_ctrl=1e-6, tol_bc=1e-4,
    )


# ── Fig 1: h(s) = s^{-3/2} Bernstein 근사 ────────────────────

def fig_h_approx():
    """s^{-3/2}의 K차 Bernstein 근사와 수렴."""
    print("Generating fig_h_approx...")

    # 시나리오 B 기준 r 범위
    a, b = 0.85**2, 1.25**2  # r_min^2, r_max^2 (근사)

    h_exact = lambda s: s**(-1.5)

    K_values = [2, 4, 6, 8, 10, 12, 16]
    s_fine = np.linspace(a, b, 500)
    h_fine = h_exact(s_fine)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), gridspec_kw={"height_ratios": [2, 1.2]})

    # 상단: 함수와 근사
    ax = axes[0]
    ax.plot(s_fine, h_fine, "k-", linewidth=1.5, label="Exact $s^{-3/2}$")

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(K_values)))
    max_errors = []

    for idx, K in enumerate(K_values):
        h_coeffs = chebyshev_to_bernstein_approx(h_exact, K, a, b)
        # [a,b] → [0,1]로 재매개변수화된 함수의 Bernstein 계수
        # 평가: s값을 [0,1]로 매핑 후 Bernstein 평가
        t_fine = (s_fine - a) / (b - a)
        h_approx = eval_bernstein(h_coeffs, t_fine)

        err = np.max(np.abs(h_fine - h_approx))
        max_errors.append(err)

        if K in [2, 4, 6, 8]:
            ax.plot(s_fine, h_approx, "--", color=colors[idx], linewidth=1,
                    label=f"$K={K}$ (err={err:.1e})")

    ax.set_xlabel("$s = r^2$")
    ax.set_ylabel("$h(s) = s^{-3/2}$")
    ax.legend(fontsize=8)
    ax.set_title("Bernstein approximation of $s^{-3/2}$")

    # 하단: 오차 수렴
    ax2 = axes[1]
    ax2.semilogy(K_values, max_errors, "ko-", markersize=5)
    ax2.set_xlabel("Approximation degree $K$")
    ax2.set_ylabel("Max absolute error")
    ax2.set_title("Exponential convergence")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_h_approx.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Max errors: {dict(zip(K_values, [f'{e:.2e}' for e in max_errors]))}")


# ── Fig 2: a_grav 비교 ───────────────────────────────────────

def fig_agrav_compare():
    """Bernstein 합성 vs 점별 평가의 a_grav(τ) 비교.

    동일한 Bernstein 궤적 위에서 두 방법을 비교 (공정 비교).
    """
    print("Generating fig_agrav_compare...")

    prob = get_scenario_b()
    N = prob.N
    t_f = prob.t_f

    result = solve_inner_loop(prob)
    P_u = result.P_u_opt
    print(f"  SCP converged: iter={result.n_iter}, cost={result.cost:.6f}")

    # Bernstein 위치 궤적 (추력 기여)
    q_r = position_control_points(P_u, prob.r0, prob.v0, t_f, N)

    tau_fine = np.linspace(0, 1, 500)

    # 기준: 동일 Bernstein 궤적 위에서 점별 a_grav
    a_ref, r_pts = pointwise_agrav_on_bernstein_traj(q_r, tau_fine)

    # Bernstein 합성
    q_r2 = (bern_product(q_r[:, 0], q_r[:, 0])
            + bern_product(q_r[:, 1], q_r[:, 1])
            + bern_product(q_r[:, 2], q_r[:, 2]))

    a_val = np.min(q_r2)
    b_val = np.max(q_r2)
    q_stilde = (q_r2 - a_val) / (b_val - a_val)

    K = 6
    h_func = lambda s: (a_val + (b_val - a_val) * s)**(-1.5)
    h_coeffs = chebyshev_to_bernstein_approx(h_func, K)

    q_rinv3_full = bern_compose(h_coeffs, q_stilde)
    R = 20
    q_rinv3 = degree_reduce_l2(q_rinv3_full, R)

    q_agrav_components = []
    for k in range(3):
        q_rk_elev = degree_elevate(q_r[:, k], R)
        q_ak = -bern_product(q_rk_elev, q_rinv3)
        q_agrav_components.append(q_ak)

    a_bern = np.column_stack([
        eval_bernstein(q_agrav_components[k], tau_fine)
        for k in range(3)
    ])

    # 그림 (3+1 패널: xyz + 잔차)
    fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 0.6]})
    labels = ["$a_x$", "$a_y$", "$a_z$"]

    for k in range(3):
        ax = axes[k]
        ax.plot(tau_fine, a_ref[:, k], "k-", linewidth=1.2, label="Pointwise (exact)")
        ax.plot(tau_fine, a_bern[:, k], "b--", linewidth=1, label=f"Bernstein ($K={K}$, $R={R}$)")
        ax.set_ylabel(labels[k])
        if k == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    # 잔차 패널
    ax = axes[3]
    for k in range(3):
        ax.semilogy(tau_fine, np.abs(a_bern[:, k] - a_ref[:, k]) + 1e-16,
                     linewidth=0.8, label=labels[k])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("$|\\Delta a|$")
    ax.set_title("Absolute error (log scale)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    axes[0].set_title(r"$\mathbf{a}_\mathrm{grav}(\tau)$: Bernstein composition vs pointwise (same trajectory)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_agrav_compare.pdf", bbox_inches="tight")
    plt.close()

    err = np.max(np.abs(a_bern - a_ref), axis=0)
    print(f"  Max |a_bern - a_ref|: x={err[0]:.2e}, y={err[1]:.2e}, z={err[2]:.2e}")


# ── Fig 3: c_v 수렴 ──────────────────────────────────────────

def pointwise_agrav_on_bernstein_traj(q_r, tau_arr):
    """Bernstein 위치 제어점으로부터 이산 점에서 a_grav 평가 (기준값)."""
    r_pts = np.column_stack([
        eval_bernstein(q_r[:, k], tau_arr) for k in range(3)
    ])
    r_mag = np.linalg.norm(r_pts, axis=1)
    a_pts = -r_pts / r_mag[:, None]**3
    return a_pts, r_pts


def fig_cv_convergence():
    """K에 따른 c_v 상대 오차의 수렴.

    공정 비교: 동일한 Bernstein 궤적 위에서
    (a) 점별 평가 + 수치 적분 (기준)
    (b) Bernstein 합성 + 제어점 평균 (테스트)
    """
    print("Generating fig_cv_convergence...")

    prob = get_scenario_b()
    N = prob.N
    t_f = prob.t_f

    result = solve_inner_loop(prob)
    P_u = result.P_u_opt

    # 위치 제어점 (추력 기여만 — SCP iteration 0의 참조 궤도에 해당)
    q_r = position_control_points(P_u, prob.r0, prob.v0, t_f, N)

    # 기준: 동일한 Bernstein 궤적 위에서 점별 a_grav 평가 + 수치 적분
    n_quad = 2000
    tau_quad = np.linspace(0, 1, n_quad + 1)
    a_pts, _ = pointwise_agrav_on_bernstein_traj(q_r, tau_quad)
    cv_ref = np.trapezoid(a_pts, tau_quad, axis=0)
    cv_ref_norm = np.linalg.norm(cv_ref)
    print(f"  c_v (pointwise ref): {cv_ref}, norm={cv_ref_norm:.6e}")

    # r² 제어점
    q_r2 = (bern_product(q_r[:, 0], q_r[:, 0])
            + bern_product(q_r[:, 1], q_r[:, 1])
            + bern_product(q_r[:, 2], q_r[:, 2]))
    a_val = np.min(q_r2)
    b_val = np.max(q_r2)
    q_stilde = (q_r2 - a_val) / (b_val - a_val)
    print(f"  r² range: [{a_val:.6f}, {b_val:.6f}], ratio b/a={b_val/a_val:.3f}")

    K_values = [2, 3, 4, 5, 6, 8, 10, 12]
    R = 20
    errors = []

    for K in K_values:
        h_func = lambda s, a=a_val, b=b_val: (a + (b - a) * s)**(-1.5)
        h_coeffs = chebyshev_to_bernstein_approx(h_func, K)

        q_rinv3_full = bern_compose(h_coeffs, q_stilde)
        q_rinv3 = degree_reduce_l2(q_rinv3_full, R)

        # a_grav 제어점 (2체 only)
        cv_bern = np.zeros(3)
        for k in range(3):
            q_rk_elev = degree_elevate(q_r[:, k], R)
            q_ak = -bern_product(q_rk_elev, q_rinv3)
            cv_bern[k] = np.mean(q_ak)  # 정적분 = 평균

        rel_err = np.linalg.norm(cv_bern - cv_ref) / cv_ref_norm
        errors.append(rel_err)
        print(f"  K={K:2d}: rel_err={rel_err:.2e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(K_values, errors, "ko-", markersize=6, label="Bernstein composition")
    ax.set_xlabel("Approximation degree $K$")
    ax.set_ylabel(r"$\|\mathbf{c}_v^{(K)} - \mathbf{c}_v^{\mathrm{ref}}\| / \|\mathbf{c}_v^{\mathrm{ref}}\|$")
    ax.set_title(r"$\mathbf{c}_v$ convergence: Bernstein composition vs pointwise evaluation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_cv_convergence.pdf", bbox_inches="tight")
    plt.close()


# ── Fig 4: 파이프라인 시각화 ──────────────────────────────────

def fig_pipeline_visual():
    """파이프라인 각 단계의 곡선과 제어점."""
    print("Generating fig_pipeline_visual...")

    prob = get_scenario_b()
    N = prob.N
    t_f = prob.t_f

    result = solve_inner_loop(prob)
    P_u = result.P_u_opt

    q_r = position_control_points(P_u, prob.r0, prob.v0, t_f, N)
    M_pos = N + 2

    tau_fine = np.linspace(0, 1, 300)

    # (a) r(τ) 3D 궤적
    r_curve = np.column_stack([
        eval_bernstein(q_r[:, k], tau_fine) for k in range(3)
    ])

    # (b) r²(τ)
    q_r2 = (bern_product(q_r[:, 0], q_r[:, 0])
            + bern_product(q_r[:, 1], q_r[:, 1])
            + bern_product(q_r[:, 2], q_r[:, 2]))
    r2_curve = eval_bernstein(q_r2, tau_fine)

    a_val = np.min(q_r2)
    b_val = np.max(q_r2)

    # (c) r^{-3}(τ)
    q_stilde = (q_r2 - a_val) / (b_val - a_val)
    K = 6
    h_func = lambda s: (a_val + (b_val - a_val) * s)**(-1.5)
    h_coeffs = chebyshev_to_bernstein_approx(h_func, K)
    q_rinv3_full = bern_compose(h_coeffs, q_stilde)
    q_rinv3_R20 = degree_reduce_l2(q_rinv3_full, 20)

    rinv3_full = eval_bernstein(q_rinv3_full, tau_fine)
    rinv3_r20 = eval_bernstein(q_rinv3_R20, tau_fine)

    # 정확한 r^{-3} (r²의 Bernstein 평가값 사용, 동일 궤적)
    r_mag = np.sqrt(np.abs(r2_curve))  # abs for safety
    rinv3_exact = np.where(r_mag > 0, r_mag**(-3), 0.0)

    # (d) a_grav,x(τ)
    q_rx_elev = degree_elevate(q_r[:, 0], 20)
    q_ax = -bern_product(q_rx_elev, q_rinv3_R20)
    ax_curve = eval_bernstein(q_ax, tau_fine)
    ax_exact = -r_curve[:, 0] * rinv3_exact

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # (a) 3D trajectory (2D projection)
    ax = axes[0, 0]
    ax.plot(r_curve[:, 0], r_curve[:, 1], "b-", linewidth=1)
    tau_cp = np.linspace(0, 1, M_pos + 1)
    ax.plot(q_r[:, 0], q_r[:, 1], "rs", markersize=4, alpha=0.6)
    ax.plot(q_r[:, 0], q_r[:, 1], "r--", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"(a) Position $\\mathbf{{r}}(\\tau)$, deg {M_pos}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    # (b) r²(τ)
    ax = axes[0, 1]
    ax.plot(tau_fine, r2_curve, "b-", linewidth=1)
    deg_r2 = len(q_r2) - 1
    tau_cp_r2 = np.linspace(0, 1, deg_r2 + 1)
    ax.plot(tau_cp_r2, q_r2, "rs", markersize=2, alpha=0.4)
    ax.axhline(a_val, color="gray", linestyle=":", linewidth=0.8, label=f"$a = {a_val:.4f}$")
    ax.axhline(b_val, color="gray", linestyle="--", linewidth=0.8, label=f"$b = {b_val:.4f}$")
    ax.fill_between(tau_fine, a_val, b_val, alpha=0.05, color="blue")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("$r^2(\\tau)$")
    ax.set_title(f"(b) $r^2(\\tau)$, deg {deg_r2}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (c) r^{-3}(τ)
    ax = axes[1, 0]
    ax.plot(tau_fine, rinv3_exact, "k-", linewidth=1.2, label="Exact")
    ax.plot(tau_fine, rinv3_full, color="gray", linewidth=0.5, alpha=0.6,
            label=f"Composed (deg {len(q_rinv3_full)-1})")
    ax.plot(tau_fine, rinv3_r20, "b--", linewidth=1,
            label=f"Reduced (deg 20)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("$r^{-3}(\\tau)$")
    ax.set_title("(c) $r^{-3}(\\tau)$: composition + reduction")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # (d) a_{grav,x}(τ)
    ax = axes[1, 1]
    ax.plot(tau_fine, ax_exact, "k-", linewidth=1.2, label="Exact")
    ax.plot(tau_fine, ax_curve, "b--", linewidth=1, label="Bernstein pipeline")
    deg_ax = len(q_ax) - 1
    tau_cp_ax = np.linspace(0, 1, deg_ax + 1)
    ax.plot(tau_cp_ax, q_ax, "rs", markersize=2, alpha=0.3)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("$a_{\\mathrm{grav},x}(\\tau)$")
    ax.set_title(f"(d) $a_{{\\mathrm{{grav}},x}}(\\tau)$, deg {deg_ax}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_pipeline_visual.pdf", bbox_inches="tight")
    plt.close()


# ── Fig 5: 차수 축소 오차 ────────────────────────────────────

def fig_degree_reduction():
    """r^{-3}(τ) 차수 축소 시 목표 차수에 따른 L² 오차."""
    print("Generating fig_degree_reduction...")

    prob = get_scenario_b()
    N = prob.N
    t_f = prob.t_f

    result = solve_inner_loop(prob)
    P_u = result.P_u_opt

    q_r = position_control_points(P_u, prob.r0, prob.v0, t_f, N)

    q_r2 = (bern_product(q_r[:, 0], q_r[:, 0])
            + bern_product(q_r[:, 1], q_r[:, 1])
            + bern_product(q_r[:, 2], q_r[:, 2]))
    a_val = np.min(q_r2)
    b_val = np.max(q_r2)
    q_stilde = (q_r2 - a_val) / (b_val - a_val)

    K = 6
    h_func = lambda s: (a_val + (b_val - a_val) * s)**(-1.5)
    h_coeffs = chebyshev_to_bernstein_approx(h_func, K)

    q_rinv3_full = bern_compose(h_coeffs, q_stilde)
    deg_full = len(q_rinv3_full) - 1
    print(f"  Full composed degree: {deg_full}")

    # 정확한 곡선 평가
    tau_fine = np.linspace(0, 1, 500)
    rinv3_full_curve = eval_bernstein(q_rinv3_full, tau_fine)

    R_values = list(range(8, min(deg_full, 50) + 1, 2))
    errors = []

    for R in R_values:
        q_reduced = degree_reduce_l2(q_rinv3_full, R)
        rinv3_reduced = eval_bernstein(q_reduced, tau_fine)
        # L² 오차 (적분 근사)
        diff = rinv3_full_curve - rinv3_reduced
        l2_err = np.sqrt(np.trapezoid(diff**2, tau_fine))
        errors.append(l2_err)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(R_values, errors, "ko-", markersize=5)
    ax.set_xlabel("Target degree $R$")
    ax.set_ylabel("$L^2$ error")
    ax.set_title(f"Degree reduction of $r^{{-3}}(\\tau)$ (original deg {deg_full})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_degree_reduction.pdf", bbox_inches="tight")
    plt.close()


def fig_KR_tradeoff():
    """K (근사 차수) × R (축소 차수) 조합에 따른 c_v 상대 오차.

    보고서 6.2절 표 생성용.
    """
    print("Generating K-R tradeoff table...")

    prob = get_scenario_b()
    N = prob.N
    t_f = prob.t_f

    result = solve_inner_loop(prob)
    P_u = result.P_u_opt

    q_r = position_control_points(P_u, prob.r0, prob.v0, t_f, N)

    # 기준값: 점별 평가 (2000점 수치 적분)
    tau_quad = np.linspace(0, 1, 2001)
    a_pts, _ = pointwise_agrav_on_bernstein_traj(q_r, tau_quad)
    cv_ref = np.trapezoid(a_pts, tau_quad, axis=0)
    cv_ref_norm = np.linalg.norm(cv_ref)

    q_r2 = (bern_product(q_r[:, 0], q_r[:, 0])
            + bern_product(q_r[:, 1], q_r[:, 1])
            + bern_product(q_r[:, 2], q_r[:, 2]))
    a_val = np.min(q_r2)
    b_val = np.max(q_r2)
    q_stilde = (q_r2 - a_val) / (b_val - a_val)

    K_values = [4, 6, 8, 10, 12]
    R_values = [16, 20, 24, 28, 32, 40]

    print(f"\n  c_v relative error (%) — K (rows) × R (columns)")
    header = "  K\\R  " + "".join(f"{R:>8d}" for R in R_values)
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for K in K_values:
        h_func = lambda s, a=a_val, b=b_val: (a + (b - a) * s)**(-1.5)
        h_coeffs = chebyshev_to_bernstein_approx(h_func, K)
        q_rinv3_full = bern_compose(h_coeffs, q_stilde)

        row = []
        for R in R_values:
            q_rinv3 = degree_reduce_l2(q_rinv3_full, R)
            cv_bern = np.zeros(3)
            for k in range(3):
                q_rk_elev = degree_elevate(q_r[:, k], R)
                q_ak = -bern_product(q_rk_elev, q_rinv3)
                cv_bern[k] = np.mean(q_ak)
            rel_err = np.linalg.norm(cv_bern - cv_ref) / cv_ref_norm
            row.append(rel_err)
            results[(K, R)] = rel_err

        row_str = f"  {K:>3d}   " + "".join(f"{e*100:>8.4f}" for e in row)
        print(row_str)

    # 시각화: heatmap-style plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, K in enumerate(K_values):
        errs = [results[(K, R)] for R in R_values]
        ax.semilogy(R_values, errs, "o-", markersize=5, label=f"$K={K}$")

    ax.set_xlabel("Reduction degree $R$")
    ax.set_ylabel(r"$\|\mathbf{c}_v - \mathbf{c}_v^{\mathrm{ref}}\| / \|\mathbf{c}_v^{\mathrm{ref}}\|$")
    ax.set_title(r"$\mathbf{c}_v$ error: approximation degree $K$ $\times$ reduction degree $R$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_KR_tradeoff.pdf", bbox_inches="tight")
    plt.close()
    print()


# ── Fig 6: SCP 수렴 비교 ─────────────────────────────────────

def fig_scp_convergence():
    """RK4 기반 SCP vs Bernstein 합성 SCP 수렴 비교.

    현재는 RK4 기반만 구현되어 있으므로, RK4 결과를 기준으로
    Bernstein 합성의 드리프트 적분 정확도가 SCP에 미치는 영향을 분석.
    """
    print("Generating fig_scp_convergence...")

    prob = get_scenario_b()
    result = solve_inner_loop(prob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # 왼쪽: 비용 이력
    ax = axes[0]
    costs = [c for c in result.cost_history if c < float("inf")]
    ax.plot(range(1, len(costs) + 1), costs, "ko-", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost $J$")
    ax.set_title("SCP cost history")
    ax.grid(True, alpha=0.2)

    # 오른쪽: ctrl_change 이력
    ax = axes[1]
    ax.semilogy(range(1, len(result.ctrl_change_history) + 1),
                result.ctrl_change_history, "ko-", markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("$\\|\\Delta\\mathbf{Z}\\|_F$")
    ax.set_title("Control point change")
    ax.grid(True, alpha=0.2)

    plt.suptitle(f"SCP convergence (Scenario B, N={prob.N}, $t_f$={prob.t_f})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_scp_convergence.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Final: iter={result.n_iter}, cost={result.cost:.6f}, bc_viol={result.bc_violation:.2e}")


# ── 메인 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    fig_h_approx()
    print()

    fig_pipeline_visual()
    print()

    fig_agrav_compare()
    print()

    fig_cv_convergence()
    print()

    fig_degree_reduction()
    print()

    fig_scp_convergence()
    print()

    fig_KR_tradeoff()
    print()

    print("=" * 60)
    print("All figures generated.")
