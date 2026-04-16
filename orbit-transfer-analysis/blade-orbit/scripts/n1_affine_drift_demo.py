"""N1: 아핀 드리프트 근사 수치 예제.

보고서 009 수치 검증.

예제 1: 참조 궤도의 중력 적분을 구간별 Taylor 전개로 근사
  → K개 구간으로 나누고, 각 구간 중점에서 1차 또는 0차 전개
  → RK4 대비 오차를 K 함수로 측정

예제 2: 아핀 드리프트를 SCP 경계조건에 적용할 때의 BC 정확도
  → 같은 제어점에서 RK4 기반 c_v,c_r vs 아핀 기반 c_v,c_r
  → 최종 경계조건 잔차 비교
"""

import numpy as np
from bezier_orbit.normalize import CanonicalUnits, J2_EARTH
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.orbit.dynamics import jacobian_twobody_j2
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import (
    solve_inner_loop, _propagate_reference, _compute_drift_integrals,
)

mu_km = 398600.4418
a0 = 6678.137
cu = CanonicalUnits(a0=a0, mu=mu_km)
Re_star = cu.R_earth_star


def gravity_accel(r):
    """2체+J2 중력 가속도 (벡터화)."""
    r_mag = np.linalg.norm(r)
    a = -r / r_mag**3
    x, y, z = r
    r2, r5 = r_mag**2, r_mag**5
    z2_r2 = z**2 / r2
    c = 1.5 * J2_EARTH * Re_star**2 / r5
    a[0] += c * x * (5.0 * z2_r2 - 1.0)
    a[1] += c * y * (5.0 * z2_r2 - 1.0)
    a[2] += c * z * (5.0 * z2_r2 - 3.0)
    return a


def gravity_jacobian_3x3(r):
    """중력 Jacobian ∂a_grav/∂r ∈ R^{3×3}."""
    x_state = np.zeros(6)
    x_state[:3] = r
    A = jacobian_twobody_j2(x_state, Re_star=Re_star, include_j2=True)
    return A[3:6, :3]


# ── 시나리오 설정 ──────────────────────────────────────────────
scenarios = [
    ("LEO 원형→1.1a₀ 원형", 1.0, 1.1, 0.0, 3.0),
    ("LEO 원형→1.2a₀ 원형", 1.0, 1.2, 0.0, 3.62),
    ("LEO 원형→1.0a₀, Δi=10°", 1.0, 1.0, 10.0, 3.0),
]

N = 12

for label, a_init, a_final, di_deg, t_f in scenarios:
    print(f"\n{'=' * 70}")
    print(f"시나리오: {label}")
    print(f"{'=' * 70}")

    i0 = 0.0
    i_f = np.radians(di_deg)
    r0, v0 = keplerian_to_cartesian(a0 * a_init, 0.0, i0, 0.0, 0.0, 0.0, mu_km)
    rf, vf = keplerian_to_cartesian(a0 * a_final, 0.0, i_f, 0.0, 0.0, 0.0, mu_km)

    prob = SCPProblem(
        r0=r0 / cu.DU, v0=v0 / cu.VU,
        rf=rf / cu.DU, vf=vf / cu.VU,
        N=N, t_f=t_f, canonical_units=cu, max_iter=60,
    )

    # SCP 풀어서 수렴된 참조 궤도 확보
    result = solve_inner_loop(prob)
    P_u_ref = result.P_u_opt
    ref_traj = _propagate_reference(prob, P_u_ref, Re_star)
    n_traj = ref_traj.shape[0]
    tau_traj = np.linspace(0, 1, n_traj)

    # Ground truth: 고해상도 수치 적분
    cv_true, cr_true = _compute_drift_integrals(prob, ref_traj, Re_star)

    print(f"  SCP: cost={result.cost:.6f}, iter={result.n_iter}, "
          f"bc_viol={result.bc_violation:.2e}")
    print(f"  c_v = [{cv_true[0]:+.8f}, {cv_true[1]:+.8f}, {cv_true[2]:+.8f}]")
    print(f"  c_r = [{cr_true[0]:+.8f}, {cr_true[1]:+.8f}, {cr_true[2]:+.8f}]")

    # ── 예제 1: 구간별 Taylor 근사로 c_v 재현 ─────────────────
    print(f"\n  --- c_v 근사 오차 (구간별 Taylor) ---")
    print(f"  {'K':>4s} | {'0차 err':>12s} | {'1차 err':>12s} | "
          f"{'0차 상대%':>10s} | {'1차 상대%':>10s}")
    print(f"  {'-' * 60}")

    cv_norm = np.linalg.norm(cv_true)
    results_cv = []

    for K in [1, 2, 4, 8, 16, 32]:
        tau_breaks = np.linspace(0, 1, K + 1)
        M_seg = max(20, 100 // K)

        cv_0th = np.zeros(3)  # 0차: 구간 중점의 a_grav 값만
        cv_1st = np.zeros(3)  # 1차: 중점 + Jacobian · (r-r*)

        for seg in range(K):
            t_lo, t_hi = tau_breaks[seg], tau_breaks[seg + 1]
            dtau = t_hi - t_lo

            # 구간 중점
            tau_mid = 0.5 * (t_lo + t_hi)
            r_mid = np.array([
                np.interp(tau_mid, tau_traj, ref_traj[:, k]) for k in range(3)
            ])
            a_mid = gravity_accel(r_mid.copy())
            A_mid = gravity_jacobian_3x3(r_mid.copy())

            # 구적점
            tau_loc = np.linspace(t_lo, t_hi, M_seg + 1)
            dt = tau_loc[1] - tau_loc[0]
            w = np.ones(M_seg + 1) * dt
            w[0] *= 0.5
            w[-1] *= 0.5

            r_loc = np.array([
                np.interp(tau_loc, tau_traj, ref_traj[:, k]) for k in range(3)
            ]).T  # (M_seg+1, 3)

            # 0차: ∫ a_grav(r_mid) dτ = a_grav(r_mid) · Δτ
            cv_0th += a_mid * dtau

            # 1차: ∫ [a_grav(r_mid) + A(r_mid)·(r(τ)-r_mid)] dτ
            for j in range(M_seg + 1):
                dr = r_loc[j] - r_mid
                cv_1st += w[j] * (a_mid + A_mid @ dr)

        err_0 = np.linalg.norm(cv_0th - cv_true)
        err_1 = np.linalg.norm(cv_1st - cv_true)
        rel_0 = err_0 / cv_norm * 100
        rel_1 = err_1 / cv_norm * 100
        results_cv.append((K, err_0, err_1, rel_0, rel_1))
        print(f"  {K:4d} | {err_0:12.2e} | {err_1:12.2e} | "
              f"{rel_0:9.4f}% | {rel_1:9.4f}%")

    # 수렴 차수 추정
    errs_0 = [(1.0 / k, e0) for k, e0, _, _, _ in results_cv if e0 > 1e-14]
    errs_1 = [(1.0 / k, e1) for k, _, e1, _, _ in results_cv if e1 > 1e-14]
    if len(errs_0) >= 3:
        h, e = np.log([x[0] for x in errs_0]), np.log([x[1] for x in errs_0])
        p0 = np.polyfit(h, e, 1)[0]
    else:
        p0 = float('nan')
    if len(errs_1) >= 3:
        h, e = np.log([x[0] for x in errs_1]), np.log([x[1] for x in errs_1])
        p1 = np.polyfit(h, e, 1)[0]
    else:
        p1 = float('nan')
    print(f"  수렴 차수: 0차 ≈ O(1/K^{p0:.1f}), 1차 ≈ O(1/K^{p1:.1f})")

    # ── 예제 1b: c_r 근사 ─────────────────────────────────────
    print(f"\n  --- c_r 근사 오차 (구간별 Taylor) ---")
    print(f"  {'K':>4s} | {'0차 err':>12s} | {'1차 err':>12s} | "
          f"{'0차 상대%':>10s} | {'1차 상대%':>10s}")
    print(f"  {'-' * 60}")

    cr_norm = np.linalg.norm(cr_true)
    results_cr = []

    for K in [1, 2, 4, 8, 16, 32]:
        tau_breaks = np.linspace(0, 1, K + 1)
        M_seg = max(20, 100 // K)

        # c_r = ∫₀¹∫₀^s a_grav(r(σ)) dσ ds
        # 구간별 구적으로 이중 적분 근사
        # 먼저 F(s) = ∫₀^s a_grav(r(σ)) dσ 를 구간별로 계산

        n_total = K * M_seg + 1
        tau_fine = np.linspace(0, 1, n_total)
        dt_fine = tau_fine[1] - tau_fine[0]

        # 0차, 1차 a_grav 근사값
        a_0th = np.zeros((n_total, 3))
        a_1st = np.zeros((n_total, 3))

        for seg in range(K):
            t_lo, t_hi = tau_breaks[seg], tau_breaks[seg + 1]
            tau_mid = 0.5 * (t_lo + t_hi)
            r_mid = np.array([
                np.interp(tau_mid, tau_traj, ref_traj[:, k]) for k in range(3)
            ])
            a_mid = gravity_accel(r_mid.copy())
            A_mid = gravity_jacobian_3x3(r_mid.copy())

            idx_lo = seg * M_seg
            idx_hi = (seg + 1) * M_seg + 1
            if seg == K - 1:
                idx_hi = n_total

            for j in range(idx_lo, idx_hi):
                r_j = np.array([
                    np.interp(tau_fine[j], tau_traj, ref_traj[:, k])
                    for k in range(3)
                ])
                a_0th[j] = a_mid
                a_1st[j] = a_mid + A_mid @ (r_j - r_mid)

        # 누적 적분 F(s) = ∫₀^s a(σ) dσ, 이중 적분 c_r = ∫₀¹ F(s) ds
        F_0 = np.cumsum(0.5 * (a_0th[:-1] + a_0th[1:]) * dt_fine, axis=0)
        F_0 = np.vstack([np.zeros((1, 3)), F_0])
        cr_0th = np.trapezoid(F_0, tau_fine, axis=0)

        F_1 = np.cumsum(0.5 * (a_1st[:-1] + a_1st[1:]) * dt_fine, axis=0)
        F_1 = np.vstack([np.zeros((1, 3)), F_1])
        cr_1st = np.trapezoid(F_1, tau_fine, axis=0)

        err_0 = np.linalg.norm(cr_0th - cr_true)
        err_1 = np.linalg.norm(cr_1st - cr_true)
        rel_0 = err_0 / cr_norm * 100 if cr_norm > 1e-15 else 0
        rel_1 = err_1 / cr_norm * 100 if cr_norm > 1e-15 else 0
        results_cr.append((K, err_0, err_1, rel_0, rel_1))
        print(f"  {K:4d} | {err_0:12.2e} | {err_1:12.2e} | "
              f"{rel_0:9.4f}% | {rel_1:9.4f}%")


# ── 예제 2: SCP 경계조건 정확도 비교 ──────────────────────────
print(f"\n{'=' * 70}")
print("예제 2: SCP BC 잔차에 미치는 영향")
print(f"{'=' * 70}")

# a0 → 1.2*a0 시나리오
r0, v0 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu_km)
rf, vf = keplerian_to_cartesian(a0 * 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, mu_km)

prob = SCPProblem(
    r0=r0 / cu.DU, v0=v0 / cu.VU,
    rf=rf / cu.DU, vf=vf / cu.VU,
    N=N, t_f=3.62, canonical_units=cu, max_iter=60,
)
result = solve_inner_loop(prob)
P_u = result.P_u_opt
ref_traj = _propagate_reference(prob, P_u, Re_star)

from bezier_orbit.bezier.basis import int_matrix, double_int_matrix

eI = int_matrix(N)[-1, :]     # I_N 의 마지막 행: ∫₀¹ 기여
eIbar = double_int_matrix(N)[-1, :]  # Ī_N 의 마지막 행

t_f = prob.t_f

# RK4 기반 c_v, c_r
cv_rk4, cr_rk4 = _compute_drift_integrals(prob, ref_traj, Re_star)

# BC 잔차 (RK4 기반)
v_pred_rk4 = prob.v0 + t_f * cv_rk4 + t_f * eI @ P_u
r_pred_rk4 = prob.r0 + t_f * prob.v0 + t_f**2 * cr_rk4 + t_f * eIbar @ P_u

v_err_rk4 = np.linalg.norm(v_pred_rk4 - prob.vf)
r_err_rk4 = np.linalg.norm(r_pred_rk4 - prob.rf)

print(f"\nRK4 기반 (ground truth):")
print(f"  속도 BC 잔차: {v_err_rk4:.2e}")
print(f"  위치 BC 잔차: {r_err_rk4:.2e}")

# 구간별 Taylor 기반 c_v, c_r
n_traj = ref_traj.shape[0]
tau_traj = np.linspace(0, 1, n_traj)

for K in [1, 4, 8, 16]:
    tau_breaks = np.linspace(0, 1, K + 1)
    M_seg = max(20, 100 // K)

    # c_v 1차 근사
    cv_approx = np.zeros(3)
    for seg in range(K):
        t_lo, t_hi = tau_breaks[seg], tau_breaks[seg + 1]
        tau_mid = 0.5 * (t_lo + t_hi)
        r_mid = np.array([np.interp(tau_mid, tau_traj, ref_traj[:, k]) for k in range(3)])
        a_mid = gravity_accel(r_mid.copy())
        A_mid = gravity_jacobian_3x3(r_mid.copy())

        tau_loc = np.linspace(t_lo, t_hi, M_seg + 1)
        dt_loc = tau_loc[1] - tau_loc[0]
        w_loc = np.ones(M_seg + 1) * dt_loc
        w_loc[0] *= 0.5; w_loc[-1] *= 0.5

        for j in range(M_seg + 1):
            r_j = np.array([np.interp(tau_loc[j], tau_traj, ref_traj[:, k]) for k in range(3)])
            cv_approx += w_loc[j] * (a_mid + A_mid @ (r_j - r_mid))

    # c_r 1차 근사
    n_fine = K * M_seg + 1
    tau_fine = np.linspace(0, 1, n_fine)
    dt_fine = tau_fine[1] - tau_fine[0]
    a_1st = np.zeros((n_fine, 3))

    for seg in range(K):
        t_lo, t_hi = tau_breaks[seg], tau_breaks[seg + 1]
        tau_mid = 0.5 * (t_lo + t_hi)
        r_mid = np.array([np.interp(tau_mid, tau_traj, ref_traj[:, k]) for k in range(3)])
        a_mid = gravity_accel(r_mid.copy())
        A_mid = gravity_jacobian_3x3(r_mid.copy())

        idx_lo = seg * M_seg
        idx_hi = min((seg + 1) * M_seg + 1, n_fine)
        for j in range(idx_lo, idx_hi):
            r_j = np.array([np.interp(tau_fine[j], tau_traj, ref_traj[:, k]) for k in range(3)])
            a_1st[j] = a_mid + A_mid @ (r_j - r_mid)

    F = np.cumsum(0.5 * (a_1st[:-1] + a_1st[1:]) * dt_fine, axis=0)
    F = np.vstack([np.zeros((1, 3)), F])
    cr_approx = np.trapezoid(F, tau_fine, axis=0)

    # BC 잔차
    v_pred = prob.v0 + t_f * cv_approx + t_f * eI @ P_u
    r_pred = prob.r0 + t_f * prob.v0 + t_f**2 * cr_approx + t_f * eIbar @ P_u

    v_err = np.linalg.norm(v_pred - prob.vf)
    r_err = np.linalg.norm(r_pred - prob.rf)

    cv_diff = np.linalg.norm(cv_approx - cv_rk4)
    cr_diff = np.linalg.norm(cr_approx - cr_rk4)

    print(f"\nK={K:2d} 구간 1차 Taylor:")
    print(f"  Δc_v={cv_diff:.2e}, Δc_r={cr_diff:.2e}")
    print(f"  속도 BC 잔차: {v_err:.2e}")
    print(f"  위치 BC 잔차: {r_err:.2e}")
