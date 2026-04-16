"""N2: 드리프트 적분 계산 시간 비교.

RK4 전파 기반 vs 구간별 Taylor 근사의 실행 시간 측정.
"""

import time
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

# 시나리오: LEO → 1.2a₀
r0, v0 = keplerian_to_cartesian(a0, 0.0, 0.0, 0.0, 0.0, 0.0, mu_km)
rf, vf = keplerian_to_cartesian(a0 * 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, mu_km)

N = 12
t_f = 3.62

prob = SCPProblem(
    r0=r0 / cu.DU, v0=v0 / cu.VU,
    rf=rf / cu.DU, vf=vf / cu.VU,
    N=N, t_f=t_f, canonical_units=cu, max_iter=60,
)

# SCP 풀어서 참조 궤도 확보
result = solve_inner_loop(prob)
P_u_ref = result.P_u_opt


def gravity_accel_vec(r_all):
    """2체+J2 중력 가속도 (벡터화, (M,3) 입력)."""
    r_mag = np.linalg.norm(r_all, axis=1)
    r3 = r_mag**3
    a = -r_all / r3[:, None]
    # J2
    r2 = r_mag**2
    r5 = r_mag**5
    z = r_all[:, 2]
    z2_r2 = z**2 / r2
    c = 1.5 * J2_EARTH * Re_star**2 / r5
    a[:, 0] += c * r_all[:, 0] * (5.0 * z2_r2 - 1.0)
    a[:, 1] += c * r_all[:, 1] * (5.0 * z2_r2 - 1.0)
    a[:, 2] += c * z * (5.0 * z2_r2 - 3.0)
    return a


def gravity_jacobian_3x3(r):
    """중력 Jacobian ∂a_grav/∂r."""
    x_state = np.zeros(6)
    x_state[:3] = r
    A = jacobian_twobody_j2(x_state, Re_star=Re_star, include_j2=True)
    return A[3:6, :3]


# ── 방법 1: RK4 전파 + 수치 적분 (현재 구현) ─────────────────
def method_rk4():
    ref_traj = _propagate_reference(prob, P_u_ref, Re_star)
    cv, cr = _compute_drift_integrals(prob, ref_traj, Re_star)
    return cv, cr


# ── 방법 2: 구간별 0차 (midpoint rule) ────────────────────────
def method_piecewise_0th(K):
    # 참조 궤도 위치를 Bernstein으로 대수적 평가
    # (실제 구현에서는 이전 반복의 궤적에서 보간하지만,
    #  여기서는 RK4 결과에서 보간하여 시간 비교에 집중)
    ref_traj = _ref_traj_cache
    n_traj = ref_traj.shape[0]
    tau_traj = np.linspace(0, 1, n_traj)
    tau_breaks = np.linspace(0, 1, K + 1)

    cv = np.zeros(3)
    for seg in range(K):
        t_lo, t_hi = tau_breaks[seg], tau_breaks[seg + 1]
        dtau = t_hi - t_lo
        tau_mid = 0.5 * (t_lo + t_hi)
        r_mid = np.array([np.interp(tau_mid, tau_traj, ref_traj[:, k]) for k in range(3)])
        cv += gravity_accel_vec(r_mid.reshape(1, 3))[0] * dtau

    # c_r: 이중 적분 근사 (사다리꼴)
    tau_fine = tau_breaks
    a_at_breaks = np.zeros((K + 1, 3))
    for j in range(K + 1):
        r_j = np.array([np.interp(tau_breaks[j], tau_traj, ref_traj[:, k]) for k in range(3)])
        a_at_breaks[j] = gravity_accel_vec(r_j.reshape(1, 3))[0]

    dt = tau_breaks[1] - tau_breaks[0]
    F = np.cumsum(0.5 * (a_at_breaks[:-1] + a_at_breaks[1:]) * dt, axis=0)
    F = np.vstack([np.zeros((1, 3)), F])
    cr = np.trapezoid(F, tau_breaks, axis=0)

    return cv, cr


# ── 방법 2b: 구간별 0차 (벡터화, 보간 제거) ──────────────────
def method_piecewise_0th_vec(K):
    ref_traj = _ref_traj_cache
    n_traj = ref_traj.shape[0]
    tau_traj = np.linspace(0, 1, n_traj)
    tau_breaks = np.linspace(0, 1, K + 1)
    tau_mids = 0.5 * (tau_breaks[:-1] + tau_breaks[1:])
    dtau = 1.0 / K

    # 구간 중점 위치 (벡터화 보간)
    r_mids = np.column_stack([
        np.interp(tau_mids, tau_traj, ref_traj[:, k]) for k in range(3)
    ])  # (K, 3)
    a_mids = gravity_accel_vec(r_mids)  # (K, 3)
    cv = np.sum(a_mids * dtau, axis=0)

    # c_r: 구간 경계에서 평가
    r_breaks = np.column_stack([
        np.interp(tau_breaks, tau_traj, ref_traj[:, k]) for k in range(3)
    ])
    a_breaks = gravity_accel_vec(r_breaks)
    dt = tau_breaks[1] - tau_breaks[0]
    F = np.cumsum(0.5 * (a_breaks[:-1] + a_breaks[1:]) * dt, axis=0)
    F = np.vstack([np.zeros((1, 3)), F])
    cr = np.trapezoid(F, tau_breaks, axis=0)

    return cv, cr


# ── 방법 3: 구간별 1차 (Jacobian 포함) ───────────────────────
def method_piecewise_1st_vec(K):
    ref_traj = _ref_traj_cache
    n_traj = ref_traj.shape[0]
    tau_traj = np.linspace(0, 1, n_traj)
    tau_breaks = np.linspace(0, 1, K + 1)
    dtau = 1.0 / K
    M_seg = max(10, 50 // K)

    cv = np.zeros(3)
    for seg in range(K):
        t_lo, t_hi = tau_breaks[seg], tau_breaks[seg + 1]
        tau_mid = 0.5 * (t_lo + t_hi)
        r_mid = np.array([np.interp(tau_mid, tau_traj, ref_traj[:, k]) for k in range(3)])
        a_mid = gravity_accel_vec(r_mid.reshape(1, 3))[0]
        A_mid = gravity_jacobian_3x3(r_mid)

        tau_loc = np.linspace(t_lo, t_hi, M_seg + 1)
        dt_loc = tau_loc[1] - tau_loc[0]
        w = np.ones(M_seg + 1) * dt_loc
        w[0] *= 0.5; w[-1] *= 0.5

        r_loc = np.column_stack([
            np.interp(tau_loc, tau_traj, ref_traj[:, k]) for k in range(3)
        ])
        dr = r_loc - r_mid
        cv += np.sum(w[:, None] * (a_mid + (A_mid @ dr.T).T), axis=0)

    # c_r 생략 (c_v와 동일 구조)
    cr = np.zeros(3)
    return cv, cr


# ── 벤치마크 ──────────────────────────────────────────────────
# 캐시: RK4 참조 궤도 (방법 2,3에서 사용)
_ref_traj_cache = _propagate_reference(prob, P_u_ref, Re_star)

# Ground truth
cv_true, cr_true = method_rk4()

n_warmup = 3
n_repeat = 50

def bench(func, label, *args):
    # 워밍업
    for _ in range(n_warmup):
        func(*args)
    # 측정
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        result = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = np.mean(times) * 1000  # ms
    std = np.std(times) * 1000
    return avg, std, result


print("=" * 70)
print("드리프트 적분 계산 시간 비교")
print("=" * 70)
print(f"시나리오: LEO → 1.2a₀, N={N}, t_f={t_f}")
print(f"측정: {n_repeat}회 반복 평균\n")

# RK4 전파 + 드리프트 적분 (전체)
t_rk4_full, s_rk4_full, _ = bench(method_rk4, "RK4 전체")
print(f"{'방법':<35s} | {'시간 (ms)':>10s} | {'c_v err':>10s} | {'c_r err':>10s}")
print("-" * 75)
print(f"{'RK4 전파 + 수치적분 (201 steps)':<35s} | {t_rk4_full:10.3f} |   {'기준':>7s} |   {'기준':>7s}")

# RK4 전파만 (드리프트 적분 제외)
def just_propagate():
    return _propagate_reference(prob, P_u_ref, Re_star)

t_prop, _, _ = bench(just_propagate, "RK4 전파만")

# 드리프트 적분만 (이미 전파된 궤적 사용)
def just_drift():
    return _compute_drift_integrals(prob, _ref_traj_cache, Re_star)

t_drift, _, _ = bench(just_drift, "드리프트 적분만")
print(f"  {'└ RK4 전파':<33s} | {t_prop:10.3f} |")
print(f"  {'└ 드리프트 적분 (벡터화)':<33s} | {t_drift:10.3f} |")

# 구간별 근사
K_values = [4, 8, 16, 32]
print()

for K in K_values:
    # 0차 벡터화
    t_0v, _, (cv_0v, cr_0v) = bench(method_piecewise_0th_vec, f"0차 K={K} (vec)", K)
    cv_err = np.linalg.norm(cv_0v - cv_true) / np.linalg.norm(cv_true) * 100
    cr_err = np.linalg.norm(cr_0v - cr_true) / np.linalg.norm(cr_true) * 100
    print(f"{'0차 midpoint K=' + str(K) + ' (벡터화)':<35s} | {t_0v:10.3f} | {cv_err:9.4f}% | {cr_err:9.4f}%")

print()
for K in K_values:
    # 1차
    t_1, _, (cv_1, _) = bench(method_piecewise_1st_vec, f"1차 K={K}", K)
    cv_err = np.linalg.norm(cv_1 - cv_true) / np.linalg.norm(cv_true) * 100
    print(f"{'1차 Taylor K=' + str(K):<35s} | {t_1:10.3f} | {cv_err:9.4f}% |")

# 속도 향상 비율
print(f"\n{'=' * 70}")
print("속도 향상 비율 (RK4 전체 대비)")
print(f"{'=' * 70}")
for K in K_values:
    t_0v, _, _ = bench(method_piecewise_0th_vec, "", K)
    speedup = t_rk4_full / t_0v
    print(f"  0차 K={K:2d}: {speedup:5.1f}x 빠름 ({t_0v:.3f} ms vs {t_rk4_full:.3f} ms)")
