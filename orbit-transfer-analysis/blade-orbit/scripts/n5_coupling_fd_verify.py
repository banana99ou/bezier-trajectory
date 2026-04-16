"""N5: 유한차분으로 정확한 커플링 행렬 ∂c_v/∂P_u 검증.

검증 1: 정확한 Jacobian vs 수치적 커플링 행렬 비교
  - 크기(norm), 방향(cos 유사도), 성분별 상관
검증 2: 시나리오별 (약/중/강 전이) 과소추정 정도
검증 3: 유한차분 Jacobian을 커플링으로 사용한 SCP 수렴 효과

출력: 콘솔 테이블 + results/n5_coupling_verify.csv
"""

import csv
import os
import time
import warnings

import numpy as np

from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.drift import (
    DriftConfig, compute_drift, compute_coupling_matrices,
)
from bezier_orbit.scp.problem import SCPProblem, build_lifted_boundary_constraints
from bezier_orbit.scp.inner_loop import solve_inner_loop, SCPResult

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def fd_jacobian_cv(prob, P_u, Re_star, eps=1e-7):
    """유한차분으로 ∂c_v/∂vec(P_u) 계산.

    Returns (3, 3(N+1)) Jacobian, c_v_base (3,).
    """
    cv_base, _, _ = compute_drift(prob, P_u, Re_star)
    n_params = P_u.size
    J = np.zeros((3, n_params))
    for j in range(n_params):
        P_pert = P_u.copy().ravel()
        P_pert[j] += eps
        cv_pert, _, _ = compute_drift(prob, P_pert.reshape(P_u.shape), Re_star)
        J[:, j] = (cv_pert - cv_base) / eps
    return J, cv_base


def fd_jacobian_cr(prob, P_u, Re_star, eps=1e-7):
    """유한차분으로 ∂c_r/∂vec(P_u) 계산."""
    _, cr_base, _ = compute_drift(prob, P_u, Re_star)
    n_params = P_u.size
    J = np.zeros((3, n_params))
    for j in range(n_params):
        P_pert = P_u.copy().ravel()
        P_pert[j] += eps
        _, cr_pert, _ = compute_drift(prob, P_pert.reshape(P_u.shape), Re_star)
        J[:, j] = (cr_pert - cr_base) / eps
    return J, cr_base


def cosine_similarity(A, B):
    """행렬 A, B의 코사인 유사도 (Frobenius 내적)."""
    dot = np.sum(A * B)
    return dot / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-30)


# ══════════════════════════════════════════════════════════════
# 시나리오 정의
# ══════════════════════════════════════════════════════════════

SCENARIOS = {
    "A_weak":   {"a_ratio": 1.1, "t_f": 3.0,  "label": "LEO→1.1a₀"},
    "B_medium": {"a_ratio": 1.2, "t_f": 3.62, "label": "LEO→1.2a₀"},
    "C_strong": {"a_ratio": 1.5, "t_f": 5.0,  "label": "LEO→1.5a₀"},
}

N = 8
Re_star = 1.0


def make_prob(scenario, cfg):
    s = SCENARIOS[scenario]
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    rf, vf = keplerian_to_cartesian(s["a_ratio"], 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
    return SCPProblem(
        r0=r0, v0=v0, rf=rf, vf=vf,
        t_f=s["t_f"], N=N, drift_config=cfg,
    )


# ══════════════════════════════════════════════════════════════
# 검증 1: 정확한 Jacobian vs 수치적 커플링 (성분별 비교)
# ══════════════════════════════════════════════════════════════

def verify_jacobian_accuracy(scenario):
    """한 시나리오에 대해 정확한 Jacobian과 수치적 커플링을 비교."""
    # 수렴된 P_u 확보 (RK4)
    prob_rk4 = make_prob(scenario, DriftConfig(method="rk4"))
    prob_rk4.max_iter = 60
    res = solve_inner_loop(prob_rk4)
    P_u = res.P_u_opt

    # Bernstein 기반 정확한 Jacobian (유한차분)
    prob_bern = make_prob(scenario, DriftConfig(
        method="bernstein", gravity_correction="algebraic", gravity_correction_K=4,
    ))
    J_exact_v, cv_base = fd_jacobian_cv(prob_bern, P_u, Re_star)
    J_exact_r, cr_base = fd_jacobian_cr(prob_bern, P_u, Re_star)

    # 수치적 커플링 행렬
    M_v_num, M_r_num = compute_coupling_matrices(prob_bern, P_u, Re_star)

    return {
        "scenario": scenario,
        "norm_Jv_exact": np.linalg.norm(J_exact_v),
        "norm_Mv_num": np.linalg.norm(M_v_num),
        "ratio_v": np.linalg.norm(J_exact_v) / (np.linalg.norm(M_v_num) + 1e-30),
        "cos_sim_v": cosine_similarity(J_exact_v, M_v_num),
        "norm_Jr_exact": np.linalg.norm(J_exact_r),
        "norm_Mr_num": np.linalg.norm(M_r_num),
        "ratio_r": np.linalg.norm(J_exact_r) / (np.linalg.norm(M_r_num) + 1e-30),
        "cos_sim_r": cosine_similarity(J_exact_r, M_r_num),
        "J_exact_v": J_exact_v,
        "J_exact_r": J_exact_r,
        "P_u": P_u,
    }


# ══════════════════════════════════════════════════════════════
# 검증 2: 유한차분 Jacobian을 커플링으로 사용한 SCP 수렴
# ══════════════════════════════════════════════════════════════

def solve_with_exact_coupling(scenario, max_iter=50, tol_ctrl=1e-3, tol_bc=0.05):
    """유한차분 Jacobian을 커플링 행렬로 사용하여 SCP를 풀고 반복 수 비교."""
    from bezier_orbit.scp.inner_loop import (
        _solve_convex_subproblem, _propagate_reference,
        _bc_violation_from_traj, _bc_violation_algebraic,
    )
    from bezier_orbit.bezier.basis import int_matrix, double_int_matrix

    s = SCENARIOS[scenario]
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 1.0, 0.0])
    rf, vf = keplerian_to_cartesian(s["a_ratio"], 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
    t_f = s["t_f"]

    cfg = DriftConfig(method="bernstein", gravity_correction="algebraic",
                      gravity_correction_K=4, use_coupling=False)
    prob = SCPProblem(
        r0=r0, v0=v0, rf=rf, vf=vf, t_f=t_f, N=N,
        max_iter=max_iter, tol_ctrl=tol_ctrl, tol_bc=tol_bc,
        drift_config=cfg,
    )

    Z_prev = np.zeros((N + 1, 3))
    delta = prob.trust_region
    prev_bc_viol = float("inf")
    prev_phi = None
    cost_history = []
    ctrl_history = []
    n_steps_final = max(200, 10 * N)

    for iteration in range(max_iter):
        P_u_ref = Z_prev / t_f

        # 드리프트 적분
        c_v, c_r, prev_phi = compute_drift(prob, P_u_ref, Re_star, prev_phi=prev_phi)

        # 유한차분 커플링 행렬 (2번째 반복부터)
        M_v, M_r, Z_ref_coupling = None, None, None
        if iteration >= 1:
            M_v, _ = fd_jacobian_cv(prob, P_u_ref, Re_star)
            _, _ = fd_jacobian_cr(prob, P_u_ref, Re_star)
            # M_r도 유한차분
            M_r_fd, _ = fd_jacobian_cr(prob, P_u_ref, Re_star)
            M_v = M_v
            M_r = M_r_fd
            Z_ref_coupling = Z_prev

        A_eq, b_eq = build_lifted_boundary_constraints(
            prob, c_v=c_v, c_r=c_r,
            M_v=M_v, M_r=M_r, Z_ref=Z_ref_coupling,
        )

        use_tr = iteration >= 2
        Z_opt, cost, _solver = _solve_convex_subproblem(
            prob, A_eq, b_eq, Re_star,
            Z_ref=Z_prev if use_tr else None,
            trust_radius=delta if use_tr else None,
        )

        if Z_opt is None:
            if use_tr:
                delta = max(delta * prob.trust_shrink, prob.trust_min)
            cost_history.append(float("inf"))
            ctrl_history.append(0.0)
            continue

        alpha_val = 1.0 if iteration == 0 else prob.relax_alpha
        Z_new = (1.0 - alpha_val) * Z_prev + alpha_val * Z_opt
        cost_history.append(cost)

        ctrl_change = np.linalg.norm(Z_new - Z_prev, "fro")
        ctrl_history.append(ctrl_change)

        P_u_new = Z_new / t_f
        bc_viol = _bc_violation_algebraic(prob, P_u_new, c_v, c_r)

        if use_tr:
            if bc_viol < prev_bc_viol:
                delta = min(delta * prob.trust_expand, 1e6)
            else:
                delta = max(delta * prob.trust_shrink, prob.trust_min)
        prev_bc_viol = bc_viol

        cost_converged = (
            len(cost_history) >= 2
            and cost_history[-2] < float("inf")
            and abs(cost - cost_history[-2]) < 1e-4 * max(abs(cost), 1e-12)
        )
        ctrl_converged = ctrl_change < tol_ctrl
        if (ctrl_converged or cost_converged) and bc_viol < tol_bc:
            ref_final = _propagate_reference(prob, P_u_new, Re_star, n_steps=n_steps_final)
            bc_viol_exact = _bc_violation_from_traj(prob, ref_final)
            return SCPResult(
                Z_opt=Z_new, P_u_opt=P_u_new,
                cost=cost, n_iter=iteration + 1,
                converged=True, bc_violation=bc_viol_exact,
                ctrl_change_history=ctrl_history, cost_history=cost_history,
            )

        Z_prev = Z_new.copy()

    P_u_opt = Z_prev / t_f
    ref_final = _propagate_reference(prob, P_u_opt, Re_star, n_steps=n_steps_final)
    bc_viol_exact = _bc_violation_from_traj(prob, ref_final)
    return SCPResult(
        Z_opt=Z_prev, P_u_opt=P_u_opt,
        cost=cost_history[-1] if cost_history else float("inf"),
        n_iter=max_iter, converged=False,
        bc_violation=bc_viol_exact,
        ctrl_change_history=ctrl_history, cost_history=cost_history,
    )


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs("scripts/results", exist_ok=True)

    # ── 검증 1: Jacobian 정확도 ──────────────────────────────
    print("=" * 70)
    print("  검증 1: 정확한 Jacobian vs 수치적 커플링 행렬")
    print("=" * 70)

    results = []
    for sc_name, sc in SCENARIOS.items():
        print(f"\n  [{sc['label']}] 계산 중...")
        r = verify_jacobian_accuracy(sc_name)
        results.append(r)

    print(f"\n  {'시나리오':<14s}  {'‖J_v‖':>8s}  {'‖M_v‖':>8s}  {'비율':>6s}  "
          f"{'cos':>6s}  {'‖J_r‖':>8s}  {'‖M_r‖':>8s}  {'비율':>6s}  {'cos':>6s}")
    print("  " + "-" * 74)
    for r in results:
        sc = SCENARIOS[r["scenario"]]
        print(f"  {sc['label']:<14s}  {r['norm_Jv_exact']:8.3f}  {r['norm_Mv_num']:8.4f}  "
              f"{r['ratio_v']:5.1f}×  {r['cos_sim_v']:6.3f}  "
              f"{r['norm_Jr_exact']:8.3f}  {r['norm_Mr_num']:8.4f}  "
              f"{r['ratio_r']:5.1f}×  {r['cos_sim_r']:6.3f}")

    # ── 검증 2: SCP 수렴 비교 ────────────────────────────────
    print("\n" + "=" * 70)
    print("  검증 2: SCP 수렴 비교 (시나리오 B)")
    print("=" * 70)

    configs_compare = {
        "coupling off":     DriftConfig(method="bernstein", gravity_correction="algebraic",
                                         gravity_correction_K=4, use_coupling=False),
        "coupling num":     DriftConfig(method="bernstein", gravity_correction="algebraic",
                                         gravity_correction_K=4, use_coupling=True),
    }

    print(f"\n  {'방법':<24s}  {'반복':>4s}  {'비용':>10s}  {'BC위반':>10s}  {'시간':>8s}")
    print("  " + "-" * 62)

    for name, cfg in configs_compare.items():
        prob = SCPProblem(
            r0=np.array([1.0, 0.0, 0.0]),
            v0=np.array([0.0, 1.0, 0.0]),
            rf=keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)[0],
            vf=keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)[1],
            t_f=3.62, N=N,
            max_iter=50, tol_ctrl=1e-3, tol_bc=0.05, drift_config=cfg,
        )
        solve_inner_loop(prob)  # warmup
        t0 = time.perf_counter()
        res = solve_inner_loop(prob)
        ms = (time.perf_counter() - t0) * 1000
        cv = "✓" if res.converged else "✗"
        print(f"  {name:<24s}  {res.n_iter:4d}  {res.cost:10.6f}  "
              f"{res.bc_violation:10.2e}  {ms:7.0f}ms {cv}")

    # 유한차분 커플링
    print("\n  유한차분 커플링으로 SCP 풀이 중 (느릴 수 있음)...")
    t0 = time.perf_counter()
    res_fd = solve_with_exact_coupling("B_medium")
    ms = (time.perf_counter() - t0) * 1000
    cv = "✓" if res_fd.converged else "✗"
    print(f"  {'coupling fd (exact)':<24s}  {res_fd.n_iter:4d}  {res_fd.cost:10.6f}  "
          f"{res_fd.bc_violation:10.2e}  {ms:7.0f}ms {cv}")

    # CSV 저장
    csv_rows = []
    for r in results:
        csv_rows.append({
            "scenario": r["scenario"],
            "norm_Jv": r["norm_Jv_exact"],
            "norm_Mv": r["norm_Mv_num"],
            "ratio_v": r["ratio_v"],
            "cos_sim_v": r["cos_sim_v"],
            "norm_Jr": r["norm_Jr_exact"],
            "norm_Mr": r["norm_Mr_num"],
            "ratio_r": r["ratio_r"],
            "cos_sim_r": r["cos_sim_r"],
        })
    with open("scripts/results/n5_coupling_verify.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  → scripts/results/n5_coupling_verify.csv 저장 완료")


if __name__ == "__main__":
    main()
