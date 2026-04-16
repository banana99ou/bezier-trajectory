"""Diagnostic: drift integral 검증.

자유 낙하 궤적에서 c_v, c_r 계산하고
실제 궤적 종말 상태와 비교하여 경계조건이 올바른지 확인.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.orbit.dynamics import propagate_rk4, eom_twobody_j2
from bezier_orbit.bezier.basis import int_matrix, double_int_matrix
from bezier_orbit.scp.problem import SCPProblem, build_lifted_boundary_constraints
from bezier_orbit.scp.inner_loop import _compute_drift_integrals, _propagate_reference


def main():
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.1, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

    print(f"r0 = {r0}")
    print(f"v0 = {v0}")
    print(f"rf = {rf}")
    print(f"vf = {vf}")

    N = 8
    t_f = 4.0
    prob = SCPProblem(r0=r0, v0=v0, rf=rf, vf=vf, t_f=t_f, N=N)
    Re_star = 1.0

    # 1. 자유 낙하 궤적 (u=0)
    print("\n=== Free-fall trajectory ===")
    ref_traj = _propagate_reference(prob, np.zeros((N+1, 3)), Re_star)
    x_final_free = ref_traj[-1]
    print(f"Free-fall final state: r={x_final_free[:3]}, v={x_final_free[3:]}")
    print(f"Target: r={rf}, v={vf}")
    print(f"Position error: {np.linalg.norm(x_final_free[:3] - rf):.6f}")
    print(f"Velocity error: {np.linalg.norm(x_final_free[3:] - vf):.6f}")

    # 2. Drift integrals
    print("\n=== Drift integrals ===")
    c_v, c_r = _compute_drift_integrals(prob, ref_traj, Re_star)
    print(f"c_v = {c_v}")
    print(f"c_r = {c_r}")
    print(f"t_f * c_v = {t_f * c_v}")
    print(f"t_f^2 * c_r = {t_f**2 * c_r}")

    # 3. 경계조건 확인 (Z=0일 때 잔차)
    print("\n=== Boundary condition residuals (Z=0) ===")
    A_eq, b_eq = build_lifted_boundary_constraints(prob, c_v=c_v, c_r=c_r)
    z_zero = np.zeros(3 * (N+1))
    residual = A_eq @ z_zero - b_eq
    print(f"Velocity residual: {residual[:3]}")
    print(f"Position residual: {residual[3:]}")
    print(f"b_eq (velocity): {b_eq[:3]}")
    print(f"b_eq (position): {b_eq[3:]}")

    # 4. 해석적 검증: v_f ≈ v_0 + t_f*c_v + 0  (Z=0이면 eI·Z=0)
    v_predicted = v0 + t_f * c_v
    r_predicted = r0 + t_f * v0 + t_f**2 * c_r

    print("\n=== Analytical check ===")
    print(f"v_predicted (v0 + t_f*c_v) = {v_predicted}")
    print(f"v_actual (free-fall end)    = {x_final_free[3:]}")
    print(f"v_target                    = {vf}")
    print(f"  v_pred vs actual error: {np.linalg.norm(v_predicted - x_final_free[3:]):.6e}")

    print(f"r_predicted (r0 + t_f*v0 + t_f²*c_r) = {r_predicted}")
    print(f"r_actual (free-fall end)               = {x_final_free[:3]}")
    print(f"r_target                               = {rf}")
    print(f"  r_pred vs actual error: {np.linalg.norm(r_predicted - x_final_free[:3]):.6e}")

    # 5. Z=0 해석: v0 + t_f*c_v이 free-fall 끝 속도와 일치해야
    # 경계조건에서 필요한 Z = ?
    I_N = int_matrix(N)
    Ibar_N = double_int_matrix(N)
    eI = I_N[N+1, :]
    eIbar = Ibar_N[N+2, :]
    print(f"\neI = {eI}")
    print(f"eIbar = {eIbar}")
    print(f"sum(eI) = {np.sum(eI):.6f}")
    print(f"sum(eIbar) = {np.sum(eIbar):.6f}")

    # eI 의 의미: ∫₀¹ B_{N+1}(τ)^T dτ 의 마지막 행
    # For Z = const, eI @ Z[:,k] = sum(eI) * Z_k → just sum(eI) times scalar
    # eI should sum to 1 (partition of unity of integrated basis)
    # eIbar should sum to 0.5 (double integration of constant)

    # 6. 필요한 ΔV 크기 추정
    dv_needed = vf - v_predicted  # v_f - (v0 + t_f*c_v) = needed correction
    dr_needed = rf - r_predicted  # r_f - (r0 + t_f*v0 + t_f²*c_r)
    print(f"\nNeeded velocity correction: {dv_needed}, norm={np.linalg.norm(dv_needed):.6f}")
    print(f"Needed position correction: {dr_needed}, norm={np.linalg.norm(dr_needed):.6f}")


if __name__ == "__main__":
    main()
