"""비공면 궤도전이 + 경로 제약 수치 검증 데이터 생성.

보고서 014용 시나리오 V4–V7 실행 및 결과 출력.

출력: 콘솔 테이블 + DuckDB
"""

import numpy as np
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop, _propagate_reference
from bezier_orbit.scp.drift import DriftConfig
from bezier_orbit.bezier.basis import bernstein_eval, double_int_matrix
from bezier_orbit.normalize import from_orbit
from bezier_orbit.db.store import SimulationStore

_RK4 = DriftConfig(method="rk4")
RESULTS_DIR = "scripts/results"
DB_PATH = f"{RESULTS_DIR}/simulations.duckdb"


def run_scenario(name, r0, v0, rf, vf, t_f, N=12, u_max=None,
                 r_min=None, K_subdiv=8, max_iter=50, store=None):
    """시나리오 실행 및 결과 반환."""
    prob = SCPProblem(
        r0=r0, v0=v0, rf=rf, vf=vf,
        t_f=t_f, N=N, u_max=u_max,
        perturbation_level=0,
        max_iter=max_iter, tol_ctrl=1e-6, tol_bc=1e-3,
        r_min=r_min, r_max=None,
        path_K_subdiv=K_subdiv,
        drift_config=_RK4,
    )
    res = solve_inner_loop(prob)

    # DB 저장
    sim_id = None
    if store is not None:
        cu = from_orbit(1.0, mu=1.0)
        sim_id = store.save_simulation(prob, res, cu)

    # 궤적 반경 프로파일 ‖r(τ)‖
    Ibar_N = double_int_matrix(N)
    ell = np.arange(N + 3) / (N + 2)
    q_r = np.empty((N + 3, 3))
    for k in range(3):
        q_r[:, k] = r0[k] + t_f * v0[k] * ell + t_f**2 * (Ibar_N @ res.P_u_opt[:, k])

    tau_plot = np.linspace(0, 1, 201)
    r_norms = np.array([np.linalg.norm(bernstein_eval(N + 2, q_r, t)) for t in tau_plot])

    # 추력 프로파일 ‖u(τ)‖
    u_norms = np.array([np.linalg.norm(bernstein_eval(N, res.P_u_opt, t)) for t in tau_plot])

    # Delta-V (정규화)
    delta_v = np.sqrt(2 * res.cost * t_f) if res.cost < float("inf") else float("inf")

    return {
        "name": name,
        "converged": res.converged,
        "n_iter": res.n_iter,
        "cost": res.cost,
        "delta_v": delta_v,
        "bc_violation": res.bc_violation,
        "tau_plot": tau_plot,
        "r_norms": r_norms,
        "u_norms": u_norms,
        "cost_history": res.cost_history,
        "ctrl_history": res.ctrl_change_history,
        "u_max": u_max,
        "r_min": r_min,
        "sim_id": sim_id,
    }


def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    store = SimulationStore(DB_PATH)

    results = {}

    # ── V4: 경사각 변경 (i=0° → i=28.5°) ──
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.0, 0.0, np.radians(28.5), 0.0, 0.0, np.pi, mu=1.0)
    results["V4"] = run_scenario("V4: 경사각 변경 (i=0→28.5°)",
                                  r0, v0, rf, vf, t_f=4.0, store=store)

    # ── V5: 원형→타원 + 경사각 ──
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, np.radians(15), 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.3, 0.3, np.radians(15), 0.0, np.radians(90), np.pi, mu=1.0)
    results["V5"] = run_scenario("V5: 원형→타원(e=0.3)+경사각15°",
                                  r0, v0, rf, vf, t_f=5.0, store=store)

    # ── V6: RAAN + 경로 제약 ──
    r0, v0 = keplerian_to_cartesian(1.0, 0.01, np.radians(10), 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.01, np.radians(30), np.radians(20), 0.0, np.pi, mu=1.0)
    results["V6"] = run_scenario("V6: 경사각+RAAN+r_min=0.9",
                                  r0, v0, rf, vf, t_f=5.0, r_min=0.9, store=store)

    # V6 비교용: 경로 제약 없는 경우
    results["V6_nopath"] = run_scenario("V6_nopath: 경사각+RAAN (제약 없음)",
                                         r0, v0, rf, vf, t_f=5.0, store=store)

    # V6 K 영향 분석
    for K in [4, 16]:
        key = f"V6_K{K}"
        results[key] = run_scenario(f"V6 (K={K})",
                                     r0, v0, rf, vf, t_f=5.0, r_min=0.9, K_subdiv=K, store=store)

    # ── V7: SOCP + 비공면 + r_min ──
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, np.radians(20), 0.0, 0.0, np.pi, mu=1.0)
    results["V7"] = run_scenario("V7: SOCP+비공면+r_min=0.9",
                                  r0, v0, rf, vf, t_f=5.0, u_max=3.0, r_min=0.9, store=store)

    # ── 출력 ──
    print("=" * 80)
    print("비공면 궤도전이 수치 검증 결과 (보고서 014)")
    print("=" * 80)

    for key in ["V4", "V5", "V6", "V7"]:
        r = results[key]
        s = "CONV" if r["converged"] else "FAIL"
        sid = f" [sim_id={r['sim_id']}]" if r["sim_id"] else ""
        print(f"\n{r['name']}")
        print(f"  [{s}] iter={r['n_iter']}, cost={r['cost']:.6f}, "
              f"ΔV*={r['delta_v']:.4f}, bc_viol={r['bc_violation']:.2e}{sid}")
        print(f"  ‖r‖ range: [{r['r_norms'].min():.4f}, {r['r_norms'].max():.4f}]")
        if r["u_max"] is not None:
            print(f"  ‖u‖ max: {r['u_norms'].max():.4f} (u_max={r['u_max']})")

    # V6 경로 제약 비교
    print(f"\n--- V6 경로 제약 비교 ---")
    r6 = results["V6"]
    r6n = results["V6_nopath"]
    print(f"  제약 있음: cost={r6['cost']:.6f}, ‖r‖_min={r6['r_norms'].min():.4f}")
    print(f"  제약 없음: cost={r6n['cost']:.6f}, ‖r‖_min={r6n['r_norms'].min():.4f}")
    cost_increase = (r6["cost"] - r6n["cost"]) / r6n["cost"] * 100
    print(f"  비용 증가: {cost_increase:.2f}%")

    # V6 K 영향
    print(f"\n--- V6 세분화 구간 수 K 영향 ---")
    for K in [4, 8, 16]:
        key = f"V6_K{K}" if K != 8 else "V6"
        r = results[key]
        n_constraints = K * (12 + 3)  # N=12
        s = "CONV" if r["converged"] else "FAIL"
        print(f"  K={K:2d}: [{s}] iter={r['n_iter']}, cost={r['cost']:.6f}, "
              f"제약 수={n_constraints}, bc={r['bc_violation']:.2e}")

    # 종합 표
    print(f"\n{'='*80}")
    print("종합 결과표")
    print(f"{'='*80}")
    print(f"{'ID':<8} {'Type':<25} {'Status':<6} {'Iter':>5} {'Cost':>10} "
          f"{'ΔV*':>8} {'BC Viol':>10} {'Constr':<15}")
    print("-" * 90)
    for key in ["V4", "V5", "V6", "V7"]:
        r = results[key]
        s = "CONV" if r["converged"] else "FAIL"
        constr = "QP"
        if r["u_max"] is not None:
            constr = f"SOCP(u≤{r['u_max']})"
        if r["r_min"] is not None:
            constr += f"+r≥{r['r_min']}"
        print(f"{key:<8} {r['name'][4:29]:<25} {s:<6} {r['n_iter']:>5} "
              f"{r['cost']:>10.6f} {r['delta_v']:>8.4f} {r['bc_violation']:>10.2e} {constr:<15}")

    store.close()
    print(f"\nDB 저장: {DB_PATH}")


if __name__ == "__main__":
    main()
