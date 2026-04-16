"""V2: SCP 수렴 특성 분석.

QP/SOCP 각각에 대해:
- 비용 함수 수렴 이력
- 제어 변화량 수렴
- BC 위반량 변화
- 반복 횟수 vs N, t_f

출력: scripts/results/v2_convergence.csv + DuckDB
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from pathlib import Path
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.normalize import from_orbit
from bezier_orbit.db.store import SimulationStore

RESULTS_DIR = Path("scripts/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = RESULTS_DIR / "simulations.duckdb"


def run_convergence_study(store):
    """SCP 수렴 특성 분석."""
    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
    cu = from_orbit(1.0, mu=1.0)

    # 파라미터 공간
    N_values = [6, 8, 12, 16]
    t_f_values = [3.0, 3.38, 4.0, 5.0]
    u_max_cases = [("QP", None), ("SOCP_5", 5.0)]

    rows = []
    for mode_name, u_max in u_max_cases:
        print(f"\n=== Mode: {mode_name} ===")
        for N in N_values:
            for t_f in t_f_values:
                prob = SCPProblem(
                    r0=r0, v0=v0, rf=rf, vf=vf,
                    t_f=t_f, N=N,
                    u_max=u_max,
                    perturbation_level=0,
                    max_iter=60,
                    tol_ctrl=1e-5, tol_bc=1e-2,
                )
                result = solve_inner_loop(prob)

                status = "CONV" if result.converged else ("FEAS" if result.cost < float("inf") else "FAIL")

                # DB 저장
                sim_id = store.save_simulation(prob, result, cu)
                store.save_param_sweep(sim_id, "N", float(N), result)
                store.save_param_sweep(sim_id, "t_f", t_f, result)

                print(f"  N={N:2d}, t_f={t_f:.1f}: cost={result.cost:.6f}, iter={result.n_iter}, bc={result.bc_violation:.2e} [{status}] [sim_id={sim_id}]")

                rows.append({
                    "mode": mode_name,
                    "N": N,
                    "t_f": t_f,
                    "cost": result.cost,
                    "n_iter": result.n_iter,
                    "converged": result.converged,
                    "bc_violation": result.bc_violation,
                    "cost_history": result.cost_history,
                    "ctrl_history": result.ctrl_change_history,
                })

    return rows


def main():
    print("=" * 70)
    print("V2: SCP 수렴 특성 분석")
    print("=" * 70)

    store = SimulationStore(DB_PATH)
    rows = run_convergence_study(store)

    # CSV 요약
    csv_path = RESULTS_DIR / "v2_convergence.csv"
    with open(csv_path, "w") as f:
        f.write("mode,N,t_f,cost,n_iter,converged,bc_violation,final_ctrl_change\n")
        for r in rows:
            final_ctrl = r["ctrl_history"][-1] if r["ctrl_history"] else float("nan")
            f.write(f"{r['mode']},{r['N']},{r['t_f']:.1f},{r['cost']:.8f},"
                    f"{r['n_iter']},{r['converged']},{r['bc_violation']:.2e},{final_ctrl:.2e}\n")
    print(f"\n결과 저장: {csv_path}")

    # 수렴 이력 상세 (최적 케이스)
    csv_detail = RESULTS_DIR / "v2_convergence_history.csv"
    with open(csv_detail, "w") as f:
        f.write("mode,N,t_f,iteration,cost,ctrl_change\n")
        for r in rows:
            for i, (c, ctrl) in enumerate(zip(r["cost_history"], r["ctrl_history"])):
                f.write(f"{r['mode']},{r['N']},{r['t_f']:.1f},{i+1},{c:.8f},{ctrl:.2e}\n")
    print(f"수렴 이력 저장: {csv_detail}")

    store.close()
    print(f"DB 저장: {DB_PATH}")


if __name__ == "__main__":
    main()
