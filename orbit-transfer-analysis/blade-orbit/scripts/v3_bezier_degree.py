"""V3: 베지어 차수(N) 연구 — 정밀도 vs 비용 트레이드오프.

N이 클수록 추력 프로파일의 자유도가 증가하여 비용이 감소하지만,
계산량과 수치 조건수도 증가한다. 최적 N 범위를 식별한다.

출력: scripts/results/v3_degree.csv + DuckDB
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from pathlib import Path
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.bezier.basis import gram_matrix, bernstein_eval
from bezier_orbit.normalize import from_orbit
from bezier_orbit.db.store import SimulationStore

RESULTS_DIR = Path("scripts/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = RESULTS_DIR / "simulations.duckdb"


def gram_condition_number(N):
    """Gram 행렬의 조건수."""
    G = gram_matrix(N)
    eigvals = np.linalg.eigvalsh(G)
    return eigvals[-1] / eigvals[0]


def compute_thrust_smoothness(P_u, N, n_eval=500):
    """추력 프로파일 부드러움 측정 (L2 norm of derivative)."""
    tau = np.linspace(0, 1, n_eval)
    u_eval = bernstein_eval(N, P_u, tau)
    u_diff = np.diff(u_eval, axis=0) / (tau[1] - tau[0])
    smoothness = np.sqrt(np.mean(np.sum(u_diff**2, axis=1)))
    return smoothness


def main():
    print("=" * 70)
    print("V3: 베지어 차수(N) 연구")
    print("=" * 70)

    r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(1.2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)
    cu = from_orbit(1.0, mu=1.0)

    N_values = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # Hohmann time for a=1→1.2
    t_f = 3.62  # ≈ π·sqrt((1+1.2)³/8)

    store = SimulationStore(DB_PATH)
    rows = []
    for N in N_values:
        cond = gram_condition_number(N)

        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=t_f, N=N,
            u_max=None,
            perturbation_level=0,
            max_iter=60,
            tol_ctrl=1e-5, tol_bc=1e-2,
        )
        result = solve_inner_loop(prob)

        if result.cost < float("inf"):
            smoothness = compute_thrust_smoothness(result.P_u_opt, N)
            # ΔV 계산
            tau = np.linspace(0, 1, 500)
            u_eval = bernstein_eval(N, result.P_u_opt, tau)
            u_norm = np.linalg.norm(u_eval, axis=1)
            delta_v = t_f * np.trapezoid(u_norm, tau)
        else:
            smoothness = float("nan")
            delta_v = float("inf")

        # DB 저장
        sim_id = store.save_simulation(prob, result, cu)
        store.save_param_sweep(sim_id, "N", float(N), result)

        status = "CONV" if result.converged else ("FEAS" if result.cost < float("inf") else "FAIL")
        print(f"  N={N:2d}: cost={result.cost:.6f}, ΔV={delta_v:.6f}, iter={result.n_iter}, "
              f"cond(G)={cond:.1e}, time={result.solve_time_s:.2f}s [{status}] [sim_id={sim_id}]")

        rows.append({
            "N": N,
            "cost": result.cost,
            "delta_v": delta_v,
            "n_iter": result.n_iter,
            "converged": result.converged,
            "bc_violation": result.bc_violation,
            "gram_cond": cond,
            "smoothness": smoothness,
            "time_s": result.solve_time_s,
        })

    # CSV
    csv_path = RESULTS_DIR / "v3_degree.csv"
    with open(csv_path, "w") as f:
        f.write("N,cost,delta_v,n_iter,converged,bc_violation,gram_cond,smoothness,time_s\n")
        for r in rows:
            f.write(f"{r['N']},{r['cost']:.8f},{r['delta_v']:.8f},{r['n_iter']},{r['converged']},"
                    f"{r['bc_violation']:.2e},{r['gram_cond']:.6e},{r['smoothness']:.6f},{r['time_s']:.3f}\n")

    store.close()
    print(f"\n결과 저장: {csv_path}")
    print(f"DB 저장: {DB_PATH}")

    # 요약
    print("\n--- 요약 ---")
    valid = [r for r in rows if r["cost"] < float("inf")]
    if valid:
        best = min(valid, key=lambda x: x["cost"])
        print(f"최소 비용: N={best['N']}, cost={best['cost']:.6f}")
        print(f"비용 범위: {min(r['cost'] for r in valid):.6f} ~ {max(r['cost'] for r in valid):.6f}")
        print(f"조건수 범위: {min(r['gram_cond'] for r in valid):.1e} ~ {max(r['gram_cond'] for r in valid):.1e}")


if __name__ == "__main__":
    main()
