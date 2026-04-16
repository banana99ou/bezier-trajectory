"""S1: 3차원 궤도 전이 시나리오 (경사각 변화 포함).

동일 평면 전이(2D)를 넘어, 궤도면 변화를 포함하는
3차원 전이의 SCP 해를 분석한다.

시나리오:
- S1a: 순수 경사각 변경 (Δi = 5°, 10°, 20°)
- S1b: 고도 + 경사각 동시 변경
- S1c: 다양한 t_f에서의 3D 전이 자유시간 최적화

출력: scripts/results/s1_3d_transfer.csv + DuckDB
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from pathlib import Path
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.scp.outer_loop import grid_search
from bezier_orbit.bezier.basis import bernstein_eval
from bezier_orbit.normalize import from_orbit
from bezier_orbit.db.store import SimulationStore

RESULTS_DIR = Path("scripts/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = RESULTS_DIR / "simulations.duckdb"


def compute_delta_v(P_u, t_f, N, n_eval=500):
    tau = np.linspace(0, 1, n_eval)
    u_eval = bernstein_eval(N, P_u, tau)
    u_norm = np.linalg.norm(u_eval, axis=1)
    return t_f * np.trapezoid(u_norm, tau)


def main():
    print("=" * 70)
    print("S1: 3차원 궤도 전이 시나리오")
    print("=" * 70)

    N = 12
    rows = []

    store = SimulationStore(DB_PATH)

    # ── S1a: 순수 경사각 변경 ──
    print("\n--- S1a: 순수 경사각 변경 ---")
    inc_changes = [5.0, 10.0, 15.0, 20.0, 28.5]
    for di_deg in inc_changes:
        di_rad = np.radians(di_deg)
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(1.0, 0.0, di_rad, 0.0, 0.0, np.pi, mu=1.0)

        # t_f 격자 탐색
        t_f_grid = np.array([3.0, 4.0, 5.0, 6.0, 8.0])
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=N, u_max=None,
            perturbation_level=0,
            max_iter=60, tol_ctrl=1e-5, tol_bc=1e-2,
        )

        outer = grid_search(prob, t_f_grid)
        best_cost = outer.inner_result.cost if outer.inner_result else float("inf")
        best_tf = outer.t_f_opt

        if outer.inner_result and best_cost < float("inf"):
            dv = compute_delta_v(outer.inner_result.P_u_opt, best_tf, N)
        else:
            dv = float("inf")

        # DB 저장
        cu = from_orbit(1.0, mu=1.0)
        prob_best = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=best_tf, N=N, u_max=None,
            perturbation_level=0,
            max_iter=60, tol_ctrl=1e-5, tol_bc=1e-2,
        )
        sim_id = store.save_simulation(prob_best, outer.inner_result, cu)
        store.save_outer_loop(sim_id, "grid", (t_f_grid[0], t_f_grid[-1]), outer)

        print(f"  Δi={di_deg:5.1f}°: t_f*={best_tf:.2f}, cost={best_cost:.6f}, ΔV={dv:.6f} [sim_id={sim_id}]")
        rows.append({
            "scenario": "S1a_inclination",
            "a_target": 1.0,
            "di_deg": di_deg,
            "da": 0.0,
            "t_f_opt": best_tf,
            "cost": best_cost,
            "delta_v": dv,
            "converged": outer.inner_result.converged if outer.inner_result else False,
        })

    # ── S1b: 고도 + 경사각 동시 변경 ──
    print("\n--- S1b: 고도 + 경사각 동시 변경 ---")
    combined = [
        (1.1, 5.0),
        (1.1, 10.0),
        (1.2, 5.0),
        (1.2, 10.0),
        (1.3, 10.0),
    ]
    for a_target, di_deg in combined:
        di_rad = np.radians(di_deg)
        r0, v0 = keplerian_to_cartesian(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
        rf, vf = keplerian_to_cartesian(a_target, 0.0, di_rad, 0.0, 0.0, np.pi, mu=1.0)

        t_f_grid = np.array([3.0, 4.0, 5.0, 6.0, 8.0])
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=4.0, N=N, u_max=None,
            perturbation_level=0,
            max_iter=60, tol_ctrl=1e-5, tol_bc=1e-2,
        )

        outer = grid_search(prob, t_f_grid)
        best_cost = outer.inner_result.cost if outer.inner_result else float("inf")
        best_tf = outer.t_f_opt

        if outer.inner_result and best_cost < float("inf"):
            dv = compute_delta_v(outer.inner_result.P_u_opt, best_tf, N)
        else:
            dv = float("inf")

        # DB 저장
        cu = from_orbit(1.0, mu=1.0)
        prob_best = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=best_tf, N=N, u_max=None,
            perturbation_level=0,
            max_iter=60, tol_ctrl=1e-5, tol_bc=1e-2,
        )
        sim_id = store.save_simulation(prob_best, outer.inner_result, cu)
        store.save_outer_loop(sim_id, "grid", (t_f_grid[0], t_f_grid[-1]), outer)

        print(f"  a={a_target}, Δi={di_deg:5.1f}°: t_f*={best_tf:.2f}, cost={best_cost:.6f}, ΔV={dv:.6f} [sim_id={sim_id}]")
        rows.append({
            "scenario": "S1b_combined",
            "a_target": a_target,
            "di_deg": di_deg,
            "da": a_target - 1.0,
            "t_f_opt": best_tf,
            "cost": best_cost,
            "delta_v": dv,
            "converged": outer.inner_result.converged if outer.inner_result else False,
        })

    # CSV (하위 호환)
    csv_path = RESULTS_DIR / "s1_3d_transfer.csv"
    with open(csv_path, "w") as f:
        f.write("scenario,a_target,di_deg,da,t_f_opt,cost,delta_v,converged\n")
        for r in rows:
            f.write(f"{r['scenario']},{r['a_target']},{r['di_deg']:.1f},{r['da']:.2f},"
                    f"{r['t_f_opt']:.4f},{r['cost']:.8f},{r['delta_v']:.8f},{r['converged']}\n")

    store.close()
    print(f"\n결과 저장: {csv_path}")
    print(f"DB 저장: {DB_PATH}")


if __name__ == "__main__":
    main()
