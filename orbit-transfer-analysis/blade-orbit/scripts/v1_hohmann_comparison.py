"""V1: Hohmann 전이와 베지어-SCP 해의 비교 검증.

Hohmann 전이는 해석적으로 최적인 2-임펄스 전이이므로,
유한 추력(연속) 최적화가 반환하는 ΔV가 이론적 하한(Hohmann)과
비교 가능한지 검증한다.

출력: scripts/results/v1_hohmann.npz + DuckDB
"""

import sys
sys.path.insert(0, "src")

import numpy as np
from pathlib import Path
from bezier_orbit.normalize import from_orbit
from bezier_orbit.orbit.elements import keplerian_to_cartesian
from bezier_orbit.scp.problem import SCPProblem
from bezier_orbit.scp.inner_loop import solve_inner_loop
from bezier_orbit.bezier.basis import bernstein_eval
from bezier_orbit.db.store import SimulationStore

RESULTS_DIR = Path("scripts/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = RESULTS_DIR / "simulations.duckdb"


def hohmann_delta_v(a1, a2, mu=1.0):
    """Hohmann 전이 ΔV (정규화 단위)."""
    v1 = np.sqrt(mu / a1)
    v2 = np.sqrt(mu / a2)
    v_t1 = np.sqrt(2 * mu * a2 / (a1 * (a1 + a2)))
    v_t2 = np.sqrt(2 * mu * a1 / (a2 * (a1 + a2)))
    dv1 = abs(v_t1 - v1)
    dv2 = abs(v2 - v_t2)
    tof = np.pi * np.sqrt((a1 + a2)**3 / (8 * mu))
    return dv1, dv2, dv1 + dv2, tof


def compute_bezier_delta_v(P_u, t_f, N, n_eval=500):
    """베지어 추력 프로파일로부터 ΔV 계산 (수치 적분)."""
    tau = np.linspace(0, 1, n_eval)
    u_eval = bernstein_eval(N, P_u, tau)  # (n_eval, 3)
    u_norm = np.linalg.norm(u_eval, axis=1)
    # ΔV = t_f * ∫₀¹ ‖u(τ)‖ dτ
    delta_v = t_f * np.trapezoid(u_norm, tau)
    return delta_v, tau, u_eval


def run_scenario(a_target, N=12, t_f_values=None, max_iter=60, store=None):
    """단일 시나리오: 원형→원형 전이."""
    a1 = 1.0  # 정규화
    a2 = a_target

    dv1_h, dv2_h, dv_total_h, tof_h = hohmann_delta_v(a1, a2)

    r0, v0 = keplerian_to_cartesian(a1, 0.0, 0.0, 0.0, 0.0, 0.0, mu=1.0)
    rf, vf = keplerian_to_cartesian(a2, 0.0, 0.0, 0.0, 0.0, np.pi, mu=1.0)

    if t_f_values is None:
        t_f_values = np.array([0.8 * tof_h, 0.9 * tof_h, tof_h, 1.1 * tof_h, 1.2 * tof_h, 1.5 * tof_h])

    cu = from_orbit(1.0, mu=1.0)
    results = []
    for t_f in t_f_values:
        prob = SCPProblem(
            r0=r0, v0=v0, rf=rf, vf=vf,
            t_f=t_f, N=N,
            u_max=None,
            perturbation_level=0,
            max_iter=max_iter,
            tol_ctrl=1e-5, tol_bc=1e-2,
        )
        result = solve_inner_loop(prob)

        if result.cost < float("inf"):
            dv_scp, tau, u_eval = compute_bezier_delta_v(result.P_u_opt, t_f, N)
        else:
            dv_scp = float("inf")
            tau, u_eval = None, None

        # DB 저장
        sim_id = None
        if store is not None:
            sim_id = store.save_simulation(prob, result, cu)
            store.save_param_sweep(sim_id, "t_f", t_f, result)

        results.append({
            "t_f": t_f,
            "t_f_ratio": t_f / tof_h,
            "cost": result.cost,
            "dv_scp": dv_scp,
            "converged": result.converged,
            "n_iter": result.n_iter,
            "bc_violation": result.bc_violation,
            "sim_id": sim_id,
        })

        status = "OK" if result.cost < float("inf") else "FAIL"
        sid = f" [sim_id={sim_id}]" if sim_id else ""
        print(f"  t_f={t_f:.3f} (ratio={t_f/tof_h:.2f}): ΔV_scp={dv_scp:.6f}, cost={result.cost:.6f}, iter={result.n_iter} [{status}]{sid}")

    return dv_total_h, tof_h, results


def main():
    print("=" * 70)
    print("V1: Hohmann 전이 vs 베지어-SCP 비교")
    print("=" * 70)

    store = SimulationStore(DB_PATH)

    scenarios = [
        ("LEO → LEO+10%", 1.1),
        ("LEO → LEO+20%", 1.2),
        ("LEO → LEO+50%", 1.5),
    ]

    all_results = {}
    for name, a_target in scenarios:
        print(f"\n--- {name} (a_target = {a_target}) ---")
        dv_h, tof_h, results = run_scenario(a_target, N=12, store=store)
        print(f"  Hohmann: ΔV = {dv_h:.6f}, ToF = {tof_h:.4f}")

        for r in results:
            if r["dv_scp"] < float("inf"):
                ratio = r["dv_scp"] / dv_h
                print(f"    t_f/ToF_H={r['t_f_ratio']:.2f}: ΔV_scp/ΔV_H = {ratio:.4f}")

        all_results[name] = {
            "a_target": a_target,
            "dv_hohmann": dv_h,
            "tof_hohmann": tof_h,
            "scp_results": results,
        }

    # 저장
    np.savez(RESULTS_DIR / "v1_hohmann.npz", **{
        k: v for k, v in [
            ("scenario_names", [s[0] for s in scenarios]),
            ("a_targets", [s[1] for s in scenarios]),
            ("dv_hohmanns", [all_results[s[0]]["dv_hohmann"] for s in scenarios]),
            ("tof_hohmanns", [all_results[s[0]]["tof_hohmann"] for s in scenarios]),
        ]
    })

    # CSV 요약
    csv_path = RESULTS_DIR / "v1_hohmann_summary.csv"
    with open(csv_path, "w") as f:
        f.write("scenario,a_target,dv_hohmann,t_f,t_f_ratio,dv_scp,dv_ratio,cost,converged,n_iter,bc_violation\n")
        for name, a_target in scenarios:
            d = all_results[name]
            for r in d["scp_results"]:
                ratio = r["dv_scp"] / d["dv_hohmann"] if r["dv_scp"] < float("inf") else float("inf")
                f.write(f"{name},{a_target},{d['dv_hohmann']:.8f},{r['t_f']:.6f},{r['t_f_ratio']:.4f},"
                        f"{r['dv_scp']:.8f},{ratio:.6f},{r['cost']:.8f},{r['converged']},{r['n_iter']},{r['bc_violation']:.2e}\n")

    store.close()
    print(f"\n결과 저장: {csv_path}")
    print(f"DB 저장: {DB_PATH}")


if __name__ == "__main__":
    main()
