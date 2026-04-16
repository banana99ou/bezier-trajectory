"""자체 파라미터 격자 생성 → BLADE-SCP 배치 실행.

collocation DB 없이, 지정된 파라미터 격자에서 케이스를 생성하여
BLADE-SCP로 풀고 blade_simulations 테이블에 저장한다.

Phase 2 (고도 격자), Phase 3 (ω + 이심률 확장) 모두 이 스크립트로 실행.

사용법:
    PYTHONPATH=src python scripts/batch_parametric_grid.py <grid_config.yaml> [옵션]

옵션:
    --workers 8       병렬 워커 수
    --resume          이미 완료된 케이스 건너뛰기
    --limit 100       최대 실행 수
    --db PATH         DB 경로 (기본: scripts/results/simulations.duckdb)

격자 설정 YAML 예시:
    batch_tag: altitude_grid_v1
    K: 12
    n: 2
    max_iter: 50
    tol_bc: 0.001
    l1_lambda: 0.0
    u_max_phys: 0.01
    grid:
      h0: [300, 350, 400, ..., 1200]
      delta_a: [-500, -222, 56, ..., 2000]
      delta_i: [0, 3.75, 7.5, 11.25, 15]
      T_max_normed: [0.15, 0.27, 0.38, ..., 1.2]
      e0: [0]
      ef: [0]
      aop_dep: [0]
      aop_arr: [0]
"""

import sys
import os
import time
import argparse
import itertools
import multiprocessing as mp
from dataclasses import dataclass

sys.path.insert(0, "src")

import numpy as np
import yaml

from bezier_orbit.normalize import from_orbit, MU_EARTH, R_EARTH
from bezier_orbit.blade.orbit import (
    OrbitBC, BLADEOrbitProblem, solve_blade_scp,
    BLADESCPResult, BLADEValidation,
)
from bezier_orbit.db.store import SimulationStore

# 피크 분류
sys.path.insert(0, "vendor")
OTA_PATH = os.environ.get("OTA_PATH", os.path.expanduser("~/gitlab/orbit-transfer-analysis"))
if os.path.isdir(os.path.join(OTA_PATH, "src")):
    sys.path.insert(0, os.path.join(OTA_PATH, "src"))

def _try_import_classifier():
    try:
        from orbit_transfer.classification.peak_detection import detect_peaks
        from orbit_transfer.classification.classifier import classify_profile
        return detect_peaks, classify_profile
    except ImportError:
        return None, None


# ── 격자 생성 ────────────────────────────────────────────────

def generate_cases(config: dict) -> list[dict]:
    """YAML 설정에서 파라미터 격자의 모든 조합을 생성."""
    grid = config["grid"]

    keys = ["h0", "delta_a", "delta_i", "T_max_normed", "e0", "ef", "aop_dep", "aop_arr"]
    arrays = []
    for k in keys:
        vals = grid.get(k, [0.0])
        if isinstance(vals, (int, float)):
            vals = [vals]
        arrays.append(vals)

    cases = []
    for combo in itertools.product(*arrays):
        cases.append(dict(zip(keys, combo)))
    return cases


# ── 워커 ─────────────────────────────────────────────────────

@dataclass
class GridResult:
    case: dict
    success: bool
    error_msg: str | None = None
    blade_converged: bool = False
    blade_cost: float = float("inf")
    blade_n_iter: int = 0
    blade_bc_violation: float = float("inf")
    blade_status: str = ""
    blade_solve_time: float = 0.0
    blade_n_peaks: int | None = None
    blade_profile_class: int | None = None
    blade_cost_history: list = None
    blade_bc_history: list = None
    blade_p_segments_flat: list = None
    blade_bc_violation_r: float | None = None
    blade_bc_violation_v: float | None = None
    blade_thrust_violation: float | None = None
    blade_ta_opt: float | None = None
    val_bc_rk4: float | None = None
    val_bc_r: float | None = None
    val_bc_v: float | None = None
    val_max_thrust: float | None = None
    val_thrust_viol: float | None = None
    val_energy_err: float | None = None
    val_passed: bool | None = None


def _solve_grid_case(args_tuple):
    """단일 격자 케이스 풀이."""
    case, K, n, max_iter, tol_bc, l1_lambda, u_max_phys = args_tuple

    try:
        h0 = case["h0"]
        a0 = R_EARTH + h0
        af = a0 + case["delta_a"]
        di_rad = np.radians(case["delta_i"])
        aop_dep = np.radians(case.get("aop_dep", 0.0))
        aop_arr = np.radians(case.get("aop_arr", 0.0))

        cu = from_orbit(a0)

        T0 = 2.0 * np.pi * np.sqrt(a0**3 / MU_EARTH)
        T_max = case["T_max_normed"] * T0
        t_f_norm = T_max / cu.TU

        u_max_norm = u_max_phys / cu.AU

        dep = OrbitBC(a=a0, e=case["e0"], inc=0.0, raan=0.0, aop=aop_dep, ta=0.0)
        arr = OrbitBC(a=af, e=case["ef"], inc=di_rad, raan=0.0, aop=aop_arr, ta=0.0)

        prob = BLADEOrbitProblem(
            dep=dep, arr=arr,
            t_f=t_f_norm, K=K, n=n,
            u_max=u_max_norm,
            canonical_units=cu,
            max_iter=max_iter, tol_bc=tol_bc,
            l1_lambda=l1_lambda,
            relax_alpha=0.5, trust_region=5.0,
            n_steps_per_seg=30, coupling_order=1,
            ta_free=True, algebraic_drift=True,
            validate=True,
        )

        t0 = time.perf_counter()
        result = solve_blade_scp(prob)
        solve_time = time.perf_counter() - t0

        # 프로파일 분류
        n_peaks, profile_class = None, None
        detect_peaks, classify_profile = _try_import_classifier()
        if detect_peaks is not None and result.converged:
            try:
                from bezier_orbit.blade.orbit import _propagate_blade_reference
                from bezier_orbit.bezier.basis import bernstein

                deltas = np.full(K, 1.0 / K)
                t_f_phys = cu.dim_time(prob.t_f)
                r0_phys, v0_phys = dep.at_time(0.0)
                x0_norm = np.concatenate([cu.nondim_pos(r0_phys), cu.nondim_vel(v0_phys)])

                seg_trajs = _propagate_blade_reference(
                    result.p_segments, n, K, deltas, prob.t_f,
                    x0_norm, cu.R_earth_star, 30,
                )
                n_per = seg_trajs[0].shape[0]
                N_total = K * (n_per - 1) + 1
                u_norm_arr = np.zeros(N_total)
                idx = 0
                for k in range(K):
                    pk = result.p_segments[k]
                    pts = n_per - 1 if k < K - 1 else n_per
                    for j in range(pts):
                        B = bernstein(n, j / (n_per - 1))
                        u_norm_arr[idx] = np.linalg.norm(B @ pk) * cu.AU
                        idx += 1
                t_arr = np.linspace(0.0, t_f_phys, N_total)
                n_peaks, _, _ = detect_peaks(t_arr, u_norm_arr, t_f_phys)
                profile_class = classify_profile(n_peaks)
            except Exception:
                pass

        gr = GridResult(
            case=case, success=True,
            blade_converged=bool(result.converged),
            blade_cost=float(result.cost),
            blade_n_iter=int(result.n_iter),
            blade_bc_violation=float(result.bc_violation),
            blade_status=result.status,
            blade_solve_time=solve_time,
            blade_n_peaks=n_peaks,
            blade_profile_class=profile_class,
            blade_cost_history=[float(x) for x in result.cost_history],
            blade_bc_history=[float(x) for x in result.bc_history],
            blade_p_segments_flat=[pk.tolist() for pk in result.p_segments],
            blade_bc_violation_r=float(result.bc_violation_r) if result.bc_violation_r is not None else None,
            blade_bc_violation_v=float(result.bc_violation_v) if result.bc_violation_v is not None else None,
            blade_thrust_violation=float(result.thrust_violation) if result.thrust_violation is not None else None,
            blade_ta_opt=float(result.ta_opt) if result.ta_opt is not None else None,
        )

        if result.validation is not None:
            v = result.validation
            gr.val_bc_rk4 = float(v.bc_violation_rk4)
            gr.val_bc_r = float(v.bc_violation_r)
            gr.val_bc_v = float(v.bc_violation_v)
            gr.val_max_thrust = float(v.max_thrust_norm)
            gr.val_thrust_viol = float(v.thrust_violation)
            gr.val_energy_err = float(v.energy_error)
            gr.val_passed = bool(v.passed)

        return gr

    except Exception as e:
        return GridResult(case=case, success=False, error_msg=str(e))


# ── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="파라미터 격자 BLADE 배치")
    parser.add_argument("config_yaml", help="격자 설정 YAML 파일")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    with open(args.config_yaml) as f:
        config = yaml.safe_load(f)

    batch_tag = config["batch_tag"]
    K = config.get("K", 12)
    n = config.get("n", 2)
    max_iter = config.get("max_iter", 50)
    tol_bc = config.get("tol_bc", 1e-3)
    l1_lambda = config.get("l1_lambda", 0.0)
    u_max_phys = config.get("u_max_phys", 0.01)
    n_workers = args.workers or mp.cpu_count()

    db_path = args.db or os.environ.get("RESULTS_DIR", "scripts/results") + "/simulations.duckdb"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    cases = generate_cases(config)
    if args.limit:
        cases = cases[:args.limit]

    print("=" * 70)
    print(f"  파라미터 격자 BLADE 배치 [{batch_tag}]")
    print("=" * 70)
    print(f"  DB: {db_path}")
    print(f"  설정: K={K}, n={n}, l1_lambda={l1_lambda}, u_max={u_max_phys}")
    print(f"  격자 케이스: {len(cases)}개, 워커: {n_workers}")

    store = SimulationStore(db_path)

    # Resume: batch_tag + case 키로 중복 방지
    completed = set()
    if args.resume:
        rows = store.con.execute(
            "SELECT dep_a, dep_e, dep_aop, arr_a, arr_e, arr_inc, arr_aop, t_f "
            "FROM blade_simulations WHERE batch_tag = ?", [batch_tag]
        ).fetchall()
        completed = {tuple(round(x, 6) if x else 0 for x in r) for r in rows}
        print(f"  이미 완료: {len(completed)}개 (건너뜀)")

    def _case_key(case):
        h0 = case["h0"]
        a0 = R_EARTH + h0
        af = a0 + case["delta_a"]
        cu = from_orbit(a0)
        T0 = 2.0 * np.pi * np.sqrt(a0**3 / MU_EARTH)
        t_f_norm = case["T_max_normed"] * T0 / cu.TU
        return (
            round(a0, 6), round(case["e0"], 6), round(np.radians(case.get("aop_dep", 0.0)), 6),
            round(af, 6), round(case["ef"], 6), round(np.radians(case["delta_i"]), 6),
            round(np.radians(case.get("aop_arr", 0.0)), 6), round(t_f_norm, 6),
        )

    if args.resume:
        pending = [c for c in cases if _case_key(c) not in completed]
    else:
        pending = cases

    n_total = len(pending)
    print(f"  실행 대상: {n_total}개")
    print("-" * 70)

    if n_total == 0:
        store.close()
        print("  실행할 케이스 없음")
        return

    work_args = [
        (case, K, n, max_iter, tol_bc, l1_lambda, u_max_phys)
        for case in pending
    ]

    t_batch_start = time.perf_counter()
    n_success = 0
    n_fail = 0
    n_done = 0

    with mp.Pool(processes=n_workers) as pool:
        for gr in pool.imap_unordered(_solve_grid_case, work_args):
            n_done += 1
            case = gr.case

            if not gr.success:
                print(f"  [{n_done}/{n_total}] h0={case['h0']:.0f} Δa={case['delta_a']:+.0f}: [ERROR] {gr.error_msg}")
                n_fail += 1
                continue

            try:
                h0 = case["h0"]
                a0 = R_EARTH + h0
                af = a0 + case["delta_a"]
                di_rad = np.radians(case["delta_i"])
                cu = from_orbit(a0)
                T0 = 2.0 * np.pi * np.sqrt(a0**3 / MU_EARTH)
                t_f_norm = case["T_max_normed"] * T0 / cu.TU
                u_max_norm = u_max_phys / cu.AU

                validation = None
                if gr.val_passed is not None:
                    validation = BLADEValidation(
                        bc_violation_rk4=gr.val_bc_rk4,
                        bc_violation_r=gr.val_bc_r, bc_violation_v=gr.val_bc_v,
                        max_thrust_norm=gr.val_max_thrust,
                        thrust_violation=gr.val_thrust_viol,
                        energy_error=gr.val_energy_err,
                        passed=gr.val_passed, details={},
                    )

                p_segments = [np.array(pk) for pk in gr.blade_p_segments_flat]
                blade_result = BLADESCPResult(
                    p_segments=p_segments,
                    cost=gr.blade_cost, converged=gr.blade_converged,
                    n_iter=gr.blade_n_iter, bc_violation=gr.blade_bc_violation,
                    bc_history=gr.blade_bc_history or [], cost_history=gr.blade_cost_history or [],
                    status=gr.blade_status,
                    bc_violation_r=gr.blade_bc_violation_r, bc_violation_v=gr.blade_bc_violation_v,
                    thrust_violation=gr.blade_thrust_violation,
                    ta_opt=gr.blade_ta_opt, validation=validation,
                )

                prob = BLADEOrbitProblem(
                    dep=OrbitBC(a=a0, e=case["e0"], inc=0.0, raan=0.0,
                                aop=np.radians(case.get("aop_dep", 0.0)), ta=0.0),
                    arr=OrbitBC(a=af, e=case["ef"], inc=di_rad, raan=0.0,
                                aop=np.radians(case.get("aop_arr", 0.0)), ta=0.0),
                    t_f=t_f_norm, K=K, n=n, u_max=u_max_norm,
                    canonical_units=cu, max_iter=max_iter, tol_bc=tol_bc,
                    l1_lambda=l1_lambda,
                    relax_alpha=0.5, trust_region=5.0,
                    n_steps_per_seg=30, coupling_order=1,
                    ta_free=True, algebraic_drift=True,
                )

                store.save_blade_simulation(prob, blade_result, cu, batch_tag=batch_tag)

                status = "CONV" if gr.blade_converged else "FAIL"
                peaks_str = f"peaks={gr.blade_n_peaks}" if gr.blade_n_peaks is not None else ""
                print(f"  [{n_done}/{n_total}] h0={h0:.0f} Δa={case['delta_a']:+.0f} Δi={case['delta_i']:.1f}° "
                      f"T/T₀={case['T_max_normed']:.2f}: [{status}] cost={gr.blade_cost:.6f} "
                      f"{gr.blade_solve_time:.1f}s {peaks_str}")

                if gr.blade_converged:
                    n_success += 1
                else:
                    n_fail += 1

            except Exception as e:
                print(f"  [{n_done}/{n_total}] [DB ERROR] {e}")
                n_fail += 1

    t_batch = time.perf_counter() - t_batch_start
    store.close()

    print("=" * 70)
    print(f"  [{batch_tag}] 완료: {n_success} 수렴 / {n_fail} 실패 / {n_total} 전체")
    print(f"  총 소요: {t_batch:.1f}초 ({t_batch/60:.1f}분)")
    print(f"  결과: {db_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
