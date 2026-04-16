"""Collocation DB의 전체 케이스를 BLADE-SCP로 재풀이하여 비교 저장.

orbit-transfer-analysis의 trajectories_all.duckdb에서 구성 파라미터를 읽고,
동일 조건으로 BLADE-SCP를 실행한 뒤 blade-orbit의 DB에 나란히 저장한다.

멀티프로세스로 CPU 코어를 최대 활용한다.

사용법:
    PYTHONPATH=src python scripts/batch_blade_vs_collocation.py [옵션]

옵션:
    --h0 400          특정 고도만 실행 (400/600/800/1000)
    --limit 10        최대 실행 케이스 수
    --converged-only  collocation에서 수렴한 케이스만
    --resume          이미 완료된 케이스 건너뛰기
    --config name     실행 설정 이름 (예: baseline_K12n2)
    --K 12            BLADE 세그먼트 수 (기본 12)
    --n 2             BLADE 세그먼트 차수 (기본 2)
    --workers 8       병렬 워커 수 (기본: CPU 코어 수)
"""

import sys
import os
import time
import argparse
import multiprocessing as mp
from dataclasses import dataclass

sys.path.insert(0, "src")
# orbit-transfer-analysis: 로컬이면 ~/gitlab/..., 컨테이너면 vendor/
OTA_PATH = os.environ.get("OTA_PATH", os.path.expanduser("~/gitlab/orbit-transfer-analysis"))
if os.path.isdir(os.path.join(OTA_PATH, "src")):
    sys.path.insert(0, os.path.join(OTA_PATH, "src"))
# vendor/ 경로 (gpu-farm 컨테이너용)
if os.path.isdir("vendor"):
    sys.path.insert(0, "vendor")

import numpy as np
import duckdb

from bezier_orbit.normalize import from_orbit, MU_EARTH, R_EARTH
from bezier_orbit.blade.orbit import (
    OrbitBC, BLADEOrbitProblem, solve_blade_scp,
    BLADESCPResult,
)
from bezier_orbit.db.store import SimulationStore

# ── 경로 설정 ──────────────────────────────────────────────────

# Collocation DB: 로컬이면 OTA 경로, 컨테이너면 data/
_default_colloc = os.path.join(OTA_PATH, "data", "trajectories_all.duckdb")
if not os.path.exists(_default_colloc):
    _default_colloc = "data/trajectories_all.duckdb"
COLLOC_DB = os.environ.get("COLLOC_DB", _default_colloc)

RESULTS_DIR = os.environ.get("RESULTS_DIR", "scripts/results")
BLADE_DB = os.path.join(RESULTS_DIR, "simulations.duckdb")


# ── 피크 분류 (import 실패 시 graceful) ───────────────────────

def _try_import_classifier():
    try:
        from orbit_transfer.classification.peak_detection import detect_peaks
        from orbit_transfer.classification.classifier import classify_profile
        return detect_peaks, classify_profile
    except ImportError:
        return None, None


# ── 데이터 로드 ───────────────────────────────────────────────

def load_collocation_cases(
    colloc_db: str,
    h0: float | None = None,
    converged_only: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """Collocation DB에서 케이스 로드."""
    con = duckdb.connect(colloc_db, read_only=True)

    query = "SELECT * FROM trajectories"
    conditions = []
    if h0 is not None:
        conditions.append(f"h0 = {h0}")
    if converged_only:
        conditions.append("converged = true")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id"
    if limit is not None:
        query += f" LIMIT {limit}"

    rows = con.execute(query).fetchall()
    columns = [desc[0] for desc in con.description]
    con.close()

    return [dict(zip(columns, row)) for row in rows]


# ── 워커 함수 (프로세스별 실행) ───────────────────────────────

@dataclass
class WorkerResult:
    """워커가 반환하는 단일 케이스 결과."""
    colloc_id: int
    case: dict
    success: bool
    error_msg: str | None = None
    # BLADE 결과 (직렬화 가능한 형태)
    blade_converged: bool = False
    blade_cost: float = float("inf")
    blade_n_iter: int = 0
    blade_bc_violation: float = float("inf")
    blade_status: str = ""
    blade_solve_time: float = 0.0
    blade_n_peaks: int | None = None
    blade_profile_class: int | None = None
    # 직렬화된 BLADE 결과 (DB 저장용)
    blade_cost_history: list = None
    blade_bc_history: list = None
    blade_p_segments_flat: list = None
    blade_bc_violation_r: float | None = None
    blade_bc_violation_v: float | None = None
    blade_thrust_violation: float | None = None
    blade_ta_opt: float | None = None
    # validation
    val_bc_rk4: float | None = None
    val_bc_r: float | None = None
    val_bc_v: float | None = None
    val_max_thrust: float | None = None
    val_thrust_viol: float | None = None
    val_energy_err: float | None = None
    val_passed: bool | None = None


def _solve_single(args_tuple):
    """단일 케이스 풀이 (워커 프로세스에서 실행).

    pickle 가능한 인자/결과만 사용해야 한다.
    """
    case, K, n, max_iter, tol_bc, l1_lambda = args_tuple
    cid = case["id"]

    try:
        h0 = case["h0"]
        a0 = R_EARTH + h0
        af = a0 + case["delta_a"]
        di_rad = np.radians(case["delta_i"])

        cu = from_orbit(a0)

        T0 = 2.0 * np.pi * np.sqrt(a0**3 / MU_EARTH)
        T_max = case["T_max_normed"] * T0
        t_f_norm = T_max / cu.TU

        u_max_phys = 0.01
        u_max_norm = u_max_phys / cu.AU

        dep = OrbitBC(a=a0, e=case["e0"], inc=0.0, raan=0.0, aop=0.0, ta=0.0)
        arr = OrbitBC(a=af, e=case["ef"], inc=di_rad, raan=0.0, aop=0.0, ta=0.0)

        prob = BLADEOrbitProblem(
            dep=dep, arr=arr,
            t_f=t_f_norm, K=K, n=n,
            u_max=u_max_norm,
            canonical_units=cu,
            max_iter=max_iter, tol_bc=tol_bc,
            relax_alpha=0.5, trust_region=5.0,
            l1_lambda=l1_lambda,
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

        wr = WorkerResult(
            colloc_id=cid, case=case, success=True,
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
            wr.val_bc_rk4 = float(v.bc_violation_rk4)
            wr.val_bc_r = float(v.bc_violation_r)
            wr.val_bc_v = float(v.bc_violation_v)
            wr.val_max_thrust = float(v.max_thrust_norm)
            wr.val_thrust_viol = float(v.thrust_violation)
            wr.val_energy_err = float(v.energy_error)
            wr.val_passed = bool(v.passed)

        return wr

    except Exception as e:
        return WorkerResult(
            colloc_id=cid, case=case, success=False, error_msg=str(e),
        )


# ── 메인 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BLADE vs Collocation 배치 비교")
    parser.add_argument("--h0", type=float, default=None, help="특정 고도만 실행")
    parser.add_argument("--limit", type=int, default=None, help="최대 케이스 수")
    parser.add_argument("--converged-only", action="store_true", help="수렴 케이스만")
    parser.add_argument("--resume", action="store_true", help="완료된 케이스 건너뛰기")
    parser.add_argument("--config", type=str, default=None,
                        help="실행 설정 이름 (예: baseline_K12n2)")
    parser.add_argument("--K", type=int, default=12, help="BLADE 세그먼트 수")
    parser.add_argument("--n", type=int, default=2, help="BLADE 세그먼트 차수")
    parser.add_argument("--max-iter", type=int, default=50, help="SCP 최대 반복")
    parser.add_argument("--tol-bc", type=float, default=1e-3, help="BC 허용치")
    parser.add_argument("--l1-lambda", type=float, default=0.0,
                        help="L1 정규화 강도 (0=에너지 최적, >0=코스팅 유도)")
    parser.add_argument("--workers", type=int, default=None,
                        help="병렬 워커 수 (기본: CPU 코어 수)")
    args = parser.parse_args()

    n_workers = args.workers or mp.cpu_count()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  BLADE vs Collocation 배치 비교 (멀티프로세스)")
    print("=" * 70)
    print(f"  Collocation DB: {COLLOC_DB}")
    print(f"  BLADE DB:       {BLADE_DB}")
    print(f"  BLADE 설정: K={args.K}, n={args.n}, max_iter={args.max_iter}, tol_bc={args.tol_bc}, l1_lambda={args.l1_lambda}")
    print(f"  워커 수: {n_workers}")

    # Collocation 케이스 로드
    cases = load_collocation_cases(
        COLLOC_DB, h0=args.h0,
        converged_only=args.converged_only,
        limit=args.limit,
    )
    print(f"  로드된 케이스: {len(cases)}개")

    store = SimulationStore(BLADE_DB)

    # 실행 설정 등록/조회
    l1_suffix = f"_l1_{args.l1_lambda:.0e}" if args.l1_lambda > 0 else ""
    config_name = args.config or f"K{args.K}_n{args.n}_tol{args.tol_bc:.0e}{l1_suffix}"
    config_id = store.get_or_create_config(
        name=config_name,
        K=args.K, n=args.n,
        max_iter=args.max_iter, tol_bc=args.tol_bc,
        l1_lambda=args.l1_lambda,
        description=f"K={args.K}, n={args.n}, max_iter={args.max_iter}, tol_bc={args.tol_bc}, l1_lambda={args.l1_lambda}",
    )
    print(f"  실행 설정: '{config_name}' (config_id={config_id})")

    # Resume: (colloc_id, h0) 기준
    completed = set()
    if args.resume:
        completed = store.get_completed_colloc_keys(config_id)
        print(f"  이미 완료: {len(completed)}개 (건너뜀)")

    pending = [c for c in cases if (c["id"], c["h0"]) not in completed]
    n_total = len(pending)
    print(f"  실행 대상: {n_total}개")
    print("-" * 70)

    if n_total == 0:
        store.close()
        print("  실행할 케이스 없음")
        return

    # ── 멀티프로세스 실행 ─────────────────────────────────────
    work_args = [
        (case, args.K, args.n, args.max_iter, args.tol_bc, args.l1_lambda)
        for case in pending
    ]

    t_batch_start = time.perf_counter()
    n_success = 0
    n_fail = 0
    n_done = 0

    with mp.Pool(processes=n_workers) as pool:
        for wr in pool.imap_unordered(_solve_single, work_args):
            n_done += 1
            cid = wr.colloc_id
            case = wr.case

            if not wr.success:
                print(f"  [{n_done}/{n_total}] colloc_id={cid}: [ERROR] {wr.error_msg}")
                n_fail += 1
                continue

            # DB 저장 (메인 프로세스에서 — DuckDB는 단일 writer)
            try:
                h0 = case["h0"]
                cu = from_orbit(R_EARTH + h0)

                # BLADESCPResult 재구성 (직렬화된 데이터로부터)
                from bezier_orbit.blade.orbit import BLADESCPResult, BLADEValidation
                p_segments = [np.array(pk) for pk in wr.blade_p_segments_flat]

                validation = None
                if wr.val_passed is not None:
                    validation = BLADEValidation(
                        bc_violation_rk4=wr.val_bc_rk4,
                        bc_violation_r=wr.val_bc_r,
                        bc_violation_v=wr.val_bc_v,
                        max_thrust_norm=wr.val_max_thrust,
                        thrust_violation=wr.val_thrust_viol,
                        energy_error=wr.val_energy_err,
                        passed=wr.val_passed,
                        details={},
                    )

                blade_result = BLADESCPResult(
                    p_segments=p_segments,
                    cost=wr.blade_cost,
                    converged=wr.blade_converged,
                    n_iter=wr.blade_n_iter,
                    bc_violation=wr.blade_bc_violation,
                    bc_history=wr.blade_bc_history or [],
                    cost_history=wr.blade_cost_history or [],
                    status=wr.blade_status,
                    bc_violation_r=wr.blade_bc_violation_r,
                    bc_violation_v=wr.blade_bc_violation_v,
                    thrust_violation=wr.blade_thrust_violation,
                    ta_opt=wr.blade_ta_opt,
                    validation=validation,
                )

                # prob 재구성 (DB 저장용)
                a0 = R_EARTH + h0
                af = a0 + case["delta_a"]
                di_rad = np.radians(case["delta_i"])
                T0 = 2.0 * np.pi * np.sqrt(a0**3 / MU_EARTH)
                t_f_norm = case["T_max_normed"] * T0 / cu.TU
                u_max_norm = 0.01 / cu.AU

                prob = BLADEOrbitProblem(
                    dep=OrbitBC(a=a0, e=case["e0"], inc=0.0, raan=0.0, aop=0.0, ta=0.0),
                    arr=OrbitBC(a=af, e=case["ef"], inc=di_rad, raan=0.0, aop=0.0, ta=0.0),
                    t_f=t_f_norm, K=args.K, n=args.n,
                    u_max=u_max_norm, canonical_units=cu,
                    max_iter=args.max_iter, tol_bc=args.tol_bc,
                    l1_lambda=args.l1_lambda,
                    relax_alpha=0.5, trust_region=5.0,
                    n_steps_per_seg=30, coupling_order=1,
                    ta_free=True, algebraic_drift=True,
                )

                blade_id = store.save_blade_simulation(prob, blade_result, cu)
                cmp_id = store.save_comparison(
                    colloc_row=case, blade_id=blade_id,
                    blade_result=blade_result,
                    colloc_db_path=COLLOC_DB,
                    blade_n_peaks=wr.blade_n_peaks,
                    blade_profile_class=wr.blade_profile_class,
                    config_id=config_id,
                )
                store.update_comparison_solve_time(cmp_id, wr.blade_solve_time)

                status = "CONV" if wr.blade_converged else "FAIL"
                cost_str = f"cost={wr.blade_cost:.6f}" if wr.blade_cost < float("inf") else "cost=inf"
                peaks_str = f"peaks={wr.blade_n_peaks}" if wr.blade_n_peaks is not None else ""
                print(f"  [{n_done}/{n_total}] colloc_id={cid}: [{status}] {cost_str}, "
                      f"iter={wr.blade_n_iter}, bc={wr.blade_bc_violation:.2e}, "
                      f"{wr.blade_solve_time:.1f}s {peaks_str}")

                if wr.blade_converged:
                    n_success += 1
                else:
                    n_fail += 1

            except Exception as e:
                print(f"  [{n_done}/{n_total}] colloc_id={cid}: [DB ERROR] {e}")
                n_fail += 1

    t_batch = time.perf_counter() - t_batch_start
    store.close()

    print("=" * 70)
    print(f"  완료: {n_success} 수렴 / {n_fail} 실패 / {n_total} 전체")
    print(f"  총 소요: {t_batch:.1f}초 ({t_batch/60:.1f}분)")
    print(f"  평균 처리량: {n_total/t_batch:.1f} 케이스/초")
    print(f"  결과: {BLADE_DB}")
    print("=" * 70)


if __name__ == "__main__":
    main()
