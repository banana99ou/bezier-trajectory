"""궤적 데이터베이스 구축 (적응적 샘플링).

Usage:
    python scripts/run_database.py --h0 400
    python scripts/run_database.py --all
    python scripts/run_database.py --all --parallel --run-id run_20260215_001
"""
import argparse
import multiprocessing as mp
import sys
import time
from datetime import datetime

sys.path.insert(0, 'src')

from orbit_transfer.config import H0_SLICES
from orbit_transfer.types import TransferConfig
from orbit_transfer.optimizer.two_pass import TwoPassOptimizer
from orbit_transfer.database.storage import TrajectoryDatabase
from orbit_transfer.sampling.adaptive_sampler import AdaptiveSampler


def _generate_run_id():
    """타임스탬프 기반 run_id 자동 생성."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _current_param_config():
    """현재 PARAM_RANGES에서 param_config 문자열 생성."""
    from orbit_transfer.config import PARAM_RANGES
    t_lo = PARAM_RANGES["T_max_normed"][0]
    n_dims = len(PARAM_RANGES)
    # 예: "T015_5D" (T_min=0.15, 5차원)
    return f"T{int(t_lo * 100):03d}_{n_dims}D"


def make_evaluate_fn(h0, db, run_id=None, param_config=None):
    """궤적 최적화 + DB 저장하는 evaluate 함수 생성 (collocation only, 후방 호환)."""
    def evaluate(T_max_normed, delta_a, delta_i, e0, ef):
        config = TransferConfig(
            h0=h0, delta_a=delta_a, delta_i=delta_i,
            T_max_normed=T_max_normed, e0=e0, ef=ef,
        )
        t0 = time.time()
        opt = TwoPassOptimizer(config)
        result = opt.solve()
        elapsed = time.time() - t0
        db.insert_result(config, result, solve_time=elapsed,
                         run_id=run_id, param_config=param_config)
        return result.profile_class
    return evaluate


def _try_solve(solver, config, db, run_id, param_config):
    """솔버 실행 + DB 저장. 실패 시 None 반환."""
    method = getattr(solver, "method_name", type(solver).__name__)
    try:
        t0 = time.time()
        result = solver.solve(config)
        elapsed = time.time() - t0
        db.insert_benchmark_result(
            config, result, solve_time=elapsed,
            run_id=run_id, param_config=param_config,
        )
        return result
    except Exception as exc:
        print(f"  [{method}] FAILED: {exc}")
        return None


def make_evaluate_fn_all(h0, db, run_id=None, param_config=None,
                          solvers=None, blade_kwargs=None):
    """5개 솔버로 궤적 최적화 + DB 저장하는 evaluate 함수 생성.

    collocation 결과의 profile_class를 GP에 반환한다.
    """
    if solvers is None:
        solvers = ["hohmann", "lambert", "collocation", "blade", "blade_collocation"]
    if blade_kwargs is None:
        blade_kwargs = {}

    def evaluate(T_max_normed, delta_a, delta_i, e0, ef):
        config = TransferConfig(
            h0=h0, delta_a=delta_a, delta_i=delta_i,
            T_max_normed=T_max_normed, e0=e0, ef=ef,
        )

        col_class = 0
        col_tf = None

        # 1) Collocation FIRST → profile_class (GP 학습) + T_f (BLADE 공정 비교)
        if "collocation" in solvers:
            from orbit_transfer.benchmark.solvers import CollocationSolver
            col_result = _try_solve(
                CollocationSolver(), config, db, run_id, param_config,
            )
            if col_result is not None:
                col_class = col_result.metrics.get("profile_class", 0) or 0
                col_tf = col_result.extra.get("T_f")

        # 2) Hohmann
        if "hohmann" in solvers:
            from orbit_transfer.benchmark.solvers import HohmannSolver
            _try_solve(HohmannSolver(), config, db, run_id, param_config)

        # 3) Lambert
        if "lambert" in solvers:
            from orbit_transfer.benchmark.solvers import LambertSolver
            _try_solve(LambertSolver(), config, db, run_id, param_config)

        # 4) BLADE-SCP (col_tf 전달)
        if "blade" in solvers:
            try:
                from orbit_transfer.benchmark.solvers import BladeSolver
                kw = dict(blade_kwargs)
                if col_tf is not None:
                    kw.setdefault("t_f_override", col_tf)
                solver = BladeSolver(**kw)
                _try_solve(solver, config, db, run_id, param_config)
            except ImportError:
                print("  [blade] bezier_orbit 패키지 미설치 — 건너뜀")

        # 5) BLADE-Collocation
        if "blade_collocation" in solvers:
            try:
                from orbit_transfer.benchmark.solvers import BladeCollocationSolver
                solver = BladeCollocationSolver(
                    blade_K=blade_kwargs.get("K", 12),
                    blade_n=blade_kwargs.get("n", 2),
                )
                _try_solve(solver, config, db, run_id, param_config)
            except ImportError:
                print("  [blade_collocation] bezier_orbit 패키지 미설치 — 건너뜀")

        return col_class

    return evaluate


def run_altitude_slice(h0, db_path='data/trajectories.duckdb', npz_dir='data/trajectories',
                       run_id=None, param_config=None,
                       solvers=None, blade_kwargs=None,
                       sampler_overrides=None):
    db = TrajectoryDatabase(db_path=db_path, npz_dir=npz_dir)
    if solvers is not None and solvers != ["collocation"]:
        evaluate_fn = make_evaluate_fn_all(
            h0, db, run_id=run_id, param_config=param_config,
            solvers=solvers, blade_kwargs=blade_kwargs,
        )
    else:
        evaluate_fn = make_evaluate_fn(h0, db, run_id=run_id, param_config=param_config)
    sampler_kw = {}
    if sampler_overrides:
        sampler_kw.update(sampler_overrides)
    sampler = AdaptiveSampler(h0=h0, evaluate_fn=evaluate_fn, **sampler_kw)

    print(f"\n{'='*60}")
    print(f"Running h0={h0}km slice")
    print(f"{'='*60}")

    t0 = time.time()
    X, y, gpc = sampler.run()
    elapsed = time.time() - t0

    import numpy as np
    print(f"  Samples: {len(y)}")
    print(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Time: {elapsed:.1f}s")

    db.close()
    return len(y)


def _run_altitude_slice_args(args):
    """multiprocessing용 래퍼."""
    h0, db_path, npz_dir, run_id, param_config, solvers, blade_kwargs, sampler_overrides = args
    return run_altitude_slice(
        h0, db_path, npz_dir, run_id=run_id, param_config=param_config,
        solvers=solvers, blade_kwargs=blade_kwargs,
        sampler_overrides=sampler_overrides,
    )


def merge_databases(db_paths, output_path):
    """여러 고도별 DB를 하나로 병합."""
    import duckdb
    import os
    if os.path.exists(output_path):
        os.remove(output_path)
    conn = duckdb.connect(output_path)
    for i, path in enumerate(db_paths):
        conn.execute(f"ATTACH '{path}' AS src (READ_ONLY)")
        if i == 0:
            conn.execute("CREATE TABLE trajectories AS SELECT * FROM src.trajectories")
        else:
            conn.execute("INSERT INTO trajectories SELECT * FROM src.trajectories")
        conn.execute("DETACH src")
    conn.close()
    print(f"Merged {len(db_paths)} databases into {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Build trajectory database')
    parser.add_argument('--h0', type=float, help='Single altitude slice [km]')
    parser.add_argument('--all', action='store_true', help='Run all altitude slices')
    parser.add_argument('--parallel', action='store_true', help='Run altitude slices in parallel')
    parser.add_argument('--db_path', type=str, default=None, help='DB file path (default: data/trajectories.duckdb)')
    parser.add_argument('--npz_dir', type=str, default=None, help='NPZ directory (default: data/trajectories)')
    parser.add_argument('--run-id', type=str, default=None, help='Run ID (default: auto-generated)')
    parser.add_argument('--solvers', type=str, default='collocation',
                        help='Comma-separated solver list or "all" (default: collocation)')
    parser.add_argument('--blade-K', type=int, default=12, help='BLADE segment count')
    parser.add_argument('--blade-n', type=int, default=2, help='BLADE segment degree')
    parser.add_argument('--blade-validate', action='store_true', help='Enable BLADE RK4 validation')
    parser.add_argument('--n-init', type=int, default=None, help='Override GP_N_INIT')
    parser.add_argument('--n-max', type=int, default=None, help='Override GP_N_MAX')
    args = parser.parse_args()

    run_id = args.run_id or _generate_run_id()
    param_config = _current_param_config()
    print(f"Run ID: {run_id}")
    print(f"Param config: {param_config}")

    # 솔버 리스트 파싱
    if args.solvers == "all":
        solvers = ["hohmann", "lambert", "collocation", "blade", "blade_collocation"]
    else:
        solvers = [s.strip() for s in args.solvers.split(",")]
    print(f"Solvers: {solvers}")

    blade_kwargs = {"K": args.blade_K, "n": args.blade_n}
    if args.blade_validate:
        blade_kwargs["validate"] = True

    # AdaptiveSampler 오버라이드 준비
    sampler_overrides = {}
    if args.n_init is not None:
        sampler_overrides["n_init"] = args.n_init
    if args.n_max is not None:
        sampler_overrides["n_max"] = args.n_max

    if args.all:
        if args.parallel:
            mp.set_start_method('spawn', force=True)
            tasks = [
                (h0, f'data/h{int(h0)}/trajectories.duckdb',
                      f'data/h{int(h0)}/trajectories',
                      run_id, param_config, solvers, blade_kwargs,
                      sampler_overrides)
                for h0 in H0_SLICES
            ]
            with mp.Pool(processes=min(4, len(H0_SLICES))) as pool:
                pool.map(_run_altitude_slice_args, tasks)

            merge_databases(
                [f'data/h{int(h0)}/trajectories.duckdb' for h0 in H0_SLICES],
                'data/trajectories_all.duckdb',
            )
        else:
            for h0 in H0_SLICES:
                run_altitude_slice(h0, run_id=run_id, param_config=param_config,
                                   solvers=solvers, blade_kwargs=blade_kwargs,
                                   sampler_overrides=sampler_overrides)
    elif args.h0:
        db_path = args.db_path or 'data/trajectories.duckdb'
        npz_dir = args.npz_dir or 'data/trajectories'
        run_altitude_slice(args.h0, db_path=db_path, npz_dir=npz_dir,
                           run_id=run_id, param_config=param_config,
                           solvers=solvers, blade_kwargs=blade_kwargs,
                           sampler_overrides=sampler_overrides)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
