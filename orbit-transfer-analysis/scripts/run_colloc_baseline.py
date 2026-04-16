"""comparison_runs 파라미터로 collocation 재실행 (λ=0 baseline 구축).

comparison_runs의 (h0, Δa, Δi, T_max_normed, e0, ef) 고유 조합에 대해
로컬 TwoPassOptimizer를 돌리고 결과를 DuckDB에 저장한다.

Usage:
    python scripts/run_colloc_baseline.py [--n-workers 8] [--limit 100]
"""

from __future__ import annotations
import sys, os, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import duckdb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

BLADE_DB = os.path.expanduser(
    "~/fmcl-database-student/student-kit/downloads/EXP-20260403-001/merged.duckdb"
)
OUT_DB = os.path.join(os.path.dirname(__file__), "../data/colloc_baseline.duckdb")
OUT_DB = os.path.normpath(OUT_DB)


# ── 스키마 ───────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS colloc_baseline (
    run_id          INTEGER PRIMARY KEY,
    h0              DOUBLE,
    delta_a         DOUBLE,
    delta_i         DOUBLE,
    T_max_normed    DOUBLE,
    e0              DOUBLE,
    ef              DOUBLE,
    converged       BOOLEAN,
    profile_class   INTEGER,
    n_peaks         INTEGER,
    cost            DOUBLE,
    solve_time      DOUBLE,
    nu0             DOUBLE,
    nuf             DOUBLE
)
"""


def init_db(path: str):
    con = duckdb.connect(path)
    con.execute(SCHEMA)
    con.close()


def already_done(path: str) -> set[tuple]:
    con = duckdb.connect(path, read_only=True)
    rows = con.execute(
        "SELECT h0, delta_a, delta_i, T_max_normed, e0, ef FROM colloc_baseline"
    ).fetchall()
    con.close()
    return {(round(r[0]), round(r[1],4), round(r[2],4),
             round(r[3],4), round(r[4],5), round(r[5],5)) for r in rows}


def load_configs(limit: int | None) -> list[dict]:
    con = duckdb.connect(BLADE_DB, read_only=True)
    q = """
        SELECT DISTINCT h0, delta_a, delta_i, T_max_normed, e0, ef
        FROM comparison_runs
        ORDER BY h0, delta_a, delta_i
    """
    if limit:
        q += f" LIMIT {limit}"
    rows = con.execute(q).fetchall()
    con.close()
    return [dict(h0=r[0], delta_a=r[1], delta_i=r[2],
                 T_max_normed=r[3], e0=r[4], ef=r[5])
            for r in rows]


# ── 단일 케이스 실행 (서브프로세스에서 호출) ────────────────
def run_one(cfg: dict) -> dict:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from orbit_transfer.types import TransferConfig
    from orbit_transfer.optimizer.two_pass import TwoPassOptimizer

    t0 = time.time()
    try:
        config = TransferConfig(**cfg)
        result = TwoPassOptimizer(config).solve()
        elapsed = time.time() - t0
        return dict(
            **cfg,
            converged=result.converged,
            profile_class=result.profile_class if result.converged else -1,
            n_peaks=result.n_peaks if result.converged else -1,
            cost=result.cost if result.converged else float("nan"),
            solve_time=elapsed,
            nu0=float(result.nu0) if result.converged else float("nan"),
            nuf=float(result.nuf) if result.converged else float("nan"),
        )
    except Exception as e:
        return dict(
            **cfg,
            converged=False,
            profile_class=-1, n_peaks=-1,
            cost=float("nan"),
            solve_time=time.time() - t0,
            nu0=float("nan"), nuf=float("nan"),
        )


# ── 결과 저장 ─────────────────────────────────────────────────
_run_id_counter = 0

def save_result(con: duckdb.DuckDBPyConnection, r: dict):
    global _run_id_counter
    _run_id_counter += 1
    con.execute("""
        INSERT INTO colloc_baseline VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        _run_id_counter,
        r["h0"], r["delta_a"], r["delta_i"], r["T_max_normed"],
        r["e0"], r["ef"],
        r["converged"], r["profile_class"], r["n_peaks"],
        r["cost"], r["solve_time"], r["nu0"], r["nuf"],
    ])


# ── main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="테스트용 케이스 수 제한")
    args = parser.parse_args()

    init_db(OUT_DB)
    done = already_done(OUT_DB)

    all_cfgs = load_configs(args.limit)
    todo = [c for c in all_cfgs
            if (round(c["h0"]), round(c["delta_a"],4), round(c["delta_i"],4),
                round(c["T_max_normed"],4), round(c["e0"],5), round(c["ef"],5))
            not in done]

    total = len(todo)
    print(f"총 {len(all_cfgs)}케이스 중 {total}개 미완료 → {args.n_workers}코어로 실행")
    if total == 0:
        print("모두 완료됨.")
        return

    con = duckdb.connect(OUT_DB)
    global _run_id_counter
    _run_id_counter = con.execute("SELECT COALESCE(MAX(run_id), 0) FROM colloc_baseline").fetchone()[0]

    t_start = time.time()
    n_done = 0
    n_conv = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(run_one, cfg): cfg for cfg in todo}
        for fut in as_completed(futures):
            r = fut.result()
            save_result(con, r)
            n_done += 1
            if r["converged"]:
                n_conv += 1

            elapsed = time.time() - t_start
            eta = elapsed / n_done * (total - n_done)
            print(f"  [{n_done}/{total}] "
                  f"h0={r['h0']:.0f} Δa={r['delta_a']:.0f} Δi={r['delta_i']:.1f} "
                  f"T={r['T_max_normed']:.2f} → "
                  f"{'✓' if r['converged'] else '✗'} "
                  f"class={r['profile_class']} "
                  f"({r['solve_time']:.1f}s) "
                  f"ETA {eta/60:.1f}min",
                  flush=True)

    con.close()
    total_time = time.time() - t_start
    print(f"\n완료: {n_conv}/{n_done} 수렴 ({100*n_conv/n_done:.1f}%)")
    print(f"총 시간: {total_time/60:.1f}분")
    print(f"결과: {OUT_DB}")


if __name__ == "__main__":
    main()
