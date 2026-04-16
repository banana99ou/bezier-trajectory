"""확장 샘플링: 전체 h0 슬라이스 × circular/eccentric 조합 실행.

기존 run_sampling.py는 h0=400 circular 1개만 실행함.
이 스크립트는 4개 고도 슬라이스 × 2개 이심률 조합을 순차 실행하여
분석용 데이터를 대폭 늘린다.

Usage:
    python scripts/run_sampling_extended.py                  # 전체 실행
    python scripts/run_sampling_extended.py --mode circular  # circular만
    python scripts/run_sampling_extended.py --mode eccentric # eccentric만
    python scripts/run_sampling_extended.py --n_max 200      # 슬라이스당 200개
    python scripts/run_sampling_extended.py --h0 400 600     # 특정 고도만
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from orbit_transfer.pipeline.evaluate import create_database, make_sampler_evaluate_fn
from orbit_transfer.sampling.adaptive_sampler import AdaptiveSampler
from orbit_transfer.config import H0_SLICES


# ── 슬라이스 설정 ─────────────────────────────────────────────────────────────
SLICE_CONFIGS = {
    "circular": {
        "db_path": "data/trajectories_circular.duckdb",
        "npz_dir": "data/trajectories_circular",
        "e0": 0.0,
        "ef": 0.0,
    },
    "eccentric": {
        "db_path": "data/trajectories_eccentric.duckdb",
        "npz_dir": "data/trajectories_eccentric",
        "e0": 0.05,
        "ef": 0.05,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="확장 샘플링 실행")
    p.add_argument("--mode", choices=["circular", "eccentric", "both"], default="both")
    p.add_argument("--n_max", type=int, default=300, help="슬라이스당 최대 샘플 수")
    p.add_argument("--n_init", type=int, default=50, help="LHS 초기 샘플 수")
    p.add_argument("--batch_size", type=int, default=5, help="GP 배치 크기")
    p.add_argument("--h0", type=float, nargs="+", default=H0_SLICES,
                   help="실행할 고도 슬라이스 [km]")
    p.add_argument("--seed_offset", type=int, default=100,
                   help="랜덤 시드 오프셋 (기존과 다른 샘플 생성)")
    return p.parse_args()


def run_slice(mode: str, h0: float, n_init: int, n_max: int,
              batch_size: int, seed: int) -> dict:
    """단일 (mode, h0) 슬라이스 샘플링 실행."""
    cfg = SLICE_CONFIGS[mode]

    db = create_database(db_path=cfg["db_path"], npz_dir=cfg["npz_dir"])

    # 기존 샘플 수 확인 — db.conn 재사용 (같은 파일 이중 연결 방지)
    existing = db.conn.execute(
        "SELECT COUNT(*) FROM trajectories WHERE h0=?", [h0]
    ).fetchone()[0]

    print(f"  기존 샘플: {existing}개  →  목표: {n_max}개")

    evaluate_fn = make_sampler_evaluate_fn(
        h0=h0, e0=cfg["e0"], ef=cfg["ef"], db=db
    )

    sampler = AdaptiveSampler(
        h0=h0,
        evaluate_fn=evaluate_fn,
        n_init=n_init,
        n_max=n_max,
        batch_size=batch_size,
        seed=seed,
    )

    t0 = time.perf_counter()
    X, y, _ = sampler.run()
    elapsed = time.perf_counter() - t0

    # 완료 후 현재 DB 상태 조회 — 동일 연결 사용
    total_now = db.conn.execute(
        "SELECT COUNT(*) FROM trajectories WHERE h0=?", [h0]
    ).fetchone()[0]
    conv_now = db.conn.execute(
        "SELECT COUNT(*) FROM trajectories WHERE h0=? AND converged=TRUE", [h0]
    ).fetchone()[0]
    class_dist = db.conn.execute(
        "SELECT profile_class, COUNT(*) FROM trajectories "
        "WHERE h0=? AND converged=TRUE GROUP BY profile_class ORDER BY profile_class",
        [h0]
    ).fetchall()
    db.close()

    return {
        "mode": mode, "h0": h0,
        "added": total_now - existing,
        "total": total_now, "converged": conv_now,
        "elapsed": elapsed,
        "class_dist": dict(class_dist),
    }


def print_summary(results: list[dict]):
    print(f"\n{'=' * 65}")
    print("  최종 요약")
    print(f"{'=' * 65}")
    print(f"  {'mode':>10}  {'h0':>5}  {'added':>6}  {'total':>6}  {'conv':>6}  "
          f"{'c0':>4}  {'c1':>4}  {'c2':>4}  {'time'}  ")
    print(f"  {'-' * 63}")
    labels = {0: "uni", 1: "bi", 2: "multi"}
    total_added = 0
    for r in results:
        cd = r["class_dist"]
        t = r["elapsed"]
        tstr = f"{t/60:.1f}m" if t >= 60 else f"{t:.0f}s"
        print(f"  {r['mode']:>10}  {r['h0']:>5.0f}  {r['added']:>6}  "
              f"{r['total']:>6}  {r['converged']:>6}  "
              f"{cd.get(0,0):>4}  {cd.get(1,0):>4}  {cd.get(2,0):>4}  {tstr}")
        total_added += r["added"]
    print(f"  {'-' * 63}")
    print(f"  총 추가 샘플: {total_added}개")


def estimate_runtime(n_slices: int, n_max: int, avg_sec: float = 2.5):
    total_sec = n_slices * n_max * avg_sec
    if total_sec < 60:
        return f"~{total_sec:.0f}초"
    elif total_sec < 3600:
        return f"~{total_sec/60:.0f}분"
    else:
        return f"~{total_sec/3600:.1f}시간"


def main():
    args = parse_args()

    modes = ["circular", "eccentric"] if args.mode == "both" else [args.mode]
    h0_list = args.h0

    n_runs = len(modes) * len(h0_list)
    est = estimate_runtime(n_runs, args.n_max)

    print(f"{'=' * 65}")
    print(f"  확장 샘플링 시작")
    print(f"  모드: {modes}")
    print(f"  고도: {h0_list} km")
    print(f"  슬라이스당 목표: {args.n_max}개  (init={args.n_init}, batch={args.batch_size})")
    print(f"  총 실행 횟수: {n_runs}개 슬라이스  예상 소요: {est}")
    print(f"{'=' * 65}\n")

    results = []
    total_t0 = time.perf_counter()

    for mode in modes:
        for i, h0 in enumerate(h0_list):
            seed = args.seed_offset + i * 7 + (0 if mode == "circular" else 50)
            print(f"[{len(results)+1}/{n_runs}] {mode}  h0={h0:.0f}km  seed={seed}")
            r = run_slice(
                mode=mode, h0=h0,
                n_init=args.n_init,
                n_max=args.n_max,
                batch_size=args.batch_size,
                seed=seed,
            )
            results.append(r)
            elapsed_total = time.perf_counter() - total_t0
            remaining = n_runs - len(results)
            avg_per_slice = elapsed_total / len(results)
            eta = avg_per_slice * remaining
            eta_str = f"{eta/60:.1f}분" if eta >= 60 else f"{eta:.0f}초"
            print(f"  완료: +{r['added']}개  수렴={r['converged']}  "
                  f"소요={r['elapsed']/60:.1f}분  남은 예상={eta_str}\n")

    print_summary(results)


if __name__ == "__main__":
    main()
