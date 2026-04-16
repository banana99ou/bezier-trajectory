#!/usr/bin/env python
"""궤도전이 기법 비교 파이프라인.

사용 예:
    # 빠른 테스트
    python scripts/run_comparison.py --case quick_test --no-collocation

    # 논문 기준 케이스 (Hohmann + Lambert + Collocation)
    python scripts/run_comparison.py --case paper_main

    # 경사각 시리즈 전체 (콜로케이션 제외 - 빠름)
    python scripts/run_comparison.py --group inclination_series --no-collocation

    # 전시간 시리즈 + 콜로케이션 포함
    python scripts/run_comparison.py --group time_series --outdir results/time_sweep

    # 케이스 목록 확인
    python scripts/run_comparison.py --list
"""

import argparse
import os
import sys
import time
import json
import csv
from pathlib import Path

# 패키지 경로 추가 (editable install이 안 된 경우 대비)
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대응

from orbit_transfer.benchmark import TransferBenchmark, BenchmarkResult
from orbit_transfer.benchmark.cases import (
    ComparisonCase, get_case, get_group, list_cases, CASE_REGISTRY
)


# ===========================================================================
# 단일 케이스 실행
# ===========================================================================

def run_case(
    case: ComparisonCase,
    outdir: str,
    methods: list[str],
    fmt: str = "pdf",
    lambert_tof: float | None = None,
    verbose: bool = True,
) -> dict:
    """단일 케이스를 실행하고 결과를 저장한다.

    Parameters
    ----------
    case : ComparisonCase
    outdir : str        결과 저장 디렉토리 (케이스별 서브디렉토리 자동 생성)
    methods : list[str] 실행할 기법 목록 ('hohmann', 'lambert', 'collocation')
    fmt : str           그림 저장 형식
    lambert_tof : float, optional  Lambert 비행시간 지정
    verbose : bool

    Returns
    -------
    dict  케이스별 지표 요약 (CSV 취합용)
    """
    case_dir = os.path.join(outdir, case.name)
    os.makedirs(case_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"케이스: {case.name}")
        print(f"설명: {case.description}")
        if case.notes:
            print(f"비고: {case.notes}")
        cfg = case.config
        print(f"  h0={cfg.h0} km, Δa={cfg.delta_a} km, Δi={cfg.delta_i}°"
              f", T_max_normed={cfg.T_max_normed}, u_max={cfg.u_max}")
        print(f"{'='*60}")

    bench = TransferBenchmark(case.config)
    timings: dict[str, float] = {}

    # --- 각 기법 실행 ---
    if "hohmann" in methods:
        t0 = time.perf_counter()
        bench.run_hohmann()
        timings["hohmann"] = time.perf_counter() - t0
        r = bench.results["hohmann"]
        if verbose:
            _print_result("Hohmann", r, timings["hohmann"])

    if "lambert" in methods:
        t0 = time.perf_counter()
        lambert_kw = case.extra.get("lambert_kwargs", {})
        bench.run_lambert(tof=lambert_tof, **lambert_kw)
        timings["lambert"] = time.perf_counter() - t0
        r = bench.results["lambert"]
        if verbose:
            _print_result("Lambert", r, timings["lambert"])

    if "collocation" in methods and case.supports_collocation:
        t0 = time.perf_counter()
        bench.run_collocation()
        timings["collocation"] = time.perf_counter() - t0
        r = bench.results["collocation"]
        if verbose:
            _print_result("Collocation", r, timings["collocation"])
    elif "collocation" in methods and not case.supports_collocation:
        if verbose:
            print("  [Collocation] 건너뜀 (RAAN 변화 미지원)")

    results = bench.results
    if not results:
        if verbose:
            print("  결과 없음. 종료.")
        return {}

    # --- 콘솔 요약 ---
    if verbose:
        bench.print_summary()

    # --- 파일 내보내기 ---
    try:
        bench.export_csv(os.path.join(case_dir, "metrics_summary.csv"))
        bench.export_trajectory_csv(case_dir)
        bench.export_figures(case_dir, fmt=fmt,
                             metrics_to_plot=["dv_total", "cost_l1", "tof_norm"])
    except Exception as e:
        if verbose:
            print(f"  [경고] 내보내기 실패: {e}")

    # --- 계산 시간 CSV ---
    timing_path = os.path.join(case_dir, "solve_times.csv")
    with open(timing_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "solve_time_s"])
        for m, t_val in timings.items():
            w.writerow([m, f"{t_val:.4f}"])

    # --- 케이스 메타 저장 ---
    meta = {
        "name": case.name,
        "description": case.description,
        "config": {
            "h0": case.config.h0,
            "delta_a": case.config.delta_a,
            "delta_i": case.config.delta_i,
            "T_max_normed": case.config.T_max_normed,
            "u_max": case.config.u_max,
        },
        "notes": case.notes,
        "extra": case.extra,
        "timings": timings,
    }
    with open(os.path.join(case_dir, "case_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  → 결과 저장: {case_dir}")

    # 취합용 요약 행 반환
    row: dict = {
        "case_name": case.name,
        "h0_km": case.config.h0,
        "delta_a_km": case.config.delta_a,
        "delta_i_deg": case.config.delta_i,
        "T_max_normed": case.config.T_max_normed,
        "u_max": case.config.u_max,
    }
    for method, res in results.items():
        prefix = method
        row[f"{prefix}_converged"] = res.converged
        row[f"{prefix}_dv_total"] = res.metrics.get("dv_total", "")
        row[f"{prefix}_cost_l1"] = res.metrics.get("cost_l1", "")
        row[f"{prefix}_cost_l2"] = res.metrics.get("cost_l2", "")
        row[f"{prefix}_tof"] = res.metrics.get("tof", "")
        row[f"{prefix}_tof_norm"] = res.metrics.get("tof_norm", "")
        row[f"{prefix}_solve_s"] = timings.get(method, "")
    return row


def _print_result(name: str, r: BenchmarkResult, elapsed: float) -> None:
    """단일 기법 결과 요약 출력."""
    dv = r.metrics.get("dv_total", float("nan"))
    l1 = r.metrics.get("cost_l1", float("nan"))
    l2 = r.metrics.get("cost_l2", float("nan"))
    tof = r.metrics.get("tof", float("nan"))
    conv = "OK" if r.converged else "FAIL"
    l2_str = f"{l2:.6f}" if not (isinstance(l2, float) and np.isnan(l2)) else "  N/A  "
    print(f"  [{name:<12}] {conv} | Δv={dv:.4f} km/s | L1={l1:.4f} km/s"
          f" | L2={l2_str} km²/s³ | ToF={tof:.0f} s | t={elapsed:.2f}s")


# ===========================================================================
# 다중 케이스 취합 CSV
# ===========================================================================

def write_aggregate_csv(rows: list[dict], outdir: str) -> None:
    """여러 케이스 결과를 하나의 CSV로 취합한다."""
    if not rows:
        return
    agg_path = os.path.join(outdir, "aggregate_results.csv")
    all_keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # NaN → 빈 문자열
            cleaned = {k: ("" if isinstance(v, float) and np.isnan(v) else v)
                       for k, v in row.items()}
            writer.writerow(cleaned)
    print(f"\n[aggregate] 전체 결과 저장: {agg_path}")


# ===========================================================================
# 쇼케이스 분석
# ===========================================================================

def analyze_showcase(rows: list[dict], outdir: str) -> None:
    """콜로케이션 L1 비용 대비 Hohmann Δv 비율을 기준으로
    제안 기법 우위가 큰 케이스를 순위별로 출력한다.

    높은 비율 = Hohmann보다 연속 추력이 연료를 크게 절약 = 좋은 쇼케이스
    """
    candidates = []
    for row in rows:
        coll_l1 = row.get("collocation_cost_l1", "")
        hohm_dv = row.get("hohmann_dv_total", "")
        if coll_l1 == "" or hohm_dv == "":
            continue
        try:
            coll_l1 = float(coll_l1)
            hohm_dv = float(hohm_dv)
        except (ValueError, TypeError):
            continue
        if coll_l1 > 0 and hohm_dv > 0:
            ratio = hohm_dv / coll_l1  # > 1 이면 연속 추력이 유리
            candidates.append((ratio, row))

    if not candidates:
        return

    candidates.sort(reverse=True)
    print(f"\n{'='*70}")
    print("쇼케이스 후보 (Hohmann Δv / Collocation L1 비율 내림차순)")
    print(f"{'순위':<5} {'케이스':<30} {'비율':>8} {'Hohmann Δv':>12} {'Coll L1':>12}")
    print("-" * 70)
    for rank, (ratio, row) in enumerate(candidates[:10], 1):
        name = row.get("case_name", "?")
        hohm = row.get("hohmann_dv_total", "?")
        coll = row.get("collocation_cost_l1", "?")
        print(f"{rank:<5} {name:<30} {ratio:>8.3f} {hohm:>12.4f} {coll:>12.4f}")

    # 최적 케이스 JSON 저장
    top = [{"rank": i+1, "ratio": r, "case": row}
           for i, (r, row) in enumerate(candidates[:5])]
    with open(os.path.join(outdir, "showcase_ranking.json"), "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[showcase] 상위 5개 케이스: {os.path.join(outdir, 'showcase_ranking.json')}")


# ===========================================================================
# CLI 진입점
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="궤도전이 기법 비교 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 케이스 선택
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--case", "-c", type=str,
                     help="실행할 케이스 이름 (예: paper_main, quick_test)")
    grp.add_argument("--group", "-g", type=str,
                     help="케이스 그룹 (paper / inclination_series / time_series / combined_series / all)")
    grp.add_argument("--list", "-l", action="store_true",
                     help="등록된 케이스 목록 출력 후 종료")

    # 기법 선택
    parser.add_argument("--methods", "-m", nargs="+",
                        default=["hohmann", "lambert", "collocation"],
                        choices=["hohmann", "lambert", "collocation"],
                        help="실행할 기법 (기본: 전체)")
    parser.add_argument("--no-collocation", action="store_true",
                        help="콜로케이션 제외 (빠른 실행)")

    # 출력
    parser.add_argument("--outdir", "-o", type=str, default="results/comparison",
                        help="결과 저장 디렉토리 (기본: results/comparison)")
    parser.add_argument("--fmt", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="그림 저장 형식 (기본: pdf)")

    # Lambert 옵션
    parser.add_argument("--lambert-tof", type=float, default=None,
                        help="Lambert 비행시간 [s]. 미지정시 Hohmann TOF 사용")

    # 쇼케이스 분석
    parser.add_argument("--showcase", action="store_true",
                        help="케이스 그룹 실행 후 쇼케이스 순위 분석")

    parser.add_argument("--quiet", "-q", action="store_true",
                        help="상세 출력 억제")

    return parser.parse_args()


def main():
    args = parse_args()

    # -- 목록 출력 --
    if args.list:
        list_cases()
        return

    # -- 기법 목록 결정 --
    methods = list(args.methods)
    if args.no_collocation and "collocation" in methods:
        methods.remove("collocation")

    if not methods:
        print("[오류] 실행할 기법이 없습니다.")
        sys.exit(1)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # -- 케이스 목록 구성 --
    if args.case:
        try:
            cases = [get_case(args.case)]
        except KeyError as e:
            print(f"[오류] {e}")
            sys.exit(1)
    elif args.group:
        try:
            from orbit_transfer.benchmark.cases import get_group
            cases = get_group(args.group)
        except KeyError as e:
            print(f"[오류] {e}")
            sys.exit(1)
    else:
        # 기본: paper_main
        cases = [get_case("paper_main")]

    verbose = not args.quiet
    if verbose:
        print(f"\n실행 기법: {methods}")
        print(f"케이스 수: {len(cases)}")
        print(f"출력 디렉토리: {outdir}\n")

    # -- 실행 --
    aggregate_rows = []
    total_t0 = time.perf_counter()

    for i, case in enumerate(cases, 1):
        if verbose and len(cases) > 1:
            print(f"\n[{i}/{len(cases)}]", end="")
        try:
            row = run_case(
                case=case,
                outdir=outdir,
                methods=methods,
                fmt=args.fmt,
                lambert_tof=args.lambert_tof,
                verbose=verbose,
            )
            if row:
                aggregate_rows.append(row)
        except Exception as e:
            print(f"\n[오류] 케이스 '{case.name}' 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.perf_counter() - total_t0

    # -- 취합 CSV --
    if len(aggregate_rows) > 1:
        write_aggregate_csv(aggregate_rows, outdir)

    # -- 쇼케이스 분석 --
    if args.showcase and aggregate_rows:
        analyze_showcase(aggregate_rows, outdir)

    if verbose:
        print(f"\n완료. 총 소요 시간: {total_elapsed:.1f}s")
        print(f"결과 위치: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
