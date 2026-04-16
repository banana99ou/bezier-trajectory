"""피크 탐지 알고리즘 진단 스크립트.

Figure 3 대표 프로파일의 분류 라벨-형태 불일치 원인을 진단한다.

Usage:
    python scripts/diagnose_peak_detection.py --db data/trajectories.duckdb --outdir manuscript/figures/diagnostics
    python scripts/diagnose_peak_detection.py --db data/trajectories.duckdb --only diag1,diag3
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.classification.classifier import classify_profile
from orbit_transfer.config import PEAK_PROMINENCE_RATIO, PEAK_MIN_DISTANCE_RATIO
from orbit_transfer.database.storage import TrajectoryDatabase


# ---------------------------------------------------------------------------
# 로컬 유틸리티
# ---------------------------------------------------------------------------

def _detect_peaks_custom(t, u_mag, T, prominence_ratio=None, min_distance_ratio=None):
    """파라미터를 변경 가능한 detect_peaks 변형.

    Returns:
        n_peaks, peak_indices, u_smooth, prominence_threshold, min_distance, smooth_window
    """
    if prominence_ratio is None:
        prominence_ratio = PEAK_PROMINENCE_RATIO
    if min_distance_ratio is None:
        min_distance_ratio = PEAK_MIN_DISTANCE_RATIO

    t = np.asarray(t, dtype=float)
    u_mag = np.asarray(u_mag, dtype=float)

    t_span = t[-1] - t[0]
    if t_span > 0:
        min_distance = int(len(t) * min_distance_ratio * (T / t_span))
    else:
        min_distance = 1
    min_distance = max(min_distance, 1)

    smooth_window = max(2 * (min_distance // 4) + 1, 5)
    u_smooth = uniform_filter1d(u_mag, size=smooth_window)

    u_max = np.max(u_smooth)
    if u_max == 0.0:
        return 0, np.array([], dtype=int), u_smooth, 0.0, min_distance, smooth_window

    prominence_threshold = prominence_ratio * u_max

    peak_indices, _ = find_peaks(
        u_smooth,
        prominence=prominence_threshold,
        distance=min_distance,
    )

    return (
        len(peak_indices),
        peak_indices,
        u_smooth,
        prominence_threshold,
        min_distance,
        smooth_window,
    )


def _cleanliness_score(t, u_mag, T, expected_peaks):
    """프로파일 깨끗함 정량화.

    점수가 낮을수록 해당 class의 "이상적" 형태에 가깝다.

    Components:
        1. smoothed 잔차 (raw - smooth)의 정규화 RMS  →  노이즈 척도
        2. |탐지된 피크 수 - expected_peaks|  →  피크 수 편차

    Returns:
        score (float): 낮을수록 깨끗함
    """
    n_peaks, _, u_smooth, _, _, _ = _detect_peaks_custom(t, u_mag, T)

    u_max = np.max(np.abs(u_mag))
    if u_max == 0:
        return float("inf")

    # 잔차 RMS (정규화)
    residual = u_mag - u_smooth
    rms = np.sqrt(np.mean(residual**2)) / u_max

    # 피크 수 편차
    peak_dev = abs(n_peaks - expected_peaks)

    return rms + 0.5 * peak_dev


def _load_converged_rows(db):
    """수렴 사례 전체 로드."""
    return db.get_results(converged=True)


def _load_trajectory(db, row):
    """DB row에서 궤적 데이터 로드."""
    traj = db.get_trajectory(row["id"])
    t = traj["t"]
    u = traj["u"]
    u_mag = np.linalg.norm(u, axis=0)
    T = t[-1] - t[0]
    return t, u_mag, T


# ---------------------------------------------------------------------------
# diag1: 대표 사례 상세 분석
# ---------------------------------------------------------------------------

def diag1_median_cases(db, outdir):
    """Figure 3과 동일한 3개 median-cost 사례의 상세 피크 탐지 시각화."""
    print("[diag1] 대표 사례 상세 분석...")

    CLASS_NAMES = ["Unimodal", "Bimodal", "Multimodal"]
    CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for cls in range(3):
        ax = axes[cls]
        rows = db.get_results(converged=True, profile_class=cls)
        if not rows:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(f"({chr(97 + cls)}) {CLASS_NAMES[cls]}")
            continue

        # median-cost 선택 (Figure 3와 동일 로직)
        rows.sort(key=lambda r: r["cost"])
        row = rows[len(rows) // 2]

        t, u_mag, T = _load_trajectory(db, row)
        n_peaks, peak_idx, u_smooth, prom_thresh, min_dist, sw = _detect_peaks_custom(
            t, u_mag, T
        )

        t_hr = t / 3600.0

        # 원시 신호
        ax.plot(t_hr, u_mag, color="gray", alpha=0.5, linewidth=0.8, label="Raw $u_{mag}$")
        # smoothed 신호
        ax.plot(t_hr, u_smooth, color=CLASS_COLORS[cls], linewidth=1.5, label="Smoothed")
        # 피크 위치
        if len(peak_idx) > 0:
            ax.plot(
                t_hr[peak_idx],
                u_smooth[peak_idx],
                "v",
                color="black",
                markersize=8,
                zorder=5,
                label=f"Peaks ({n_peaks})",
            )
        # prominence threshold
        ax.axhline(
            prom_thresh,
            color="red",
            linestyle="--",
            alpha=0.5,
            linewidth=0.8,
            label=f"Prom. thr. ({prom_thresh:.2e})",
        )

        ax.set_xlabel("Time [hr]")
        if cls == 0:
            ax.set_ylabel(r"$\|\mathbf{u}(t)\|$ [km/s$^2$]")
        ax.set_title(
            f"({chr(97 + cls)}) {CLASS_NAMES[cls]} (peaks={n_peaks})\n"
            f"id={row['id']}, cost={row['cost']:.4e}\n"
            f"sw={sw}, min_dist={min_dist}, N={len(t)}"
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "diag1_median_cases.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# diag2: 전체 분류 통계
# ---------------------------------------------------------------------------

def diag2_npeaks_distribution(db, outdir):
    """전체 수렴 사례의 n_peaks 분포 및 DB 저장값과의 일치 검증."""
    print("[diag2] 전체 분류 통계...")

    rows = _load_converged_rows(db)
    print(f"  수렴 사례 수: {len(rows)}")

    # 재계산
    db_peaks = []
    recomputed_peaks = []
    db_class = []
    recomputed_class = []
    mismatches = []

    for row in rows:
        try:
            t, u_mag, T = _load_trajectory(db, row)
            n_pk, _, _ = detect_peaks(t, u_mag, T)
            cls = classify_profile(n_pk)

            db_peaks.append(row["n_peaks"])
            recomputed_peaks.append(n_pk)
            db_class.append(row["profile_class"])
            recomputed_class.append(cls)

            if row["n_peaks"] != n_pk or row["profile_class"] != cls:
                mismatches.append(
                    {
                        "id": row["id"],
                        "db_peaks": row["n_peaks"],
                        "recomputed_peaks": n_pk,
                        "db_class": row["profile_class"],
                        "recomputed_class": cls,
                    }
                )
        except Exception as e:
            print(f"  Warning: id={row['id']} 로드 실패: {e}")

    db_peaks = np.array(db_peaks)
    recomputed_peaks = np.array(recomputed_peaks)
    db_class = np.array(db_class)
    recomputed_class = np.array(recomputed_class)

    print(f"  DB-재계산 불일치: {len(mismatches)}건")
    if mismatches:
        print("  불일치 사례 (최대 20건):")
        for m in mismatches[:20]:
            print(
                f"    id={m['id']}: DB peaks={m['db_peaks']} vs recomp={m['recomputed_peaks']}, "
                f"DB class={m['db_class']} vs recomp={m['recomputed_class']}"
            )

    # 시각화: 2x3 그리드
    CLASS_NAMES = ["Unimodal (cls=0)", "Bimodal (cls=1)", "Multimodal (cls=2)"]
    CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 상단: DB 저장 n_peaks 분포 (class별)
    for cls in range(3):
        ax = axes[0, cls]
        mask = db_class == cls
        if np.any(mask):
            peaks_vals = db_peaks[mask]
            max_pk = max(peaks_vals.max(), 10)
            bins = np.arange(-0.5, max_pk + 1.5, 1)
            ax.hist(peaks_vals, bins=bins, color=CLASS_COLORS[cls], alpha=0.7, edgecolor="white")
        ax.set_xlabel("n_peaks (DB)")
        ax.set_ylabel("Count")
        ax.set_title(f"DB: {CLASS_NAMES[cls]} (n={np.sum(mask)})")
        ax.grid(True, alpha=0.3)

    # 하단: 재계산 n_peaks 분포 (class별)
    for cls in range(3):
        ax = axes[1, cls]
        mask = recomputed_class == cls
        if np.any(mask):
            peaks_vals = recomputed_peaks[mask]
            max_pk = max(peaks_vals.max(), 10)
            bins = np.arange(-0.5, max_pk + 1.5, 1)
            ax.hist(peaks_vals, bins=bins, color=CLASS_COLORS[cls], alpha=0.7, edgecolor="white")

        # 의심 사례 표시
        suspicious = np.sum((db_class == cls) & (recomputed_class != cls))
        ax.set_xlabel("n_peaks (recomputed)")
        ax.set_ylabel("Count")
        ax.set_title(f"Recomp: {CLASS_NAMES[cls]} (n={np.sum(mask)}, mismatch={suspicious})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Peak Count Distribution (total={len(db_peaks)}, mismatches={len(mismatches)})",
        fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(outdir, "diag2_npeaks_distribution.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

    # 의심 사례 목록 출력
    print("\n  [의심 사례: class=0 (unimodal) 인데 recomputed n_peaks >= 3]")
    suspicious_uni = [
        m
        for m in mismatches
        if m["db_class"] == 0 and m["recomputed_peaks"] >= 3
    ]
    # DB 기준으로도 확인
    suspicious_uni_db = []
    for i in range(len(db_peaks)):
        if db_class[i] == 0 and db_peaks[i] >= 3:
            suspicious_uni_db.append({"idx": i, "n_peaks": db_peaks[i]})
    print(f"    DB 기준: {len(suspicious_uni_db)}건")
    print(f"    재계산 기준: {len(suspicious_uni)}건")

    return path


# ---------------------------------------------------------------------------
# diag3: prominence ratio 민감도
# ---------------------------------------------------------------------------

def diag3_sensitivity(db, outdir):
    """prominence ratio 변화에 따른 분류 민감도 분석."""
    print("[diag3] Prominence ratio 민감도 분석...")

    ratios = [0.05, 0.1, 0.15, 0.2, 0.3]
    rows = _load_converged_rows(db)

    # ratio별 class 분포 수집
    class_counts = {r: {0: 0, 1: 0, 2: 0} for r in ratios}

    for row in rows:
        try:
            t, u_mag, T = _load_trajectory(db, row)
            for ratio in ratios:
                n_pk, _, _, _, _, _ = _detect_peaks_custom(
                    t, u_mag, T, prominence_ratio=ratio
                )
                cls = classify_profile(n_pk)
                class_counts[ratio][cls] += 1
        except Exception:
            continue

    # Stacked bar chart
    CLASS_NAMES = ["Unimodal", "Bimodal", "Multimodal"]
    CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ratios))
    bar_width = 0.5

    bottoms = np.zeros(len(ratios))
    for cls in range(3):
        counts = np.array([class_counts[r][cls] for r in ratios])
        ax.bar(
            x,
            counts,
            bar_width,
            bottom=bottoms,
            color=CLASS_COLORS[cls],
            label=CLASS_NAMES[cls],
            alpha=0.85,
            edgecolor="white",
        )
        # 숫자 레이블
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(
                    x[i],
                    bottoms[i] + c / 2,
                    str(c),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in ratios])
    ax.set_xlabel("Prominence Ratio")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Classification Sensitivity to Prominence Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 현재 기본값 표시
    default_idx = ratios.index(PEAK_PROMINENCE_RATIO) if PEAK_PROMINENCE_RATIO in ratios else None
    if default_idx is not None:
        ax.axvline(
            x[default_idx],
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"Current default ({PEAK_PROMINENCE_RATIO})",
        )
        ax.legend()

    fig.tight_layout()
    path = os.path.join(outdir, "diag3_sensitivity.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # 텍스트 통계
    print("  Ratio → (Unimodal, Bimodal, Multimodal):")
    for r in ratios:
        marker = " ← current" if r == PEAK_PROMINENCE_RATIO else ""
        print(
            f"    {r:.2f}: ({class_counts[r][0]:4d}, {class_counts[r][1]:4d}, {class_counts[r][2]:4d}){marker}"
        )

    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# diag4: 깨끗한 대표 후보 탐색
# ---------------------------------------------------------------------------

def diag4_clean_candidates(db, outdir):
    """각 class별 깨끗한 프로파일 상위 5개 후보를 시각화."""
    print("[diag4] 깨끗한 대표 후보 탐색...")

    CLASS_NAMES = ["Unimodal", "Bimodal", "Multimodal"]
    CLASS_COLORS = ["#2196F3", "#FF9800", "#F44336"]
    EXPECTED_PEAKS = {0: 1, 1: 2, 2: 4}  # 각 class의 기대 피크 수
    N_CANDIDATES = 5

    fig, axes = plt.subplots(3, N_CANDIDATES, figsize=(20, 10), sharey="row")

    for cls in range(3):
        rows = db.get_results(converged=True, profile_class=cls)
        if not rows:
            for j in range(N_CANDIDATES):
                axes[cls, j].text(0.5, 0.5, "No data", transform=axes[cls, j].transAxes, ha="center")
            continue

        # 모든 사례의 cleanliness score 계산
        scored = []
        for row in rows:
            try:
                t, u_mag, T = _load_trajectory(db, row)
                score = _cleanliness_score(t, u_mag, T, EXPECTED_PEAKS[cls])
                scored.append((score, row, t, u_mag, T))
            except Exception:
                continue

        # 점수 오름차순 정렬 (낮을수록 깨끗)
        scored.sort(key=lambda x: x[0])

        # median-cost 사례 id (비교용)
        rows_sorted = sorted(rows, key=lambda r: r["cost"])
        median_row = rows_sorted[len(rows_sorted) // 2]

        for j in range(min(N_CANDIDATES, len(scored))):
            score, row, t, u_mag, T = scored[j]
            ax = axes[cls, j]

            n_pk, peak_idx, u_smooth, prom_thresh, _, _ = _detect_peaks_custom(t, u_mag, T)
            t_hr = t / 3600.0

            ax.plot(t_hr, u_mag, color="gray", alpha=0.4, linewidth=0.6)
            ax.plot(t_hr, u_smooth, color=CLASS_COLORS[cls], linewidth=1.2)
            if len(peak_idx) > 0:
                ax.plot(t_hr[peak_idx], u_smooth[peak_idx], "v", color="black", markersize=6)

            is_median = row["id"] == median_row["id"]
            title = f"id={row['id']}, pk={n_pk}\nscore={score:.3f}"
            if is_median:
                title += " [MEDIAN]"
                ax.set_facecolor("#FFFFDD")

            ax.set_title(title, fontsize=8)
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel(f"{CLASS_NAMES[cls]}\n" + r"$\|u\|$", fontsize=9)
            if cls == 2:
                ax.set_xlabel("Time [hr]", fontsize=8)

        # 빈 서브플롯 처리
        for j in range(len(scored), N_CANDIDATES):
            axes[cls, j].set_visible(False)

    fig.suptitle(
        "Clean Representative Candidates (top 5 per class, lowest cleanliness score)",
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(outdir, "diag4_clean_candidates.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DIAGNOSTICS = {
    "diag1": diag1_median_cases,
    "diag2": diag2_npeaks_distribution,
    "diag3": diag3_sensitivity,
    "diag4": diag4_clean_candidates,
}


def main():
    parser = argparse.ArgumentParser(
        description="피크 탐지 알고리즘 진단",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/diagnose_peak_detection.py --db data/trajectories.duckdb
  python scripts/diagnose_peak_detection.py --db data/trajectories.duckdb --only diag1,diag3
  python scripts/diagnose_peak_detection.py --db data/trajectories.duckdb --outdir ./diag_output
        """,
    )
    parser.add_argument("--db", type=str, default="data/trajectories.duckdb", help="DuckDB 경로")
    parser.add_argument(
        "--outdir",
        type=str,
        default="manuscript/figures/diagnostics",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="실행할 진단 항목 (쉼표 구분, e.g. diag1,diag3)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    db = TrajectoryDatabase(db_path=args.db)
    total = len(db.get_results(converged=True))
    print(f"Database: {total} converged cases")
    print(f"Output:   {args.outdir}")
    print(f"Config:   PEAK_PROMINENCE_RATIO={PEAK_PROMINENCE_RATIO}, "
          f"PEAK_MIN_DISTANCE_RATIO={PEAK_MIN_DISTANCE_RATIO}")
    print()

    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
    else:
        keys = list(DIAGNOSTICS.keys())

    for key in keys:
        if key in DIAGNOSTICS:
            try:
                DIAGNOSTICS[key](db, args.outdir)
            except Exception as e:
                print(f"  {key}: FAILED - {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  Unknown diagnostic: {key}")
        print()

    db.close()
    print("Done.")


if __name__ == "__main__":
    main()
