"""profile_class 0/1/2 가 unimodal/bimodal/multimodal로 올바르게 구현됐는지 검증.

검증 논리:
  - DB에서 수렴한 케이스를 로드
  - 각 케이스의 npz에서 u(t) 읽어 피크 재탐지
  - 재탐지한 n_peaks vs 저장된 profile_class 비교
  - 불일치 케이스, 경계 케이스, 클래스별 통계 출력

Usage:
    python scripts/validate_profile_classification.py
    python scripts/validate_profile_classification.py --db data/trajectories_circular.duckdb
    python scripts/validate_profile_classification.py --n_samples 200 --plot
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from orbit_transfer.classification.peak_detection import detect_peaks
from orbit_transfer.classification.classifier import classify_profile


# ─── 기본 DB 경로 ──────────────────────────────────────────────────────────────
DEFAULT_DB = "data/trajectories.duckdb"
RESULTS_DIR = Path("results")
FIG_DIR = Path("fig/validate_classification")


def parse_args():
    p = argparse.ArgumentParser(description="profile_class 분류 검증")
    p.add_argument("--db", default=DEFAULT_DB, help="DuckDB 경로")
    p.add_argument("--n_samples", type=int, default=None,
                   help="검증할 최대 샘플 수 (None=전체)")
    p.add_argument("--plot", action="store_true",
                   help="불일치 케이스 플롯 저장")
    p.add_argument("--plot_examples", action="store_true",
                   help="각 클래스별 대표 케이스 플롯 저장")
    return p.parse_args()


def load_converged(db_path: str, n_samples: int | None) -> pd.DataFrame:
    """수렴한 케이스 로드."""
    conn = duckdb.connect(db_path, read_only=True)
    q = "SELECT * FROM trajectories WHERE converged = TRUE ORDER BY id"
    if n_samples:
        q += f" LIMIT {n_samples}"
    df = conn.execute(q).df()
    conn.close()
    return df


def redetect_peaks(row: pd.Series) -> dict:
    """npz에서 u(t) 읽어 피크 재탐지."""
    traj_file = row["trajectory_file"]
    if not traj_file or not Path(traj_file).exists():
        return {"status": "no_file", "redetected_n_peaks": None, "redetected_class": None}

    data = np.load(traj_file)
    t = data["t"]
    u = data["u"]  # shape (3, N)

    if t.ndim != 1 or u.ndim != 2 or u.shape[0] != 3:
        return {"status": "bad_shape", "redetected_n_peaks": None, "redetected_class": None}

    u_mag = np.linalg.norm(u, axis=0)
    T = t[-1] - t[0]

    if T <= 0 or len(t) < 5:
        return {"status": "bad_data", "redetected_n_peaks": None, "redetected_class": None}

    n_peaks, peak_times, peak_widths = detect_peaks(t, u_mag, T)
    redetected_class = classify_profile(n_peaks)

    return {
        "status": "ok",
        "redetected_n_peaks": int(n_peaks),
        "redetected_class": int(redetected_class),
        "t": t,
        "u_mag": u_mag,
        "peak_times": peak_times,
    }


def check_class_consistency(n_peaks_stored: int, profile_class_stored: int,
                             n_peaks_redetected: int, profile_class_redetected: int) -> dict:
    """저장 값과 재탐지 값의 일치 여부 판단."""
    class_match = (profile_class_stored == profile_class_redetected)

    # 경계 케이스: n_peaks가 클래스 경계 근처 (n_peaks=1 vs 2, n_peaks=2 vs 3)
    boundary = (n_peaks_stored in [1, 2] or n_peaks_redetected in [1, 2])

    return {
        "class_match": class_match,
        "boundary": boundary,
    }


def print_summary(df: pd.DataFrame, results: list[dict]):
    ok_results = [r for r in results if r["status"] == "ok"]
    no_file = sum(1 for r in results if r["status"] == "no_file")
    bad = sum(1 for r in results if r["status"] in ("bad_shape", "bad_data"))

    print(f"\n{'=' * 65}")
    print(f"  profile_class 검증 결과")
    print(f"{'=' * 65}")
    print(f"  총 수렴 케이스       : {len(results):>6}")
    print(f"  파일 없음            : {no_file:>6}")
    print(f"  데이터 오류          : {bad:>6}")
    print(f"  재탐지 성공          : {len(ok_results):>6}")

    if not ok_results:
        print("  재탐지 성공 케이스 없음 – 종료")
        return

    n_match = sum(1 for r in ok_results if r["class_match"])
    n_mismatch = len(ok_results) - n_match
    accuracy = n_match / len(ok_results) * 100

    print(f"\n  일치 (class_match)   : {n_match:>6}  ({accuracy:.1f}%)")
    print(f"  불일치 (mismatch)    : {n_mismatch:>6}  ({100 - accuracy:.1f}%)")

    # ─── 저장 class별 n_peaks 분포 ─────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  [저장 class별 n_peaks 분포]")
    print(f"  {'class':>8}  {'label':>11}  {'n_samples':>9}  {'n_peaks mean':>12}  "
          f"{'n_peaks range':>14}")

    class_labels = {0: "unimodal", 1: "bimodal", 2: "multimodal"}
    for cls in [0, 1, 2]:
        rows_cls = [r for r in ok_results if r.get("stored_class") == cls]
        if not rows_cls:
            continue
        peaks = [r["redetected_n_peaks"] for r in rows_cls if r["redetected_n_peaks"] is not None]
        if peaks:
            print(f"  {cls:>8}  {class_labels[cls]:>11}  {len(rows_cls):>9}  "
                  f"{np.mean(peaks):>12.2f}  [{min(peaks)} – {max(peaks)}]")

    # ─── 클래스별 일치율 ───────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  [저장 class별 재탐지 일치율]")
    print(f"  {'stored':>8}  {'redetected':>10}  {'count':>7}  {'match%':>7}")

    for cls in [0, 1, 2]:
        rows_cls = [r for r in ok_results if r.get("stored_class") == cls]
        if not rows_cls:
            continue
        for rcls in [0, 1, 2]:
            cnt = sum(1 for r in rows_cls if r["redetected_class"] == rcls)
            if cnt == 0:
                continue
            match_sym = "✓" if cls == rcls else "✗"
            pct = cnt / len(rows_cls) * 100
            print(f"  {cls:>8}  {rcls:>10}  {cnt:>7}  {pct:>6.1f}%  {match_sym}")

    # ─── 불일치 케이스 샘플 ────────────────────────────────────────────────
    mismatch_rows = [r for r in ok_results if not r["class_match"]]
    if mismatch_rows:
        print(f"\n{'─' * 65}")
        print(f"  [불일치 케이스 (최대 20개)]")
        print(f"  {'id':>6}  {'h0':>6}  {'da':>6}  {'di':>5}  "
              f"{'Tn':>5}  {'stored_np':>9}  {'redet_np':>8}  "
              f"{'stored_cls':>10}  {'redet_cls':>9}")
        for r in mismatch_rows[:20]:
            print(f"  {r['id']:>6}  {r['h0']:>6.0f}  {r['delta_a']:>6.0f}  "
                  f"{r['delta_i']:>5.1f}  {r['T_normed']:>5.2f}  "
                  f"{r['stored_n_peaks']:>9}  {r['redetected_n_peaks']:>8}  "
                  f"{r['stored_class']:>10}  {r['redetected_class']:>9}")

    # ─── 경계 케이스 통계 ─────────────────────────────────────────────────
    boundary_mismatch = [r for r in mismatch_rows if r.get("boundary")]
    print(f"\n  경계 케이스 불일치 (n_peaks ∈ {{1,2}}): {len(boundary_mismatch)}")
    print(f"  비경계 불일치: {len(mismatch_rows) - len(boundary_mismatch)}")


def save_mismatch_csv(ok_results: list[dict], out_path: Path):
    """불일치 케이스를 CSV로 저장."""
    mismatch = [r for r in ok_results if not r["class_match"]]
    if not mismatch:
        print("  불일치 케이스 없음 – CSV 저장 생략")
        return
    cols = ["id", "h0", "delta_a", "delta_i", "T_normed", "e0", "ef",
            "stored_n_peaks", "stored_class", "redetected_n_peaks", "redetected_class", "boundary"]
    rows = [{k: r.get(k) for k in cols} for r in mismatch]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  불일치 CSV 저장: {out_path}")


def plot_mismatch_cases(ok_results: list[dict], n_plot: int = 6):
    """불일치 케이스 추력 프로파일 플롯."""
    mismatch = [r for r in ok_results if not r["class_match"] and "t" in r]
    if not mismatch:
        print("  플롯할 불일치 케이스 없음")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    samples = mismatch[:n_plot]
    n = len(samples)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, r in zip(axes, samples):
        t, u_mag = r["t"], r["u_mag"]
        ax.plot(t, u_mag, "b-", lw=1.2, label="||u(t)||")
        for pt in r.get("peak_times", []):
            ax.axvline(pt, color="r", ls="--", lw=0.8)
        title = (f"id={r['id']}  stored={r['stored_class']}({r['stored_n_peaks']}pk) "
                 f"→ redet={r['redetected_class']}({r['redetected_n_peaks']}pk)\n"
                 f"h0={r['h0']:.0f}km  Δa={r['delta_a']:.0f}km  "
                 f"Δi={r['delta_i']:.1f}°  Tn={r['T_normed']:.2f}")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("t [s]", fontsize=8)
        ax.set_ylabel("||u|| [km/s²]", fontsize=8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("profile_class 불일치 케이스", fontsize=11, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "mismatch_cases.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  불일치 플롯 저장: {out}")


def plot_class_examples(ok_results: list[dict]):
    """각 클래스별 대표 케이스 추력 프로파일 플롯."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    class_labels = {0: "unimodal", 1: "bimodal", 2: "multimodal"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for cls, ax in zip([0, 1, 2], axes):
        candidates = [r for r in ok_results
                      if r.get("stored_class") == cls and r["class_match"] and "t" in r]
        if not candidates:
            ax.text(0.5, 0.5, f"class {cls}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"class {cls}: {class_labels[cls]}")
            continue

        # n_peaks가 클래스 기댓값에 정확히 맞는 케이스 우선
        expected_peaks = {0: 1, 1: 2, 2: 3}
        exact = [r for r in candidates if r["redetected_n_peaks"] == expected_peaks[cls]]
        sample = exact[0] if exact else candidates[0]

        t, u_mag = sample["t"], sample["u_mag"]
        ax.plot(t, u_mag, "b-", lw=1.5)
        for pt in sample.get("peak_times", []):
            ax.axvline(pt, color="r", ls="--", lw=1.0, alpha=0.7)
        ax.fill_between(t, 0, u_mag, alpha=0.15)
        ax.set_title(
            f"class {cls}: {class_labels[cls]}\n"
            f"id={sample['id']}  n_peaks={sample['redetected_n_peaks']}  "
            f"Tn={sample['T_normed']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("t [s]")
        ax.set_ylabel("||u(t)|| [km/s²]")

    fig.suptitle("각 profile_class 대표 케이스", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "class_examples.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  클래스 예시 플롯 저장: {out}")


def main():
    args = parse_args()

    print(f"DB 로드: {args.db}")
    df = load_converged(args.db, args.n_samples)
    print(f"수렴 케이스: {len(df)}")

    if len(df) == 0:
        print("수렴 케이스가 없습니다.")
        return

    results = []
    for _, row in df.iterrows():
        r = redetect_peaks(row)
        # 메타 정보 합치기
        r["id"] = int(row["id"])
        r["h0"] = float(row["h0"])
        r["delta_a"] = float(row["delta_a"])
        r["delta_i"] = float(row["delta_i"])
        r["T_normed"] = float(row["T_normed"])
        r["e0"] = float(row["e0"]) if "e0" in row else 0.0
        r["ef"] = float(row["ef"]) if "ef" in row else 0.0
        r["stored_n_peaks"] = int(row["n_peaks"]) if pd.notna(row["n_peaks"]) else -1
        r["stored_class"] = int(row["profile_class"]) if pd.notna(row["profile_class"]) else -1

        if r["status"] == "ok":
            check = check_class_consistency(
                r["stored_n_peaks"], r["stored_class"],
                r["redetected_n_peaks"], r["redetected_class"],
            )
            r.update(check)

        results.append(r)

    print_summary(df, results)

    # ─── CSV 저장 ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ok_results = [r for r in results if r["status"] == "ok"]
    save_mismatch_csv(ok_results, RESULTS_DIR / "profile_class_mismatch.csv")

    # ─── 플롯 ──────────────────────────────────────────────────────────────
    if args.plot:
        plot_mismatch_cases(ok_results, n_plot=9)

    if args.plot_examples:
        plot_class_examples(ok_results)

    # ─── 전체 검증 통과 여부 ───────────────────────────────────────────────
    n_mismatch = sum(1 for r in ok_results if not r["class_match"])
    n_ok = len(ok_results)
    accuracy = (n_ok - n_mismatch) / n_ok * 100 if n_ok else 0
    print(f"\n{'=' * 65}")
    if accuracy >= 95.0:
        print(f"  ✓ 검증 통과: 일치율 {accuracy:.1f}% (≥95%)")
    else:
        print(f"  ✗ 검증 주의: 일치율 {accuracy:.1f}% (<95%)")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
