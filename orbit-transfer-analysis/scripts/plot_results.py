import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import math
import matplotlib.pyplot as plt

from orbit_transfer.pipeline.evaluate import create_database


def class_label(profile_class: int) -> str:
    if profile_class == 0:
        return "class 0"
    elif profile_class == 1:
        return "class 1"
    elif profile_class == 2:
        return "class 2"
    return f"class {profile_class}"


def main():
    h0 = 400.0

    # T_normed를 특정 값 근처만 보고 싶으면 설정
    # 예: 5.0 근처만 보고 싶으면 target_T = 5.0, tol_T = 0.3
    # 전체 다 보려면 target_T = None
    target_T = 2.0
    tol_T = 0.5

    db = create_database(
        db_path="data/trajectories.duckdb",
        npz_dir="data/trajectories",
    )

    rows = db.conn.execute(
        """
        SELECT
            h0, delta_a, delta_i, T_normed,
            converged, profile_class, n_peaks, cost, solve_time
        FROM trajectories
        WHERE h0 = ?
        ORDER BY id ASC
        """,
        [h0],
    ).fetchall()

    if len(rows) == 0:
        print("DB에 데이터가 없음")
        return

    filtered = []
    for row in rows:
        row_h0, delta_a, delta_i, T_normed, converged, profile_class, n_peaks, cost, solve_time = row

        if target_T is not None:
            if abs(T_normed - target_T) > tol_T:
                continue

        filtered.append({
            "delta_a": delta_a,
            "delta_i": delta_i,
            "T_normed": T_normed,
            "converged": converged,
            "profile_class": profile_class,
            "n_peaks": n_peaks,
            "cost": cost,
            "solve_time": solve_time,
        })

    if len(filtered) == 0:
        print("필터 조건에 맞는 데이터가 없음")
        return

    plt.figure(figsize=(9, 7))

    # 클래스별 + 성공/실패별로 나눠서 그림
    classes = sorted(set(row["profile_class"] for row in filtered))

    for c in classes:
        success_x = []
        success_y = []
        fail_x = []
        fail_y = []

        for row in filtered:
            if row["profile_class"] != c:
                continue

            if row["converged"]:
                success_x.append(row["delta_a"])
                success_y.append(row["delta_i"])
            else:
                fail_x.append(row["delta_a"])
                fail_y.append(row["delta_i"])

        if len(success_x) > 0:
            plt.scatter(
                success_x,
                success_y,
                label=f"{class_label(c)} / success",
                marker="o",
                s=70,
                alpha=0.8,
            )

        if len(fail_x) > 0:
            plt.scatter(
                fail_x,
                fail_y,
                label=f"{class_label(c)} / fail",
                marker="x",
                s=90,
                alpha=0.9,
            )

    title = f"Orbit-transfer results at h0={h0} km"
    if target_T is not None:
        title += f" (T_normed ≈ {target_T} ± {tol_T})"

    plt.title(title)
    plt.xlabel("delta_a [km]")
    plt.ylabel("delta_i [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()