import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

from orbit_transfer.pipeline.evaluate import create_database
from orbit_transfer.database.query import get_statistics, get_boundary_cases


def main():
    h0 = 400.0

    db = create_database(
        db_path="data/trajectories.duckdb",
        npz_dir="data/trajectories",
    )

    stats = get_statistics(db, h0=h0)

    print("=" * 60)
    print("DB 요약")
    print("=" * 60)
    print("전체 개수:", stats["total"])
    print("수렴 개수:", stats["converged"])
    print("수렴률:", stats["convergence_rate"])
    print("클래스 분포:", stats["class_distribution"])
    print("평균 비용:", stats["mean_cost"])
    print()

    print("=" * 60)
    print("경계 케이스 후보")
    print("=" * 60)
    try:
        boundary_cases = get_boundary_cases(db, h0=h0)
        if len(boundary_cases) == 0:
            print("경계 케이스 없음")
        else:
            for i, row in enumerate(boundary_cases[:10], start=1):
                print(f"[{i}] {row}")
    except Exception as e:
        print("경계 케이스 조회 중 오류:", e)

    print()

    print("=" * 60)
    print("직접 SQL로 최근 결과 일부 보기")
    print("=" * 60)
    try:
        rows = db.conn.execute(
            """
            SELECT
                h0, delta_a, delta_i, T_normed,
                converged, profile_class, n_peaks, cost, solve_time
            FROM trajectories
            WHERE h0 = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            [h0],
        ).fetchall()

        for i, row in enumerate(rows, start=1):
            print(f"[{i}] {row}")
    except Exception as e:
        print("최근 결과 조회 중 오류:", e)

    print()

    print("=" * 60)
    print("실패 케이스 보기")
    print("=" * 60)
    try:
        failed_rows = db.conn.execute(
            """
            SELECT
                h0, delta_a, delta_i, T_normed,
                converged, profile_class, n_peaks, cost, solve_time
            FROM trajectories
            WHERE h0 = ? AND converged = FALSE
            ORDER BY id DESC
            LIMIT 10
            """,
            [h0],
        ).fetchall()

        if len(failed_rows) == 0:
            print("실패 케이스 없음")
        else:
            for i, row in enumerate(failed_rows, start=1):
                print(f"[{i}] {row}")
    except Exception as e:
        print("실패 케이스 조회 중 오류:", e)


if __name__ == "__main__":
    main()