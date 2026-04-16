import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import duckdb
import pandas as pd
import numpy as np


CIRC_DB = "data/trajectories_circular.duckdb"
ECC_DB = "data/trajectories_eccentric.duckdb"


def load_df(db_path: str) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM trajectories ORDER BY id").df()
    conn.close()
    return df


def main():
    circ = load_df(CIRC_DB)
    ecc = load_df(ECC_DB)

    # 비교에 쓸 key
    keys = ["h0", "delta_a", "delta_i", "T_normed"]

    # 필요한 컬럼만 남기고 이름 바꾸기
    circ_sub = circ[keys + ["converged", "profile_class", "cost", "n_peaks"]].copy()
    circ_sub = circ_sub.rename(columns={
        "converged": "converged_circ",
        "profile_class": "class_circ",
        "cost": "cost_circ",
        "n_peaks": "n_peaks_circ",
    })

    ecc_sub = ecc[keys + ["converged", "profile_class", "cost", "n_peaks"]].copy()
    ecc_sub = ecc_sub.rename(columns={
        "converged": "converged_ecc",
        "profile_class": "class_ecc",
        "cost": "cost_ecc",
        "n_peaks": "n_peaks_ecc",
    })

    # 같은 조건끼리 merge
    merged = pd.merge(circ_sub, ecc_sub, on=keys, how="inner")

    print("\n=== merged summary ===")
    print("총 매칭된 점 수:", len(merged))

    # 둘 다 수렴한 점만
    both_ok = merged[
        (merged["converged_circ"] == True) &
        (merged["converged_ecc"] == True)
    ].copy()

    print("둘 다 converged인 점 수:", len(both_ok))

    # class 바뀐 점
    switched = both_ok[both_ok["class_circ"] != both_ok["class_ecc"]].copy()

    print("class switch 개수:", len(switched))

    if len(switched) == 0:
        print("class switch 없음")
        return

    # cost 차이 계산
    switched["cost_diff"] = switched["cost_ecc"] - switched["cost_circ"]

    # 보기 좋게 정렬
    switched = switched.sort_values(
        by=["T_normed", "delta_a", "delta_i"]
    ).reset_index(drop=True)

    print("\n=== class switched cases (first 20) ===")
    print(
        switched[
            [
                "h0", "delta_a", "delta_i", "T_normed",
                "class_circ", "class_ecc",
                "n_peaks_circ", "n_peaks_ecc",
                "cost_circ", "cost_ecc", "cost_diff"
            ]
        ].head(20).to_string(index=False)
    )

    # switch 패턴 요약
    print("\n=== switch pattern counts ===")
    pattern_counts = (
        switched.groupby(["class_circ", "class_ecc"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    print(pattern_counts.to_string(index=False))

    # T별 switch 개수
    print("\n=== switch count by T_normed ===")
    t_counts = (
        switched.groupby("T_normed")
        .size()
        .reset_index(name="count")
        .sort_values("T_normed")
    )
    print(t_counts.to_string(index=False))

    # csv 저장
    switched.to_csv("results/class_switch_cases.csv", index=False)
    pattern_counts.to_csv("results/class_switch_pattern_counts.csv", index=False)
    t_counts.to_csv("results/class_switch_count_by_T.csv", index=False)

    print("\n저장 완료:")
    print(" - results/class_switch_cases.csv")
    print(" - results/class_switch_pattern_counts.csv")
    print(" - results/class_switch_count_by_T.csv")


if __name__ == "__main__":
    main()