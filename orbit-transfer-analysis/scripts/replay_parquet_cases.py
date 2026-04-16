import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

from pathlib import Path
import time
import numpy as np
import pandas as pd

from orbit_transfer.pipeline.evaluate import evaluate_transfer


PARQUET_PATH = "/Users/heewon/Desktop/무제 폴더/trajectories_h400.parquet"
OUT_DIR = "results/replay_parquet"
OUT_CSV = f"{OUT_DIR}/replay_comparison.csv"


def pick_subset(df: pd.DataFrame, h0=400.0, n_cases=30, random_seed=42) -> pd.DataFrame:
    """
    parquet에서 replay할 부분집합 선택.
    기본:
    - h0=400만 사용
    - converged True/False 섞이게
    - T_normed 분포도 어느 정도 다양하게
    """
    sub = df[df["h0"] == h0].copy()

    # 혹시 컬럼명이 parquet에서 T_max_normed인 경우
    if "T_max_normed" in sub.columns and "T_normed" not in sub.columns:
        sub["T_normed"] = sub["T_max_normed"]

    # 중복 입력 제거 (같은 입력 여러 번 방지)
    key_cols = ["h0", "delta_a", "delta_i", "T_normed", "e0", "ef"]
    sub = sub.drop_duplicates(subset=key_cols).copy()

    # 성공/실패 나눠서 균형 있게
    ok = sub[sub["converged"] == True].copy()
    fail = sub[sub["converged"] == False].copy()

    n_half = n_cases // 2
    rng = np.random.default_rng(random_seed)

    ok_n = min(len(ok), n_half)
    fail_n = min(len(fail), n_cases - ok_n)

    ok_idx = rng.choice(ok.index, size=ok_n, replace=False) if ok_n > 0 else []
    fail_idx = rng.choice(fail.index, size=fail_n, replace=False) if fail_n > 0 else []

    chosen = pd.concat([ok.loc[ok_idx], fail.loc[fail_idx]], axis=0)

    # 아직 부족하면 남은 것에서 채우기
    remain = n_cases - len(chosen)
    if remain > 0:
        used_idx = set(chosen.index)
        leftover = sub[~sub.index.isin(used_idx)]
        add_n = min(len(leftover), remain)
        if add_n > 0:
            add_idx = rng.choice(leftover.index, size=add_n, replace=False)
            chosen = pd.concat([chosen, leftover.loc[add_idx]], axis=0)

    # 보기 좋게 정렬
    chosen = chosen.sort_values(["T_normed", "delta_a", "delta_i"]).reset_index(drop=True)
    return chosen


def replay_one_case(row: pd.Series) -> dict:
    """
    parquet row 하나를 현재 코드로 replay.
    """
    h0 = float(row["h0"])
    delta_a = float(row["delta_a"])
    delta_i = float(row["delta_i"])
    T_normed = float(row["T_normed"])
    e0 = float(row["e0"])
    ef = float(row["ef"])

    t0 = time.perf_counter()
    label, result = evaluate_transfer(
        h0=h0,
        delta_a=delta_a,
        delta_i=delta_i,
        T_normed=T_normed,
        e0=e0,
        ef=ef,
        db=None,
    )
    dt = time.perf_counter() - t0

    return {
        "replay_converged": bool(result.converged),
        "replay_class": int(label),
        "replay_n_peaks": int(result.n_peaks),
        "replay_cost": float(result.cost),
        "replay_nu0": float(result.nu0),
        "replay_nuf": float(result.nuf),
        "replay_runtime": float(dt),
    }


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)

    # parquet 컬럼명 통일
    if "T_max_normed" in df.columns and "T_normed" not in df.columns:
        df["T_normed"] = df["T_max_normed"]

    subset = pick_subset(df, h0=400.0, n_cases=30, random_seed=42)

    print("선택된 케이스 수:", len(subset))
    print("저장 경로:", OUT_CSV)

    rows = []

    for i, row in subset.iterrows():
        print(f"\n[{i+1}/{len(subset)}] replay 중...")
        print(
            f"h0={row['h0']}, delta_a={row['delta_a']}, delta_i={row['delta_i']}, "
            f"T_normed={row['T_normed']}, e0={row['e0']}, ef={row['ef']}"
        )

        replay = replay_one_case(row)

        out = {
            # 입력
            "id": row.get("id", np.nan),
            "h0": row["h0"],
            "delta_a": row["delta_a"],
            "delta_i": row["delta_i"],
            "T_normed": row["T_normed"],
            "e0": row["e0"],
            "ef": row["ef"],

            # parquet 결과
            "parquet_converged": row.get("converged", np.nan),
            "parquet_class": row.get("profile_class", np.nan),
            "parquet_n_peaks": row.get("n_peaks", np.nan),
            "parquet_cost": row.get("cost", np.nan),
            "parquet_nu0": row.get("nu0", np.nan),
            "parquet_nuf": row.get("nuf", np.nan),

            # replay 결과
            **replay,
        }

        # 비교 컬럼
        out["same_converged"] = (
            pd.notna(out["parquet_converged"]) and
            bool(out["parquet_converged"]) == bool(out["replay_converged"])
        )

        out["same_class"] = (
            pd.notna(out["parquet_class"]) and
            int(out["parquet_class"]) == int(out["replay_class"])
        )

        try:
            out["cost_diff"] = float(out["replay_cost"]) - float(out["parquet_cost"])
        except Exception:
            out["cost_diff"] = np.nan

        rows.append(out)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False)

    print("\n완료")
    print("비교 csv 저장:", OUT_CSV)

    print("\n요약:")
    print("same_converged rate =", result_df["same_converged"].mean())
    print("same_class rate     =", result_df["same_class"].mean())

    finite_cost_diff = result_df["cost_diff"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(finite_cost_diff) > 0:
        print("mean cost diff      =", finite_cost_diff.mean())


if __name__ == "__main__":
    main()