import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import duckdb
import pandas as pd


CIRC_DB = "data/trajectories_circular.duckdb"
ECC_DB = "data/trajectories_eccentric.duckdb"


def load_df(db_path: str) -> pd.DataFrame:
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM trajectories ORDER BY id").df()
    conn.close()
    return df


def summarize(df: pd.DataFrame, name: str):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("총 샘플 수:", len(df))

    if "converged" in df.columns:
        conv_rate = df["converged"].mean()
        print("수렴률:", conv_rate)

    if "profile_class" in df.columns:
        print("\nclass 분포:")
        print(df["profile_class"].value_counts().sort_index())

    if "n_peaks" in df.columns:
        print("\n평균 n_peaks:")
        print(df["n_peaks"].mean())

    if "cost" in df.columns:
        finite_cost = df["cost"].replace([float("inf"), -float("inf")], pd.NA).dropna()
        print("\n유한 cost 개수:", len(finite_cost))
        if len(finite_cost) > 0:
            print("평균 cost:", finite_cost.mean())

    if "T_f" in df.columns:
        print("\n평균 T_f:")
        print(df["T_f"].mean())

    if "e0" in df.columns and "ef" in df.columns:
        print("\ne0, ef unique:")
        print("e0:", sorted(df["e0"].unique()))
        print("ef:", sorted(df["ef"].unique()))


def compare_two(circ: pd.DataFrame, ecc: pd.DataFrame):
    print("\n" + "=" * 70)
    print("요약 비교")
    print("=" * 70)

    print("circular 총 샘플:", len(circ))
    print("eccentric 총 샘플:", len(ecc))

    if "converged" in circ.columns and "converged" in ecc.columns:
        print("\n수렴률 비교")
        print("circular :", circ["converged"].mean())
        print("eccentric:", ecc["converged"].mean())

    if "profile_class" in circ.columns and "profile_class" in ecc.columns:
        circ_cls = circ["profile_class"].value_counts().sort_index()
        ecc_cls = ecc["profile_class"].value_counts().sort_index()

        print("\nclass 분포 비교")
        print("[circular]")
        print(circ_cls)
        print("[eccentric]")
        print(ecc_cls)

    if "cost" in circ.columns and "cost" in ecc.columns:
        circ_cost = circ["cost"].replace([float("inf"), -float("inf")], pd.NA).dropna()
        ecc_cost = ecc["cost"].replace([float("inf"), -float("inf")], pd.NA).dropna()

        print("\n평균 cost 비교")
        if len(circ_cost) > 0:
            print("circular :", circ_cost.mean())
        if len(ecc_cost) > 0:
            print("eccentric:", ecc_cost.mean())

    if "n_peaks" in circ.columns and "n_peaks" in ecc.columns:
        print("\n평균 n_peaks 비교")
        print("circular :", circ["n_peaks"].mean())
        print("eccentric:", ecc["n_peaks"].mean())


def main():
    circ = load_df(CIRC_DB)
    ecc = load_df(ECC_DB)

    summarize(circ, "CIRCULAR SLICE")
    summarize(ecc, "ECCENTRIC SLICE")
    compare_two(circ, ecc)


if __name__ == "__main__":
    main()