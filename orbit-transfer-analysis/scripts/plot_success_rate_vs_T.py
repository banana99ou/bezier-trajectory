import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import duckdb
import numpy as np
import matplotlib.pyplot as plt


CIRC_DB = "data/trajectories_circular.duckdb"
ECC_DB = "data/trajectories_eccentric.duckdb"


def load_df(db_path: str):
    conn = duckdb.connect(db_path, read_only=True)
    df = conn.execute("SELECT * FROM trajectories ORDER BY id").df()
    conn.close()
    return df


def compute_success_rate(df, h0, target_T, tol_T):
    sub = df[
        (df["h0"] == h0) &
        (np.abs(df["T_normed"] - target_T) <= tol_T)
    ].copy()

    if len(sub) == 0:
        return np.nan, 0

    success_rate = sub["converged"].mean()
    return float(success_rate), len(sub)


def main():
    h0 = 400.0
    tol_T = 0.5
    T_targets = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    circ = load_df(CIRC_DB)
    ecc = load_df(ECC_DB)

    circ_rates = []
    ecc_rates = []
    circ_counts = []
    ecc_counts = []

    print("\n=== success rate by T ===")
    for T in T_targets:
        circ_rate, circ_n = compute_success_rate(circ, h0, T, tol_T)
        ecc_rate, ecc_n = compute_success_rate(ecc, h0, T, tol_T)

        circ_rates.append(circ_rate)
        ecc_rates.append(ecc_rate)
        circ_counts.append(circ_n)
        ecc_counts.append(ecc_n)

        print(f"T≈{T}±{tol_T}")
        print(f"  circular : rate={circ_rate}, n={circ_n}")
        print(f"  eccentric: rate={ecc_rate}, n={ecc_n}")

    plt.figure(figsize=(9, 6))
    plt.plot(T_targets, circ_rates, marker="o", label="circular")
    plt.plot(T_targets, ecc_rates, marker="o", label="eccentric")

    for x, y, n in zip(T_targets, circ_rates, circ_counts):
        if not np.isnan(y):
            plt.text(x, y + 0.02, f"n={n}", ha="center", fontsize=9)

    for x, y, n in zip(T_targets, ecc_rates, ecc_counts):
        if not np.isnan(y):
            plt.text(x, y - 0.06, f"n={n}", ha="center", fontsize=9)

    plt.ylim(0.0, 1.05)
    plt.xlabel("T_normed target")
    plt.ylabel("success rate")
    plt.title(f"Success Rate vs T_normed (h0={h0}, tol={tol_T})")
    plt.grid(True)
    plt.legend()

    plt.savefig("fig/success_rate_vs_T.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\n저장 완료: fig/success_rate_vs_T.png")


if __name__ == "__main__":
    main()