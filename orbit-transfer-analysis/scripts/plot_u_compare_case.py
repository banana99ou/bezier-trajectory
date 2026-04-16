import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import duckdb
import numpy as np
import matplotlib.pyplot as plt


CIRC_DB = "data/trajectories_circular.duckdb"
ECC_DB = "data/trajectories_eccentric.duckdb"


def load_matching_row(db_path, h0, delta_a, delta_i, T_normed, tol=1e-6):
    conn = duckdb.connect(db_path, read_only=True)

    query = """
    SELECT *
    FROM trajectories
    WHERE ABS(h0 - ?) < ?
      AND ABS(delta_a - ?) < ?
      AND ABS(delta_i - ?) < ?
      AND ABS(T_normed - ?) < ?
    ORDER BY ABS(delta_a - ?) + ABS(delta_i - ?) + ABS(T_normed - ?)
    LIMIT 1
    """

    row = conn.execute(
        query,
        [
            h0, tol,
            delta_a, tol,
            delta_i, tol,
            T_normed, tol,
            delta_a, delta_i, T_normed,
        ],
    ).fetchdf()

    conn.close()

    if len(row) == 0:
        return None
    return row.iloc[0]


def load_npz(npz_path):
    data = np.load(npz_path)
    return data["t"], data["u"]


def main():
    # =========================
    # 여기 값을 class_switch_cases.csv에서 복붙해서 넣으면 됨
    # =========================
    h0 = 400.0
    delta_a = 1531.75176884137
    delta_i = 11.785864026188
    T_normed = 1.53396816819833

    circ_row = load_matching_row(CIRC_DB, h0, delta_a, delta_i, T_normed)
    ecc_row = load_matching_row(ECC_DB, h0, delta_a, delta_i, T_normed)

    if circ_row is None:
        print("circular DB에서 해당 케이스를 못 찾음")
        return
    if ecc_row is None:
        print("eccentric DB에서 해당 케이스를 못 찾음")
        return

    t_c, u_c = load_npz(circ_row["trajectory_file"])
    t_e, u_e = load_npz(ecc_row["trajectory_file"])

    u_mag_c = np.linalg.norm(u_c, axis=0)
    u_mag_e = np.linalg.norm(u_e, axis=0)

    print("\n=== circular ===")
    print("class     =", circ_row["profile_class"])
    print("n_peaks   =", circ_row["n_peaks"])
    print("cost      =", circ_row["cost"])
    print("converged =", circ_row["converged"])
    print("file      =", circ_row["trajectory_file"])

    print("\n=== eccentric ===")
    print("class     =", ecc_row["profile_class"])
    print("n_peaks   =", ecc_row["n_peaks"])
    print("cost      =", ecc_row["cost"])
    print("converged =", ecc_row["converged"])
    print("file      =", ecc_row["trajectory_file"])

    plt.figure(figsize=(10, 6))
    plt.plot(t_c, u_mag_c, label=f"circular (class={int(circ_row['profile_class'])}, peaks={int(circ_row['n_peaks'])})")
    plt.plot(t_e, u_mag_e, label=f"eccentric (class={int(ecc_row['profile_class'])}, peaks={int(ecc_row['n_peaks'])})")

    plt.xlabel("time [s]")
    plt.ylabel("||u(t)||")
    plt.title(
        "u(t) comparison\n"
        f"h0={h0}, delta_a={delta_a}, delta_i={delta_i}, T_normed={T_normed}"
    )
    plt.grid(True)
    plt.legend()

    plt.savefig("fig/u_compare_case.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\n저장 완료: fig/u_compare_case.png")


if __name__ == "__main__":
    main()