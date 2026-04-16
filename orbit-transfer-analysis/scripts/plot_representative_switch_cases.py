import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import os
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CIRC_DB = "data/trajectories_circular.duckdb"
ECC_DB = "data/trajectories_eccentric.duckdb"
SWITCH_CSV = "results/class_switch_cases.csv"

OUT_DIR = "fig/representative_switch_cases"
OUT_TABLE = "results/representative_switch_cases.csv"


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


def choose_representative_cases(df: pd.DataFrame):
    # 대표적으로 보고 싶은 패턴
    target_patterns = [(2, 0), (2, 1), (1, 0)]

    selected = []

    for c_from, c_to in target_patterns:
        sub = df[
            (df["class_circ"] == c_from) &
            (df["class_ecc"] == c_to)
        ].copy()

        if len(sub) == 0:
            print(f"[SKIP] pattern {c_from}->{c_to} 없음")
            continue

        # 대표 선택 기준: |cost_diff| 가장 큰 케이스
        sub["abs_cost_diff"] = np.abs(sub["cost_diff"])
        row = sub.sort_values("abs_cost_diff", ascending=False).iloc[0]
        selected.append(row)

    if len(selected) == 0:
        return pd.DataFrame()

    return pd.DataFrame(selected).reset_index(drop=True)


def plot_case(case_row, case_idx):
    h0 = float(case_row["h0"])
    delta_a = float(case_row["delta_a"])
    delta_i = float(case_row["delta_i"])
    T_normed = float(case_row["T_normed"])

    class_circ = int(case_row["class_circ"])
    class_ecc = int(case_row["class_ecc"])

    circ_row = load_matching_row(CIRC_DB, h0, delta_a, delta_i, T_normed)
    ecc_row = load_matching_row(ECC_DB, h0, delta_a, delta_i, T_normed)

    if circ_row is None:
        print(f"[SKIP] circular row 못 찾음: case {case_idx}")
        return
    if ecc_row is None:
        print(f"[SKIP] eccentric row 못 찾음: case {case_idx}")
        return

    t_c, u_c = load_npz(circ_row["trajectory_file"])
    t_e, u_e = load_npz(ecc_row["trajectory_file"])

    # 시간 단위: 시간[hr]로 표시
    t_c_hr = t_c / 3600.0
    t_e_hr = t_e / 3600.0

    u_mag_c = np.linalg.norm(u_c, axis=0)
    u_mag_e = np.linalg.norm(u_e, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"class {class_circ} -> {class_ecc} representative case | "
        f"h0={h0:.1f} km, delta_a={delta_a:.6f} km, "
        f"delta_i={delta_i:.6f} deg, T_normed={T_normed:.6f}"
    )

    # ||u(t)||
    ax = axes[0, 0]
    ax.plot(t_c_hr, u_mag_c, label="circular")
    ax.plot(t_e_hr, u_mag_e, label="eccentric")
    ax.set_title("||u(t)||")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    # u_x
    ax = axes[0, 1]
    ax.plot(t_c_hr, u_c[0, :], label="circular")
    ax.plot(t_e_hr, u_e[0, :], label="eccentric")
    ax.set_title("u_x")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    # u_y
    ax = axes[1, 0]
    ax.plot(t_c_hr, u_c[1, :], label="circular")
    ax.plot(t_e_hr, u_e[1, :], label="eccentric")
    ax.set_title("u_y")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    # u_z
    ax = axes[1, 1]
    ax.plot(t_c_hr, u_c[2, :], label="circular")
    ax.plot(t_e_hr, u_e[2, :], label="eccentric")
    ax.set_title("u_z")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    filename = (
        f"case_{case_idx+1}_class_{class_circ}_to_{class_ecc}"
        f"_T_{T_normed:.3f}.png"
    )
    save_path = os.path.join(OUT_DIR, filename)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {save_path}")


def main():
    df = pd.read_csv(SWITCH_CSV)

    print("총 class switch rows:", len(df))

    reps = choose_representative_cases(df)
    if len(reps) == 0:
        print("대표 케이스 없음")
        return

    os.makedirs("results", exist_ok=True)
    reps.to_csv(OUT_TABLE, index=False)

    print("\n대표 케이스:")
    print(
        reps[
            [
                "h0", "delta_a", "delta_i", "T_normed",
                "class_circ", "class_ecc",
                "n_peaks_circ", "n_peaks_ecc",
                "cost_circ", "cost_ecc", "cost_diff"
            ]
        ].to_string(index=False)
    )

    print(f"\n저장 완료: {OUT_TABLE}")

    for i, row in reps.iterrows():
        plot_case(row, i)


if __name__ == "__main__":
    main()