import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import os
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from orbit_transfer.pipeline.evaluate import evaluate_transfer


# 파일 경로
PARQUET_PATH = "/Users/heewon/Desktop/무제 폴더/trajectories_h400.parquet"
REPLAY_CSV = "results/replay_parquet/replay_comparison.csv"

OUT_DIR = "fig/replay_mismatch_cases"
OUT_TABLE = "results/replay_parquet/replay_mismatch_cases.csv"


def find_parquet_row(df, h0, delta_a, delta_i, T_normed, e0, ef, tol=1e-6):
    sub = df[
        (np.abs(df["h0"] - h0) < tol) &
        (np.abs(df["delta_a"] - delta_a) < tol) &
        (np.abs(df["delta_i"] - delta_i) < tol) &
        (np.abs(df["T_normed"] - T_normed) < tol) &
        (np.abs(df["e0"] - e0) < tol) &
        (np.abs(df["ef"] - ef) < tol)
    ].copy()

    if len(sub) == 0:
        return None
    return sub.iloc[0]


def load_parquet_trajectory(base_dir, rel_path):
    full_path = Path(base_dir) / rel_path

    if not full_path.exists():
        return None, None, str(full_path)

    data = np.load(full_path)
    return data["t"], data["u"], str(full_path)


def save_replay_trajectory(case_idx, t, u):
    os.makedirs(f"{OUT_DIR}/npz", exist_ok=True)
    path = f"{OUT_DIR}/npz/replay_case_{case_idx+1:02d}.npz"
    np.savez_compressed(path, t=t, u=u)
    return path


def plot_case(case_idx, parquet_row, replay_row, parquet_base_dir):
    h0 = float(replay_row["h0"])
    delta_a = float(replay_row["delta_a"])
    delta_i = float(replay_row["delta_i"])
    T_normed = float(replay_row["T_normed"])
    e0 = float(replay_row["e0"])
    ef = float(replay_row["ef"])

    # parquet trajectory
    t_p, u_p, parquet_npz_path = load_parquet_trajectory(
    parquet_base_dir,
    parquet_row["trajectory_file"]
    )

    if t_p is None or u_p is None:
        print(f"[SKIP] parquet npz 없음: {parquet_npz_path}")
        return None
    

    # replay trajectory: 현재 코드로 다시 돌림
    label, result = evaluate_transfer(
        h0=h0,
        delta_a=delta_a,
        delta_i=delta_i,
        T_normed=T_normed,
        e0=e0,
        ef=ef,
        db=None,
    )

    t_r = result.t
    u_r = result.u
    replay_npz_path = save_replay_trajectory(case_idx, t_r, u_r)

    u_mag_p = np.linalg.norm(u_p, axis=0)
    u_mag_r = np.linalg.norm(u_r, axis=0)

    # 시간 단위 hr
    t_p_hr = t_p / 3600.0
    t_r_hr = t_r / 3600.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Replay mismatch case {case_idx+1}\n"
        f"h0={h0:.1f}, delta_a={delta_a:.6f}, delta_i={delta_i:.6f}, "
        f"T_normed={T_normed:.6f}, e0={e0:.6f}, ef={ef:.6f}\n"
        f"parquet class={int(replay_row['parquet_class'])}, "
        f"replay class={int(replay_row['replay_class'])}"
    )

    # ||u(t)||
    ax = axes[0, 0]
    ax.plot(t_p_hr, u_mag_p, label=f"parquet (class={int(replay_row['parquet_class'])}, peaks={int(replay_row['parquet_n_peaks'])})")
    ax.plot(t_r_hr, u_mag_r, label=f"replay (class={int(replay_row['replay_class'])}, peaks={int(replay_row['replay_n_peaks'])})")
    ax.set_title("||u(t)||")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    # u_x
    ax = axes[0, 1]
    ax.plot(t_p_hr, u_p[0, :], label="parquet")
    ax.plot(t_r_hr, u_r[0, :], label="replay")
    ax.set_title("u_x")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    # u_y
    ax = axes[1, 0]
    ax.plot(t_p_hr, u_p[1, :], label="parquet")
    ax.plot(t_r_hr, u_r[1, :], label="replay")
    ax.set_title("u_y")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    # u_z
    ax = axes[1, 1]
    ax.plot(t_p_hr, u_p[2, :], label="parquet")
    ax.plot(t_r_hr, u_r[2, :], label="replay")
    ax.set_title("u_z")
    ax.set_xlabel("Time [hr]")
    ax.set_ylabel("Thrust [km/s^2]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig_path = f"{OUT_DIR}/mismatch_case_{case_idx+1:02d}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {fig_path}")

    return {
        "case_idx": case_idx + 1,
        "h0": h0,
        "delta_a": delta_a,
        "delta_i": delta_i,
        "T_normed": T_normed,
        "e0": e0,
        "ef": ef,
        "parquet_class": int(replay_row["parquet_class"]),
        "replay_class": int(replay_row["replay_class"]),
        "parquet_n_peaks": int(replay_row["parquet_n_peaks"]),
        "replay_n_peaks": int(replay_row["replay_n_peaks"]),
        "parquet_cost": float(replay_row["parquet_cost"]),
        "replay_cost": float(replay_row["replay_cost"]),
        "parquet_npz_path": parquet_npz_path,
        "replay_npz_path": replay_npz_path,
        "figure_path": fig_path,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("results/replay_parquet", exist_ok=True)

    replay_df = pd.read_csv(REPLAY_CSV)
    parquet_df = pd.read_parquet(PARQUET_PATH)

    # 컬럼명 통일
    if "T_max_normed" in parquet_df.columns and "T_normed" not in parquet_df.columns:
        parquet_df["T_normed"] = parquet_df["T_max_normed"]

    # mismatch + 둘 다 성공
    mismatch = replay_df[
        (replay_df["parquet_converged"] == True) &
        (replay_df["replay_converged"] == True) &
        (replay_df["parquet_class"] != replay_df["replay_class"])
    ].copy()

    print("mismatch rows:", len(mismatch))

    if len(mismatch) == 0:
        print("비교할 mismatch 케이스 없음")
        return

    # parquet class가 큰 것(1,2)부터 보게 정렬
    mismatch = mismatch.sort_values(
        by=["parquet_class", "T_normed", "delta_a", "delta_i"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    # 최대 5개까지만 그림 저장
    n_plot = min(len(mismatch), 5)

    records = []
    parquet_base_dir = str(Path(PARQUET_PATH).parent)

    for i in range(n_plot):
        row = mismatch.iloc[i]

        parquet_row = find_parquet_row(
            parquet_df,
            h0=float(row["h0"]),
            delta_a=float(row["delta_a"]),
            delta_i=float(row["delta_i"]),
            T_normed=float(row["T_normed"]),
            e0=float(row["e0"]),
            ef=float(row["ef"]),
        )

        if parquet_row is None:
            print(f"[SKIP] parquet row 못 찾음: index={i}")
            continue

        rec = plot_case(i, parquet_row, row, parquet_base_dir)
        if rec is not None:
            records.append(rec)

    if len(records) > 0:
        out_df = pd.DataFrame(records)
        out_df.to_csv(OUT_TABLE, index=False)
        print(f"\n저장 완료: {OUT_TABLE}")


if __name__ == "__main__":
    main()