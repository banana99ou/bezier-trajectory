import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

if "--save" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

CIRC_DB = ROOT / "data/trajectories_circular.duckdb"
ECC_DB = ROOT / "data/trajectories_eccentric.duckdb"
SWITCH_CSV = ROOT / "results/class_switch_cases.csv"


def load_df(db_path: Path) -> pd.DataFrame:
    conn = duckdb.connect(str(db_path), read_only=True)
    df = conn.execute("SELECT * FROM trajectories ORDER BY id").df()
    conn.close()
    return df


def build_switch_cases() -> pd.DataFrame:
    circ = load_df(CIRC_DB)
    ecc = load_df(ECC_DB)

    keys = ["h0", "delta_a", "delta_i", "T_normed"]

    circ_sub = circ[keys + ["converged", "profile_class", "cost", "n_peaks"]].copy()
    circ_sub = circ_sub.rename(
        columns={
            "converged": "converged_circ",
            "profile_class": "class_circ",
            "cost": "cost_circ",
            "n_peaks": "n_peaks_circ",
        }
    )

    ecc_sub = ecc[keys + ["converged", "profile_class", "cost", "n_peaks"]].copy()
    ecc_sub = ecc_sub.rename(
        columns={
            "converged": "converged_ecc",
            "profile_class": "class_ecc",
            "cost": "cost_ecc",
            "n_peaks": "n_peaks_ecc",
        }
    )

    merged = pd.merge(circ_sub, ecc_sub, on=keys, how="inner")
    merged = merged[
        (merged["converged_circ"] == True) & (merged["converged_ecc"] == True)
    ].copy()
    merged["cost_diff"] = merged["cost_ecc"] - merged["cost_circ"]
    return merged


def select_representative_case(df: pd.DataFrame) -> pd.Series:
    candidates = df[(df["class_circ"] == 2) & (df["class_ecc"] == 0)].copy()
    if len(candidates) == 0:
        raise RuntimeError("No class 2 -> 0 cases found.")

    candidates["peak_gap"] = candidates["n_peaks_circ"] - candidates["n_peaks_ecc"]
    candidates["abs_cost_diff"] = candidates["cost_diff"].abs()

    candidates = candidates.sort_values(
        by=["peak_gap", "abs_cost_diff", "T_normed"],
        ascending=[False, False, True],
    )
    return candidates.iloc[0]


def fetch_row(
    db_path: Path,
    h0: float,
    delta_a: float,
    delta_i: float,
    T_normed: float,
    expected_class: int | None = None,
    tol: float = 1e-6,
):
    conn = duckdb.connect(str(db_path), read_only=True)
    rows = conn.execute(
        """
        SELECT id, trajectory_file, profile_class, n_peaks, cost, converged
        FROM trajectories
        WHERE ABS(h0 - ?) < ?
          AND ABS(delta_a - ?) < ?
          AND ABS(delta_i - ?) < ?
          AND ABS(T_normed - ?) < ?
        ORDER BY id
        """,
        [h0, tol, delta_a, tol, delta_i, tol, T_normed, tol],
    ).fetchall()
    conn.close()

    if expected_class is not None:
        rows = [row for row in rows if row[2] == expected_class]

    if len(rows) == 0:
        raise RuntimeError("No matching row found in DB.")

    row = rows[0]
    return {
        "id": row[0],
        "trajectory_file": row[1],
        "profile_class": row[2],
        "n_peaks": row[3],
        "cost": row[4],
        "converged": row[5],
    }


def load_trajectory(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["t"], data["u"]


def ensure_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        return ROOT / path
    return path


def plot_comparison(
    t_circ: np.ndarray,
    u_circ: np.ndarray,
    t_ecc: np.ndarray,
    u_ecc: np.ndarray,
    case_info: str,
    save_path: Path | None = None,
):
    t_circ_hr = t_circ / 3600.0
    t_ecc_hr = t_ecc / 3600.0

    u_circ_mag = np.linalg.norm(u_circ, axis=0)
    u_ecc_mag = np.linalg.norm(u_ecc, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.flatten()

    axes[0].plot(t_circ_hr, u_circ_mag, label="circular", lw=2)
    axes[0].plot(t_ecc_hr, u_ecc_mag, label="eccentric", lw=2)
    axes[0].set_title("||u(t)||")
    axes[0].set_xlabel("Time [hr]")
    axes[0].set_ylabel("Thrust [km/s^2]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    labels = ["u_x", "u_y", "u_z"]
    for i, label in enumerate(labels, start=1):
        axes[i].plot(t_circ_hr, u_circ[i - 1], label="circular", lw=2)
        axes[i].plot(t_ecc_hr, u_ecc[i - 1], label="eccentric", lw=2)
        axes[i].set_title(label)
        axes[i].set_xlabel("Time [hr]")
        axes[i].set_ylabel("Thrust [km/s^2]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    fig.suptitle(case_info, fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def main():
    if SWITCH_CSV.exists():
        cases = pd.read_csv(SWITCH_CSV)
    else:
        cases = build_switch_cases()

    case = select_representative_case(cases)

    h0 = float(case["h0"])
    delta_a = float(case["delta_a"])
    delta_i = float(case["delta_i"])
    T_normed = float(case["T_normed"])

    circ_row = fetch_row(
        CIRC_DB,
        h0=h0,
        delta_a=delta_a,
        delta_i=delta_i,
        T_normed=T_normed,
        expected_class=2,
    )
    ecc_row = fetch_row(
        ECC_DB,
        h0=h0,
        delta_a=delta_a,
        delta_i=delta_i,
        T_normed=T_normed,
        expected_class=0,
    )

    circ_path = ensure_path(circ_row["trajectory_file"])
    ecc_path = ensure_path(ecc_row["trajectory_file"])

    if not circ_path.exists():
        raise RuntimeError(f"Missing circular npz: {circ_path}")
    if not ecc_path.exists():
        raise RuntimeError(f"Missing eccentric npz: {ecc_path}")

    t_circ, u_circ = load_trajectory(circ_path)
    t_ecc, u_ecc = load_trajectory(ecc_path)

    case_info = (
        "class 2 -> 0 representative case | "
        f"h0={h0:.1f} km, delta_a={delta_a:.6f} km, "
        f"delta_i={delta_i:.6f} deg, T_normed={T_normed:.6f}"
    )

    save_path = None
    if "--save" in sys.argv:
        save_path = ROOT / "fig" / "compare_ut_class2_to_0.png"

    plot_comparison(t_circ, u_circ, t_ecc, u_ecc, case_info, save_path=save_path)


if __name__ == "__main__":
    main()
