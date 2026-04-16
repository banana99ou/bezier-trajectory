import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from orbit_transfer.database.storage import TrajectoryDatabase


def plot_failure_map(db_path, npz_dir, h0, target_T, tol_T, save_path, slice_name):
    db = TrajectoryDatabase(
        db_path=db_path,
        npz_dir=npz_dir,
    )

    rows = db.conn.execute(
        """
        SELECT delta_a, delta_i, T_normed, converged, e0, ef
        FROM trajectories
        WHERE h0 = ?
        """,
        [h0],
    ).fetchall()

    data = []
    e0_values = []
    ef_values = []

    for r in rows:
        delta_a, delta_i, T_normed, converged, e0, ef = r

        if abs(T_normed - target_T) > tol_T:
            continue

        # converged=True -> 1, False -> 0
        label = 1 if converged else 0

        data.append((delta_a, delta_i, label))
        e0_values.append(e0)
        ef_values.append(ef)

    if len(data) < 3:
        print(f"[{slice_name}] 데이터 부족")
        return

    data = np.array(data)
    X = data[:, :2]
    y = data[:, 2]

    # success/failure 2-class 분류기
    clf = SVC(kernel="rbf", gamma="auto")
    clf.fit(X, y)

    a_min, a_max = X[:, 0].min() - 200, X[:, 0].max() + 200
    i_min, i_max = X[:, 1].min() - 2, X[:, 1].max() + 2

    aa, ii = np.meshgrid(
        np.linspace(a_min, a_max, 200),
        np.linspace(i_min, i_max, 200)
    )

    grid = np.c_[aa.ravel(), ii.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(aa.shape)

    plt.figure(figsize=(9, 7))
    plt.contourf(aa, ii, Z, alpha=0.25)

    # 실패/성공 점 표시
    labels = {
        0: "fail",
        1: "success",
    }

    for cls in np.unique(y):
        mask = y == cls
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            label=labels[int(cls)],
            s=80
        )

    e0_unique = sorted(set(e0_values))
    ef_unique = sorted(set(ef_values))

    plt.xlabel("delta_a [km]")
    plt.ylabel("delta_i [deg]")
    plt.title(
        f"Failure Map ({slice_name})\n"
        f"h0={h0}, T_normed≈{target_T}, e0={e0_unique}, ef={ef_unique}"
    )
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[{slice_name}] 저장 완료: {save_path}")

    # 화면에 띄우지 않고 바로 닫기
    plt.close()

    db.close()


def main():
    h0 = 400.0
    target_T = 2.0
    tol_T = 0.5

    # circular slice
    plot_failure_map(
        db_path="data/trajectories_circular.duckdb",
        npz_dir="data/trajectories_circular",
        h0=h0,
        target_T=target_T,
        tol_T=tol_T,
        save_path="fig/failure_map_circular.png",
        slice_name="circular",
    )

    # eccentric slice
    plot_failure_map(
        db_path="data/trajectories_eccentric.duckdb",
        npz_dir="data/trajectories_eccentric",
        h0=h0,
        target_T=target_T,
        tol_T=tol_T,
        save_path="fig/failure_map_eccentric.png",
        slice_name="eccentric",
    )


if __name__ == "__main__":
    main()