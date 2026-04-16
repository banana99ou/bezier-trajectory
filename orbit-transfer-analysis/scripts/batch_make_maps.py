import sys
sys.path.append("/Users/heewon/Desktop/무제 폴더")

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from orbit_transfer.database.storage import TrajectoryDatabase


def load_slice_data(db_path, npz_dir, h0, target_T, tol_T, mode="class"):
    db = TrajectoryDatabase(
        db_path=db_path,
        npz_dir=npz_dir,
    )

    rows = db.conn.execute(
        """
        SELECT delta_a, delta_i, T_normed, profile_class, converged, e0, ef
        FROM trajectories
        WHERE h0 = ?
        """,
        [h0],
    ).fetchall()

    data = []
    e0_values = []
    ef_values = []

    for r in rows:
        delta_a, delta_i, T_normed, profile_class, converged, e0, ef = r

        if abs(T_normed - target_T) > tol_T:
            continue

        if mode == "class":
            if not converged:
                continue
            label = profile_class
        elif mode == "failure":
            label = 1 if converged else 0
        else:
            raise ValueError("mode must be 'class' or 'failure'")

        data.append((delta_a, delta_i, label))
        e0_values.append(e0)
        ef_values.append(ef)

    db.close()

    return data, sorted(set(e0_values)), sorted(set(ef_values))


def make_map(data, e0_values, ef_values, h0, target_T, tol_T, save_path, title, mode="class"):
    if len(data) < 3:
        print(f"[SKIP] 데이터 부족: {title}")
        return

    data = np.array(data)
    X = data[:, :2]
    y = data[:, 2]

    unique_classes = np.unique(y)

    if mode == "class":
        label_names = {0: "class 0", 1: "class 1", 2: "class 2"}
    else:
        label_names = {0: "fail", 1: "success"}

    # 클래스가 1개뿐이면 SVM 없이 scatter만 저장
    if len(unique_classes) < 2:
        print(f"[SKIP-SVM] 클래스가 1개뿐임: {title} -> classes={unique_classes}")

        plt.figure(figsize=(9, 7))

        for cls in unique_classes:
            mask = y == cls
            plt.scatter(
                X[mask, 0],
                X[mask, 1],
                label=label_names.get(int(cls), f"label {int(cls)}"),
                s=80
            )

        plt.xlabel("delta_a [km]")
        plt.ylabel("delta_i [deg]")
        plt.title(
            f"{title}\n"
            f"h0={h0}, T_normed≈{target_T}±{tol_T}, e0={e0_values}, ef={ef_values}"
        )
        plt.legend()
        plt.grid(True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[SAVE-scatter-only] {save_path}")
        return

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

    for cls in unique_classes:
        mask = y == cls
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            label=label_names.get(int(cls), f"label {int(cls)}"),
            s=80
        )

    plt.xlabel("delta_a [km]")
    plt.ylabel("delta_i [deg]")
    plt.title(
        f"{title}\n"
        f"h0={h0}, T_normed≈{target_T}±{tol_T}, e0={e0_values}, ef={ef_values}"
    )
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {save_path}")


def process_slice(slice_name, db_path, npz_dir, h0, target_T_list, tol_T):
    for target_T in target_T_list:
        # classification map
        data, e0_values, ef_values = load_slice_data(
            db_path=db_path,
            npz_dir=npz_dir,
            h0=h0,
            target_T=target_T,
            tol_T=tol_T,
            mode="class",
        )

        save_path = f"fig/classification_map_{slice_name}_T{target_T}.png"
        title = f"Classification Map ({slice_name})"
        make_map(
            data=data,
            e0_values=e0_values,
            ef_values=ef_values,
            h0=h0,
            target_T=target_T,
            tol_T=tol_T,
            save_path=save_path,
            title=title,
            mode="class",
        )

        # failure map
        data, e0_values, ef_values = load_slice_data(
            db_path=db_path,
            npz_dir=npz_dir,
            h0=h0,
            target_T=target_T,
            tol_T=tol_T,
            mode="failure",
        )

        save_path = f"fig/failure_map_{slice_name}_T{target_T}.png"
        title = f"Failure Map ({slice_name})"
        make_map(
            data=data,
            e0_values=e0_values,
            ef_values=ef_values,
            h0=h0,
            target_T=target_T,
            tol_T=tol_T,
            save_path=save_path,
            title=title,
            mode="failure",
        )


def main():
    h0 = 400.0
    tol_T = 0.5
    target_T_list = [1.0, 2.0, 4.0]

    # circular slice
    process_slice(
        slice_name="circular",
        db_path="data/trajectories_circular.duckdb",
        npz_dir="data/trajectories_circular",
        h0=h0,
        target_T_list=target_T_list,
        tol_T=tol_T,
    )

    # eccentric slice
    process_slice(
        slice_name="eccentric",
        db_path="data/trajectories_eccentric.duckdb",
        npz_dir="data/trajectories_eccentric",
        h0=h0,
        target_T_list=target_T_list,
        tol_T=tol_T,
    )


if __name__ == "__main__":
    main()