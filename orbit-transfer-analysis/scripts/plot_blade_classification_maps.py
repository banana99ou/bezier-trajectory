"""BLADE classification map 생성 (comparison_runs 기반).

기존 fig3~fig5를 BLADE profile_class로 재생성:
  fig3_blade_classification_da_di.pdf   — Δa vs Δi  (4 h0 슬라이스)
  fig4_blade_classification_da_T.pdf    — Δa vs T_max_normed
  fig5_blade_classification_di_T.pdf    — Δi vs T_max_normed
"""

import os
import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH = os.path.expanduser(
    "~/fmcl-database-student/student-kit/downloads/EXP-20260403-001/merged.duckdb"
)
OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../manuscript/figures")
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── 스타일 (기존 그림과 동일) ──────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

CLASS_STYLE = {
    0: dict(label="Unimodal",   color="#4C72B0", zorder=4, s=18, alpha=0.85),
    1: dict(label="Bimodal",    color="#DD8452", zorder=3, s=18, alpha=0.75),
    2: dict(label="Multimodal", color="#C44E52", zorder=2, s=18, alpha=0.75),
}
H0_LIST = [400, 600, 800, 1000]


def load_data(con):
    rows = con.execute("""
        SELECT h0, delta_a, delta_i, T_max_normed,
               blade_profile_class,
               colloc_profile_class
        FROM comparison_runs
        WHERE blade_converged
        ORDER BY h0
    """).fetchall()
    return rows


def make_fig(rows, x_col, y_col, xlabel, ylabel,
             xlim, ylim, fname):
    """x_col, y_col: 인덱스 (1=delta_a, 2=delta_i, 3=T_max_normed, 4=blade_class)"""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, h0 in zip(axes, H0_LIST):
        subset = [r for r in rows if abs(r[0] - h0) < 1]

        for cls, style in CLASS_STYLE.items():
            xs = [r[x_col] for r in subset if r[4] == cls]
            ys = [r[y_col] for r in subset if r[4] == cls]
            ax.scatter(xs, ys, **style)

        ax.set_title(f"$h_0 = {h0}$ km")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", markerscale=1.3)

    # 공통 축 레이블
    for ax in axes[2:]:
        ax.set_xlabel(xlabel)
    for ax in [axes[0], axes[2]]:
        ax.set_ylabel(ylabel)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    rows = load_data(con)
    con.close()

    # Fig 3: Δa vs Δi
    make_fig(rows,
             x_col=1, y_col=2,
             xlabel=r"$\Delta a$ [km]",
             ylabel=r"$\Delta i$ [deg]",
             xlim=(-600, 2100), ylim=(-0.5, 15.5),
             fname="fig3_blade_classification_da_di.pdf")

    # Fig 4: Δa vs T_max_normed
    make_fig(rows,
             x_col=1, y_col=3,
             xlabel=r"$\Delta a$ [km]",
             ylabel=r"$T_\mathrm{max}/T_0$",
             xlim=(-600, 2100), ylim=(0.05, 1.30),
             fname="fig4_blade_classification_da_T.pdf")

    # Fig 5: Δi vs T_max_normed
    make_fig(rows,
             x_col=2, y_col=3,
             xlabel=r"$\Delta i$ [deg]",
             ylabel=r"$T_\mathrm{max}/T_0$",
             xlim=(-0.5, 15.5), ylim=(0.05, 1.30),
             fname="fig5_blade_classification_di_T.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
