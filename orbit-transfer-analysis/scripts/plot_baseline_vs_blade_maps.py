"""Baseline colloc vs BLADE classification map 비교 그림 생성.

fig_compare_da_di.pdf   — Δa vs Δi  (baseline colloc | BLADE λ=0.01)
fig_compare_da_T.pdf    — Δa vs T   (baseline colloc | BLADE λ=0.01)
fig_compare_di_T.pdf    — Δi vs T   (baseline colloc | BLADE λ=0.01)
"""

import os, sys
import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLADE_DB = os.path.expanduser(
    "~/fmcl-database-student/student-kit/downloads/EXP-20260403-001/merged.duckdb"
)
BASE_DB  = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../data/colloc_baseline.duckdb")
)
OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../manuscript/figures")
)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5,
})

CLASS_STYLE = {
    0: dict(label="Unimodal",   color="#4C72B0", zorder=4, s=14, alpha=0.85),
    1: dict(label="Bimodal",    color="#DD8452", zorder=3, s=14, alpha=0.75),
    2: dict(label="Multimodal", color="#C44E52", zorder=2, s=14, alpha=0.75),
}
H0_LIST = [400, 600, 800, 1000]


def load_data(lam: float = 0.01):
    con = duckdb.connect(BLADE_DB, read_only=True)
    con.execute(f"ATTACH '{BASE_DB}' AS base (READ_ONLY)")

    rows = con.execute(f"""
        SELECT cr.h0, cr.delta_a, cr.delta_i, cr.T_max_normed,
               cb.profile_class  AS base_class,
               cr.blade_profile_class AS blade_class
        FROM comparison_runs cr
        JOIN blade_simulations b ON cr.blade_id = b.blade_id
        JOIN base.colloc_baseline cb
            ON ROUND(cr.h0,1)          = ROUND(cb.h0,1)
            AND ROUND(cr.delta_a,3)    = ROUND(cb.delta_a,3)
            AND ROUND(cr.delta_i,3)    = ROUND(cb.delta_i,3)
            AND ROUND(cr.T_max_normed,3) = ROUND(cb.T_max_normed,3)
            AND ROUND(cr.e0,4)         = ROUND(cb.e0,4)
            AND ROUND(cr.ef,4)         = ROUND(cb.ef,4)
        WHERE cr.blade_converged AND cb.converged
          AND b.l1_lambda = {lam}
        ORDER BY cr.h0
    """).fetchall()
    con.close()
    return rows


def make_compare_fig(rows, x_idx, y_idx, xlabel, ylabel, xlim, ylim, fname, lam):
    """좌: baseline colloc, 우: BLADE(λ=lam) — 4 h0 × 2열 = 8 subplot"""
    fig, axes = plt.subplots(4, 2, figsize=(9, 12), sharex=True, sharey=True)

    for row_i, h0 in enumerate(H0_LIST):
        sub = [r for r in rows if abs(r[0] - h0) < 1]

        for col_i, (cls_idx, title_sfx) in enumerate([(4, "Collocation (λ=0)"),
                                                        (5, f"BLADE (λ={lam})")]):
            ax = axes[row_i][col_i]

            for cls, style in CLASS_STYLE.items():
                xs = [r[x_idx] for r in sub if r[cls_idx] == cls]
                ys = [r[y_idx] for r in sub if r[cls_idx] == cls]
                ax.scatter(xs, ys, **style)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="upper right", markerscale=1.2)

            if row_i == 0:
                ax.set_title(title_sfx, fontweight="bold", pad=6)

            # h0 레이블 (왼쪽 열만)
            if col_i == 0:
                ax.set_ylabel(f"$h_0={h0}$ km\n{ylabel}")
            if row_i == 3:
                ax.set_xlabel(xlabel)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def main():
    rows = load_data(lam=0.01)
    print(f"데이터: {len(rows)}건")

    make_compare_fig(rows,
        x_idx=1, y_idx=2,
        xlabel=r"$\Delta a$ [km]", ylabel=r"$\Delta i$ [deg]",
        xlim=(-600, 2100), ylim=(-0.5, 15.5),
        fname="fig_compare_da_di.pdf", lam=0.01)

    make_compare_fig(rows,
        x_idx=1, y_idx=3,
        xlabel=r"$\Delta a$ [km]", ylabel=r"$T_\mathrm{max}/T_0$",
        xlim=(-600, 2100), ylim=(0.05, 1.30),
        fname="fig_compare_da_T.pdf", lam=0.01)

    make_compare_fig(rows,
        x_idx=2, y_idx=3,
        xlabel=r"$\Delta i$ [deg]", ylabel=r"$T_\mathrm{max}/T_0$",
        xlim=(-0.5, 15.5), ylim=(0.05, 1.30),
        fname="fig_compare_di_T.pdf", lam=0.01)

    print("Done.")


if __name__ == "__main__":
    main()
