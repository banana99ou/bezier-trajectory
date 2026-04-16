"""EXP-20260403-001 BLADE DB 분석 시각화.

생성 그림:
  fig_blade_batch_conv.pdf      - 배치별 수렴률
  fig_blade_l1_profile.pdf      - L1 lambda별 profile_class 분포
  fig_blade_confusion.pdf       - Colloc vs BLADE 혼동행렬
  fig_blade_altitude_conv.pdf   - altitude_grid_v1 h0별 수렴률
"""

import os
import duckdb
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

DB_PATH = os.path.expanduser(
    "~/fmcl-database-student/student-kit/downloads/EXP-20260403-001/merged.duckdb"
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "../manuscript/figures")
OUT_DIR = os.path.normpath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}
CLASS_LABELS = {0: "Class 0\n(coasting)", 1: "Class 1\n(unimodal)", 2: "Class 2\n(bimodal)"}

# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def connect():
    return duckdb.connect(DB_PATH, read_only=True)


# ──────────────────────────────────────────────
# Fig 1: 배치별 수렴률
# ──────────────────────────────────────────────
def plot_batch_convergence(con):
    rows = con.execute("""
        SELECT
            COALESCE(batch_tag, 'baseline/l1') as tag,
            COUNT(*) as n,
            SUM(CASE WHEN converged THEN 1 ELSE 0 END) as n_conv
        FROM blade_simulations
        GROUP BY tag
        ORDER BY tag
    """).fetchall()

    tags = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    convs = [r[2] for r in rows]
    fails = [t - c for t, c in zip(totals, convs)]
    pcts = [100.0 * c / t for c, t in zip(convs, totals)]

    # 짧은 레이블
    short = {
        "altitude_grid_v1": "Altitude grid\n(h₀=300–1200 km)",
        "baseline/l1": "Baseline / L1 sweep\n(λ=0.01–1.0)",
        "ecc_extended_v1": "Eccentric extension\n(e∈[0.1, 0.3])",
        "omega_screen_v1": "ω sensitivity\nscreen",
    }

    x = np.arange(len(tags))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    bars_c = ax.bar(x, convs, label="Converged", color="#4C72B0", zorder=3)
    bars_f = ax.bar(x, fails, bottom=convs, label="Failed", color="#C44E52",
                    alpha=0.6, zorder=3)

    # 수렴률 텍스트
    for i, (c, t, p) in enumerate(zip(convs, totals, pcts)):
        ax.text(i, t + 80, f"{p:.1f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([short.get(t, t) for t in tags], fontsize=8)
    ax.set_ylabel("Number of cases")
    ax.set_title("Convergence rate by batch — BLADE-SCP database (EXP-20260403-001)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(0, max(totals) * 1.18)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_blade_batch_conv.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 2: L1 lambda → profile_class 분포
# ──────────────────────────────────────────────
def plot_l1_profile(con):
    rows = con.execute("""
        SELECT b.l1_lambda, cr.blade_profile_class, COUNT(*) as n
        FROM comparison_runs cr
        JOIN blade_simulations b ON cr.blade_id = b.blade_id
        WHERE cr.blade_converged
        GROUP BY b.l1_lambda, cr.blade_profile_class
        ORDER BY b.l1_lambda, cr.blade_profile_class
    """).fetchall()

    from collections import defaultdict
    by_l = defaultdict(lambda: {0: 0, 1: 0, 2: 0})
    for lam, cls, n in rows:
        by_l[lam][cls] = n

    lambdas = sorted(by_l.keys())
    c0 = [by_l[l][0] for l in lambdas]
    c1 = [by_l[l][1] for l in lambdas]
    c2 = [by_l[l][2] for l in lambdas]
    totals = [a + b + c for a, b, c in zip(c0, c1, c2)]

    # 비율로 변환
    c0_pct = [100 * a / t for a, t in zip(c0, totals)]
    c1_pct = [100 * b / t for b, t in zip(c1, totals)]
    c2_pct = [100 * c / t for c, t in zip(c2, totals)]

    x = np.arange(len(lambdas))
    width = 0.55

    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    b0 = ax.bar(x, c0_pct, width, label="Class 0 (coasting)", color=PALETTE[0], zorder=3)
    b1 = ax.bar(x, c1_pct, width, bottom=c0_pct, label="Class 1 (unimodal)",
                color=PALETTE[1], zorder=3)
    b2 = ax.bar(x, c2_pct, width,
                bottom=[a + b for a, b in zip(c0_pct, c1_pct)],
                label="Class 2 (bimodal)", color=PALETTE[2], zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={l}" for l in lambdas])
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Effect of L1 regularization (λ) on thrust profile classification")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # 건수 표기 — 바 바깥 위
    for i, t in enumerate(totals):
        ax.text(i, 101.5, f"n={t}", ha="center", va="bottom", fontsize=7.5,
                color="black", zorder=4)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_blade_l1_profile.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 3: Colloc vs BLADE 혼동행렬
# ──────────────────────────────────────────────
def plot_confusion(con):
    rows = con.execute("""
        SELECT colloc_profile_class, blade_profile_class, COUNT(*) as n
        FROM comparison_runs
        WHERE colloc_converged AND blade_converged
        GROUP BY colloc_profile_class, blade_profile_class
        ORDER BY colloc_profile_class, blade_profile_class
    """).fetchall()

    classes = [0, 1, 2]
    mat = np.zeros((3, 3), dtype=int)
    for cp, bp, n in rows:
        mat[cp][bp] = n

    # 행 기준 정규화 (recall)
    mat_norm = mat.astype(float) / mat.sum(axis=1, keepdims=True) * 100

    cmap = LinearSegmentedColormap.from_list("blue_white", ["white", "#4C72B0"])

    total = mat.sum()
    match = mat.trace()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), gridspec_kw={"wspace": 0.42})
    fig.suptitle(
        f"Collocation vs. BLADE-SCP profile classification agreement"
        f"  (overall accuracy: {match}/{total} = {100*match/total:.1f}%)",
        fontsize=9, y=1.01,
    )

    for ax, data, title, fmt in zip(
        axes,
        [mat, mat_norm],
        ["(a) Count", "(b) Row-normalized (%)"],
        ["{:d}", "{:.1f}%"],
    ):
        im = ax.imshow(data, cmap=cmap, aspect="equal",
                       vmin=0, vmax=data.max())
        plt.colorbar(im, ax=ax, shrink=0.85)

        for i in range(3):
            for j in range(3):
                val = data[i, j]
                color = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Class 0", "Class 1", "Class 2"])
        ax.set_yticklabels(["Class 0", "Class 1", "Class 2"])
        ax.set_xlabel("BLADE prediction")
        ax.set_ylabel("Collocation (reference)")
        ax.set_title(title, pad=8)

    out = os.path.join(OUT_DIR, "fig_blade_confusion.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ──────────────────────────────────────────────
# Fig 4: altitude_grid_v1 h0별 수렴률
# ──────────────────────────────────────────────
def plot_altitude_conv(con):
    rows = con.execute("""
        SELECT ROUND(dep_a - 6371, 0) as h0_km,
               COUNT(*) as n,
               SUM(CASE WHEN converged THEN 1 ELSE 0 END) as n_conv
        FROM blade_simulations
        WHERE batch_tag = 'altitude_grid_v1'
        GROUP BY h0_km
        ORDER BY h0_km
    """).fetchall()

    h0s = [r[0] for r in rows]
    pcts = [100.0 * r[2] / r[1] for r in rows]
    ns = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 3.2))

    ax.plot(h0s, pcts, "o-", color="#4C72B0", markersize=5, linewidth=1.4, zorder=3)
    ax.fill_between(h0s, pcts, alpha=0.15, color="#4C72B0")

    ax.axhline(np.mean(pcts), color="#C44E52", linestyle="--", linewidth=1,
               label=f"Mean: {np.mean(pcts):.1f}%")

    ax.set_xlabel("Initial altitude $h_0$ [km]")
    ax.set_ylabel("Convergence rate (%)")
    ax.set_title("BLADE-SCP convergence rate vs. initial altitude\n"
                 "(altitude_grid_v1 batch, h₀ = 300–1200 km)")
    ax.set_ylim(75, 95)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100))
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    ax.legend()

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig_blade_altitude_conv.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    con = connect()
    plot_batch_convergence(con)
    plot_l1_profile(con)
    plot_confusion(con)
    plot_altitude_conv(con)
    con.close()
    print("Done.")
