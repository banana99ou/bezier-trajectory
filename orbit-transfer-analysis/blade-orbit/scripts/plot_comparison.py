"""Collocation vs BLADE 비교 시각화.

scripts/results/simulations.duckdb의 comparison_runs 테이블을 읽어
비교 차트를 생성한다.

사용법:
    PYTHONPATH=src python scripts/plot_comparison.py
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import duckdb

rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

DB_PATH = "scripts/results/simulations.duckdb"
OUTDIR = "scripts/results/figures"

import os
os.makedirs(OUTDIR, exist_ok=True)


def load_data(config_name: str | None = None):
    con = duckdb.connect(DB_PATH, read_only=True)
    if config_name:
        rows = con.execute("""
            SELECT c.* FROM comparison_runs c
            JOIN run_configs r ON c.config_id = r.config_id
            WHERE r.name = ?
            ORDER BY c.cmp_id
        """, [config_name]).fetchall()
    else:
        rows = con.execute("""
            SELECT * FROM comparison_runs ORDER BY cmp_id
        """).fetchall()
    cols = [d[0] for d in con.description]
    con.close()
    return [dict(zip(cols, r)) for r in rows]


def plot_cost_comparison(data):
    """Fig 1: Collocation cost vs BLADE cost (scatter)."""
    colloc_cost = np.array([d["colloc_cost"] for d in data if d["colloc_cost"] is not None])
    blade_cost = np.array([d["blade_cost"] for d in data if d["colloc_cost"] is not None])

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(colloc_cost, blade_cost, c="steelblue", s=50, alpha=0.8, edgecolors="navy", lw=0.5)

    # 1:1 line
    lo = min(colloc_cost.min(), blade_cost.min()) * 0.5
    hi = max(colloc_cost.max(), blade_cost.max()) * 2.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Collocation Cost (L²)")
    ax.set_ylabel("BLADE Cost (L²)")
    ax.set_title("Cost Comparison: Collocation vs BLADE-SCP")
    ax.legend()
    ax.set_aspect("equal")

    # annotation: how many BLADE < colloc
    n_better = np.sum(blade_cost < colloc_cost)
    n_total = len(colloc_cost)
    ax.text(0.05, 0.95, f"BLADE < Colloc: {n_better}/{n_total}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/cost_comparison.png")
    fig.savefig(f"{OUTDIR}/cost_comparison.pdf")
    print(f"Saved: cost_comparison")
    plt.close(fig)


def plot_cost_ratio_vs_params(data):
    """Fig 2: Cost ratio (BLADE/Colloc) vs 5D parameters."""
    ratios = []
    params = {"delta_a": [], "delta_i": [], "T_max_normed": [], "e0": [], "ef": []}
    for d in data:
        if d["colloc_cost"] and d["colloc_cost"] > 0 and d["blade_cost"] is not None:
            r = d["blade_cost"] / d["colloc_cost"]
            ratios.append(r)
            for k in params:
                params[k].append(d[k])

    ratios = np.array(ratios)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    labels = {
        "delta_a": "Δa [km]",
        "delta_i": "Δi [deg]",
        "T_max_normed": "T_max / T₀",
        "e0": "e₀",
        "ef": "e_f",
    }

    for ax, (key, vals) in zip(axes, params.items()):
        vals = np.array(vals)
        colors = np.where(ratios < 1.0, "green", "tomato")
        ax.scatter(vals, ratios, c=colors, s=40, alpha=0.7, edgecolors="gray", lw=0.3)
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel(labels[key])
        ax.set_ylabel("Cost Ratio (BLADE / Colloc)")
        ax.set_yscale("log")

    fig.suptitle("Cost Ratio vs Configuration Parameters (green: BLADE better)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUTDIR}/cost_ratio_vs_params.png")
    fig.savefig(f"{OUTDIR}/cost_ratio_vs_params.pdf")
    print(f"Saved: cost_ratio_vs_params")
    plt.close(fig)


def plot_profile_class_agreement(data):
    """Fig 3: Profile class agreement matrix."""
    # confusion matrix: colloc class vs blade class
    classes = [0, 1, 2]
    labels_map = {0: "Unimodal", 1: "Bimodal", 2: "Multimodal"}
    matrix = np.zeros((3, 3), dtype=int)

    for d in data:
        cc = d["colloc_profile_class"]
        bc = d["blade_profile_class"]
        if cc is not None and bc is not None and cc in classes and bc in classes:
            matrix[cc][bc] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > matrix.max() * 0.5 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([labels_map[c] for c in classes])
    ax.set_yticklabels([labels_map[c] for c in classes])
    ax.set_xlabel("BLADE Profile Class")
    ax.set_ylabel("Collocation Profile Class")
    ax.set_title("Profile Classification Agreement")

    total = matrix.sum()
    agree = np.trace(matrix)
    ax.text(0.95, 0.05, f"Agreement: {agree}/{total} ({100*agree/total:.0f}%)",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/profile_class_agreement.png")
    fig.savefig(f"{OUTDIR}/profile_class_agreement.pdf")
    print(f"Saved: profile_class_agreement")
    plt.close(fig)


def plot_solve_time_comparison(data):
    """Fig 4: Solve time comparison."""
    colloc_t = []
    blade_t = []
    for d in data:
        if d["colloc_converged"] and d["blade_converged"]:
            ct = d.get("colloc_solve_time")
            bt = d.get("blade_solve_time")
            if ct and bt and ct > 0 and bt > 0:
                colloc_t.append(ct)
                blade_t.append(bt)

    if not colloc_t:
        print("No solve time data available")
        return

    colloc_t = np.array(colloc_t)
    blade_t = np.array(blade_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Scatter
    ax1.scatter(colloc_t, blade_t, c="steelblue", s=50, alpha=0.8, edgecolors="navy", lw=0.5)
    lo = min(colloc_t.min(), blade_t.min()) * 0.5
    hi = max(colloc_t.max(), blade_t.max()) * 2.0
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Collocation Solve Time [s]")
    ax1.set_ylabel("BLADE Solve Time [s]")
    ax1.set_title("Solve Time Comparison")
    ax1.legend()

    # Speedup histogram
    speedup = colloc_t / blade_t
    ax2.hist(speedup, bins=15, color="steelblue", edgecolor="navy", alpha=0.8)
    ax2.axvline(np.median(speedup), color="red", ls="--", lw=1.5,
                label=f"Median: {np.median(speedup):.1f}×")
    ax2.set_xlabel("Speedup (Collocation / BLADE)")
    ax2.set_ylabel("Count")
    ax2.set_title("BLADE Speedup Distribution")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/solve_time_comparison.png")
    fig.savefig(f"{OUTDIR}/solve_time_comparison.pdf")
    print(f"Saved: solve_time_comparison")
    plt.close(fig)


def plot_summary_table(data):
    """Fig 5: Summary statistics as text figure."""
    n_total = len(data)
    n_both_conv = sum(1 for d in data if d["colloc_converged"] and d["blade_converged"])
    n_blade_only = sum(1 for d in data if not d["colloc_converged"] and d["blade_converged"])
    n_colloc_only = sum(1 for d in data if d["colloc_converged"] and not d["blade_converged"])
    n_neither = sum(1 for d in data if not d["colloc_converged"] and not d["blade_converged"])

    # Cost comparison (both converged)
    ratios = []
    for d in data:
        if d["colloc_converged"] and d["blade_converged"] and d["colloc_cost"] and d["colloc_cost"] > 0:
            ratios.append(d["blade_cost"] / d["colloc_cost"])
    ratios = np.array(ratios) if ratios else np.array([])

    n_better = np.sum(ratios < 1.0) if len(ratios) > 0 else 0

    # BC violation
    bc_viols = [d["blade_bc_violation"] for d in data if d["blade_converged"]]
    bc_viols = np.array(bc_viols) if bc_viols else np.array([])

    # Solve times
    blade_times = [d["blade_solve_time"] for d in data if d["blade_converged"] and d["blade_solve_time"]]
    blade_times = np.array(blade_times) if blade_times else np.array([])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    lines = [
        f"BLADE vs Collocation — Summary (n={n_total})",
        f"{'─' * 50}",
        f"",
        f"Convergence:",
        f"  Both converged:       {n_both_conv:4d}",
        f"  BLADE only:           {n_blade_only:4d}",
        f"  Collocation only:     {n_colloc_only:4d}",
        f"  Neither:              {n_neither:4d}",
        f"",
        f"Cost (both converged, n={len(ratios)}):",
        f"  BLADE < Colloc:       {n_better}/{len(ratios)}  ({100*n_better/max(len(ratios),1):.0f}%)",
        f"  Median ratio:         {np.median(ratios):.3f}" if len(ratios) > 0 else "  Median ratio:         N/A",
        f"  Min / Max ratio:      {ratios.min():.3f} / {ratios.max():.3f}" if len(ratios) > 0 else "",
        f"",
        f"BLADE BC Violation (n={len(bc_viols)}):",
        f"  Mean:                 {bc_viols.mean():.2e}" if len(bc_viols) > 0 else "",
        f"  Max:                  {bc_viols.max():.2e}" if len(bc_viols) > 0 else "",
        f"",
        f"BLADE Solve Time (n={len(blade_times)}):",
        f"  Mean:                 {blade_times.mean():.2f} s" if len(blade_times) > 0 else "",
        f"  Median:               {np.median(blade_times):.2f} s" if len(blade_times) > 0 else "",
    ]

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top",
            fontsize=11, fontfamily="monospace",
            bbox=dict(boxstyle="round", fc="ivory", ec="gray", alpha=0.9))

    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/summary_table.png")
    fig.savefig(f"{OUTDIR}/summary_table.pdf")
    print(f"Saved: summary_table")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="특정 설정 이름으로 필터링")
    args = parser.parse_args()

    data = load_data(args.config)
    if not data:
        print("No comparison data found.")
        return

    label = f" (config={args.config})" if args.config else ""
    print(f"Loaded {len(data)} comparison records{label}")
    print(f"Output: {OUTDIR}/\n")

    plot_cost_comparison(data)
    plot_cost_ratio_vs_params(data)
    plot_profile_class_agreement(data)
    plot_solve_time_comparison(data)
    plot_summary_table(data)

    print(f"\nAll figures saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
