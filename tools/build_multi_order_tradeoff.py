#!/usr/bin/env python3
"""
Degree trade-off figure at n_seg = 16 (paper 그림 6).

Plots control cost and runtime against the Bezier degree, read from the SAME
committed CSV that fills 표 5.

The previous version disagreed with its own caption. The caption says
"$N=6,7,8$ 차수에 대한 제어 비용(좌)과 계산 시간(우)의 추세" and section 5.3 says
"제어 비용은 차수에 대해 단조 감소하고 계산 시간은 단조 증가한다" -- a trend
against DEGREE. The figure instead drew three curves against n_seg, from a
pre-canonical-SCvx cache, using `cost_true_energy * T * 1e6` where 표 5 reports
mean control acceleration in m/s^2. Reading the CSV fixes the source, the
quantity, and the independent variable at once.

Usage:
    python tools/build_multi_order_tradeoff.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.paper_tables_csv import T5_SIGNATURE, block

# Tableau Colorblind 10, the palette the concept figures use.
C_INK, C_COST, C_TIME, C_GRID = "#333333", "#006BA4", "#C85200", "#CFCFCF"

PANELS = [
    ("ctrl_cost_ms2", "Control cost (m/s$^2$)", "Control cost vs degree", C_COST),
    ("runtime_s",     "Runtime (s)",            "Runtime vs degree",      C_TIME),
]


def main():
    rows = block(T5_SIGNATURE, "그림 6 (degree sweep, n_seg=16)")
    degrees = np.array([r["degree"] for r in rows])
    n_ctrl = np.array([r["n_ctrl"] for r in rows])
    n_seg = rows[0]["n_seg"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)

    for ax, (key, ylabel, title, color) in zip(axes, PANELS):
        y = np.array([r[key] for r in rows])
        ax.plot(degrees, y, "-o", color=color, lw=1.8, ms=8, zorder=3)

        # Three points is few enough that the reader should see the values.
        span = y.max() - y.min()
        for d, v in zip(degrees, y):
            ax.annotate(f"{v:.3f}", xy=(d, v), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=9,
                        color=color)
        ax.set_ylim(y.min() - 0.30 * span, y.max() + 0.55 * span)

        # Each panel auto-zooms to its own range, which makes a 0.4% change look
        # as steep as a 43% one. Section 5.3's claim is precisely that the cost
        # difference is the SMALLER of the two, so state both relative spans.
        rel = 100.0 * (y[-1] - y[0]) / y[0]
        ax.text(0.5, 0.94, f"$N=6 \\rightarrow 8$:  {rel:+.1f}%",
                transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
                color=color)

        ax.set_xticks(degrees)
        ax.set_xticklabels([f"$N={d}$\n({c} ctrl pts)" for d, c in zip(degrees, n_ctrl)])
        ax.set_xlabel("Bézier degree", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=C_INK, pad=8)
        ax.grid(True, axis="y", color=C_GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.tick_params(labelsize=9, colors="#666666")

    fig.suptitle(f"120 deg phase lag, $n_{{\\mathrm{{seg}}}}={n_seg}$",
                 fontsize=10, color="#666666")

    out = ROOT / "figures" / "multi_order_tradeoff_N678.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"rows read from doc/results/paper_tables.csv ({len(rows)}):")
    for r in rows:
        print(f"  N={r['degree']}  n_ctrl={r['n_ctrl']}  n_seg={r['n_seg']}  "
              f"cost={r['ctrl_cost_ms2']:.3f} m/s^2  runtime={r['runtime_s']:.3f} s")
    print(f"\nSaved -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
