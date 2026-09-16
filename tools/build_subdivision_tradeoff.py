#!/usr/bin/env python3
"""Build Figure 5 from the committed measurements used in Table 3.

Compare measured and estimated minimum clearances, control cost, and runtime for
n_seg = 8, 16, 32, 64 at N = 7. These configurations are certified and attain
their closest approach to the obstacle between the endpoints. Table 3 retains
the complete sweep, including the uncertified and endpoint cases at n_seg = 2
and 4.

Usage:
    python tools/build_subdivision_tradeoff.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, ScalarFormatter

from tools.paper_tables_csv import T3_SIGNATURE, block

C_INK, C_DATA, C_PRED, C_GRID = "#333333", "#006BA4", "#444444", "#DDDDDD"
PLOT_SEGS = (8, 16, 32, 64)
PANELS = (
    ("margin_km", "Minimum clearance (km)", "(a) Minimum clearance", ".2f"),
    ("ctrl_cost_ms2", "Control cost (m/s$^2$)", "(b) Control cost", ".3f"),
    ("runtime_s", "Runtime (s)", "(c) Runtime", ".3f"),
)
# Keep labels away from the line approaching or leaving each measurement.
LABEL_OFFSETS = {
    ("margin_km", 16): (9, 7, "left"),
    ("margin_km", 32): (9, 7, "left"),
    ("ctrl_cost_ms2", 16): (9, 7, "left"),
    ("runtime_s", 32): (-9, 8, "right"),
}


def main():
    all_rows = block(T3_SIGNATURE, "그림 5 (subdivision sweep, N=7)")
    rows = [r for r in all_rows if r["n_seg"] in PLOT_SEGS]
    if tuple(r["n_seg"] for r in rows) != PLOT_SEGS:
        raise ValueError(f"Figure 5 requires subdivision counts {PLOT_SEGS}")
    if not all(r["certified"] and 0.0 < float(r["tau_star"]) < 1.0
               for r in rows):
        raise ValueError("Figure 5 requires certified runs with interior minima")

    segs = np.array([r["n_seg"] for r in rows])
    predicted = np.array([float(r["pred_margin_km"]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    for ax, (key, ylabel, title, fmt) in zip(axes, PANELS):
        y = np.array([r[key] for r in rows])
        ax.plot(segs, y, color=C_DATA, linewidth=2.0, marker="o",
                markersize=5.5, label="Measured", zorder=3)

        if key == "margin_km":
            # Different line styles preserve the comparison in grayscale.
            ax.plot(segs, predicted, color=C_PRED, linewidth=1.4,
                    linestyle=(0, (5, 3)), label="Estimated", zorder=4)
            ax.set_yscale("log")
            ax.set_yticks([1, 4, 16, 64])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylim(0.65, 115)
            ax.legend(loc="upper right", frameon=False, fontsize=10,
                      handlelength=2.8)
        elif key == "ctrl_cost_ms2":
            ax.set_ylim(4.54, 4.94)
            ax.set_yticks([4.6, 4.7, 4.8, 4.9])
        else:
            ax.set_ylim(0.075, 0.245)
            ax.set_yticks([0.08, 0.12, 0.16, 0.20, 0.24])

        for xs, ys in zip(segs, y):
            dx, dy, align = LABEL_OFFSETS.get((key, int(xs)), (0, 9, "center"))
            ax.annotate(format(ys, fmt), (xs, ys), textcoords="offset points",
                        xytext=(dx, dy), ha=align, va="bottom", fontsize=9,
                        color=C_INK, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xticks(segs)
        ax.set_xticklabels([str(int(s)) for s in segs])
        ax.set_xlim(segs[0] / 1.3, segs[-1] * 1.3)
        ax.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, color=C_INK, pad=10)
        ax.grid(True, which="major", color=C_GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#999999")
        ax.tick_params(labelsize=10, colors=C_INK)

    out = ROOT / "figures" / "subdivision_tradeoff_N7.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figure 5: {len(rows)} of {len(all_rows)} Table 3 configurations")
    for r in rows:
        print(f"  n_seg={r['n_seg']:2d}  measured={r['margin_km']:.2f} km  "
              f"estimated={float(r['pred_margin_km']):.2f} km  "
              f"cost={r['ctrl_cost_ms2']:.3f} m/s^2  "
              f"runtime={r['runtime_s']:.3f} s")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
