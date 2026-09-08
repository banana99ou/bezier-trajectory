#!/usr/bin/env python3
"""
Subdivision-count trade-off figure for N = 7 (paper 그림 5).

Plots the three quantities the caption names -- safety margin, control cost and
runtime -- against the subdivision count, read from the SAME committed CSV that
fills 표 4. Reading `doc/results/paper_tables.csv` rather than the solver cache is
the point of this script: the previous version loaded `cache/opt_*.pkl` written
before the canonical-SCvx change, so it plotted runtimes three orders of
magnitude away from the table printed beside it, and it plotted
`cost_true_energy * T * 1e6` labelled as an energy integral where 표 4 reports
mean control acceleration in m/s^2. Both are fixed by reading the CSV.

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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from tools.paper_tables_csv import T4_SIGNATURE, block, endpoint_attained

# Tableau Colorblind 10, the palette the concept figures use.
C_INK, C_DATA, C_FLAG, C_GRID = "#333333", "#006BA4", "#C85200", "#CFCFCF"

# Light-to-dark ramp anchored on C_DATA: the line's own colour carries the
# direction of the sweep, so the trend reads before the axis labels do.
CMAP = LinearSegmentedColormap.from_list(
    "seg", ["#A6CEE3", "#4B97C6", C_DATA, "#003F62"])

# Hand-placed label offsets in points, one per n_seg, per panel. Automatic
# placement put every label on the line; these keep each clear of it.
LABEL_OFFSETS = {
    "margin_km":     [(2, 13), (2, 13), (14, 7), (14, 7), (14, 7), (2, 13)],
    "ctrl_cost_ms2": [(2, 13), (14, 6), (0, -17), (0, -17), (0, -17), (0, -17)],
    "runtime_s":     [(2, 13), (-2, -17), (0, -17), (-14, -6), (-14, 2), (2, 13)],
}

PANELS = [
    ("margin_km",     "Safety margin (km)",     "Safety margin", True),
    ("ctrl_cost_ms2", "Control cost (m/s$^2$)", "Control cost",  True),
    ("runtime_s",     "Runtime (s)",            "Runtime",       True),
]


def main():
    rows = block(T4_SIGNATURE, "그림 5 (subdivision sweep, N=7)")
    segs = np.array([r["n_seg"] for r in rows])
    certified = np.array([r["certified"] for r in rows])
    endpoint = np.array(endpoint_attained(rows))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)

    for ax, (key, ylabel, title, logy) in zip(axes, PANELS):
        y = np.array([r[key] for r in rows])
        # Gradient polyline: each segment coloured by where it sits in the sweep.
        pts = np.column_stack([segs, y]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1),
                            cmap=CMAP, norm=Normalize(0, len(segs) - 2),
                            linewidths=2.4, zorder=3)
        lc.set_array(np.arange(len(segs) - 1))
        ax.add_collection(lc)
        # Certified runs are filled; the uncertified one is hollow, so the
        # reader can see at a glance which points the guarantee covers.
        ax.scatter(segs[certified], y[certified], c=np.arange(len(segs))[certified],
                   cmap=CMAP, norm=Normalize(0, len(segs) - 1), s=52, zorder=4)
        ax.plot(segs[~certified], y[~certified], "o", mfc="white", mec=C_FLAG,
                mew=1.8, ms=8, zorder=5)

        # Print each measured value beside its marker: the log axes make the
        # trend legible but not the magnitude, and the text quotes these numbers.
        fmt = {"margin_km": "{:.2f}", "ctrl_cost_ms2": "{:.3f}",
               "runtime_s": "{:.3f}"}[key]
        for xs, ys, off in zip(segs, y, LABEL_OFFSETS[key]):
            ax.annotate(fmt.format(ys), (xs, ys), textcoords="offset points",
                        xytext=off, ha="center", fontsize=8, color=C_INK,
                        zorder=8)

        ax.set_xscale("log", base=2)
        ax.set_xticks(segs)
        ax.set_xticklabels([str(int(s)) for s in segs])
        ax.set_xmargin(0.12)
        ax.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=C_INK, pad=8)
        if logy:
            ax.set_yscale("log")
            ax.set_ymargin(0.18)
        ax.grid(True, which="both", color=C_GRID, lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.tick_params(labelsize=9, colors="#666666")

        if key == "margin_km" and endpoint.any():
            # Where the minimum radius sits at an endpoint the margin measures
            # the departure orbit, not the method -- see section 5.2.
            ax.plot(segs[endpoint], y[endpoint], "x", color=C_FLAG, ms=9,
                    mew=1.8, zorder=6)
            ax.text(0.04, 0.10,
                    "\u00d7  minimum at an endpoint:\n"
                    "    departure altitude, not clearance",
                    transform=ax.transAxes, fontsize=8, color=C_FLAG,
                    ha="left", va="center")

    n_unc = int((~certified).sum())
    if n_unc:
        bad = ", ".join(f"$n_{{\\mathrm{{seg}}}}={int(s)}$" for s in segs[~certified])
        fig.text(0.005, -0.02,
                 f"hollow marker: Proposition 1 certificate not attained ({bad})",
                 fontsize=8.5, color=C_FLAG, ha="left", va="top")

    out = ROOT / "figures" / "subdivision_tradeoff_N7.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"rows read from doc/results/paper_tables.csv ({len(rows)}):")
    for r in rows:
        print(f"  n_seg={r['n_seg']:2d}  certified={str(r['certified']):5s}  "
              f"margin={r['margin_km']:8.2f} km  cost={r['ctrl_cost_ms2']:8.3f} "
              f"m/s^2  runtime={r['runtime_s']:.3f} s")
    print(f"\nSaved -> {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
