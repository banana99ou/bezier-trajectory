#!/usr/bin/env python3
"""
Subdivision-count trade-off figure for N = 7 (paper 그림 5).

Plots the three quantities the caption names -- safety margin, control cost and
runtime -- against the subdivision count, read from the SAME committed CSV that
fills 표 3. Reading `doc/results/paper_tables.csv` rather than the solver cache is
the point of this script: the previous version loaded `cache/opt_*.pkl` written
before the canonical-SCvx change, so it plotted runtimes three orders of
magnitude away from the table printed beside it, and it plotted
`cost_true_energy * T * 1e6` labelled as an energy integral where 표 3 reports
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

from tools.paper_tables_csv import T3_SIGNATURE, block, endpoint_attained

# Tableau Colorblind 10, the palette the concept figures use.
C_INK, C_DATA, C_FLAG, C_GRID = "#333333", "#006BA4", "#C85200", "#CFCFCF"

PANELS = [
    ("margin_km",     "Safety margin (km)",     "Safety margin", True),
    ("ctrl_cost_ms2", "Control cost (m/s$^2$)", "Control cost",  True),
    ("runtime_s",     "Runtime (s)",            "Runtime",       True),
]


def main():
    rows = block(T3_SIGNATURE, "그림 5 (subdivision sweep, N=7)")
    segs = np.array([r["n_seg"] for r in rows])
    certified = np.array([r["certified"] for r in rows])
    endpoint = np.array(endpoint_attained(rows))

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)

    for ax, (key, ylabel, title, logy) in zip(axes, PANELS):
        y = np.array([r[key] for r in rows])
        ax.plot(segs, y, "-", color=C_DATA, lw=1.8, zorder=3)
        # Certified runs are filled; the uncertified one is hollow, so the
        # reader can see at a glance which points the guarantee covers.
        ax.plot(segs[certified], y[certified], "o", color=C_DATA, ms=7, zorder=4)
        ax.plot(segs[~certified], y[~certified], "o", mfc="white", mec=C_FLAG,
                mew=1.8, ms=8, zorder=5)

        ax.set_xscale("log", base=2)
        ax.set_xticks(segs)
        ax.set_xticklabels([str(int(s)) for s in segs])
        ax.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=C_INK, pad=8)
        if logy:
            ax.set_yscale("log")
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
            ax.text(0.04, 0.72,
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
