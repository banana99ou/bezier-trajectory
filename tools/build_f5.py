#!/usr/bin/env python3
"""
Build F5: Multi-order performance trends for N=6,7,8.

Two-panel figure showing:
  Left:  Delta-v proxy vs subdivision count, stratified by degree
  Right: Runtime vs subdivision count, stratified by degree

Uses the 120-deg phase-lag cache files (same as build_f3.py / build_csv.py).

Usage:
    python tools/build_f5.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orbital_docking import constants, generate_initial_control_points
from orbital_docking.cache import get_cache_key, get_cache_path, load_from_cache

# ── Reuse 120-deg geometry from build_csv.py ─────────────────────────
from build_csv import get_120deg_endpoints, get_cache_path_120

DEGREES = [6, 7, 8]
SEGS = [2, 4, 8, 16, 32, 64]
COLORS = {6: "tab:blue", 7: "tab:red", 8: "tab:green"}
MARKERS = {6: "o", 7: "s", 8: "^"}


def main():
    P_start, v0, P_end, v1 = get_120deg_endpoints()

    data = {}  # data[N] = {"segs": [], "dvs": [], "runtimes": [], "feasible": []}
    for N in DEGREES:
        segs, dvs, rts, feas = [], [], [], []
        for n_seg in SEGS:
            path = get_cache_path_120(N, n_seg, P_start, v0, P_end, v1)
            result = load_from_cache(path)
            if result is None:
                continue
            _, info = result
            segs.append(n_seg)
            dvs.append(info["dv_proxy_m_s"])
            rts.append(info["elapsed_time"])
            feas.append(info["feasible"])
        data[N] = {
            "segs": np.array(segs), "dvs": np.array(dvs),
            "runtimes": np.array(rts), "feasible": np.array(feas),
        }
        print(f"N={N}: {len(segs)} points loaded")

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for N in DEGREES:
        d = data[N]
        mask = d["feasible"]

        # Effort panel: feasible points only (infeasible seg=2 values
        # are 10x larger and destroy the y-axis scale)
        ax1.plot(d["segs"][mask], d["dvs"][mask],
                 f"{MARKERS[N]}-", color=COLORS[N], linewidth=2,
                 markersize=7, label=f"$N={N}$")

        # Runtime panel: all points
        ax2.plot(d["segs"], d["runtimes"],
                 f"{MARKERS[N]}-", color=COLORS[N], linewidth=2,
                 markersize=7, label=f"$N={N}$")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SEGS)
        ax.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in SEGS])
        ax.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.8, title="Degree")

    ax1.set_ylabel("$\\Delta v$ proxy (m/s)")
    ax1.set_title("Effort trend across degree")

    ax2.set_ylabel("Runtime (s)")
    ax2.set_title("Runtime trend across degree")

    fig.tight_layout()

    out = ROOT / "figures" / "f5_multi_order_tradeoff_N678.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
