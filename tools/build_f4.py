#!/usr/bin/env python3
"""
Build F4: Subdivision-count tradeoff figure for N=7.

Two-panel figure showing:
  Left:  Runtime vs subdivision count
  Right: Delta-v proxy and safety margin vs subdivision count

Uses the 120-deg phase-lag cache files (same as build_f3.py / build_csv.py).

Usage:
    python tools/build_f4.py
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

N = 7
SEGS = [2, 4, 8, 16, 32, 64]


def main():
    P_start, v0, P_end, v1 = get_120deg_endpoints()

    segs, runtimes, dvs, margins = [], [], [], []
    for n_seg in SEGS:
        path = get_cache_path_120(N, n_seg, P_start, v0, P_end, v1)
        result = load_from_cache(path)
        if result is None:
            print(f"  MISSING N={N} seg={n_seg}")
            continue
        _, info = result
        min_r = info["min_radius"]
        safety = min_r - constants.KOZ_RADIUS
        segs.append(n_seg)
        runtimes.append(info["elapsed_time"])
        dvs.append(info["dv_proxy_m_s"])
        margins.append(safety)
        tag = "INFEAS" if not info["feasible"] else "OK"
        print(f"  {tag}: seg={n_seg:2d}  dv={info['dv_proxy_m_s']:10.3f}  "
              f"safety={safety:8.3f} km  rt={info['elapsed_time']:.1f}s")

    segs = np.array(segs)
    runtimes = np.array(runtimes)
    dvs = np.array(dvs)
    margins = np.array(margins)

    # ── Figure ────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: Runtime
    ax1.plot(segs, runtimes, "o-", color="tab:red", linewidth=2, markersize=7)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(SEGS)
    ax1.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in SEGS])
    ax1.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_title("Runtime vs subdivision count")
    ax1.grid(True, alpha=0.3)

    # Right panel: dv proxy + safety margin (dual y-axis)
    color_dv = "tab:blue"
    color_sm = "tab:green"

    ax2.plot(segs, dvs, "o-", color=color_dv, linewidth=2, markersize=7,
             label="$\\Delta v$ proxy (m/s)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(SEGS)
    ax2.set_xticklabels([f"$2^{{{int(np.log2(s))}}}$" for s in SEGS])
    ax2.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$")
    ax2.set_ylabel("$\\Delta v$ proxy (m/s)", color=color_dv)
    ax2.tick_params(axis="y", labelcolor=color_dv)
    ax2.set_title("Outcome metrics vs subdivision count")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(segs, margins, "s--", color=color_sm, linewidth=2, markersize=7,
              label="Safety margin (km)")
    ax2b.set_ylabel("Safety margin (km)", color=color_sm)
    ax2b.tick_params(axis="y", labelcolor=color_sm)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               fontsize=9, framealpha=0.8)

    fig.tight_layout()

    out = ROOT / "figures" / "f4_subdivision_tradeoff_N7.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
