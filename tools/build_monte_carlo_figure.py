#!/usr/bin/env python3
"""Build Figure 5 and Table 4 from the near-baseline Monte Carlo.

Both come from the committed doc/results/monte_carlo_subdivision/runs.csv with the
selection rules of tools/build_monte_carlo.py, so the figure and the table cannot
disagree. Panel (c) is the baseline instance's runtime from Table 3's committed
timing (minimum of 15 solves on the paper's machine): the Monte Carlo ran in
parallel on another machine and does not measure runtime.

  (a) measured / estimated minimum clearance, over certified solves whose
      constraint is active, whose closest approach is interior, and whose
      closest sub-arc lies on its supporting plane -- section 3.1's assumption.
      n_seg = 4 has no such solve: the closest point is always the departure.
  (b) control cost J above the same instance at n_seg = 64, over certified
      solves with the constraint active.
  (c) baseline runtime.

Boxes span the quartiles; whiskers the 5th to 95th percentile.

Usage:
    .venv/bin/python tools/build_monte_carlo_figure.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullLocator

from tools import build_monte_carlo as MC
from tools.paper_tables_csv import T3_SIGNATURE, block

RUNS = ROOT / "doc" / "results" / "monte_carlo_subdivision" / "runs.csv"
OUT = ROOT / "figures" / "monte_carlo_subdivision.png"
NSEGS = (4, 8, 16, 32, 64)
TOP = 64

C_INK, C_BOX, C_BASE, C_GRID = "#333333", "#006BA4", "#222222", "#DDDDDD"


def load():
    rows = [MC._typed(r) for r in MC.read_rows(RUNS)]
    by = {(r["instance"], r["n_seg"]): r for r in rows if r["kind"] == "sample"}
    ids = sorted({i for i, _ in by})
    return by, ids


def ratios(by, ids, s):
    return [by[(i, s)]["ratio"] for i in ids
            if by[(i, s)]["certified"] and MC._active(by[(i, s)])
            and by[(i, s)]["koz_binds"] and MC._pressed(by[(i, s)])]


def cost_increase(by, ids, s):
    return [100 * (by[(i, s)]["objective"] / by[(i, TOP)]["objective"] - 1) for i in ids
            if by[(i, s)]["certified"] and by[(i, TOP)]["certified"] and MC._active(by[(i, s)])]


def q(xs, p):
    return float(np.percentile(xs, p))


def table_rows(by, ids):
    """Markdown rows of Table 4, one per n_seg."""
    out = []
    for s in NSEGS:
        cert = sum(1 for i in ids if by[(i, s)]["certified"])
        if cert != len(ids):
            raise ValueError(f"n_seg={s}: {len(ids) - cert} uncertified; Table 4 assumes none")
        r = ratios(by, ids, s)
        c = cost_increase(by, ids, s)
        it = [by[(i, s)]["iters"] for i in ids]
        rc = f"{q(r, 50):.3f} [{q(r, 5):.3f}, {q(r, 95):.3f}]" if r else "—"
        cc = "—" if s == TOP else f"{q(c, 50):.3g} [{q(c, 5):.3g}, {q(c, 95):.3g}]"
        out.append(f"| {s} | {len(r)} | {rc} | {cc} | {q(it, 50):.0f} [{q(it, 5):.0f}, {q(it, 95):.0f}] |")
    return out


def style(ax, ylabel, title):
    ax.set_xticks(range(len(NSEGS)))
    ax.set_xticklabels([str(s) for s in NSEGS])
    ax.set_xlim(-0.6, len(NSEGS) - 0.4)
    ax.set_xlabel("Subdivision count $n_{\\mathrm{seg}}$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, color=C_INK, pad=10)
    ax.grid(True, axis="y", which="major", color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(labelsize=10, colors=C_INK)


def boxes(ax, data):
    pos = [k for k, d in enumerate(data) if d]
    bp = ax.boxplot([d for d in data if d], positions=pos, widths=0.5, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=C_INK, linewidth=1.8),
                    boxprops=dict(facecolor="#DCE9F3", edgecolor=C_BOX, linewidth=1.2),
                    whiskerprops=dict(color=C_BOX, linewidth=1.2),
                    capprops=dict(color=C_BOX, linewidth=1.2))
    return bp


def main():
    by, ids = load()
    base = {r["n_seg"]: r for r in block(T3_SIGNATURE, "Figure 5 (c), baseline runtime")}
    base_cost = {s: 100 * (base[s]["objective"] / base[TOP]["objective"] - 1) for s in NSEGS}

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)

    ax = axes[0]
    boxes(ax, [ratios(by, ids, s) for s in NSEGS])
    ax.axhline(1.0, color=C_BASE, linewidth=1.2, linestyle=(0, (5, 3)), zorder=1)
    ax.text(len(NSEGS) - 0.45, 1.0, "estimate", ha="right", va="bottom", fontsize=9, color=C_INK)
    ax.text(0, 0.95, "endpoint\nminimum", ha="center", va="center", fontsize=9, color="#777777")
    ax.set_ylim(0.88, 1.02)
    style(ax, "Measured / estimated clearance", "(a) Clearance vs. estimate")

    ax = axes[1]
    data = [cost_increase(by, ids, s) if s != TOP else [] for s in NSEGS]
    boxes(ax, data)
    xs = [k for k, s in enumerate(NSEGS) if s != TOP]
    ax.plot(xs, [base_cost[NSEGS[k]] for k in xs], linestyle="none", marker="D",
            markersize=6, color=C_BASE, label="Baseline", zorder=4)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1, 3, 10, 30, 100, 300])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.25, 700)
    ax.text(len(NSEGS) - 1, 0.32, "reference", ha="center", va="bottom", fontsize=9, color="#777777")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    style(ax, f"Control cost $J$ above $n_{{\\mathrm{{seg}}}}={TOP}$ (%)", "(b) Control cost")

    ax = axes[2]
    rt = [base[s]["runtime_s"] for s in NSEGS]
    ax.plot(range(len(NSEGS)), rt, color=C_BASE, linewidth=1.6, marker="D", markersize=6,
            label="Baseline", zorder=3)
    for k, y in enumerate(rt):
        # The rising segment after n_seg = 32 would cross a centred label.
        dx, ha = (-9, "right") if NSEGS[k] == 32 else (0, "center")
        ax.annotate(f"{y:.3f}", (k, y), textcoords="offset points", xytext=(dx, 8),
                    ha=ha, va="bottom", fontsize=9, color=C_INK)
    ax.set_ylim(0.07, 0.25)
    ax.set_yticks([0.08, 0.12, 0.16, 0.20, 0.24])
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    style(ax, "Runtime (s)", "(c) Runtime")

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Figure 5 from {len(ids)} instances -> {OUT}")
    print("Table 4 rows (n_seg | compared | ratio | J above 64 (%) | iterations):")
    for line in table_rows(by, ids):
        print(line)
    print("baseline J above 64 (%):", {s: round(v, 2) for s, v in base_cost.items()})


if __name__ == "__main__":
    main()
