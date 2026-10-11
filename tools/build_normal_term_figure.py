#!/usr/bin/env python3
"""Plot the stored normal-derivative comparison without rerunning the solver.

Data and selection match tools/build_normal_term_ablation.py:
  (a) fraction with final hull violation <= 1e-6 km AND stop reason 1 or 4;
  (b) iterations until termination, including runs that fail that criterion;
  (c) paired change in J, 100 * (J_without / J_with - 1), using cases whose
      final hull violation is <= 1e-6 km in BOTH arms, irrespective of stop.
      n_seg=4 is omitted from the cost comparison, as in manuscript Table 5.

Panels (b) and (c) show medians and the 5th to 95th percentiles. A recorded
solver stop is not an independent optimality or stationarity verification.
The CSV files are read directly, so no solver extension is imported.

Usage:
    .venv/bin/python tools/build_normal_term_figure.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.paper_style import SMALL_PT, apply as apply_paper_style

DATA = ROOT / "doc/results/normal_term_ablation"
OUT = ROOT / "figures/normal_term_comparison.png"
ARMS = ("with", "without")
# Same certificate and accepted stops as tools/build_tables.py. Verify that the
# stored classification agrees rather than silently redefining the metric.
CERT_TOL_KM = 1e-6
ACCEPTED_STOPS = (1, 4)
FIG_W, FIG_H = 12.0, 5.0
HANGUL = "NanumBarunGothic"
C_INK, C_WITH, C_WITHOUT, C_GRID = "#333333", "#006BA4", "#555555", "#DDDDDD"
ARM_STYLES = {
    "with": dict(color=C_WITH, marker="o", linestyle="-", label="법선 변화 포함"),
    "without": dict(color=C_WITHOUT, marker="s", linestyle="--", label="법선 고정"),
}


def load():
    """Read matched sample rows and validate the recorded stopping metric."""
    meta = {arm: json.loads((DATA / arm / "meta.json").read_text()) for arm in ARMS}
    for key in ("n", "nsegs", "seed", "range_set", "degree"):
        if meta["with"][key] != meta["without"][key]:
            raise ValueError(f"The comparison arms differ in metadata: {key}")
    nsegs = tuple(meta["with"]["nsegs"])
    runs = {}
    for arm in ARMS:
        by = {}
        with (DATA / arm / "runs.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["kind"] != "sample":
                    continue
                if row["error"]:
                    raise ValueError(f"{arm}: failed sample {row['instance']}: {row['error']}")
                key = (int(row["instance"]), int(row["n_seg"]))
                if key in by:
                    raise ValueError(f"{arm}: duplicate sample {key}")
                r = dict(iters=int(row["iters"]), objective=float(row["objective"]),
                         hull=float(row["hull_violation_km"]), stop=int(row["stop"]))
                if not all(np.isfinite(r[k]) for k in ("iters", "objective", "hull")):
                    raise ValueError(f"{arm}: non-finite result for {key}")
                r["satisfied"] = r["hull"] <= CERT_TOL_KM and r["stop"] in ACCEPTED_STOPS
                if r["satisfied"] != (row["certified"] == "True"):
                    raise ValueError(f"{arm}: recorded stopping classification disagrees for {key}")
                by[key] = r
        runs[arm] = by
    ids = sorted({i for i, _ in runs["with"]})
    expected = {(i, s) for i in ids for s in nsegs}
    if len(ids) != meta["with"]["n"] or any(set(runs[a]) != expected for a in ARMS):
        raise ValueError("The comparison must contain the same complete sample grid in both arms")
    return runs, ids, nsegs


def summarize(runs, ids, nsegs):
    rates, iterations, costs = {}, {}, {}
    for arm in ARMS:
        rates[arm] = [100 * np.mean([runs[arm][i, s]["satisfied"] for i in ids]) for s in nsegs]
        iterations[arm] = {s: [runs[arm][i, s]["iters"] for i in ids] for s in nsegs}
    for s in nsegs:
        if s == 4:
            continue
        pairs = [(runs["with"][i, s], runs["without"][i, s]) for i in ids]
        costs[s] = [100 * (b["objective"] / a["objective"] - 1) for a, b in pairs
                    if a["hull"] <= CERT_TOL_KM and b["hull"] <= CERT_TOL_KM]
        if not costs[s] or np.percentile(costs[s], 5) <= 0:
            raise ValueError(f"n_seg={s}: log scale cannot show this cost percentile interval")
    return rates, iterations, costs


def style(ax, nsegs, title, ylabel):
    ax.set_xticks(range(len(nsegs)), [str(s) for s in nsegs])
    ax.set_xlim(-0.55, len(nsegs) - 0.45)
    ax.set_xlabel(r"분할 수 $n_{\mathrm{seg}}$", fontfamily=HANGUL)
    ax.set_ylabel(ylabel, fontfamily=HANGUL if "$" in ylabel else None)
    ax.set_title(title, color=C_INK, pad=12)
    ax.grid(axis="y", which="major", color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color("#999999")
    ax.tick_params(colors=C_INK)


def intervals(ax, xs, values, arm):
    lo, med, hi = np.percentile(values, [5, 50, 95], axis=1)
    ax.errorbar(xs, med, yerr=[med - lo, hi - med], capsize=4,
                linewidth=1.7, elinewidth=1.2, markersize=6,
                markerfacecolor="white" if arm == "without" else C_WITH,
                **ARM_STYLES[arm])


def table_rows(rates, iterations, costs, nsegs):
    """Reproduce the displayed Table 5 precision for direct review."""
    def cell(values, fmt):
        lo, med, hi = np.percentile(values, [5, 50, 95])
        return f"{med:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

    for index, s in enumerate(nsegs):
        cost = cell(costs[s], ".3g") if s in costs else "—"
        yield f"| {s} | {rates['without'][index]:.1f} | {cell(iterations['without'][s], '.0f')} | {cost} |"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    runs, ids, nsegs = load()
    rates, iterations, costs = summarize(runs, ids, nsegs)
    k_pt = apply_paper_style(FIG_W)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.062, right=0.99, top=0.77, bottom=0.29, wspace=0.39)
    xs = np.arange(len(nsegs))

    for arm in ARMS:
        axes[0].plot(xs, rates[arm], linewidth=1.7, markersize=6,
                     markerfacecolor="white" if arm == "without" else C_WITH,
                     **ARM_STYLES[arm])
    axes[0].set_ylim(-5, 106)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    style(axes[0], nsegs, "(a) 종료 기준 만족 비율", "비율 (%)")

    for arm, offset in (("with", -0.10), ("without", 0.10)):
        intervals(axes[1], xs + offset, [iterations[arm][s] for s in nsegs], arm)
    axes[1].set_ylim(0, 47)
    axes[1].set_yticks([0, 10, 20, 30, 40])
    style(axes[1], nsegs, "(b) 종료까지 반복 횟수", "반복 횟수")

    # Cost sample sizes differ by subdivision, so summarize each list separately.
    selected = [s for s in nsegs if s in costs]
    quantiles = np.array([np.percentile(costs[s], [5, 50, 95]) for s in selected]).T
    lo, med, hi = quantiles
    axes[2].errorbar([nsegs.index(s) for s in selected], med,
                     yerr=[med - lo, hi - med], capsize=4, linewidth=1.7,
                     elinewidth=1.2, markersize=6, markerfacecolor="white", **ARM_STYLES["without"])
    axes[2].set_yscale("log")
    axes[2].set_ylim(0.0005, 10)
    axes[2].set_yticks([0.001, 0.01, 0.1, 1, 10])
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[2].yaxis.set_minor_locator(NullLocator())
    axes[2].text(0, 0.07, "비교\n제외", ha="center", va="center", color="#777777",
                 fontsize=SMALL_PT * k_pt)
    style(axes[2], nsegs, "(c) 제어 비용 증가율", r"$J$ 증가율 (%)")

    handles = [Line2D([], [], linewidth=1.7, markersize=6,
                      markerfacecolor="white" if arm == "without" else C_WITH,
                      **ARM_STYLES[arm]) for arm in ARMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.97),
               ncol=2, frameon=False, handlelength=2.5, columnspacing=2.2)
    note_size = SMALL_PT * k_pt
    fig.text(0.5, 0.13,
             f"(a), (b): 분할 수별 {len(ids)}개 사례.  (b), (c): 중앙값과 5–95번째 백분위수.",
             ha="center", va="center", fontsize=note_size, color=C_INK)
    fig.text(0.5, 0.058,
             "(c): 두 해가 모두 명제 1의 조건을 만족한 사례에서, 법선 변화 포함 대비 법선 고정의 비용 증가율.",
             ha="center", va="center", fontsize=note_size, color=C_INK)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Figure from {len(ids)} paired instances -> {args.out}")
    print("Table 5 rows:")
    print("\n".join(table_rows(rates, iterations, costs, nsegs)))
    print("Paired feasible cost samples:", {s: len(costs[s]) for s in costs})
    print("With-normal stopping-criterion rates:", rates["with"])


if __name__ == "__main__":
    main()
