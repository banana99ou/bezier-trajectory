"""Build T6 downstream-comparison CSV and F6 speedup figure.

Reads per-case JSONs under ``results/dcm_pass1_replace_full/`` and writes:

- ``artifacts/paper_artifacts/t6_downstream_comparison.csv``
- ``figures/f6_downstream_speedup.png``

Also prints median/min/max speedup and cost-delta summary to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "dcm_pass1_replace_full"
DEFAULT_CSV = REPO / "artifacts" / "paper_artifacts" / "t6_downstream_comparison.csv"
DEFAULT_FIG = REPO / "figures" / "f6_downstream_speedup.png"

T6_COLUMNS = [
    "case_id",
    "T_normed",
    "h0_km",
    "delta_a_km",
    "delta_i_deg",
    "baseline_converged",
    "proposed_converged",
    "baseline_time_s",
    "bezier_time_s",
    "pass2_time_s",
    "proposed_total_time_s",
    "speedup",
    "baseline_cost",
    "proposed_cost",
    "cost_delta",
    "baseline_n_peaks",
    "proposed_n_peaks",
]


def load_case_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for json_path in sorted(results_dir.glob("case_*.json")):
        payload = json.loads(json_path.read_text())
        cfg = payload.get("config", {})
        base = payload.get("baseline", {})
        prop = payload.get("proposed", {})

        baseline_time = base.get("solve_time_s", float("nan"))
        bezier_time = prop.get("bezier_time_s", float("nan"))
        pass2_time = prop.get("pass2_time_s", float("nan"))
        total_time = prop.get("total_time_s")
        if total_time is None:
            if not (math.isnan(bezier_time) or math.isnan(pass2_time)):
                total_time = bezier_time + pass2_time
            else:
                total_time = float("nan")

        speedup = baseline_time / total_time if total_time and total_time > 0 else float("nan")
        baseline_cost = base.get("cost", float("nan"))
        proposed_cost = prop.get("cost", float("nan"))
        cost_delta = proposed_cost - baseline_cost if not (
            math.isnan(baseline_cost) or math.isnan(proposed_cost)
        ) else float("nan")

        rows.append(
            {
                "case_id": payload.get("case_id"),
                "T_normed": cfg.get("T_max_normed"),
                "h0_km": cfg.get("h0"),
                "delta_a_km": cfg.get("delta_a"),
                "delta_i_deg": cfg.get("delta_i"),
                "baseline_converged": bool(base.get("converged")),
                "proposed_converged": bool(prop.get("converged")),
                "baseline_time_s": baseline_time,
                "bezier_time_s": bezier_time,
                "pass2_time_s": pass2_time,
                "proposed_total_time_s": total_time,
                "speedup": speedup,
                "baseline_cost": baseline_cost,
                "proposed_cost": proposed_cost,
                "cost_delta": cost_delta,
                "baseline_n_peaks": base.get("n_peaks"),
                "proposed_n_peaks": prop.get("n_peaks"),
            }
        )
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=T6_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in T6_COLUMNS})


def build_figure(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    both_converged = [r for r in rows if r["baseline_converged"] and r["proposed_converged"]]
    if not both_converged:
        raise RuntimeError("No cases with both pipelines converged; cannot build F6.")

    sorted_rows = sorted(both_converged, key=lambda r: r["speedup"])
    labels = [f"{r['case_id']}" for r in sorted_rows]
    speedups = np.array([r["speedup"] for r in sorted_rows])
    baseline_times = np.array([r["baseline_time_s"] for r in sorted_rows])
    bezier_times = np.array([r["bezier_time_s"] for r in sorted_rows])
    pass2_times = np.array([r["pass2_time_s"] for r in sorted_rows])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = ["#1f77b4" if s >= 1 else "#d62728" for s in speedups]
    ax_left.bar(labels, speedups, color=colors)
    ax_left.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_left.set_ylabel("Speedup (baseline / proposed)")
    ax_left.set_xlabel("Case id (sorted by speedup)")
    ax_left.set_title("Per-case end-to-end speedup")
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)

    x = np.arange(len(labels))
    width = 0.38
    ax_right.bar(x - width / 2, baseline_times, width, label="Baseline (Pass 1 + Pass 2)", color="#9467bd")
    ax_right.bar(x + width / 2, bezier_times, width, label="Bézier upstream", color="#2ca02c")
    ax_right.bar(x + width / 2, pass2_times, width, bottom=bezier_times, label="Pass 2", color="#17becf")
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels)
    ax_right.set_ylabel("Wall-clock time (s)")
    ax_right.set_xlabel("Case id")
    ax_right.set_title("Runtime composition")
    ax_right.legend(loc="upper left", fontsize=8)
    ax_right.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("F6. Bézier-Pass-1-replacement vs full two-pass DCM (matched Pass 2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    both = [r for r in rows if r["baseline_converged"] and r["proposed_converged"]]
    speedups = np.array([r["speedup"] for r in both])
    cost_deltas = np.array([abs(r["cost_delta"]) for r in both if not math.isnan(r["cost_delta"])])

    n_total = len(rows)
    n_both = len(both)
    n_base_only = sum(1 for r in rows if r["baseline_converged"] and not r["proposed_converged"])
    n_prop_only = sum(1 for r in rows if not r["baseline_converged"] and r["proposed_converged"])
    n_both_fail = sum(1 for r in rows if not r["baseline_converged"] and not r["proposed_converged"])

    print(f"Cases total: {n_total}")
    print(f"  Both converged: {n_both}")
    print(f"  Baseline only : {n_base_only}")
    print(f"  Proposed only : {n_prop_only}")
    print(f"  Both failed   : {n_both_fail}")

    if len(speedups):
        print(
            "Speedup (both-converged): "
            f"min={speedups.min():.2f}x  median={np.median(speedups):.2f}x  "
            f"max={speedups.max():.2f}x  >=1x: {int((speedups >= 1).sum())}/{len(speedups)}"
        )
    if len(cost_deltas):
        print(
            "|cost_delta|: "
            f"max={cost_deltas.max():.3e}  median={np.median(cost_deltas):.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()

    rows = load_case_rows(args.results)
    if not rows:
        raise SystemExit(f"No case_*.json files under {args.results}")

    write_csv(rows, args.csv)
    build_figure(rows, args.figure)
    print_summary(rows)
    print(f"\nCSV:    {args.csv}")
    print(f"Figure: {args.figure}")


if __name__ == "__main__":
    main()
