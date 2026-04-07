#!/usr/bin/env python3
"""
Compare the repo's J2 gravity logic against a normalized reference dataset.

The script validates both:
- shared gravity path: orbital_docking.gravity._accel_total
- visualization gravity path: orbital_docking.visualization.accel_gravity_total_km_s2

Optionally writes summary plots for visual inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbital_docking import constants
from orbital_docking.gravity import _accel_total
from orbital_docking.j2_validation import error_summary, load_reference_dataset
from orbital_docking.visualization import accel_gravity_total_km_s2

DEFAULT_DATASET = REPO_ROOT / "tests" / "data" / "j2_reference" / "egm2008_degree2_samples.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "j2_validation"

plt.rcParams["axes.unicode_minus"] = False


def _fmt(v: float) -> str:
    return f"{float(v):.3e}"


def _summarize_rows(rows: list[dict], key: str) -> dict:
    vals = np.array([float(r[key]) for r in rows], dtype=float)
    return {
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
    }


def _print_summary(rows: list[dict]) -> None:
    opt_abs = _summarize_rows(rows, "opt_abs_norm")
    opt_rel = _summarize_rows(rows, "opt_rel_norm")
    viz_abs = _summarize_rows(rows, "viz_abs_norm")
    viz_rel = _summarize_rows(rows, "viz_rel_norm")
    xv_abs = _summarize_rows(rows, "cross_abs_norm")

    print("=" * 108)
    print(
        "sample_id".ljust(20),
        "alt_km".rjust(8),
        "opt_abs".rjust(14),
        "opt_rel".rjust(14),
        "viz_abs".rjust(14),
        "viz_rel".rjust(14),
        "opt_viz".rjust(14),
    )
    print("-" * 108)
    for row in rows:
        print(
            row["sample_id"].ljust(20),
            f"{row['altitude_km']:8.1f}",
            _fmt(row["opt_abs_norm"]).rjust(14),
            _fmt(row["opt_rel_norm"]).rjust(14),
            _fmt(row["viz_abs_norm"]).rjust(14),
            _fmt(row["viz_rel_norm"]).rjust(14),
            _fmt(row["cross_abs_norm"]).rjust(14),
        )
    print("-" * 108)
    print(
        "gravity_helper".ljust(20),
        "max".rjust(8),
        _fmt(opt_abs["max"]).rjust(14),
        _fmt(opt_rel["max"]).rjust(14),
    )
    print(
        "visualization".ljust(20),
        "max".rjust(8),
        _fmt(viz_abs["max"]).rjust(42),
        _fmt(viz_rel["max"]).rjust(14),
    )
    print(
        "gravity_vs_viz".ljust(20),
        "max".rjust(8),
        _fmt(xv_abs["max"]).rjust(70),
    )
    print("=" * 108)


def _make_plot(rows: list[dict], output_dir: Path, show: bool) -> Path:
    sample_ids = [r["sample_id"] for r in rows]
    xs = np.arange(len(rows))
    opt_abs = np.array([r["opt_abs_norm"] for r in rows], dtype=float)
    viz_abs = np.array([r["viz_abs_norm"] for r in rows], dtype=float)
    cross_abs = np.array([r["cross_abs_norm"] for r in rows], dtype=float)
    ref_mag = np.array([r["ref_norm"] for r in rows], dtype=float)
    j2_mag = np.array([r["ref_j2_norm"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(xs, ref_mag, marker="o", label="|a_total_ref|")
    axes[0, 0].plot(xs, j2_mag, marker="s", label="|a_j2_ref|")
    axes[0, 0].set_title("Reference Magnitudes")
    axes[0, 0].set_xticks(xs, sample_ids, rotation=30, ha="right")
    axes[0, 0].set_ylabel("km/s^2")
    axes[0, 0].legend()

    axes[0, 1].semilogy(xs, opt_abs + 1e-30, marker="o", label="optimizer vs ref")
    axes[0, 1].semilogy(xs, viz_abs + 1e-30, marker="s", label="visualization vs ref")
    axes[0, 1].set_title("Absolute Error Norms")
    axes[0, 1].set_xticks(xs, sample_ids, rotation=30, ha="right")
    axes[0, 1].set_ylabel("km/s^2")
    axes[0, 1].legend()

    axes[1, 0].semilogy(xs, cross_abs + 1e-30, marker="^", color="tab:green")
    axes[1, 0].set_title("Optimizer vs Visualization")
    axes[1, 0].set_xticks(xs, sample_ids, rotation=30, ha="right")
    axes[1, 0].set_ylabel("km/s^2")

    opt_rel = np.array([r["opt_rel_norm"] for r in rows], dtype=float)
    viz_rel = np.array([r["viz_rel_norm"] for r in rows], dtype=float)
    axes[1, 1].semilogy(xs, opt_rel + 1e-30, marker="o", label="optimizer")
    axes[1, 1].semilogy(xs, viz_rel + 1e-30, marker="s", label="visualization")
    axes[1, 1].set_title("Relative Error Norms")
    axes[1, 1].set_xticks(xs, sample_ids, rotation=30, ha="right")
    axes[1, 1].legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "j2_validation_summary.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify J2 logic against a normalized baseline dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Reference dataset JSON path.")
    parser.add_argument("--plot", action="store_true", help="Write a summary plot to the output directory.")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively when --plot is used.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for plot output.")
    parser.add_argument("--max-abs-error", type=float, default=2e-11, help="Maximum allowed absolute norm error in km/s^2.")
    parser.add_argument("--max-rel-error", type=float, default=5e-9, help="Maximum allowed relative norm error.")
    parser.add_argument(
        "--max-cross-error",
        type=float,
        default=1e-15,
        help="Maximum allowed optimizer-vs-visualization absolute norm difference in km/s^2.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any threshold is violated.")
    args = parser.parse_args()

    dataset = load_reference_dataset(args.dataset)
    rows = []
    for sample in dataset["samples"]:
        r_km = np.array(sample["r_km"], dtype=float)
        ref_total = np.array(sample["a_total_km_s2"], dtype=float)
        ref_j2 = np.array(sample["a_j2_km_s2"], dtype=float)
        opt_total = _accel_total(
            r_km,
            constants.EARTH_MU_SCALED,
            constants.EARTH_RADIUS_KM,
            constants.EARTH_J2,
        )
        viz_total = accel_gravity_total_km_s2(r_km)

        opt_err = error_summary(opt_total, ref_total)
        viz_err = error_summary(viz_total, ref_total)
        cross_err = error_summary(opt_total, viz_total)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "altitude_km": float(sample["altitude_km"]),
                "ref_norm": float(np.linalg.norm(ref_total)),
                "ref_j2_norm": float(np.linalg.norm(ref_j2)),
                "opt_abs_norm": opt_err["abs_norm"],
                "opt_rel_norm": opt_err["rel_norm"],
                "viz_abs_norm": viz_err["abs_norm"],
                "viz_rel_norm": viz_err["rel_norm"],
                "cross_abs_norm": cross_err["abs_norm"],
            }
        )

    _print_summary(rows)

    if args.plot:
        plot_path = _make_plot(rows, Path(args.output_dir), args.show)
        print(f"plot: {plot_path}")

    failed = []
    for row in rows:
        if row["opt_abs_norm"] > args.max_abs_error:
            failed.append(f"{row['sample_id']} gravity helper abs error {row['opt_abs_norm']}")
        if row["opt_rel_norm"] > args.max_rel_error:
            failed.append(f"{row['sample_id']} gravity helper rel error {row['opt_rel_norm']}")
        if row["viz_abs_norm"] > args.max_abs_error:
            failed.append(f"{row['sample_id']} visualization abs error {row['viz_abs_norm']}")
        if row["viz_rel_norm"] > args.max_rel_error:
            failed.append(f"{row['sample_id']} visualization rel error {row['viz_rel_norm']}")
        if row["cross_abs_norm"] > args.max_cross_error:
            failed.append(f"{row['sample_id']} gravity-helper-vs-visualization error {row['cross_abs_norm']}")

    if failed:
        print("threshold failures:")
        for msg in failed:
            print(f"  - {msg}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("all thresholds satisfied")


if __name__ == "__main__":
    main()
