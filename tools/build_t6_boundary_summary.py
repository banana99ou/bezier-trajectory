"""Summarize the Bézier-upstream-feasibility boundary from the full-range run.

Reads per-case JSONs under ``results/dcm_pass1_replace_boundary/`` and emits a
paragraph-ready table stratified by ``T_normed`` bucket. Used to populate the
boundary appendix in ``doc/dcm_downstream_pack.md`` and the limitations section.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / "results" / "dcm_pass1_replace_boundary"

BUCKETS: list[tuple[str, float, float]] = [
    ("T <= 0.5",  0.0, 0.5),
    ("0.5 < T <= 1.0", 0.5, 1.0),
    ("1.0 < T <= 2.0", 1.0, 2.0),
    ("T > 2.0", 2.0, float("inf")),
]


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("case_*.json")):
        d = json.loads(path.read_text())
        cfg = d.get("config", {})
        base = d.get("baseline", {})
        prop = d.get("proposed", {})
        up = d.get("bezier_upstream", {})
        rows.append(
            {
                "case_id": d.get("case_id"),
                "T_normed": cfg.get("T_max_normed"),
                "e0": cfg.get("e0", 0.0),
                "ef": cfg.get("ef", 0.0),
                "bezier_feasible": bool(up.get("feasible")),
                "baseline_converged": bool(base.get("converged")),
                "proposed_converged": bool(prop.get("converged")),
            }
        )
    return rows


def summarize(rows: list[dict]) -> str:
    out: list[str] = []
    out.append(f"Total cases: {len(rows)}")
    bz_feas = sum(1 for r in rows if r["bezier_feasible"])
    base_conv = sum(1 for r in rows if r["baseline_converged"])
    prop_conv = sum(1 for r in rows if r["proposed_converged"])
    both = sum(1 for r in rows if r["baseline_converged"] and r["proposed_converged"])
    out.append(
        f"Bézier upstream feasible: {bz_feas}/{len(rows)}  "
        f"Baseline converged: {base_conv}/{len(rows)}  "
        f"Proposed converged: {prop_conv}/{len(rows)}  "
        f"Both converged: {both}/{len(rows)}"
    )

    out.append("")
    out.append("| T_normed bucket | N | Bézier feasible | Baseline conv. | Both conv. | Bézier feas. & circular (e0=ef=0) | Bézier feas. & any ecc. |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, lo, hi in BUCKETS:
        bucket = [r for r in rows if (r["T_normed"] is not None) and (lo < r["T_normed"] <= hi) and r["T_normed"] <= hi]
        if not bucket and lo == 0.0:
            bucket = [r for r in rows if (r["T_normed"] is not None) and (r["T_normed"] <= hi)]
        n = len(bucket)
        bf = sum(1 for r in bucket if r["bezier_feasible"])
        bc = sum(1 for r in bucket if r["baseline_converged"])
        both_b = sum(1 for r in bucket if r["baseline_converged"] and r["proposed_converged"])
        bf_circ = sum(1 for r in bucket if r["bezier_feasible"] and r["e0"] == 0 and r["ef"] == 0)
        bf_ecc = sum(1 for r in bucket if r["bezier_feasible"] and (r["e0"] > 0 or r["ef"] > 0))
        out.append(f"| `{label}` | {n} | {bf} | {bc} | {both_b} | {bf_circ} | {bf_ecc} |")

    out.append("")
    out.append("Eccentricity breakdown of Bézier-feasible cases (all buckets combined):")
    bf_rows = [r for r in rows if r["bezier_feasible"]]
    circular = sum(1 for r in bf_rows if r["e0"] == 0 and r["ef"] == 0)
    mild_ecc = sum(1 for r in bf_rows if 0 < max(r["e0"], r["ef"]) <= 0.05)
    higher_ecc = sum(1 for r in bf_rows if max(r["e0"], r["ef"]) > 0.05)
    out.append(f"  circular (e0 = ef = 0): {circular}")
    out.append(f"  mild (0 < max(e) <= 0.05): {mild_ecc}")
    out.append(f"  higher (max(e) > 0.05): {higher_ecc}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"No case_*.json under {args.results}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
