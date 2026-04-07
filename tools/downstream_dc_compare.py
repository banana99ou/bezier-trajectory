#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbital_docking.downstream_collocation import (
    DirectCollocationConfig,
    build_demo_bezier_warm_start,
    make_demo_problem,
    run_downstream_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the narrow T6 downstream direct-collocation comparison."
    )
    parser.add_argument("--degree", type=int, default=4, choices=[2, 3, 4])
    parser.add_argument("--n-seg", type=int, default=16)
    parser.add_argument("--n-intervals", type=int, default=12)
    parser.add_argument("--dc-maxiter", type=int, default=200)
    parser.add_argument("--dc-gtol", type=float, default=1e-6)
    parser.add_argument("--dc-barrier-tol", type=float, default=1e-6)
    parser.add_argument("--upstream-maxiter", type=int, default=120)
    parser.add_argument("--upstream-tol", type=float, default=1e-8)
    parser.add_argument("--objective", type=str, default="energy", choices=["energy", "dv"])
    parser.add_argument("--ignore-upstream-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    problem = make_demo_problem(n_intervals=args.n_intervals)
    config = DirectCollocationConfig(
        maxiter=args.dc_maxiter,
        gtol=args.dc_gtol,
        barrier_tol=args.dc_barrier_tol,
        verbose=False,
    )

    control_points, upstream_info = build_demo_bezier_warm_start(
        degree=args.degree,
        n_seg=args.n_seg,
        objective_mode=args.objective,
        max_iter=args.upstream_maxiter,
        tol=args.upstream_tol,
        use_cache=True,
        ignore_existing_cache=args.ignore_upstream_cache,
    )
    results = run_downstream_comparison(control_points, problem=problem, config=config)
    results["upstream_warm_start"] = {
        "degree": int(args.degree),
        "n_seg": int(args.n_seg),
        "objective": args.objective,
        "optimizer_info": {
            "iterations": int(upstream_info.get("iterations", -1)),
            "feasible": bool(upstream_info.get("feasible", False)),
            "cost_true_energy": float(upstream_info.get("cost_true_energy", upstream_info.get("cost", float("nan")))),
            "elapsed_time": float(upstream_info.get("elapsed_time", float("nan"))),
        },
    }

    naive = results["comparison"]["naive"]
    warm = results["comparison"]["bezier_warm_start"]
    summary = {
        "naive_success": naive["success"],
        "warm_success": warm["success"],
        "naive_time_s": naive["solve_time_s"],
        "warm_time_s": warm["solve_time_s"],
        "naive_iterations": naive["iteration_count"],
        "warm_iterations": warm["iteration_count"],
        "naive_objective": naive["final_objective"],
        "warm_objective": warm["final_objective"],
        "naive_max_eq_violation": naive["constraint_satisfaction"]["max_eq_violation"],
        "warm_max_eq_violation": warm["constraint_satisfaction"]["max_eq_violation"],
        "naive_min_node_margin_km": naive["constraint_satisfaction"]["min_node_margin_km"],
        "warm_min_node_margin_km": warm["constraint_satisfaction"]["min_node_margin_km"],
    }
    results["summary"] = summary

    text = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()
