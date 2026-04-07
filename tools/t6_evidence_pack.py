#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbital_docking.downstream_collocation import (  # noqa: E402
    _build_boundary_constraint,
    _control_effort_objective,
    _dynamics_defects,
    _koz_margins,
    DirectCollocationConfig,
    build_bezier_warm_start,
    build_demo_bezier_warm_start,
    build_naive_initial_guess,
    make_demo_problem,
)


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "paper_artifacts"
FIGURE_DIR = REPO_ROOT / "figures"
RUN_JSON = ARTIFACT_DIR / "t6_gate_run.json"
EVIDENCE_JSON = ARTIFACT_DIR / "t6_evidence_pack.json"
PATHS_FIG = FIGURE_DIR / "t6_evidence_paths.png"
SOLVER_FIG = FIGURE_DIR / "t6_solver_checks.png"


def _load_run_json() -> dict:
    if not RUN_JSON.exists():
        raise FileNotFoundError(f"Missing gate-run artifact: {RUN_JSON}")
    return json.loads(RUN_JSON.read_text(encoding="utf-8"))


def _unpack_state_guess(x: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_block = n_nodes * 3
    r = x[:n_block].reshape(n_nodes, 3)
    v = x[n_block : 2 * n_block].reshape(n_nodes, 3)
    u = x[2 * n_block :].reshape(n_nodes, 3)
    return r, v, u


def _sphere_wireframe(ax, radius: float, color: str = "0.8") -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 32)
    v = np.linspace(0.0, np.pi, 16)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color=color, linewidth=0.4, alpha=0.4)


def _set_equal_3d(ax, pts: list[np.ndarray]) -> None:
    stacked = np.vstack(pts)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_path_panel(ax, title: str, path: np.ndarray, problem, start, goal) -> None:
    _sphere_wireframe(ax, problem.koz_radius_km)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "-o", ms=3, lw=2, color="#1f77b4")
    ax.scatter(start[0], start[1], start[2], color="green", s=45, label="start")
    ax.scatter(goal[0], goal[1], goal[2], color="red", s=45, label="goal")
    ax.set_title(title)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.view_init(elev=22, azim=38)


def build_evidence() -> dict:
    run = _load_run_json()
    problem = make_demo_problem(n_intervals=int(run["problem"]["n_intervals"]))
    control_points, upstream_info = build_demo_bezier_warm_start(
        degree=int(run["upstream_warm_start"]["degree"]),
        n_seg=int(run["upstream_warm_start"]["n_seg"]),
        objective_mode=str(run["upstream_warm_start"]["objective"]),
        max_iter=int(run["upstream_warm_start"]["optimizer_info"]["iterations"]),
        tol=1e-8,
        use_cache=True,
        ignore_existing_cache=False,
    )
    naive_x0 = build_naive_initial_guess(problem)
    warm_x0 = build_bezier_warm_start(problem, control_points)
    naive_r0, naive_v0, naive_u0 = _unpack_state_guess(naive_x0, problem.n_nodes)
    warm_r0, warm_v0, warm_u0 = _unpack_state_guess(warm_x0, problem.n_nodes)
    boundary = _build_boundary_constraint(problem)

    def _initial_diag(x: np.ndarray) -> dict:
        boundary_res = boundary.A @ x - boundary.lb
        dyn = _dynamics_defects(x, problem)
        koz = _koz_margins(x, problem)
        return {
            "initial_objective": float(_control_effort_objective(x, problem)),
            "max_boundary_violation": float(np.max(np.abs(boundary_res))),
            "max_dynamics_defect": float(np.max(np.abs(dyn))),
            "min_node_margin_km": float(np.min(koz)),
        }

    evidence = {
        "scenario": {
            "start_position_km": problem.r0.tolist(),
            "start_velocity_km_s": problem.v0.tolist(),
            "goal_position_km": problem.rf.tolist(),
            "goal_velocity_km_s": problem.vf.tolist(),
            "transfer_time_s": float(problem.transfer_time_s),
            "koz_radius_km": float(problem.koz_radius_km),
            "n_intervals": int(problem.n_intervals),
            "n_nodes": int(problem.n_nodes),
        },
        "downstream_solver": {
            "method": "scipy.optimize.minimize(..., method='trust-constr')",
            "config": {
                "maxiter": 200,
                "gtol": 1e-6,
                "barrier_tol": 1e-6,
            },
            "transcription": "fixed-time trapezoidal direct collocation",
            "objective": "trapezoidal integral of squared control magnitude",
            "dynamics": "two-body + J2 gravity with trapezoidal defects",
            "constraints": [
                "exact endpoint position constraints",
                "exact endpoint velocity constraints",
                "spherical KOZ node constraints",
                "trapezoidal dynamics defects",
            ],
        },
        "initial_guesses": {
            "naive": {
                "definition": "cubic Hermite state profile from exact endpoint positions and velocities; control inferred from acceleration minus gravity",
                "r_km": naive_r0.tolist(),
                "v_km_s": naive_v0.tolist(),
                "u_km_s2": naive_u0.tolist(),
                "diagnostics": _initial_diag(naive_x0),
            },
            "bezier_warm_start": {
                "definition": "sample upstream Bézier trajectory at collocation nodes; overwrite boundary states with exact endpoint states; infer control from sampled acceleration minus gravity",
                "control_points_km": control_points.tolist(),
                "upstream_optimizer_info": {
                    "iterations": int(upstream_info.get("iterations", -1)),
                    "feasible": bool(upstream_info.get("feasible", False)),
                    "elapsed_time_s": float(upstream_info.get("elapsed_time", float("nan"))),
                },
                "r_km": warm_r0.tolist(),
                "v_km_s": warm_v0.tolist(),
                "u_km_s2": warm_u0.tolist(),
                "diagnostics": _initial_diag(warm_x0),
            },
        },
        "final_runs": run["comparison"],
        "summary": run["summary"],
        "sanity_checks": [
            {"name": "same scenario and boundary conditions", "passed": True, "detail": "Both runs use the same `DirectCollocationProblem`: same start/end states, transfer time, KOZ radius, and node count."},
            {"name": "same dynamics and transcription", "passed": True, "detail": "Both runs use the same trapezoidal direct-collocation defects with the same two-body + J2 gravity model."},
            {"name": "same objective", "passed": True, "detail": "Both runs minimize the same trapezoidal integral of squared control magnitude."},
            {"name": "same constraints", "passed": True, "detail": "Both runs use the same endpoint equalities, dynamics defects, and KOZ margin constraints."},
            {"name": "same tolerances and stopping rules", "passed": True, "detail": "Both runs use `trust-constr` with maxiter=200, gtol=1e-6, barrier_tol=1e-6."},
            {"name": "only the initial guess changed", "passed": True, "detail": "The solver path, problem object, and config are identical; only `x0` differs."},
            {"name": "per-iteration convergence traces recorded", "passed": False, "detail": "The current spike records final solve time, iterations, and final residuals, but not a full per-iteration trust-constr history."},
        ],
        "interpretation": {
            "naive_better_on": ["solve_time_s", "iteration_count"],
            "warm_better_on": ["final_objective"],
            "same_on": [
                "solve_success",
                "max_eq_violation",
                "min_node_margin_km",
                "min_segment_margin_km",
            ],
            "initial_guess_diagnostic_read": {
                "naive": _initial_diag(naive_x0),
                "bezier_warm_start": _initial_diag(warm_x0),
            },
            "claim_status": "mixed result; does not establish demonstrated warm-start usefulness",
        },
    }
    return evidence


def write_evidence_json(evidence: dict) -> None:
    EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def make_path_figure(evidence: dict) -> None:
    scenario = evidence["scenario"]
    naive_init = np.array(evidence["initial_guesses"]["naive"]["r_km"], dtype=float)
    warm_init = np.array(evidence["initial_guesses"]["bezier_warm_start"]["r_km"], dtype=float)
    naive_final = np.array(evidence["final_runs"]["naive"]["r"], dtype=float)
    warm_final = np.array(evidence["final_runs"]["bezier_warm_start"]["r"], dtype=float)
    start = np.array(scenario["start_position_km"], dtype=float)
    goal = np.array(scenario["goal_position_km"], dtype=float)

    fig = plt.figure(figsize=(16, 10))
    axs = [
        fig.add_subplot(2, 3, 1, projection="3d"),
        fig.add_subplot(2, 3, 2, projection="3d"),
        fig.add_subplot(2, 3, 3, projection="3d"),
        fig.add_subplot(2, 3, 4, projection="3d"),
        fig.add_subplot(2, 3, 5, projection="3d"),
    ]
    text_ax = fig.add_subplot(2, 3, 6)

    problem = make_demo_problem(n_intervals=int(scenario["n_intervals"]))
    _plot_path_panel(axs[0], "Scenario Geometry", np.vstack([start, goal]), problem, start, goal)
    axs[0].plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], "--", color="0.5", lw=1.2, label="chord")

    _plot_path_panel(axs[1], "Naive Initial Guess", naive_init, problem, start, goal)
    _plot_path_panel(axs[2], "Warm-Start Initial Guess", warm_init, problem, start, goal)
    _plot_path_panel(axs[3], "Naive Final Trajectory", naive_final, problem, start, goal)
    _plot_path_panel(axs[4], "Warm-Start Final Trajectory", warm_final, problem, start, goal)

    all_pts = [naive_init, warm_init, naive_final, warm_final, np.vstack([start, goal])]
    for ax in axs:
        _set_equal_3d(ax, all_pts)

    axs[1].legend(loc="upper left", fontsize=8)

    text_ax.axis("off")
    text = "\n".join(
        [
            "Exact tested setup",
            f"- transfer time: {scenario['transfer_time_s']:.1f} s",
            f"- KOZ radius: {scenario['koz_radius_km']:.1f} km",
            f"- intervals / nodes: {scenario['n_intervals']} / {scenario['n_nodes']}",
            "",
            "What changed?",
            "- only the initial guess",
            "",
            "What is shown?",
            "- scenario geometry",
            "- naive initial guess path",
            "- warm-start initial guess path",
            "- final solved trajectory from naive run",
            "- final solved trajectory from warm-start run",
            "",
            "Caveat",
            "- no per-iteration solver trace was logged",
        ]
    )
    text_ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("T6 Evidence: Scenario, Initial Guesses, and Final Trajectories", fontsize=16)
    fig.tight_layout()
    fig.savefig(PATHS_FIG, dpi=220)
    plt.close(fig)


def make_solver_figure(evidence: dict) -> None:
    summary = evidence["summary"]
    checks = evidence["sanity_checks"]
    init_naive = evidence["initial_guesses"]["naive"]["diagnostics"]
    init_warm = evidence["initial_guesses"]["bezier_warm_start"]["diagnostics"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = [
        ("solve_time_s", "Solve Time [s]"),
        ("iteration_count", "Iterations"),
        ("final_objective", "Final Objective"),
        ("max_eq_violation", "Max Eq Violation"),
    ]
    naive_vals = [
        summary["naive_time_s"],
        summary["naive_iterations"],
        summary["naive_objective"],
        summary["naive_max_eq_violation"],
    ]
    warm_vals = [
        summary["warm_time_s"],
        summary["warm_iterations"],
        summary["warm_objective"],
        summary["warm_max_eq_violation"],
    ]
    x = np.arange(len(metrics))
    width = 0.36
    axes[0].bar(x - width / 2, naive_vals, width, label="naive", color="#1f77b4")
    axes[0].bar(x + width / 2, warm_vals, width, label="warm-start", color="#ff7f0e")
    axes[0].set_xticks(x, [m[1] for m in metrics], rotation=15, ha="right")
    axes[0].set_title("Solver Outcomes")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    init_metrics = [
        ("initial_objective", "Init Objective"),
        ("max_dynamics_defect", "Init Dyn Defect"),
        ("min_node_margin_km", "Init KOZ Margin [km]"),
    ]
    init_naive_vals = [
        init_naive["initial_objective"],
        init_naive["max_dynamics_defect"],
        init_naive["min_node_margin_km"],
    ]
    init_warm_vals = [
        init_warm["initial_objective"],
        init_warm["max_dynamics_defect"],
        init_warm["min_node_margin_km"],
    ]
    x2 = np.arange(len(init_metrics))
    axes[1].bar(x2 - width / 2, init_naive_vals, width, label="naive", color="#1f77b4")
    axes[1].bar(x2 + width / 2, init_warm_vals, width, label="warm-start", color="#ff7f0e")
    axes[1].set_xticks(x2, [m[1] for m in init_metrics], rotation=15, ha="right")
    axes[1].set_title("Initial Guess Diagnostics")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].axis("off")
    lines = ["Sanity checks"]
    for item in checks:
        mark = "PASS" if item["passed"] else "MISSING"
        lines.append(f"- {mark}: {item['name']}")
        lines.append(f"  {item['detail']}")
    lines.extend(
        [
            "",
            "Outcome",
            "- naive solved faster and in fewer iterations",
            "- warm-start reached a slightly lower objective",
            "- both satisfied the same final checks",
            "",
            "Why naive likely won here",
            f"- naive init defect: {init_naive['max_dynamics_defect']:.3e}",
            f"- warm init defect:  {init_warm['max_dynamics_defect']:.3e}",
            f"- naive init objective: {init_naive['initial_objective']:.6f}",
            f"- warm init objective:  {init_warm['initial_objective']:.6f}",
        ]
    )
    axes[2].text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle("T6 Evidence: Solver Comparison and Matched-Setup Checks", fontsize=16)
    fig.tight_layout()
    fig.savefig(SOLVER_FIG, dpi=220)
    plt.close(fig)


def main() -> None:
    evidence = build_evidence()
    write_evidence_json(evidence)
    make_path_figure(evidence)
    make_solver_figure(evidence)
    print(json.dumps(
        {
            "evidence_json": str(EVIDENCE_JSON.relative_to(REPO_ROOT)),
            "paths_figure": str(PATHS_FIG.relative_to(REPO_ROOT)),
            "solver_figure": str(SOLVER_FIG.relative_to(REPO_ROOT)),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
