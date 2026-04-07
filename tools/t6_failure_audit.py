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

from orbital_docking.bezier import BezierCurve  # noqa: E402
from orbital_docking.downstream_collocation import (  # noqa: E402
    _build_boundary_constraint,
    _dynamics_defects,
    _koz_margins,
    _unpack_variables,
    build_bezier_warm_start,
    build_demo_bezier_warm_start,
    build_naive_initial_guess,
    make_demo_problem,
)


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "paper_artifacts"
FIGURE_DIR = REPO_ROOT / "figures"
EVIDENCE_JSON = ARTIFACT_DIR / "t6_evidence_pack.json"
GATE_JSON = ARTIFACT_DIR / "t6_gate_run.json"
AUDIT_JSON = ARTIFACT_DIR / "t6_failure_audit.json"
AUDIT_FIG = FIGURE_DIR / "t6_failure_audit.png"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pack_variables(r: np.ndarray, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.concatenate([r.reshape(-1), v.reshape(-1), u.reshape(-1)])


def _per_interval_defects(x: np.ndarray, problem) -> dict:
    defects = _dynamics_defects(x, problem).reshape(problem.n_intervals, 2, 3)
    pos = defects[:, 0, :]
    vel = defects[:, 1, :]
    return {
        "position_defect_norms": np.linalg.norm(pos, axis=1).tolist(),
        "velocity_defect_norms": np.linalg.norm(vel, axis=1).tolist(),
        "max_position_defect_norm": float(np.max(np.linalg.norm(pos, axis=1))),
        "max_velocity_defect_norm": float(np.max(np.linalg.norm(vel, axis=1))),
        "first_interval": {
            "position": pos[0].tolist(),
            "velocity": vel[0].tolist(),
        },
        "last_interval": {
            "position": pos[-1].tolist(),
            "velocity": vel[-1].tolist(),
        },
    }


def _boundary_violation(x: np.ndarray, problem) -> dict:
    bc = _build_boundary_constraint(problem)
    residual = (bc.A @ x) - bc.lb
    return {
        "max_abs_residual": float(np.max(np.abs(residual))),
        "residual_vector": residual.tolist(),
    }


def _raw_bezier_sample(problem, control_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    curve = BezierCurve(control_points)
    taus = problem.tau_grid
    r = np.array([curve.point(t) for t in taus])
    v = np.array([curve.velocity(t) for t in taus]) / problem.transfer_time_s
    a = np.array([curve.acceleration(t) for t in taus]) / (problem.transfer_time_s**2)
    return taus, r, v, a


def _make_raw_warm_guess(problem, control_points: np.ndarray) -> np.ndarray:
    _, r, v, a = _raw_bezier_sample(problem, control_points)
    # Import lazily to avoid widening the main dependency surface.
    from orbital_docking.downstream_collocation import _gravity_accel

    u = np.array([a_i - _gravity_accel(r_i) for r_i, a_i in zip(r, a)])
    return _pack_variables(r, v, u)


def _export_mismatch(problem, control_points: np.ndarray) -> dict:
    _, r_raw, v_raw, a_raw = _raw_bezier_sample(problem, control_points)
    warm_overwritten = build_bezier_warm_start(problem, control_points)
    r_over, v_over, u_over = _unpack_variables(warm_overwritten, problem.n_nodes)

    raw_u = []
    from orbital_docking.downstream_collocation import _gravity_accel

    for r_i, a_i in zip(r_raw, a_raw):
        raw_u.append(a_i - _gravity_accel(r_i))
    raw_u = np.array(raw_u)

    return {
        "raw_endpoint_position_error_km": {
            "start": float(np.linalg.norm(r_raw[0] - problem.r0)),
            "end": float(np.linalg.norm(r_raw[-1] - problem.rf)),
        },
        "raw_endpoint_velocity_error_km_s": {
            "start": float(np.linalg.norm(v_raw[0] - problem.v0)),
            "end": float(np.linalg.norm(v_raw[-1] - problem.vf)),
        },
        "overwrite_change_norms": {
            "start_position_change_km": float(np.linalg.norm(r_over[0] - r_raw[0])),
            "end_position_change_km": float(np.linalg.norm(r_over[-1] - r_raw[-1])),
            "start_velocity_change_km_s": float(np.linalg.norm(v_over[0] - v_raw[0])),
            "end_velocity_change_km_s": float(np.linalg.norm(v_over[-1] - v_raw[-1])),
        },
        "raw_sample_initial_guess": {
            "boundary_violation": _boundary_violation(_pack_variables(r_raw, v_raw, raw_u), problem),
            "per_interval_defects": _per_interval_defects(_pack_variables(r_raw, v_raw, raw_u), problem),
            "min_node_margin_km": float(np.min(_koz_margins(_pack_variables(r_raw, v_raw, raw_u), problem))),
        },
        "overwritten_warm_start": {
            "boundary_violation": _boundary_violation(warm_overwritten, problem),
            "per_interval_defects": _per_interval_defects(warm_overwritten, problem),
            "min_node_margin_km": float(np.min(_koz_margins(warm_overwritten, problem))),
        },
    }


def build_failure_audit() -> dict:
    evidence = _load_json(EVIDENCE_JSON)
    gate = _load_json(GATE_JSON)

    problem = make_demo_problem(n_intervals=int(evidence["scenario"]["n_intervals"]))
    control_points, upstream_info = build_demo_bezier_warm_start(
        degree=int(gate["upstream_warm_start"]["degree"]),
        n_seg=int(gate["upstream_warm_start"]["n_seg"]),
        objective_mode=str(gate["upstream_warm_start"]["objective"]),
        max_iter=int(gate["upstream_warm_start"]["optimizer_info"]["iterations"]),
        tol=1e-8,
        use_cache=True,
        ignore_existing_cache=False,
    )

    naive_x0 = build_naive_initial_guess(problem)
    warm_x0 = build_bezier_warm_start(problem, control_points)

    export = _export_mismatch(problem, control_points)

    naive_per_interval = _per_interval_defects(naive_x0, problem)
    warm_per_interval = _per_interval_defects(warm_x0, problem)

    hypotheses = {
        "F1_export_inconsistency": {
            "status": "confirmed",
            "reason": "The sampled upstream Bézier endpoint velocities do not match the downstream fixed endpoint velocities, and the overwritten warm start still carries large interval defects.",
        },
        "F2_comparison_fairness_mismatch": {
            "status": "confirmed",
            "reason": "The downstream solver call is matched, but the upstream warm-start source solves a different problem: no downstream velocity BCs, node-vs-halfspace KOZ mismatch, and upstream prograde bias absent downstream.",
        },
        "F3_downstream_problem_too_weak": {
            "status": "confirmed",
            "reason": "The downstream NLP enforces node KOZ and discrete dynamics but no thrust bounds, no continuous KOZ guarantee, and no orbital-credibility constraints; final solutions remain near-stalling / retrograde relative to local prograde circular reference.",
        },
        "F4_only_visual_or_reference_issue": {
            "status": "rejected",
            "reason": "The pathology is not merely visual: the velocity audit shows extreme speed-ratio collapse and retrograde nodes, while the code confirms nothing in the downstream problem forbids such behavior.",
        },
    }

    confirmed = [
        "Same downstream problem object, solver, tolerances, objective, and constraints are used for naive and warm-start runs.",
        "Only x0 changes inside run_downstream_comparison.",
        "Warm-start initial max dynamics defect is about 404.6 versus about 3.0 for naive.",
        "Upstream warm-start source does not enforce downstream endpoint velocity constraints (v0=None, v1=None) and does enforce prograde shaping (enforce_prograde=True).",
        "Final naive and warm solutions satisfy the downstream discrete constraints to numerical tolerance.",
        "Both final solutions still show near-stall / retrograde behavior relative to a local prograde circular reference.",
    ]
    suspected = [
        "Endpoint overwrite is a major contributor to the warm-start defect spike, especially near the first and last intervals.",
        "The warm start is penalized partly because it is generated from a different feasible set than the downstream NLP.",
    ]
    unknown = [
        "Whether the final trajectories violate continuous-time dynamics between nodes rather than only looking non-orbital under a reference audit.",
        "Whether a denser downstream mesh would materially change the same qualitative pathology.",
        "Whether the solver iteration gap is dominated by export inconsistency or by general warm-start mismatch to the downstream feasible set.",
    ]

    ranked_root_causes = [
        {
            "rank": 1,
            "cause": "Warm-start export is not transcription-consistent with the downstream direct-collocation defect equations.",
            "evidence": [
                f"Raw sampled upstream endpoint velocity mismatch start/end = {export['raw_endpoint_velocity_error_km_s']['start']:.3f} / {export['raw_endpoint_velocity_error_km_s']['end']:.3f} km/s",
                f"Overwritten warm-start max position/velocity defect norms = {export['overwritten_warm_start']['per_interval_defects']['max_position_defect_norm']:.3f} / {export['overwritten_warm_start']['per_interval_defects']['max_velocity_defect_norm']:.3f}",
            ],
        },
        {
            "rank": 2,
            "cause": "The comparison is only matched at the downstream call level; the upstream warm-start source solves a materially different problem.",
            "evidence": [
                "Upstream solve uses v0=None and v1=None.",
                "Upstream solve uses enforce_prograde=True, which downstream does not replicate.",
                "Upstream KOZ handling differs from downstream node-only spherical KOZ constraints.",
            ],
        },
        {
            "rank": 3,
            "cause": "The downstream direct-collocation problem is too weak to enforce physically credible orbital-transfer behavior.",
            "evidence": [
                "No thrust bounds, no continuous KOZ guarantee, no prograde or orbital-credibility constraints.",
                "Final curves still have retrograde nodes and minimum speed ratios near 0.05 versus local circular speed.",
            ],
        },
        {
            "rank": 4,
            "cause": "Visualization ambiguity is secondary, not primary.",
            "evidence": [
                "Velocity audit reproduces the issue numerically from stored r,v arrays.",
                "The odd behavior is present in raw metrics, not only in 3D plots.",
            ],
        },
    ]

    missing_diagnostics = [
        "Dense between-node continuous-time replay of the final trajectories under the same control and gravity model.",
        "Per-iteration trust-constr history for the downstream runs.",
        "Mesh-refinement sensitivity check for the same downstream problem.",
        "A direct comparison of raw sampled warm-start defects before and after endpoint overwrite, split into first/last vs interior intervals.",
    ]

    return {
        "hypotheses": hypotheses,
        "evidence_map": {
            "confirmed": confirmed,
            "suspected": suspected,
            "unknown": unknown,
        },
        "export_consistency": {
            "upstream_optimizer_info": {
                "iterations": int(upstream_info.get("iterations", -1)),
                "feasible": bool(upstream_info.get("feasible", False)),
            },
            "raw_endpoint_velocity_error_km_s": export["raw_endpoint_velocity_error_km_s"],
            "overwrite_change_norms": export["overwrite_change_norms"],
            "naive_initial_per_interval": naive_per_interval,
            "warm_initial_per_interval": warm_per_interval,
            "raw_sample_initial_per_interval": export["raw_sample_initial_guess"]["per_interval_defects"],
        },
        "fairness_audit": {
            "downstream_matched": True,
            "upstream_problem_match": False,
            "mismatch_sources": [
                "Upstream endpoint velocity constraints absent while downstream endpoint velocity constraints are exact.",
                "Upstream prograde shaping present while downstream prograde shaping is absent.",
                "Upstream KOZ construction differs from downstream node-only KOZ constraints.",
            ],
        },
        "model_adequacy": {
            "node_only_koz": True,
            "thrust_bounds_present": False,
            "continuous_path_guarantee_present": False,
            "orbital_credibility_constraints_present": False,
            "final_curve_pathology_reference": {
                "naive_final": {
                    "min_speed_ratio": 0.04804783466059857,
                    "retrograde_node_count": 3,
                    "max_angle_to_prograde_deg": 165.57371765411824,
                },
                "warm_final": {
                    "min_speed_ratio": 0.054586830758615285,
                    "retrograde_node_count": 3,
                    "max_angle_to_prograde_deg": 165.6421695529449,
                },
            },
        },
        "missing_diagnostics": missing_diagnostics,
        "ranked_root_causes": ranked_root_causes,
    }


def write_audit_json(audit: dict) -> None:
    AUDIT_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def make_audit_figure(audit: dict) -> None:
    naive = audit["export_consistency"]["naive_initial_per_interval"]
    warm = audit["export_consistency"]["warm_initial_per_interval"]
    raw = audit["export_consistency"]["raw_sample_initial_per_interval"]
    idx = np.arange(1, len(naive["position_defect_norms"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(idx, naive["position_defect_norms"], "-o", label="naive init")
    axes[0, 0].plot(idx, raw["position_defect_norms"], "-o", label="warm raw sample")
    axes[0, 0].plot(idx, warm["position_defect_norms"], "-o", label="warm overwritten")
    axes[0, 0].set_title("Position Defect Norm by Interval")
    axes[0, 0].set_xlabel("interval")
    axes[0, 0].set_ylabel("km")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(idx, naive["velocity_defect_norms"], "-o", label="naive init")
    axes[0, 1].plot(idx, raw["velocity_defect_norms"], "-o", label="warm raw sample")
    axes[0, 1].plot(idx, warm["velocity_defect_norms"], "-o", label="warm overwritten")
    axes[0, 1].set_title("Velocity Defect Norm by Interval")
    axes[0, 1].set_xlabel("interval")
    axes[0, 1].set_ylabel("km/s")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    overwrite = audit["export_consistency"]["overwrite_change_norms"]
    endpoint_v = audit["export_consistency"]["raw_endpoint_velocity_error_km_s"]
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.98,
        "\n".join(
            [
                "Warm-start export mismatch",
                f"- raw start velocity error: {endpoint_v['start']:.3f} km/s",
                f"- raw end velocity error:   {endpoint_v['end']:.3f} km/s",
                f"- start overwrite change:   {overwrite['start_velocity_change_km_s']:.3f} km/s",
                f"- end overwrite change:     {overwrite['end_velocity_change_km_s']:.3f} km/s",
                "",
                "Interpretation",
                "- the sampled upstream curve does not satisfy downstream endpoint velocity BCs",
                "- overwrite repairs boundary states but leaves the interior on a different trajectory",
                "- the resulting x0 is not transcription-consistent",
            ]
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    axes[1, 1].axis("off")
    ranked = audit["ranked_root_causes"]
    lines = ["Ranked root causes"]
    for item in ranked:
        lines.append(f"{item['rank']}. {item['cause']}")
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )

    fig.suptitle("T6 Failure Audit: Export Consistency and Root Causes", fontsize=16)
    fig.tight_layout()
    fig.savefig(AUDIT_FIG, dpi=220)
    plt.close(fig)


def main() -> None:
    audit = build_failure_audit()
    write_audit_json(audit)
    make_audit_figure(audit)
    print(json.dumps(
        {
            "audit_json": str(AUDIT_JSON.relative_to(REPO_ROOT)),
            "audit_figure": str(AUDIT_FIG.relative_to(REPO_ROOT)),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
