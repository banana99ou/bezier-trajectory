#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orbital_docking.dymos_t6 import DymosT6Config, run_dymos_t6_experiment  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "paper_artifacts"
FIGURE_DIR = REPO_ROOT / "figures"
SWEEP_JSON = ARTIFACT_DIR / "t6_dymos_time_sweep.json"
INIT_HTML = FIGURE_DIR / "t6_dymos_time_sweep_init.html"
FINAL_HTML = FIGURE_DIR / "t6_dymos_time_sweep_final.html"
TIME_CANDIDATES_S = [900.0, 1200.0, 1500.0, 1800.0, 2100.0]


def _sphere_surface(radius: float) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.12,
        showscale=False,
        colorscale="Blues",
        hoverinfo="skip",
        showlegend=False,
    )


def _path_trace(path: np.ndarray, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=path[:, 2],
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 5},
        marker={"size": 2.8, "color": color},
    )


def _endpoint_trace(point: np.ndarray, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[point[0]],
        y=[point[1]],
        z=[point[2]],
        mode="markers+text",
        name=name,
        marker={"size": 6, "color": color},
        text=[name],
        textposition="top center",
        showlegend=False,
    )


def _scene_layout() -> dict:
    return {
        "xaxis_title": "x [km]",
        "yaxis_title": "y [km]",
        "zaxis_title": "z [km]",
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.35, "y": 1.2, "z": 0.85}},
    }


def _subplot_grid(n: int) -> tuple[int, int]:
    cols = 3
    rows = int(np.ceil(n / cols))
    return rows, cols


def _case_title(case: dict) -> str:
    result = case["result"]
    t_s = int(round(case["transfer_time_s"]))
    naive = result["summary"]["naive"]
    warm = result["summary"]["warm_start"]
    return (
        f"{t_s} s"
        f"<br>naive sr={naive['min_speed_ratio']:.2f}, warm sr={warm['min_speed_ratio']:.2f}"
    )


def _make_paths_html(cases: list[dict], *, path_kind: str, output_path: Path, title: str) -> None:
    rows, cols = _subplot_grid(len(cases))
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    titles = [_case_title(case) for case in cases] + [""] * (rows * cols - len(cases))
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=tuple(titles))

    showlegend = True
    for idx, case in enumerate(cases):
        row = idx // cols + 1
        col = idx % cols + 1
        result = case["result"]
        scenario = result["experiment_spec"]["scenario"]
        start = np.array(scenario["start_position_km"], dtype=float)
        goal = np.array(scenario["goal_position_km"], dtype=float)
        koz_r = float(scenario["koz_radius_km"])
        if path_kind == "initial":
            naive_path = np.array(result["initial_guesses"]["naive"]["r_km"], dtype=float)
            warm_path = np.array(result["initial_guesses"]["warm_start"]["r_km"], dtype=float)
            naive_name = "Naive init"
            warm_name = "Warm init"
        else:
            naive_path = np.array(result["runs"]["naive"]["timeseries"]["simulate"]["r_km"], dtype=float)
            warm_path = np.array(result["runs"]["warm_start"]["timeseries"]["simulate"]["r_km"], dtype=float)
            naive_name = "Naive final"
            warm_name = "Warm final"

        fig.add_trace(_sphere_surface(koz_r), row=row, col=col)
        fig.add_trace(
            _path_trace(naive_path, naive_name, "#1f77b4").update(showlegend=showlegend),
            row=row,
            col=col,
        )
        fig.add_trace(
            _path_trace(warm_path, warm_name, "#ff7f0e").update(showlegend=showlegend),
            row=row,
            col=col,
        )
        fig.add_trace(_endpoint_trace(start, "start", "green"), row=row, col=col)
        fig.add_trace(_endpoint_trace(goal, "goal", "red"), row=row, col=col)
        showlegend = False

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=420 * rows,
        width=1400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 20, "r": 20, "t": 90, "b": 30},
    )
    fig.update_scenes(**_scene_layout())
    output_path.write_text(fig.to_html(include_plotlyjs=True, full_html=True), encoding="utf-8")


def _summary_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        result = case["result"]
        rows.append(
            {
                "transfer_time_s": case["transfer_time_s"],
                "comparison_valid": result["validity_gates"]["comparison_valid"],
                "objective_recompute_gate": result["validity_gates"]["objective_recompute_gate"],
                "orbital_credibility_gate": result["validity_gates"]["orbital_credibility_gate"],
                "naive_success": result["summary"]["naive"]["success"],
                "warm_success": result["summary"]["warm_start"]["success"],
                "naive_runtime_s": result["summary"]["naive"]["runtime_s"],
                "warm_runtime_s": result["summary"]["warm_start"]["runtime_s"],
                "naive_min_speed_ratio": result["summary"]["naive"]["min_speed_ratio"],
                "warm_min_speed_ratio": result["summary"]["warm_start"]["min_speed_ratio"],
                "naive_retrograde_nodes": result["summary"]["naive"]["retrograde_node_count"],
                "warm_retrograde_nodes": result["summary"]["warm_start"]["retrograde_node_count"],
                "artifact_dir": case["artifact_dir"],
            }
        )
    return rows


def run_sweep() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    base_config = DymosT6Config()
    for transfer_time_s in TIME_CANDIDATES_S:
        case_dir = ARTIFACT_DIR / f"t6_dymos_time_{int(round(transfer_time_s))}s"
        config = replace(base_config, transfer_time_s=float(transfer_time_s))
        result = run_dymos_t6_experiment(artifact_dir=case_dir, config=config)
        cases.append(
            {
                "transfer_time_s": float(transfer_time_s),
                "artifact_dir": str(case_dir.relative_to(REPO_ROOT)),
                "result": result,
            }
        )

    bundle = {
        "time_candidates_s": TIME_CANDIDATES_S,
        "cases": cases,
        "summary_rows": _summary_rows(cases),
        "figures": {
            "initial_paths_html": str(INIT_HTML.relative_to(REPO_ROOT)),
            "final_paths_html": str(FINAL_HTML.relative_to(REPO_ROOT)),
        },
    }
    SWEEP_JSON.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    _make_paths_html(cases, path_kind="initial", output_path=INIT_HTML, title="T6 Dymos Initial Paths Across Transfer Times")
    _make_paths_html(cases, path_kind="final", output_path=FINAL_HTML, title="T6 Dymos Final Simulated Paths Across Transfer Times")
    return bundle


def main() -> None:
    bundle = run_sweep()
    print(
        json.dumps(
            {
                "sweep_json": str(SWEEP_JSON.relative_to(REPO_ROOT)),
                "initial_paths_html": str(INIT_HTML.relative_to(REPO_ROOT)),
                "final_paths_html": str(FINAL_HTML.relative_to(REPO_ROOT)),
                "time_candidates_s": bundle["time_candidates_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
