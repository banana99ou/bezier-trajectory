#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "paper_artifacts"
FIGURE_DIR = REPO_ROOT / "figures"
EVIDENCE_JSON = ARTIFACT_DIR / "t6_evidence_pack.json"
OUTPUT_HTML = FIGURE_DIR / "t6_interactive_paths.html"


def _load_evidence() -> dict:
    if not EVIDENCE_JSON.exists():
        raise FileNotFoundError(
            f"Missing evidence pack: {EVIDENCE_JSON}. Run tools/t6_evidence_pack.py first."
        )
    return json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))


def _sphere_surface(radius: float, color_scale: str = "Blues") -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.18,
        showscale=False,
        colorscale=color_scale,
        hoverinfo="skip",
        name="KOZ sphere",
    )


def _path_trace(path: np.ndarray, name: str, color: str) -> go.Scatter3d:
    radii = np.linalg.norm(path, axis=1)
    hover = [
        f"{name}<br>node={i}<br>x={p[0]:.1f} km<br>y={p[1]:.1f} km<br>z={p[2]:.1f} km<br>r={r:.1f} km"
        for i, (p, r) in enumerate(zip(path, radii))
    ]
    return go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=path[:, 2],
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 6},
        marker={"size": 4, "color": color},
        text=hover,
        hoverinfo="text",
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
        hovertemplate=(
            f"{name}<br>x={point[0]:.1f} km<br>y={point[1]:.1f} km<br>z={point[2]:.1f} km<extra></extra>"
        ),
    )


def _scene_layout() -> dict:
    return {
        "xaxis_title": "x [km]",
        "yaxis_title": "y [km]",
        "zaxis_title": "z [km]",
        "aspectmode": "data",
        "camera": {
            "eye": {"x": 1.5, "y": 1.35, "z": 0.8}
        },
    }


def _summary_html(evidence: dict) -> str:
    scenario = evidence["scenario"]
    naive = evidence["summary"]
    init_naive = evidence["initial_guesses"]["naive"]["diagnostics"]
    init_warm = evidence["initial_guesses"]["bezier_warm_start"]["diagnostics"]
    return (
        f"<b>Matched setup</b><br>"
        f"transfer time: {scenario['transfer_time_s']:.1f} s<br>"
        f"KOZ radius: {scenario['koz_radius_km']:.1f} km<br>"
        f"intervals / nodes: {scenario['n_intervals']} / {scenario['n_nodes']}<br>"
        f"solver: trust-constr<br><br>"
        f"<b>Final outcomes</b><br>"
        f"naive: {naive['naive_time_s']:.4f} s, {naive['naive_iterations']} iters, obj={naive['naive_objective']:.6f}<br>"
        f"warm: {naive['warm_time_s']:.4f} s, {naive['warm_iterations']} iters, obj={naive['warm_objective']:.6f}<br><br>"
        f"<b>Why the warm start struggled</b><br>"
        f"naive initial defect: {init_naive['max_dynamics_defect']:.3e}<br>"
        f"warm initial defect: {init_warm['max_dynamics_defect']:.3e}<br>"
        f"naive initial objective: {init_naive['initial_objective']:.6f}<br>"
        f"warm initial objective: {init_warm['initial_objective']:.6f}<br><br>"
        f"<b>Caveat</b><br>"
        f"No per-iteration trust-constr trace was logged."
    )


def main() -> None:
    evidence = _load_evidence()
    scenario = evidence["scenario"]
    start = np.array(scenario["start_position_km"], dtype=float)
    goal = np.array(scenario["goal_position_km"], dtype=float)
    koz_r = float(scenario["koz_radius_km"])

    naive_init = np.array(evidence["initial_guesses"]["naive"]["r_km"], dtype=float)
    warm_init = np.array(evidence["initial_guesses"]["bezier_warm_start"]["r_km"], dtype=float)
    naive_final = np.array(evidence["final_runs"]["naive"]["r"], dtype=float)
    warm_final = np.array(evidence["final_runs"]["bezier_warm_start"]["r"], dtype=float)

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Naive Initial Guess",
            "Warm-Start Initial Guess",
            "Naive Final Trajectory",
            "Warm-Start Final Trajectory",
        ),
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    panels = [
        (1, 1, naive_init, "Naive Init", "#1f77b4"),
        (1, 2, warm_init, "Warm Init", "#ff7f0e"),
        (2, 1, naive_final, "Naive Final", "#2ca02c"),
        (2, 2, warm_final, "Warm Final", "#d62728"),
    ]

    for row, col, path, name, color in panels:
        fig.add_trace(_sphere_surface(koz_r), row=row, col=col)
        fig.add_trace(_path_trace(path, name, color), row=row, col=col)
        fig.add_trace(_endpoint_trace(start, "start", "green"), row=row, col=col)
        fig.add_trace(_endpoint_trace(goal, "goal", "red"), row=row, col=col)
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], goal[0]],
                y=[start[1], goal[1]],
                z=[start[2], goal[2]],
                mode="lines",
                line={"color": "gray", "width": 3, "dash": "dash"},
                name="start-goal chord",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="T6 Interactive Path Audit",
        template="plotly_white",
        height=980,
        width=1280,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        annotations=list(fig.layout.annotations) + [
            {
                "text": _summary_html(evidence),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.08,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
                "bordercolor": "rgba(0,0,0,0.15)",
                "borderwidth": 1,
                "bgcolor": "rgba(245,245,245,0.9)",
            }
        ],
    )

    fig.update_scenes(**_scene_layout())
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs=True, full_html=True)
    print(json.dumps({"interactive_html": str(OUTPUT_HTML.relative_to(REPO_ROOT))}, indent=2))


if __name__ == "__main__":
    main()
