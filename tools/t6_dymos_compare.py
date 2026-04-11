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

from orbital_docking.dymos_t6 import (  # noqa: E402
    DymosT6Config,
    run_dymos_t6_experiment,
    validity_memo_markdown,
)
from orbital_docking import constants  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "paper_artifacts"
DOC_DIR = REPO_ROOT / "doc"
FIGURE_DIR = REPO_ROOT / "figures"
RESULT_JSON = ARTIFACT_DIR / "t6_dymos_experiment.json"
VALIDITY_MEMO = DOC_DIR / "t6_dymos_validity_memo.md"
PROTOCOL_MEMO = DOC_DIR / "t6_dymos_protocol.md"
PATHS_HTML = FIGURE_DIR / "t6_dymos_paths.html"
VELOCITY_HTML = FIGURE_DIR / "t6_dymos_velocity.html"


def _sphere_surface(radius: float) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.16,
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
        marker={"size": 3.5, "color": color},
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
        "camera": {"eye": {"x": 1.45, "y": 1.3, "z": 0.8}},
    }


def _summary_html(result: dict) -> str:
    gates = result["validity_gates"]
    summary = result["summary"]
    return (
        f"<b>Validity</b><br>"
        f"comparison valid: {gates['comparison_valid']}<br>"
        f"physics gate: {gates['physics_gate']}<br>"
        f"fairness gate: {gates['fairness_gate']}<br>"
        f"objective gate: {gates['objective_recompute_gate']}<br>"
        f"dense KOZ gate: {gates['dense_koz_gate']}<br>"
        f"orbital credibility gate: {gates['orbital_credibility_gate']}<br><br>"
        f"<b>Naive</b><br>"
        f"success={summary['naive']['success']}, iters={summary['naive']['iter_count']}, "
        f"time={summary['naive']['runtime_s']:.3f}s, obj={summary['naive']['objective']:.6f}<br>"
        f"<b>Warm</b><br>"
        f"success={summary['warm_start']['success']}, iters={summary['warm_start']['iter_count']}, "
        f"time={summary['warm_start']['runtime_s']:.3f}s, obj={summary['warm_start']['objective']:.6f}"
    )


def make_paths_html(result: dict) -> None:
    start = np.array(result["experiment_spec"]["scenario"]["start_position_km"], dtype=float)
    goal = np.array(result["experiment_spec"]["scenario"]["goal_position_km"], dtype=float)
    koz_r = float(result["experiment_spec"]["scenario"]["koz_radius_km"])

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            "Naive Initial Guess",
            "Warm Initial Guess",
            "Naive Final (Dymos)",
            "Warm Final (Dymos)",
        ),
    )

    panels = [
        (1, 1, np.array(result["initial_guesses"]["naive"]["r_km"], dtype=float), "Naive Init", "#1f77b4"),
        (1, 2, np.array(result["initial_guesses"]["warm_start"]["r_km"], dtype=float), "Warm Init", "#ff7f0e"),
        (2, 1, np.array(result["runs"]["naive"]["timeseries"]["collocation"]["r_km"], dtype=float), "Naive Final", "#2ca02c"),
        (2, 2, np.array(result["runs"]["warm_start"]["timeseries"]["collocation"]["r_km"], dtype=float), "Warm Final", "#d62728"),
    ]
    for row, col, path, name, color in panels:
        fig.add_trace(_sphere_surface(koz_r), row=row, col=col)
        fig.add_trace(_path_trace(path, name, color), row=row, col=col)
        fig.add_trace(_endpoint_trace(start, "start", "green"), row=row, col=col)
        fig.add_trace(_endpoint_trace(goal, "goal", "red"), row=row, col=col)

    fig.update_layout(
        title="T6 Dymos Path Comparison",
        template="plotly_white",
        height=980,
        width=1280,
        annotations=list(fig.layout.annotations) + [
            {
                "text": _summary_html(result),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.08,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
                "bordercolor": "rgba(0,0,0,0.15)",
                "borderwidth": 1,
                "bgcolor": "rgba(245,245,245,0.92)",
            }
        ],
    )
    fig.update_scenes(**_scene_layout())
    fig.write_html(str(PATHS_HTML), include_plotlyjs=True, full_html=True)


def make_velocity_html(result: dict) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Speed vs Circular Speed",
            "Prograde Velocity",
            "Radial Velocity",
            "Angle to Prograde",
        ),
    )
    colors = {"naive": "#1f77b4", "warm_start": "#ff7f0e"}
    labels = {"naive": "Naive Final", "warm_start": "Warm Final"}
    for key in ("naive", "warm_start"):
        audit = result["runs"][key]["velocity_audit"]["simulate"]
        x = list(range(len(audit["speed_km_s"])))
        color = colors[key]
        label = labels[key]
        fig.add_trace(go.Scatter(x=x, y=audit["speed_km_s"], mode="lines+markers", name=f"{label} speed", line={"color": color}), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=audit["circular_speed_km_s"], mode="lines", name=f"{label} circ", line={"color": color, "dash": "dash"}, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=audit["prograde_velocity_km_s"], mode="lines+markers", name=f"{label} prograde", line={"color": color}), row=1, col=2)
        fig.add_trace(go.Scatter(x=x, y=audit["radial_velocity_km_s"], mode="lines+markers", name=f"{label} radial", line={"color": color}), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=audit["angle_to_prograde_deg"], mode="lines+markers", name=f"{label} angle", line={"color": color}), row=2, col=2)

    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=1, col=2)
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=90.0, line_dash="dot", line_color="gray", row=2, col=2)
    fig.update_layout(
        title="T6 Dymos Velocity Audit",
        template="plotly_white",
        height=900,
        width=1300,
        annotations=list(fig.layout.annotations) + [
            {
                "text": _summary_html(result),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.13,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
                "bordercolor": "rgba(0,0,0,0.15)",
                "borderwidth": 1,
                "bgcolor": "rgba(245,245,245,0.92)",
            }
        ],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 20, "t": 90, "b": 130},
    )
    fig.update_xaxes(title_text="Node index", row=1, col=1)
    fig.update_xaxes(title_text="Node index", row=1, col=2)
    fig.update_xaxes(title_text="Node index", row=2, col=1)
    fig.update_xaxes(title_text="Node index", row=2, col=2)
    fig.update_yaxes(title_text="km/s", row=1, col=1)
    fig.update_yaxes(title_text="km/s", row=1, col=2)
    fig.update_yaxes(title_text="km/s", row=2, col=1)
    fig.update_yaxes(title_text="deg", row=2, col=2)
    fig.write_html(str(VELOCITY_HTML), include_plotlyjs=True, full_html=True)


def protocol_markdown(result: dict) -> str:
    spec = result["experiment_spec"]
    lines = [
        "# T6 Dymos Protocol",
        "",
        "## Frozen Framework",
        "",
        f"- Dymos version: `{spec['framework']['versions']['dymos']}`",
        f"- OpenMDAO version: `{spec['framework']['versions']['openmdao']}`",
        f"- transcription: `Radau`, segments `{spec['framework']['num_segments']}`, order `{spec['framework']['transcription_order']}`",
        f"- driver: `ScipyOptimizeDriver` with optimizer `{spec['framework']['optimizer']}`",
        "",
        "## Fairness Rule",
        "",
        "- The Dymos phase, mesh, objective, path constraint, boundary conditions, and optimizer settings are identical between runs.",
        "- Only the initial guess differs: naive Hermite vs Rust-backed matched Bezier warm start.",
        "",
        "## Saved Audit Artifacts",
        "",
        "- `artifacts/paper_artifacts/t6_dymos_experiment.json`",
        "- `artifacts/paper_artifacts/t6_dymos_naive.sql`",
        "- `artifacts/paper_artifacts/t6_dymos_warm.sql`",
        "- `figures/t6_dymos_paths.html`",
        "- `figures/t6_dymos_velocity.html`",
        "- `doc/t6_dymos_validity_memo.md`",
        "",
        "## Robustness Follow-up",
        "",
        "- Predeclared grid only if the single-case experiment is valid but inconclusive.",
        "- Vary phase lag over `20, 30, 40 deg` and transfer time over `1200, 1500, 1800 s`.",
        "- Report paired success counts, median deltas, win/loss/tie counts, and bootstrap intervals if sample count is large enough.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    result = run_dymos_t6_experiment(artifact_dir=ARTIFACT_DIR, config=DymosT6Config())
    RESULT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    VALIDITY_MEMO.write_text(validity_memo_markdown(result), encoding="utf-8")
    PROTOCOL_MEMO.write_text(protocol_markdown(result), encoding="utf-8")
    make_paths_html(result)
    make_velocity_html(result)

    print(
        json.dumps(
            {
                "result_json": str(RESULT_JSON.relative_to(REPO_ROOT)),
                "validity_memo": str(VALIDITY_MEMO.relative_to(REPO_ROOT)),
                "protocol_memo": str(PROTOCOL_MEMO.relative_to(REPO_ROOT)),
                "paths_html": str(PATHS_HTML.relative_to(REPO_ROOT)),
                "velocity_html": str(VELOCITY_HTML.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
