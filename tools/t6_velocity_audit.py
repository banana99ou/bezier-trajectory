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

from orbital_docking import constants  # noqa: E402


ARTIFACT_DIR = REPO_ROOT / "artifacts" / "paper_artifacts"
FIGURE_DIR = REPO_ROOT / "figures"
EVIDENCE_JSON = ARTIFACT_DIR / "t6_evidence_pack.json"
SUMMARY_JSON = ARTIFACT_DIR / "t6_velocity_audit.json"
OUTPUT_HTML = FIGURE_DIR / "t6_velocity_audit.html"


def _load_evidence() -> dict:
    if not EVIDENCE_JSON.exists():
        raise FileNotFoundError(
            f"Missing evidence pack: {EVIDENCE_JSON}. Run tools/t6_evidence_pack.py first."
        )
    return json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))


def _case_arrays(evidence: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "naive_init": (
            np.array(evidence["initial_guesses"]["naive"]["r_km"], dtype=float),
            np.array(evidence["initial_guesses"]["naive"]["v_km_s"], dtype=float),
        ),
        "warm_init": (
            np.array(evidence["initial_guesses"]["bezier_warm_start"]["r_km"], dtype=float),
            np.array(evidence["initial_guesses"]["bezier_warm_start"]["v_km_s"], dtype=float),
        ),
        "naive_final": (
            np.array(evidence["final_runs"]["naive"]["r"], dtype=float),
            np.array(evidence["final_runs"]["naive"]["v"], dtype=float),
        ),
        "warm_final": (
            np.array(evidence["final_runs"]["bezier_warm_start"]["r"], dtype=float),
            np.array(evidence["final_runs"]["bezier_warm_start"]["v"], dtype=float),
        ),
    }


def _reference_plane_normal(evidence: dict) -> np.ndarray:
    r0 = np.array(evidence["scenario"]["start_position_km"], dtype=float)
    v0 = np.array(evidence["scenario"]["start_velocity_km_s"], dtype=float)
    h = np.cross(r0, v0)
    return h / np.linalg.norm(h)


def _audit_case(r: np.ndarray, v: np.ndarray, h_hat: np.ndarray) -> dict:
    radii = np.linalg.norm(r, axis=1)
    speed = np.linalg.norm(v, axis=1)
    r_hat = r / radii[:, None]
    t_hat = np.cross(h_hat[None, :], r_hat)
    t_hat = t_hat / np.linalg.norm(t_hat, axis=1)[:, None]
    v_circ = np.sqrt(constants.EARTH_MU_SCALED / radii)
    v_radial = np.einsum("ij,ij->i", v, r_hat)
    v_prograde = np.einsum("ij,ij->i", v, t_hat)
    speed_ratio = speed / v_circ
    angle_to_prograde_deg = np.degrees(
        np.arccos(
            np.clip(
                np.einsum("ij,ij->i", v / speed[:, None], t_hat),
                -1.0,
                1.0,
            )
        )
    )
    altitude_km = radii - constants.EARTH_RADIUS_KM
    return {
        "radius_km": radii.tolist(),
        "altitude_km": altitude_km.tolist(),
        "speed_km_s": speed.tolist(),
        "circular_speed_km_s": v_circ.tolist(),
        "speed_ratio": speed_ratio.tolist(),
        "radial_velocity_km_s": v_radial.tolist(),
        "prograde_velocity_km_s": v_prograde.tolist(),
        "angle_to_prograde_deg": angle_to_prograde_deg.tolist(),
        "summary": {
            "min_speed_km_s": float(np.min(speed)),
            "max_speed_km_s": float(np.max(speed)),
            "min_circular_speed_km_s": float(np.min(v_circ)),
            "max_circular_speed_km_s": float(np.max(v_circ)),
            "min_speed_ratio": float(np.min(speed_ratio)),
            "max_speed_ratio": float(np.max(speed_ratio)),
            "min_prograde_velocity_km_s": float(np.min(v_prograde)),
            "max_prograde_velocity_km_s": float(np.max(v_prograde)),
            "min_radial_velocity_km_s": float(np.min(v_radial)),
            "max_radial_velocity_km_s": float(np.max(v_radial)),
            "max_angle_to_prograde_deg": float(np.max(angle_to_prograde_deg)),
            "retrograde_node_count": int(np.sum(v_prograde < 0.0)),
        },
    }


def build_audit() -> dict:
    evidence = _load_evidence()
    h_hat = _reference_plane_normal(evidence)
    cases = _case_arrays(evidence)
    audit = {
        "reference": {
            "plane_normal": h_hat.tolist(),
            "definition": "local prograde circular reference uses the start-orbit angular-momentum direction and speed sqrt(mu/r) at the same radius",
        },
        "cases": {},
    }
    for name, (r, v) in cases.items():
        audit["cases"][name] = _audit_case(r, v, h_hat)
    return audit


def write_audit_json(audit: dict) -> None:
    SUMMARY_JSON.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def make_plotly_figure(audit: dict) -> None:
    case_order = ["naive_init", "warm_init", "naive_final", "warm_final"]
    colors = {
        "naive_init": "#1f77b4",
        "warm_init": "#ff7f0e",
        "naive_final": "#2ca02c",
        "warm_final": "#d62728",
    }
    labels = {
        "naive_init": "Naive Init",
        "warm_init": "Warm Init",
        "naive_final": "Naive Final",
        "warm_final": "Warm Final",
    }
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Speed vs Local Circular Speed",
            "Prograde Tangential Velocity",
            "Radial Velocity",
            "Angle to Local Prograde Direction",
        ),
    )

    for case in case_order:
        c = audit["cases"][case]
        x = list(range(len(c["speed_km_s"])))
        color = colors[case]
        label = labels[case]
        fig.add_trace(
            go.Scatter(x=x, y=c["speed_km_s"], mode="lines+markers", name=f"{label} speed", line={"color": color}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=c["circular_speed_km_s"],
                mode="lines",
                name=f"{label} v_circ",
                line={"color": color, "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=c["prograde_velocity_km_s"], mode="lines+markers", name=f"{label} v_pro", line={"color": color}),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=x, y=c["radial_velocity_km_s"], mode="lines+markers", name=f"{label} v_rad", line={"color": color}),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=c["angle_to_prograde_deg"], mode="lines+markers", name=f"{label} angle", line={"color": color}),
            row=2,
            col=2,
        )

    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=1, col=2)
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=90.0, line_dash="dot", line_color="gray", row=2, col=2)

    summary_lines = []
    for case in case_order:
        s = audit["cases"][case]["summary"]
        summary_lines.append(
            f"{labels[case]}: min speed ratio={s['min_speed_ratio']:.3f}, retrograde nodes={s['retrograde_node_count']}, max angle={s['max_angle_to_prograde_deg']:.1f} deg"
        )

    fig.update_layout(
        template="plotly_white",
        title="T6 Velocity Audit Against Local Prograde Circular-Orbit Reference",
        height=900,
        width=1300,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        annotations=list(fig.layout.annotations) + [
            {
                "text": "<br>".join(summary_lines),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.12,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
                "bordercolor": "rgba(0,0,0,0.15)",
                "borderwidth": 1,
                "bgcolor": "rgba(245,245,245,0.9)",
            }
        ],
        margin={"l": 60, "r": 20, "t": 90, "b": 120},
    )
    fig.update_xaxes(title_text="Node index", row=1, col=1)
    fig.update_xaxes(title_text="Node index", row=1, col=2)
    fig.update_xaxes(title_text="Node index", row=2, col=1)
    fig.update_xaxes(title_text="Node index", row=2, col=2)
    fig.update_yaxes(title_text="km/s", row=1, col=1)
    fig.update_yaxes(title_text="km/s", row=1, col=2)
    fig.update_yaxes(title_text="km/s", row=2, col=1)
    fig.update_yaxes(title_text="deg", row=2, col=2)
    fig.write_html(str(OUTPUT_HTML), include_plotlyjs=True, full_html=True)


def main() -> None:
    audit = build_audit()
    write_audit_json(audit)
    make_plotly_figure(audit)
    print(json.dumps(
        {
            "audit_json": str(SUMMARY_JSON.relative_to(REPO_ROOT)),
            "interactive_html": str(OUTPUT_HTML.relative_to(REPO_ROOT)),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
