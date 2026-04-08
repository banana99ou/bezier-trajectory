"""
Integration tests for the space-time Bezier optimizer package.
"""

import argparse
from pathlib import Path

import numpy as np
import pytest

import spacetime_bezier.io as spacetime_io
from spacetime_bezier.io import load_outputs, save_outputs
from spacetime_bezier.optimize import compute_min_clearance, optimize_scenario, optimize_spacetime


def _toy_scenario() -> dict:
    return {
        "name": "toy",
        "title": "Toy Scenario",
        "init_curve": {
            "mode": "quadratic_bow",
            "bow": 2.0,
            "side": 1.0,
            "workspace_center": [5.0, 5.0],
        },
        "obstacles": [
            {"pos0": [4.5, 4.0], "vel": [0.0, 0.0], "r": 0.7, "name": "O1"},
        ],
        "start": [0.5, 1.0, 0.0],
        "end": [8.5, 8.5, 10.0],
        "T": 10.0,
    }


def test_optimize_spacetime_python_keeps_time_monotone():
    obstacles = [{"pos0": [20.0, 20.0], "vel": [0.0, 0.0], "r": 1.0}]
    P_opt = optimize_spacetime(
        N=4,
        dim=3,
        p_start=[0.5, 1.0, 0.0],
        p_end=[8.5, 8.5, 10.0],
        obstacles=obstacles,
        n_seg=4,
        max_iter=10,
        tol=1e-6,
        scp_prox_weight=0.3,
        min_dt=0.1,
        verbose=False,
        backend="python",
        init_curve={"mode": "quadratic_bow", "bow": 2.0, "side": 1.0, "workspace_center": [5.0, 5.0]},
    )

    assert P_opt.shape == (5, 3)
    assert np.all(np.diff(P_opt[:, -1]) >= 0.1 - 1e-9)
    assert compute_min_clearance(P_opt, obstacles, dim=3, n_eval=400) > 0.0


def test_toy_scenario_small_config_is_feasible_with_python_backend():
    output = optimize_scenario(
        _toy_scenario(),
        configs=[(4, 4)],
        backend="python",
        max_iter=8,
        scp_prox_weight=0.3,
        verbose=False,
    )

    assert output["best"] == "N4_seg4"
    best = output["results"][output["best"]]
    assert best["feasible"] is True
    assert best["min_clearance"] > 0.0
    control_points = np.array(best["control_points"], dtype=float)
    assert np.all(np.diff(control_points[:, -1]) >= 0.1 - 1e-9)
    assert output["name"] == "toy"
    assert set(output.keys()) == {"name", "title", "best", "obstacles", "start", "end", "T", "init_curve", "results"}


def test_toy_scenario_small_config_is_feasible_with_rust_backend():
    bezier_opt = pytest.importorskip("bezier_opt")
    if not hasattr(bezier_opt, "optimize_spacetime_bezier"):
        pytest.skip("Rust spacetime optimizer is not installed")

    output = optimize_scenario(
        _toy_scenario(),
        configs=[(4, 4)],
        backend="rust",
        max_iter=8,
        scp_prox_weight=0.3,
        verbose=False,
    )

    best = output["results"][output["best"]]
    assert best["feasible"] is True
    assert best["min_clearance"] > 0.0
    control_points = np.array(best["control_points"], dtype=float)
    assert np.all(np.diff(control_points[:, -1]) >= 0.1 - 1e-9)


def test_spacetime_output_json_round_trip(tmp_path: Path):
    payload = {
        "toy": {
            "name": "toy",
            "title": "Toy Scenario",
            "best": "N4_seg4",
            "obstacles": [],
            "start": [0.5, 1.0, 0.0],
            "end": [8.5, 8.5, 10.0],
            "T": 10.0,
            "init_curve": {"mode": "straight"},
            "results": {
                "N4_seg4": {
                    "N": 4,
                    "n_seg": 4,
                    "control_points": [[0.5, 1.0, 0.0], [8.5, 8.5, 10.0]],
                    "min_clearance": 0.25,
                    "feasible": True,
                }
            },
        }
    }
    out_path = tmp_path / "spacetime_scenarios.json"

    saved_path = save_outputs(payload, out_path)
    loaded = load_outputs(saved_path)

    assert saved_path.exists()
    assert "toy" in loaded
    assert loaded["toy"]["best"] == "N4_seg4"
    assert loaded["toy"]["results"]["N4_seg4"]["N"] == 4
    assert loaded["toy"]["results"]["N4_seg4"]["n_seg"] == 4


def test_main_with_degree_override_runs_fixed_segment_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_optimize_scenarios(
        scenario_names,
        scenario_map,
        existing_outputs=None,
        backend="auto",
        max_iter=200,
        tol=1e-6,
        scp_prox_weight=0.3,
        scp_trust_radius=0.0,
        min_dt=0.1,
        verbose=True,
    ):
        captured["scenario_names"] = list(scenario_names)
        captured["configs"] = scenario_map["original"][1]
        return {
            "original": {
                "title": "Original",
                "best": "N6_seg2",
                "results": {
                    "N6_seg2": {
                        "min_clearance": 1.0,
                        "control_points": [[0.0, 0.0, 0.0]],
                    }
                },
            }
        }

    monkeypatch.setattr(spacetime_io, "optimize_scenarios", fake_optimize_scenarios)

    spacetime_io.main(
        [
            "original",
            "-N",
            "6",
            "8",
            "--output",
            str(tmp_path / "spacetime_scenarios.json"),
            "--no-open",
        ]
    )

    assert captured["scenario_names"] == ["original"]
    assert captured["configs"] == [
        (6, 2),
        (6, 8),
        (6, 32),
        (6, 64),
        (8, 2),
        (8, 8),
        (8, 32),
        (8, 64),
    ]


def test_degree_override_parser_requires_positive_integer():
    parser = spacetime_io.build_arg_parser()

    args = parser.parse_args(spacetime_io._normalize_degree_args(["original", "-N", "17", "--no-open"]))
    assert args.N == [17]

    args = parser.parse_args(
        spacetime_io._normalize_degree_args(["original", "-N", "6", "8", "10", "12", "--no-open"])
    )
    assert args.N == [6, 8, 10, 12]

    with pytest.raises(argparse.ArgumentTypeError):
        spacetime_io._normalize_degree_args(["original", "-N", "0", "--no-open"])

    with pytest.raises(argparse.ArgumentTypeError):
        spacetime_io._normalize_degree_args(["original", "-N", "-3", "--no-open"])
