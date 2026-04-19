"""
Unit tests for the sandbox server's solve handler.

Exercises ``spacetime_bezier.sandbox``'s pure-Python entry points without
spinning up the HTTP server, so the test is fast and deterministic.
"""

from __future__ import annotations

import pytest

from spacetime_bezier import sandbox as spacetime_sandbox


@pytest.fixture
def _require_rust():
    bezier_opt = pytest.importorskip("bezier_opt")
    if not hasattr(bezier_opt, "optimize_spacetime_bezier"):
        pytest.skip("Rust spacetime optimizer is not installed")


def test_scenario_catalog_shape():
    catalog = spacetime_sandbox.scenario_catalog()
    assert "original" in catalog
    entry = catalog["original"]
    for key in ("name", "title", "obstacles", "start", "end", "T", "default_N", "default_n_seg"):
        assert key in entry, f"missing {key} in catalog entry"
    assert isinstance(entry["default_N"], int)
    assert isinstance(entry["default_n_seg"], int)
    assert len(entry["start"]) == len(entry["end"])

    # Obstacles are returned in BezierObstacle wire shape (control_points + radius).
    obstacles = entry["obstacles"]
    assert obstacles, "original scenario should have obstacles"
    for obs in obstacles:
        assert "control_points" in obs and "radius" in obs
        cps = obs["control_points"]
        assert isinstance(cps, list) and len(cps) >= 2
        for cp in cps:
            assert isinstance(cp, list) and len(cp) == len(entry["start"])


def test_solve_from_payload_returns_jsonable_response(_require_rust):
    payload = {
        "scenario_name": "original",
        "N": 8,
        "n_seg": 8,
        "scp_prox_weight": 0.5,
        "scp_trust_radius": 0.0,
        "time_ub_scale": 1.5,
    }
    response = spacetime_sandbox.solve_from_payload(payload)

    assert response["scenario_name"] == "original"
    assert response["N"] == 8 and response["n_seg"] == 8
    cps = response["control_points"]
    assert isinstance(cps, list) and len(cps) == 9  # N+1 control points
    assert all(isinstance(p, list) and len(p) == 3 for p in cps)

    # Response echoes obstacles in the same BezierObstacle shape the client sent
    # (or the preset shape when the client omitted them on first paint).
    assert isinstance(response["obstacles"], list) and response["obstacles"]
    for obs in response["obstacles"]:
        assert "control_points" in obs and "radius" in obs

    info = response["info"]
    assert isinstance(info["feasible"], bool)
    assert isinstance(info["iterations"], int)
    assert "min_clearance" in info
    assert "backend" in info and info["backend"] == "rust"

    # Response must round-trip through json without custom encoders.
    import json
    json.dumps(response)


def test_solve_uses_payload_obstacles_over_preset(_require_rust):
    """When the client sends explicit ``obstacles``, they override the named preset.

    This is the core of Step 1: the client is the source of truth for problem
    state. A preset is only consulted to backfill missing fields on first paint.
    """
    catalog = spacetime_sandbox.scenario_catalog()
    entry = catalog["original"]

    # A brand-new, far-away obstacle that the preset doesn't have. If the server
    # still used the preset, the optimizer would return the preset's answer; with
    # the edited obstacle, the solve is easier and the clearance must be larger.
    edited = [{
        "control_points": [[15.0, 15.0, 0.0], [15.0, 15.0, entry["T"]]],
        "radius": 0.1,
        "name": "solo",
    }]

    payload = {
        "scenario_name": "original",
        "obstacles": edited,
        "start": entry["start"],
        "end": entry["end"],
        "T": entry["T"],
        "N": 6,
        "n_seg": 4,
        "max_iter": 15,
    }
    response = spacetime_sandbox.solve_from_payload(payload)
    assert len(response["obstacles"]) == 1
    assert response["obstacles"][0]["name"] == "solo"
    assert response["info"]["min_clearance"] > 1.0  # far-away obstacle, generous clearance


def test_solve_rejects_payload_with_no_problem_state():
    with pytest.raises(ValueError, match="missing 'obstacles'"):
        spacetime_sandbox.solve_from_payload({
            "scenario_name": "does-not-exist",
            "N": 8,
            "n_seg": 8,
        })
