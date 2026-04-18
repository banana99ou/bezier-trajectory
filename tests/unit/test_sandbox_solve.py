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

    info = response["info"]
    assert isinstance(info["feasible"], bool)
    assert isinstance(info["iterations"], int)
    assert "min_clearance" in info
    assert "backend" in info and info["backend"] == "rust"

    # Response must round-trip through json without custom encoders.
    import json
    json.dumps(response)


def test_solve_cache_reuses_identical_payload(_require_rust):
    spacetime_sandbox.clear_solve_cache()
    payload = {
        "scenario_name": "original",
        "N": 6,
        "n_seg": 4,
        "scp_prox_weight": 0.5,
        "scp_trust_radius": 0.0,
        "time_ub_scale": 1.5,
        "capsule_time_scale": 0.5,
        "max_iter": 10,
        "tol": 1e-6,
        "min_dt": 0.1,
    }
    first = spacetime_sandbox.solve_from_payload(payload)
    assert first["cached"] is False
    second = spacetime_sandbox.solve_from_payload(payload)
    assert second["cached"] is True
    assert second["control_points"] == first["control_points"]

    # Any change in a solve-affecting parameter must miss the cache.
    changed = spacetime_sandbox.solve_from_payload({**payload, "N": 8})
    assert changed["cached"] is False


def test_solve_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenario"):
        spacetime_sandbox.solve_from_payload({
            "scenario_name": "does-not-exist",
            "N": 8,
            "n_seg": 8,
        })
