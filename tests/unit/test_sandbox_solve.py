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

    # Response must round-trip as *strict* JSON. Plain `json.dumps(response)`
    # was the assertion here and it could not fail: allow_nan defaults to True,
    # so it emits the bare tokens NaN/Infinity/-Infinity, which are not JSON and
    # which every browser's JSON.parse rejects. It passed for months while the
    # page died on them. allow_nan=False is what makes this check able to fail.
    import json
    json.dumps(response, allow_nan=False)


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


def _reject_non_json_constants(token):
    raise ValueError(f"non-JSON token {token!r}")


def test_json_safe_replaces_non_finite_floats():
    """Non-finite floats become null, at every depth."""
    import json
    import math

    payload = {
        "inf": math.inf,
        "neg_inf": -math.inf,
        "nan": math.nan,
        "ok": 1.5,
        "nested": {"deep": [math.inf, 2.0]},
    }
    safe = spacetime_sandbox._json_safe(payload)
    assert safe["inf"] is None and safe["neg_inf"] is None and safe["nan"] is None
    assert safe["ok"] == 1.5
    assert safe["nested"]["deep"] == [None, 2.0]

    # The point of the exercise: strict serialization now succeeds.
    json.dumps(safe, allow_nan=False)


def test_infeasible_solve_response_is_strict_json(_require_rust):
    """A scenario where no iterate is ever feasible must still emit valid JSON.

    `wall` leaves ``best_clearance`` at its -inf sentinel. That reached the wire
    as the literal ``-Infinity``, so the browser's JSON.parse rejected the whole
    document and the scenario was unusable in the UI ("solve failed: bad JSON")
    -- while every Python-side check passed, because json.loads accepts those
    tokens by default. This test parses the way a browser does: parse_constant
    fires on NaN/Infinity/-Infinity, so it fails if one is ever emitted again.
    """
    import json

    response = spacetime_sandbox.solve_from_payload({
        "scenario_name": "wall",
        "N": 8,
        "n_seg": 2,
        "max_iter": 3,
    })
    assert response["info"]["feasible"] is False, "wall is expected to be infeasible"

    encoded = json.dumps(spacetime_sandbox._json_safe(response), allow_nan=False)
    json.loads(encoded, parse_constant=_reject_non_json_constants)
