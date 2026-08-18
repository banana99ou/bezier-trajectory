"""
Unit tests for the sandbox's trace/diagnosis plumbing.

Covers the Phase 3 contract: every ``/api/solve`` emits a ``trace_id`` that
resolves via ``get_trace(trace_id)`` to the full per-stage frame list, plus a
one-line ``diagnosis`` that is ``None`` on feasible solves and a human-readable
string on infeasible ones. The cache is single-slot; a new solve evicts the
previous entry.
"""

from __future__ import annotations

import json

import pytest

from spacetime_bezier import sandbox as spacetime_sandbox


@pytest.fixture
def _require_rust():
    bezier_opt = pytest.importorskip("bezier_opt")
    if not hasattr(bezier_opt, "SpacetimeScpContext"):
        pytest.skip("Rust SpacetimeScpContext not installed")


def _base_payload() -> dict:
    return {
        "scenario_name": "original",
        "N": 6,
        "n_seg": 4,
        "max_iter": 10,
        "scp_prox_weight": 0.5,
        "scp_trust_radius": 0.0,
        "time_ub_scale": 1.5,
    }


def test_solve_response_carries_trace_id_and_resolves(_require_rust):
    response = spacetime_sandbox.solve_from_payload(_base_payload())

    assert isinstance(response["trace_id"], str) and response["trace_id"]
    assert isinstance(response["trace_frame_count"], int) and response["trace_frame_count"] > 0

    frames = spacetime_sandbox.get_trace(response["trace_id"])
    assert isinstance(frames, list)
    assert len(frames) == response["trace_frame_count"]

    # Frames must round-trip through json (drawer consumes them over HTTP).
    json.dumps(frames)

    # Every frame carries the minimum shape the drawer depends on.
    for frame in frames:
        assert "stage" in frame and "label" in frame
        assert "iteration" in frame  # None allowed for init/finalize
        assert "payload" in frame and isinstance(frame["payload"], dict)

    # First and last frames match the canonical stepper pipeline.
    assert frames[0]["stage"] == "init-guess"
    assert frames[-1]["stage"] == "finalize"


def test_feasible_solve_has_no_diagnosis(_require_rust):
    response = spacetime_sandbox.solve_from_payload(_base_payload())
    assert response["info"]["feasible"] is True
    assert response["diagnosis"] is None


def test_infeasible_solve_produces_diagnosis_string(_require_rust):
    # Squeeze the problem so no feasible path exists: huge obstacle blocking the
    # straight line between start and end, tight trust region, few iterations.
    payload = {
        "scenario_name": "original",
        "obstacles": [
            {
                "control_points": [[5.0, 5.0, 0.0], [5.0, 5.0, 10.0]],
                "radius": 6.0,  # swallows the whole workspace
                "name": "wall",
            }
        ],
        "start": [0.5, 1.0, 0.0],
        "end": [8.5, 8.5, 10.0],
        "T": 10.0,
        "N": 6,
        "n_seg": 4,
        "max_iter": 5,
        "scp_prox_weight": 0.5,
        "scp_trust_radius": 0.0,
    }

    response = spacetime_sandbox.solve_from_payload(payload)

    assert response["info"]["feasible"] is False
    diagnosis = response["diagnosis"]
    assert isinstance(diagnosis, str) and diagnosis.startswith("infeasible")
    # The one-line summary must name the obstacle, segment, iteration, and cp.
    assert "wall" in diagnosis
    assert "segment" in diagnosis
    assert "iteration" in diagnosis
    assert "cp=" in diagnosis


def test_trace_cache_is_single_slot(_require_rust):
    first = spacetime_sandbox.solve_from_payload(_base_payload())
    second = spacetime_sandbox.solve_from_payload(_base_payload())

    # The second solve must evict the first: old trace_id returns None, the
    # newest trace_id resolves. This matches the drawer's expectation that the
    # latest /api/solve response is always the source of truth.
    assert first["trace_id"] != second["trace_id"]
    assert spacetime_sandbox.get_trace(first["trace_id"]) is None
    assert spacetime_sandbox.get_trace(second["trace_id"]) is not None


def test_get_trace_returns_none_for_unknown_id():
    assert spacetime_sandbox.get_trace("does-not-exist") is None
    assert spacetime_sandbox.get_trace("") is None
