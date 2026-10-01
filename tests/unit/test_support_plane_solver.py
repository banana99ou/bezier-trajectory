"""Geometric counterexamples and optimizer-status independence for free planes."""
from types import SimpleNamespace

import numpy as np
import pytest

from tools.experiments import support_plane_solver as solver


def tube(radius=0.5):
    return [{"control_points": [[0.0, 0.0], [0.0, 2.0]], "radius": radius}]


def straight(x=2.0):
    return np.array([[x, 0.0], [x, 1.0], [x, 2.0]])


def audit(p, normals=None, obstacles=None, **kwargs):
    return solver.verify(p, np.array([[[1.0, 0.0]]]) if normals is None else normals,
                         tube() if obstacles is None else obstacles, 1,
                         p[0], p[-1], **kwargs)


def test_valid_certificate_is_invariant_to_normal_magnitude():
    p = straight()
    for scale in (1e-4, 1.0, 100.0):
        result = audit(p, np.array([[[scale, 0.0]]]))
        assert result["certified"]
        assert result["minimum_margin"] == pytest.approx(1.5)


def test_large_normal_cannot_hide_a_real_collision():
    # Without unit normalization, 100*.4-.5 would falsely pass.
    result = audit(straight(0.4), np.array([[[100.0, 0.0]]]))
    assert not result["certified"]
    assert result["minimum_margin"] == pytest.approx(-0.1)


@pytest.mark.parametrize("normal", [[[0.0, 0.0]], [[np.nan, 0.0]]])
def test_degenerate_or_nonfinite_normal_is_rejected(normal):
    with pytest.raises(ValueError):
        audit(straight(), np.array([normal]))


def test_curved_obstacle_interior_control_point_cannot_be_omitted():
    # At t=1 the quadratic obstacle centre is (1.5,1): distance .5 < .6.
    # Its endpoints alone suggest a spurious 1.4 separation margin.
    obstacle = [{"control_points": [[0., 0.], [3., 1.], [0., 2.]], "radius": .6}]
    midpoint = (np.array(obstacle[0]["control_points"])*np.array([.25, .5, .25])[:, None]).sum(0)
    assert np.linalg.norm(midpoint-np.array([2., 1.])) < .6
    result = audit(straight(), obstacles=obstacle)
    assert not result["certified"]
    assert result["minimum_margin"] == pytest.approx(-1.6)


def test_clear_endpoints_do_not_certify_a_midpoint_collision():
    p = np.array([[2., 0.], [-2., 1.], [2., 2.]])
    assert (.25*p[0]+.5*p[1]+.25*p[2])[0] == 0.
    result = audit(p)
    assert not result["certified"]


def test_tangency_is_not_positive_separation():
    result = audit(straight(.5))
    assert result["minimum_margin"] == 0.
    assert not result["certified"]


@pytest.mark.parametrize("which", ["time_order", "spatial_bound", "time_bound", "endpoint"])
def test_clear_obstacles_do_not_override_hard_constraints(which):
    p = straight()
    kwargs = {}
    start, end = p[0].copy(), p[-1].copy()
    if which == "time_order":
        p[1, 1] = 2.1
    elif which == "spatial_bound":
        kwargs["coord_bounds"] = [(0., 1.5)]
    elif which == "time_bound":
        kwargs["time_ub"] = 1.5
    else:
        start[0] += 1.
    result = solver.verify(p, np.array([[[1., 0.]]]), tube(), 1, start, end, **kwargs)
    assert result["minimum_margin"] > 0.
    assert result["hard_violation"] > 0.
    assert not result["certified"]


def test_positive_slack_and_optimizer_success_cannot_certify_collision(monkeypatch):
    calls = []
    def fake_minimize(fun, x, **kwargs):
        calls.append(1)
        return SimpleNamespace(x=x.copy(), success=True, nit=1, message="claimed success")
    monkeypatch.setattr(solver, "minimize", fake_minimize)
    result = solver.solve(straight(.25), tube(), 1)
    assert result["phase1_slack"] > 0.
    assert result["status"] == "uncertified_candidate"
    assert not result["certificate"]["certified"]
    assert len(calls) == 1


@pytest.mark.parametrize("invalid", ["collision", "zero_normal", "nan_control", "clearance_shortfall"])
def test_verified_phase1_incumbent_survives_invalid_refinement(monkeypatch, invalid):
    p = straight()
    calls = []
    def fake_minimize(fun, x, **kwargs):
        z = x.copy()
        if calls:
            if invalid == "collision":
                z[2] = -2.  # middle trajectory control x
            elif invalid == "zero_normal":
                z[p.size:p.size+2] = 0.
            elif invalid == "clearance_shortfall":
                # Still collision-free, but violates the requested .0001 margin.
                z[2] = .50005
            else:
                z[2] = np.nan
        calls.append(1)
        return SimpleNamespace(x=z, success=True, nit=1, message="claimed success")
    monkeypatch.setattr(solver, "minimize", fake_minimize)
    result = solver.solve(p, tube(), 1)
    assert len(calls) == 2
    assert result["status"] == "certified_feasible"
    assert not result["phase2_certificate"]["certified"]
    np.testing.assert_array_equal(result["control_points"], p)
    assert result["certificate"]["minimum_margin"] == pytest.approx(1.5)


@pytest.mark.parametrize("obstacle", [
    {"control_points": [[0., 0.], [0., 2.]], "radius": -1.},
    {"control_points": [[0., 0.], [np.nan, 2.]], "radius": .5},
])
def test_verifier_rejects_invalid_obstacle_geometry(obstacle):
    with pytest.raises(ValueError):
        audit(straight(), obstacles=[obstacle])


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_nlp_jacobians_match_independent_finite_differences(monkeypatch, dim):
    # Exercise subdivision, padded obstacle vertices, slack and normal columns.
    p = np.zeros((4, dim))
    p[:, 0] = [2., 2.2, 1.8, 2.]
    p[:, -1] = np.linspace(0., 3., 4)
    controls = np.zeros((3, dim))
    controls[:, -1] = [0., 1.5, 3.]
    obstacles = [{"control_points": controls.tolist(), "radius": .25},
                 {"control_points": [controls[1].tolist()], "radius": .1}]
    calls = []
    def checking_minimize(fun, x, jac, constraints, **kwargs):
        h = 1e-5
        for constraint in constraints[:2]:
            analytic = constraint["jac"](x)
            numeric = np.empty_like(analytic)
            for j in range(len(x)):
                plus, minus = x.copy(), x.copy()
                plus[j] += h
                minus[j] -= h
                numeric[:, j] = (constraint["fun"](plus)-constraint["fun"](minus))/(2*h)
            np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-7)
        numeric_objective = np.empty_like(x)
        for j in range(len(x)):
            plus, minus = x.copy(), x.copy()
            plus[j] += h
            minus[j] -= h
            numeric_objective[j] = (fun(plus)-fun(minus))/(2*h)
        np.testing.assert_allclose(jac(x), numeric_objective, atol=1e-7, rtol=1e-7)
        calls.append(1)
        return SimpleNamespace(x=x.copy(), success=True, nit=1, message="Jacobian checked")
    monkeypatch.setattr(solver, "minimize", checking_minimize)
    result = solver.solve(p, obstacles, 2)
    assert len(calls) == 2
    assert result["certificate"]["certified"]


def _local_result(certified):
    return {"status": "certified_feasible" if certified else "uncertified_candidate",
            "certificate": {"certified": certified},
            "phase1_iterations": 1, "phase2_iterations": int(certified)}


def test_timing_retries_stop_immediately_for_certified_requested_seed(monkeypatch):
    calls = []
    def fake_solve(p, *args, **kwargs):
        calls.append(np.array(p))
        return _local_result(True)
    monkeypatch.setattr(solver, "solve", fake_solve)
    p = straight()
    result = solver.solve_with_timing_restarts(p, tube(), 1)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], p)
    assert result["selected_seed"] == "requested"
    assert [a["seed"] for a in result["attempts"]] == ["requested"]


def test_timing_retries_record_changed_seeds_and_preserve_endpoints(monkeypatch):
    calls = []
    def fake_solve(p, *args, **kwargs):
        calls.append(np.array(p).copy())
        return _local_result(len(calls) == 3)
    monkeypatch.setattr(solver, "solve", fake_solve)
    p = np.column_stack([np.linspace(2., 8., 5), [0., .1, .4, 1.5, 3.]])
    result = solver.solve_with_timing_restarts(p, tube(), 1)
    assert len(calls) == 3
    assert result["selected_seed"] == "wait_power_3"
    assert [a["seed"] for a in result["attempts"]] == ["requested", "wait_power_2", "wait_power_3"]
    for attempt, actual in zip(result["attempts"], calls):
        np.testing.assert_array_equal(attempt["initial_controls"], actual)
        np.testing.assert_array_equal(actual[[0, -1]], p[[0, -1]])
    assert not np.array_equal(calls[0], calls[1])
    assert not np.array_equal(calls[1], calls[2])
    # Geometry-level property: waiting seeds have less midpoint chord progress
    # than the original straight spatial seed, independently of construction code.
    weights = np.array([1., 4., 6., 4., 1.])/16.
    midpoints = [weights @ seed[:, 0] for seed in calls]
    assert midpoints[2] < midpoints[1] < midpoints[0]


def test_timing_portfolio_never_turns_claimed_optimizer_success_into_certificate(monkeypatch):
    calls = []
    def fake_minimize(fun, x, **kwargs):
        calls.append(1)
        return SimpleNamespace(x=x.copy(), success=True, nit=1, message="claimed success")
    monkeypatch.setattr(solver, "minimize", fake_minimize)
    p = np.column_stack([np.linspace(2., 8., 4), np.linspace(0., 3., 4)])
    obstacle = [{"control_points": [[0., 0.], [0., 3.]], "radius": 10.}]
    # The fixed start is inside the tube, so no retry can repair this problem.
    assert np.linalg.norm(p[0]-np.array([0., 0.])) < 10.
    result = solver.solve_with_timing_restarts(p, obstacle, 1)
    assert len(calls) == 4
    assert result["status"] == "uncertified_candidate"
    assert not result["certificate"]["certified"]
    assert result["selected_seed"] == "hurry_sqrt_coefficients"
    assert all(not attempt["certificate"]["certified"] for attempt in result["attempts"])
