import numpy as np

from orbital_docking.downstream_collocation import (
    _local_frame,
    _pack_variables,
    _prograde_audit_summary,
    _prograde_speed_floor_margins,
    _reference_plane_normal,
    _segment_koz_margins,
    build_bezier_warm_start,
    build_naive_initial_guess,
    make_demo_problem,
    sample_bezier_state_profile,
)
from orbital_docking.optimization import generate_initial_control_points


def test_naive_initial_guess_matches_boundary_states():
    problem = make_demo_problem(n_intervals=6)
    x0 = build_naive_initial_guess(problem)
    n_block = problem.n_nodes * 3

    r = x0[:n_block].reshape(problem.n_nodes, 3)
    v = x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)
    u = x0[2 * n_block :].reshape(problem.n_nodes, 3)

    assert np.allclose(r[0], problem.r0)
    assert np.allclose(r[-1], problem.rf)
    assert np.allclose(v[0], problem.v0)
    assert np.allclose(v[-1], problem.vf)
    assert np.all(np.isfinite(u))


def test_bezier_warm_start_overrides_boundaries_and_populates_controls():
    problem = make_demo_problem(n_intervals=6)
    control_points = generate_initial_control_points(4, problem.r0, problem.rf)
    x0 = build_bezier_warm_start(problem, control_points)
    n_block = problem.n_nodes * 3

    r = x0[:n_block].reshape(problem.n_nodes, 3)
    v = x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)
    u = x0[2 * n_block :].reshape(problem.n_nodes, 3)

    assert np.allclose(r[0], problem.r0)
    assert np.allclose(r[-1], problem.rf)
    assert np.allclose(v[0], problem.v0)
    assert np.allclose(v[-1], problem.vf)
    assert np.all(np.isfinite(u))


def test_bezier_warm_start_can_leave_raw_boundary_mismatch_when_requested():
    problem = make_demo_problem(n_intervals=6)
    control_points = generate_initial_control_points(4, problem.r0, problem.rf)

    raw_x0 = build_bezier_warm_start(
        problem,
        control_points,
        overwrite_boundaries=False,
    )
    fixed_x0 = build_bezier_warm_start(
        problem,
        control_points,
        overwrite_boundaries=True,
    )
    n_block = problem.n_nodes * 3

    raw_v = raw_x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)
    fixed_v = fixed_x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)

    assert not np.allclose(raw_v[0], problem.v0)
    assert not np.allclose(raw_v[-1], problem.vf)
    assert np.allclose(fixed_v[0], problem.v0)
    assert np.allclose(fixed_v[-1], problem.vf)


def test_sampled_bezier_profile_has_expected_shapes():
    problem = make_demo_problem(n_intervals=6)
    control_points = generate_initial_control_points(4, problem.r0, problem.rf)
    r, v, a = sample_bezier_state_profile(problem, control_points)

    assert r.shape == (problem.n_nodes, 3)
    assert v.shape == (problem.n_nodes, 3)
    assert a.shape == (problem.n_nodes, 3)


def test_segment_koz_margins_return_one_value_per_interval():
    problem = make_demo_problem(n_intervals=6)
    x0 = build_naive_initial_guess(problem)
    margins = _segment_koz_margins(x0, problem)

    assert margins.shape == (problem.n_intervals,)
    assert np.all(np.isfinite(margins))


def test_prograde_floor_margin_detects_retrograde_node():
    problem = make_demo_problem(n_intervals=4)
    x0 = build_naive_initial_guess(problem)
    n_block = problem.n_nodes * 3
    r = x0[:n_block].reshape(problem.n_nodes, 3)
    v = x0[n_block : 2 * n_block].reshape(problem.n_nodes, 3)
    u = x0[2 * n_block :].reshape(problem.n_nodes, 3)

    mid = problem.n_nodes // 2
    plane_normal = _reference_plane_normal(problem)
    _r_hat, t_hat = _local_frame(r[mid], plane_normal)
    v[mid] = -2.0 * t_hat

    x_bad = _pack_variables(r, v, u)
    margins = _prograde_speed_floor_margins(x_bad, problem, 0.0)
    summary = _prograde_audit_summary(problem, r, v, floor_ratio=0.0)

    assert margins[mid] < 0.0
    assert summary["retrograde_node_count"] >= 1
